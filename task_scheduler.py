"""
Task Scheduler for LLM Processing
Handles scheduled tasks for processing patient-trial matching using LLM
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
import sys
import os
from asyncio.subprocess import PIPE

from config import (
    SCHEDULER_INTERVAL_MINUTES,
    MAX_CONCURRENT_TASKS,
    TASK_TIMEOUT_SECONDS,
    USE_LLM_PROCESSING,
    USE_DATABASE,
    ENABLE_FIXED_IDS,
    FIXED_TRIAL_IDS,
    FIXED_PATIENT_IDS,
    RUN_JOBS_ON_START
)

# Database services will be imported lazily in initialize() to avoid import errors when optional

class TaskScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': AsyncIOExecutor()},
            job_defaults={'coalesce': True, 'max_instances': 1}
        )
        self.running = False
        
        # Initialize services
        self.patient_db = None
        self.trial_db = None
        
        # Locks to prevent overlapping job executions
        self._patient_job_lock = asyncio.Lock()
        self._trial_job_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize all services and database connections"""
        logger.info("Initializing Task Scheduler services...")
        
        try:
            # Initialize database services
            if USE_DATABASE:
                try:
                    from database.patient_db import PatientDB  # type: ignore
                    from database.trial_database_service import TrialDatabaseService  # type: ignore
                    self.patient_db = PatientDB()
                    self.trial_db = TrialDatabaseService()
                    await self.patient_db.initialize()
                    await self.trial_db.initialize()
                    logger.info("Database services initialized")
                except ModuleNotFoundError:
                    logger.warning("Database service modules not found; proceeding without database integration")
                    self.patient_db = None
                    self.trial_db = None
            
            # Initialize LLM services
            if USE_LLM_PROCESSING:
                logger.info("LLM processing enabled - using modular pipeline services")
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
            
    async def start(self):
        """Start the task scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        logger.info("Starting task scheduler...")
        
        # Add scheduled jobs
        self._add_scheduled_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        self.running = True
        
        logger.info(f"Task scheduler started with {len(self.scheduler.get_jobs())} jobs")
        
        # Optionally kick off both jobs once immediately
        if RUN_JOBS_ON_START:
            logger.info("Running initial trial and patient jobs immediately on start")
            try:
                await self.process_trial_patient_matches()
            except Exception as e:
                logger.error(f"Initial trial job failed: {e}")
            try:
                await self.process_patient_trial_matches()
            except Exception as e:
                logger.error(f"Initial patient job failed: {e}")
        
        # Keep the scheduler running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            
    async def stop(self):
        """Stop the task scheduler"""
        if not self.running:
            return
            
        logger.info("Stopping task scheduler...")
        self.running = False
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            
        logger.info("Task scheduler stopped")
        
    def _add_scheduled_jobs(self):
        """Add all scheduled jobs to the scheduler"""
        
        # Process pending patient-trial matches every 30 minutes
        self.scheduler.add_job(
            self.process_patient_trial_matches,
            trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MINUTES),
            id='process_patient_trial_matches',
            name='Process Patient-Trial Matches',
            replace_existing=True
        )
        
        # Process pending trial-patient matches every 30 minutes
        self.scheduler.add_job(
            self.process_trial_patient_matches,
            trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MINUTES),
            id='process_trial_patient_matches',
            name='Process Trial-Patient Matches',
            replace_existing=True
        )
        
        # Update trial eligibility assessments every hour
        self.scheduler.add_job(
            self.update_trial_eligibility,
            trigger=CronTrigger(minute=0),  # Every hour at minute 0
            id='update_trial_eligibility',
            name='Update Trial Eligibility Assessments',
            replace_existing=True
        )
        
        # Clean up old results every day at 2 AM
        self.scheduler.add_job(
            self.cleanup_old_results,
            trigger=CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id='cleanup_old_results',
            name='Cleanup Old Results',
            replace_existing=True
        )
        
        logger.info(f"Added {len(self.scheduler.get_jobs())} scheduled jobs")
        
    async def process_patient_trial_matches(self):
        """Process pending patient-to-trial matching tasks"""
        logger.info("Starting patient-trial matching task...")
        
        try:
            async with self._patient_job_lock:
                # Only check database if not using fixed IDs and database is required
                if not ENABLE_FIXED_IDS and USE_DATABASE and not self.patient_db:
                    logger.warning("Required database services not initialized")
                    return
                
                # Get pending patient IDs that need trial matching
                pending_patients = await self._get_pending_patient_trial_tasks()
            
                if not pending_patients:
                    logger.info("No pending patient-trial tasks found")
                    return
                
                logger.info(f"Processing {len(pending_patients)} patient-trial tasks")
            
                # Process each patient sequentially
                for patient_id in pending_patients:
                    try:
                        await self._process_single_patient_trial_match(patient_id)
                    except Exception as e:
                        logger.error(f"Failed to process patient {patient_id}: {e}")
                        continue
                    
                logger.info("Patient-trial matching task completed")
            
        except Exception as e:
            logger.error(f"Error in patient-trial matching task: {e}")
            
    async def process_trial_patient_matches(self):
        """Process pending trial-to-patient matching tasks"""
        logger.info("Starting trial-patient matching task...")
        
        try:
            async with self._trial_job_lock:
                # Only check database if not using fixed IDs and database is required
                if not ENABLE_FIXED_IDS and USE_DATABASE and not self.trial_db:
                    logger.warning("Required database services not initialized")
                    return
                
                # Get pending trial IDs that need patient matching
                pending_trials = await self._get_pending_trial_patient_tasks()
            
                if not pending_trials:
                    logger.info("No pending trial-patient tasks found")
                    return
                
                logger.info(f"Processing {len(pending_trials)} trial-patient tasks")
            
                # Process each trial sequentially
                for trial_id in pending_trials:
                    try:
                        await self._process_single_trial_patient_match(trial_id)
                    except Exception as e:
                        logger.error(f"Failed to process trial {trial_id}: {e}")
                        continue
                    
                logger.info("Trial-patient matching task completed")
            
        except Exception as e:
            logger.error(f"Error in trial-patient matching task: {e}")
            
    async def update_trial_eligibility(self):
        """Update trial eligibility assessments"""
        logger.info("Starting trial eligibility update task...")
        
        try:
            if not self.trial_db:
                logger.warning("Required database services not initialized")
                return
                
            # Get trials that need eligibility updates
            trials_to_update = await self._get_trials_needing_eligibility_update()
            
            if not trials_to_update:
                logger.info("No trials need eligibility updates")
                return
                
            logger.info(f"Updating eligibility for {len(trials_to_update)} trials")
            
            # Process each trial
            for trial_id in trials_to_update:
                try:
                    await self._update_single_trial_eligibility(trial_id)
                except Exception as e:
                    logger.error(f"Failed to update trial {trial_id} eligibility: {e}")
                    continue
                    
            logger.info("Trial eligibility update task completed")
            
        except Exception as e:
            logger.error(f"Error in trial eligibility update task: {e}")
            
    async def cleanup_old_results(self):
        """Clean up old results and temporary data"""
        logger.info("Starting cleanup task...")
        
        try:
            # Clean up results older than 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            cleaned_count = await self._cleanup_old_data(cutoff_date)
            
            logger.info(f"Cleanup completed: {cleaned_count} old records removed")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            
    # Helper methods for database operations
    async def _get_pending_patient_trial_tasks(self) -> List[str]:
        """Get list of patient IDs that need trial matching"""
        if ENABLE_FIXED_IDS:
            return list(FIXED_PATIENT_IDS)
        # Placeholder for database-backed retrieval
        return []
        
    async def _get_pending_trial_patient_tasks(self) -> List[str]:
        """Get list of trial IDs that need patient matching"""
        if ENABLE_FIXED_IDS:
            return list(FIXED_TRIAL_IDS)
        # Placeholder for database-backed retrieval
        return []
        
    async def _get_trials_needing_eligibility_update(self) -> List[str]:
        """Get list of trial IDs that need eligibility updates"""
        # This would query your database for trials needing updates
        # For now, return empty list as placeholder
        return []
        
    async def _process_single_patient_trial_match(self, patient_id: str):
        """Process a single patient-trial matching task"""
        logger.info(f"Processing patient-trial match for patient {patient_id}")
        
        # Execute: python patient_to_trial_pipeline.py --patient-id <id>
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'patient_to_trial_pipeline.py'),
            '--patient-id', str(patient_id)
        ]
        await self._run_pipeline_cmd(cmd, context={"patient_id": patient_id})
        
    async def _process_single_trial_patient_match(self, trial_id: str):
        """Process a single trial-patient matching task"""
        logger.info(f"Processing trial-patient match for trial {trial_id}")
        
        # Execute: python trial_to_patient_pipeline.py --trial-id <id>
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'trial_to_patient_pipeline.py'),
            '--trial-id', str(trial_id)
        ]
        await self._run_pipeline_cmd(cmd, context={"trial_id": trial_id})
        
    async def _update_single_trial_eligibility(self, trial_id: str):
        """Update eligibility for a single trial"""
        logger.info(f"Updating eligibility for trial {trial_id}")
        
        # Implement the actual eligibility update logic here
        # This would use the eligibility_service to update the trial
        
    async def _cleanup_old_data(self, cutoff_date: datetime) -> int:
        """Clean up old data older than cutoff_date"""
        # Implement cleanup logic here
        # This would remove old results, temporary files, etc.
        return 0
        
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "running": self.running,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                }
                for job in self.scheduler.get_jobs()
            ],
            "services_initialized": {
                "patient_db": self.patient_db is not None,
                "trial_db": self.trial_db is not None
            }
        }

    async def _run_pipeline_cmd(self, cmd: List[str], context: Optional[Dict[str, Any]] = None) -> None:
        """Run a pipeline command with timeout and detailed logging, streaming output in real-time."""
        context = context or {}
        cmd_str = ' '.join(cmd)
        logger.info(f"Running command: {cmd_str} | context={context}")
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=PIPE, 
                stderr=PIPE,
                bufsize=0  # Unbuffered
            )
            
            async def stream_output(stream, prefix="OUT"):
                """Stream output from subprocess in real-time."""
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded = line.decode(errors='ignore').rstrip()
                        if decoded:  # Only log non-empty lines
                            logger.info(f"[{prefix}] {decoded}")
                except Exception as e:
                    logger.error(f"Error reading {prefix} stream: {e}")
            
            # Start streaming stdout and stderr concurrently
            stdout_task = asyncio.create_task(stream_output(process.stdout, "STDOUT"))
            stderr_task = asyncio.create_task(stream_output(process.stderr, "STDERR"))
            
            try:
                # Wait for process to complete with timeout
                returncode = await asyncio.wait_for(process.wait(), timeout=TASK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error(f"Command timed out after {TASK_TIMEOUT_SECONDS}s: {cmd_str} | context={context}")
                process.kill()
                await process.wait()
                stdout_task.cancel()
                stderr_task.cancel()
                return
            
            # Wait for output streaming to complete
            await stdout_task
            await stderr_task
            
            duration = time.time() - start_time
            if returncode == 0:
                logger.info(f"✓ Command succeeded in {duration:.2f}s | context={context}")
            else:
                logger.error(f"✗ Command failed (exit {returncode}) in {duration:.2f}s | context={context}")
                
        except Exception as e:
            logger.error(f"Failed to execute command: {cmd_str} | context={context} | error={e}")


async def _main():
    scheduler = TaskScheduler()
    await scheduler.initialize()
    await scheduler.start()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
