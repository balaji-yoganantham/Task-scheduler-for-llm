"""
Automatic Evaluation Monitor
Continuously monitors database for unevaluated patients and trials,
and automatically processes them through their respective pipelines.
Runs independently from task_scheduler.py
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Set, Optional
from loguru import logger
from asyncio.subprocess import PIPE

from config import (
    AUTO_EVAL_POLL_INTERVAL,
    AUTO_EVAL_PATIENT_BATCH_SIZE,
    AUTO_EVAL_TRIAL_BATCH_SIZE,
    AUTO_EVAL_ENABLE_PATIENTS,
    AUTO_EVAL_ENABLE_TRIALS,
    AUTO_EVAL_PATIENT_TIMEOUT,
    AUTO_EVAL_TRIAL_TIMEOUT,
    USE_DATABASE
)
from services.shared.database_utils import DatabaseUtils


class AutoEvaluationMonitor:
    """Monitors and automatically processes unevaluated patients and trials"""
    
    def __init__(self):
        self.db_utils: Optional[DatabaseUtils] = None
        self.running = False
        self.processed_patients: Set[int] = set()  # Track patients being processed
        self.processed_trials: Set[str] = set()  # Track trials being processed
        self._patient_lock = asyncio.Lock()
        self._trial_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize database connection"""
        logger.info("Initializing Auto Evaluation Monitor...")
        
        if not USE_DATABASE:
            logger.error("USE_DATABASE is False - cannot run monitor without database")
            raise ValueError("Database must be enabled for auto evaluation monitor")
        
        try:
            self.db_utils = DatabaseUtils()
            logger.info("✅ DatabaseUtils initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DatabaseUtils: {e}")
            raise
    
    async def start(self):
        """Start the monitoring loop"""
        if self.running:
            logger.warning("Monitor is already running")
            return
        
        logger.info("=" * 80)
        logger.info("🚀 Starting Auto Evaluation Monitor")
        logger.info("=" * 80)
        logger.info(f"📊 Configuration:")
        logger.info(f"   - Polling Interval: {AUTO_EVAL_POLL_INTERVAL} seconds")
        logger.info(f"   - Patient Processing: {'ENABLED' if AUTO_EVAL_ENABLE_PATIENTS else 'DISABLED'}")
        logger.info(f"   - Trial Processing: {'ENABLED' if AUTO_EVAL_ENABLE_TRIALS else 'DISABLED'}")
        logger.info(f"   - Patient Batch Size: {AUTO_EVAL_PATIENT_BATCH_SIZE}")
        logger.info(f"   - Trial Batch Size: {AUTO_EVAL_TRIAL_BATCH_SIZE}")
        logger.info(f"   - Patient Timeout: {AUTO_EVAL_PATIENT_TIMEOUT}s")
        logger.info(f"   - Trial Timeout: {AUTO_EVAL_TRIAL_TIMEOUT}s")
        logger.info("=" * 80)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # Check and process unevaluated items
                    await self._check_and_process()
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                
                # Wait before next check
                await asyncio.sleep(AUTO_EVAL_POLL_INTERVAL)
                
        except asyncio.CancelledError:
            logger.info("Monitor loop cancelled")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt - shutting down...")
        finally:
            self.running = False
            logger.info("Auto Evaluation Monitor stopped")
    
    async def stop(self):
        """Stop the monitoring loop"""
        logger.info("Stopping Auto Evaluation Monitor...")
        self.running = False
    
    async def _check_and_process(self):
        """Check for unevaluated items and process them"""
        if not self.db_utils:
            logger.error("Database utils not initialized")
            return
        
        # Process patients and trials concurrently
        tasks = []
        
        if AUTO_EVAL_ENABLE_PATIENTS:
            tasks.append(self._process_unevaluated_patients())
        
        if AUTO_EVAL_ENABLE_TRIALS:
            tasks.append(self._process_unevaluated_trials())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_unevaluated_patients(self):
        """Process unevaluated patients through patient-to-trial pipeline"""
        try:
            # Get unevaluated patient IDs
            patient_ids = self.db_utils.get_unevaluated_patient_ids(limit=AUTO_EVAL_PATIENT_BATCH_SIZE)
            
            if not patient_ids:
                return
            
            # Filter out patients already being processed
            async with self._patient_lock:
                new_patients = [pid for pid in patient_ids if pid not in self.processed_patients]
                if not new_patients:
                    return
                
                # Mark as being processed
                self.processed_patients.update(new_patients)
            
            logger.info(f"📋 Found {len(new_patients)} unevaluated patients to process")
            
            # Process each patient
            for patient_id in new_patients:
                try:
                    await self._run_patient_pipeline(patient_id)
                except Exception as e:
                    logger.error(f"Failed to process patient {patient_id}: {e}")
                finally:
                    # Remove from processing set
                    async with self._patient_lock:
                        self.processed_patients.discard(patient_id)
                    
        except Exception as e:
            logger.error(f"Error processing unevaluated patients: {e}")
    
    async def _process_unevaluated_trials(self):
        """Process unevaluated trials through trial-to-patient pipeline"""
        try:
            # Get unevaluated trial IDs
            trial_ids = self.db_utils.get_unevaluated_trial_ids(limit=AUTO_EVAL_TRIAL_BATCH_SIZE)
            
            if not trial_ids:
                return
            
            # Filter out trials already being processed
            async with self._trial_lock:
                new_trials = [tid for tid in trial_ids if tid not in self.processed_trials]
                if not new_trials:
                    return
                
                # Mark as being processed
                self.processed_trials.update(new_trials)
            
            logger.info(f"📋 Found {len(new_trials)} unevaluated trials to process")
            
            # Process each trial
            for trial_id in new_trials:
                try:
                    await self._run_trial_pipeline(trial_id)
                except Exception as e:
                    logger.error(f"Failed to process trial {trial_id}: {e}")
                finally:
                    # Remove from processing set
                    async with self._trial_lock:
                        self.processed_trials.discard(trial_id)
                    
        except Exception as e:
            logger.error(f"Error processing unevaluated trials: {e}")
    
    async def _run_patient_pipeline(self, patient_id: int):
        """Run patient-to-trial pipeline for a single patient"""
        logger.info(f"🔄 Processing patient {patient_id} through patient-to-trial pipeline...")
        
        # Get patient MRN first
        patient_data = self.db_utils.get_patient_by_id(patient_id)
        if not patient_data:
            logger.warning(f"Patient {patient_id} not found in database")
            return
        
        mrn = patient_data.get('mrn')
        if not mrn:
            logger.warning(f"Patient {patient_id} has no MRN")
            return
        
        # Execute: python patient_to_trial_pipeline.py --patient-id <mrn>
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'patient_to_trial_pipeline.py'),
            '--patient-id', str(mrn)
        ]
        
        start_time = time.time()
        try:
            await asyncio.wait_for(
                self._run_pipeline_cmd(cmd, context={"patient_id": patient_id, "mrn": mrn}),
                timeout=AUTO_EVAL_PATIENT_TIMEOUT
            )
            duration = time.time() - start_time
            logger.info(f"✅ Patient {patient_id} (MRN: {mrn}) processed successfully in {duration:.2f}s")
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Patient {patient_id} (MRN: {mrn}) processing timed out after {AUTO_EVAL_PATIENT_TIMEOUT}s")
        except Exception as e:
            logger.error(f"❌ Error processing patient {patient_id} (MRN: {mrn}): {e}")
    
    async def _run_trial_pipeline(self, trial_id: str):
        """Run trial-to-patient pipeline for a single trial"""
        logger.info(f"🔄 Processing trial {trial_id} through trial-to-patient pipeline...")
        
        # Execute: python trial_to_patient_pipeline.py --trial-id <trial_id>
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'trial_to_patient_pipeline.py'),
            '--trial-id', str(trial_id)
        ]
        
        start_time = time.time()
        try:
            await asyncio.wait_for(
                self._run_pipeline_cmd(cmd, context={"trial_id": trial_id}),
                timeout=AUTO_EVAL_TRIAL_TIMEOUT
            )
            duration = time.time() - start_time
            logger.info(f"✅ Trial {trial_id} processed successfully in {duration:.2f}s")
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Trial {trial_id} processing timed out after {AUTO_EVAL_TRIAL_TIMEOUT}s")
        except Exception as e:
            logger.error(f"❌ Error processing trial {trial_id}: {e}")
    
    async def _run_pipeline_cmd(self, cmd: List[str], context: Optional[dict] = None) -> None:
        """Run a pipeline command with detailed logging"""
        context = context or {}
        cmd_str = ' '.join(cmd)
        logger.debug(f"Running command: {cmd_str} | context={context}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=PIPE,
                stderr=PIPE,
                bufsize=0  # Unbuffered
            )
            
            async def stream_output(stream, prefix="OUT"):
                """Stream output from subprocess in real-time"""
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded = line.decode(errors='ignore').rstrip()
                        if decoded:  # Only log non-empty lines
                            logger.debug(f"[{prefix}] {decoded}")
                except Exception as e:
                    logger.error(f"Error reading {prefix} stream: {e}")
            
            # Start streaming stdout and stderr concurrently
            stdout_task = asyncio.create_task(stream_output(process.stdout, "STDOUT"))
            stderr_task = asyncio.create_task(stream_output(process.stderr, "STDERR"))
            
            # Wait for process to complete
            returncode = await process.wait()
            
            # Wait for output streaming to complete
            await stdout_task
            await stderr_task
            
            if returncode == 0:
                logger.debug(f"[SUCCESS] Command succeeded | context={context}")
            else:
                logger.warning(f"[FAILED] Command failed (exit {returncode}) | context={context}")
                
        except Exception as e:
            logger.error(f"Failed to execute command: {cmd_str} | context={context} | error={e}")
            raise


async def main():
    """Main entry point"""
    monitor = AutoEvaluationMonitor()
    
    try:
        await monitor.initialize()
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        await monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Monitor crashed: {e}")
        sys.exit(1)

