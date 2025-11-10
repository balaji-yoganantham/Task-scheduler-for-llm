"""
LLM Accuracy Evaluation Script
Evaluates LLM accuracy by re-evaluating eligible matches and comparing with original decisions.
Uses JSON files directly (no PostgreSQL).
"""

import json
import sys
import re
import os
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import time

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.shared.llm_utils import LLMUtils
from config import GEMINI_API_KEY, PATIENT_TRIAL_LLM_BATCH_SIZE

class LLMAccuracyEvaluator:
    def __init__(self):
        self.llm_utils = LLMUtils()
        # Create results directory in testing folder
        script_dir = Path(__file__).parent
        self.results_dir = script_dir / "llm_accuracy_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def format_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format patient data from JSON to match LLM expected format"""
        patient_info = patient_data.get("Patient Information", {})
        hpi = patient_data.get("History of Present Illness", {})
        pmh = patient_data.get("Past Medical History", {})
        family_history = patient_data.get("Family History", "")
        social_history = patient_data.get("Social History", {})
        lab_results = patient_data.get("Laboratory & Imaging Results", {})
        assessment = patient_data.get("Assessment", {})
        
        # Build combined_text similar to database format
        combined_text = f"""Patient Name: Not specified
Patient MRN: {patient_info.get('MRN', 'Not specified')}
Age: Not specified, Gender: Not specified
Date of Visit: {patient_info.get('Date of Visit', 'Not specified')}
Oncologist: {patient_info.get('Oncologist', 'Not specified')}

Chief Complaint: Not specified

History of Present Illness: {hpi.get('Date of initial colorectal cancer diagnosis', '')} - {hpi.get('Site of primary tumor', '')} - {hpi.get('Date and initial TNM stage', '')} - {hpi.get('Date and nature of metastatic disease diagnosis', '')} - {hpi.get('Summary of prior treatments', '')} - {hpi.get('Performance status', '')}

Past Medical History: {pmh.get('Significant comorbidities', 'Not specified')}

Family History: {family_history}

Social History: {social_history.get('Smoking', 'Not specified')}

Review of Systems: Not specified

Medications and Allergies: Not specified

Physical Examination: Not specified

Laboratory and Imaging Results: Total bilirubin: {lab_results.get('Total_bilirubin', 'Not specified')}, Molecular markers: {lab_results.get('Molecular_markers', 'Not specified')}

Imaging: Not specified

Vital Signs: Not specified

Assessment: {assessment.get('Diagnosis', 'Not specified')} - {assessment.get('Disease_status', 'Not specified')}"""
        
        return {
            "patient_id": patient_info.get('MRN', 0),
            "mrn": str(patient_info.get('MRN', '')),
            "age": None,  # Age not in JSON
            "gender": None,  # Gender not in JSON
            "oncologist": patient_info.get('Oncologist', 'Not specified'),
            "date_of_visit": patient_info.get('Date of Visit', 'Not specified'),
            "combined_text": combined_text,
            "performance_status": hpi.get('Performance status', ''),
            "molecular_markers": lab_results.get('Molecular_markers', ''),
            "prior_treatments": hpi.get('Summary of prior treatments', ''),
            "diagnosis": assessment.get('Diagnosis', '')
        }
    
    def format_trial_data(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trial data from JSON to match LLM expected format"""
        conditions = trial_data.get('conditions', [])
        condition_text = ', '.join(conditions) if isinstance(conditions, list) else str(conditions)
        
        return {
            "trial_id": trial_data.get('nct_id', ''),
            "title": trial_data.get('study_title', ''),
            "condition": condition_text,
            "phase": trial_data.get('study_phase', ''),
            "status": trial_data.get('overall_status', ''),
            "investigator": trial_data.get('lead_sponsor_name', ''),
            "minimum_age": trial_data.get('minimum_age', ''),
            "maximum_age": trial_data.get('maximum_age', ''),
            "sex": trial_data.get('sex', ''),
            "brief_summary": trial_data.get('brief_summary', ''),
            "detailed_description": trial_data.get('detailed_description', ''),
            "inclusion_criteria": trial_data.get('inclusion_criteria', ''),
            "exclusion_criteria": trial_data.get('exclusion_criteria', ''),
            "eligibility_criteria": trial_data.get('eligibility_criteria', '')
        }
    
    def load_data(self):
        """Load all required JSON files"""
        print("Loading data files...")
        
        # Get script directory
        script_dir = Path(__file__).parent
        
        # Load trials
        with open(script_dir / 'trial.json', 'r', encoding='utf-8') as f:
            self.trials_data = json.load(f)
        
        # Load patients
        with open(script_dir / 'patients_5_realistic_final.json', 'r', encoding='utf-8') as f:
            patients_json = json.load(f)
            self.patients_data = patients_json.get('patients', [])
        
        # Load eligible matches
        with open(script_dir / 'patients_5_realistic_final_matches_eligible_only.json', 'r', encoding='utf-8') as f:
            self.matches_data = json.load(f)
        
        # Create lookups
        self.trial_lookup = {trial['nct_id']: trial for trial in self.trials_data}
        self.patient_lookup = {}
        for patient in self.patients_data:
            mrn = patient['Patient Information']['MRN']
            self.patient_lookup[mrn] = patient
        
        print(f"Loaded {len(self.trials_data)} trials, {len(self.patients_data)} patients, {len(self.matches_data['matches'])} patient matches")
    
    def evaluate_all_matches(self, existing_evaluations: List[Dict] = None, existing_stats: Dict[str, Any] = None):
        """Evaluate all eligible matches using LLM"""
        print("\n" + "="*80)
        print("Starting LLM Accuracy Evaluation")
        print("="*80)
        
        all_evaluations = existing_evaluations or []
        accuracy_stats = existing_stats or {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "need_more_info": 0,
            "errors": 0,
            "by_patient": {},
            "by_trial": {}
        }
        
        # Create set of already evaluated matches (skip only successful evaluations, not quota errors)
        evaluated_set = set()
        for eval_item in all_evaluations:
            # Only skip if it was successfully evaluated (not an error due to quota)
            if eval_item.get('llm_decision') not in ['ERROR', 'EXCEPTION']:
                key = (eval_item.get('mrn'), eval_item.get('nct_id'))
                evaluated_set.add(key)
            # Also skip if it's a quota error that we want to retry later
            elif eval_item.get('error_type') == 'ResourceExhausted' and 'quota' in str(eval_item.get('error', '')).lower():
                # Don't add to evaluated_set - we'll retry these when quota resets
                pass
        
        total_matches = sum(len(m['results']) for m in self.matches_data['matches'])
        # Count only successful evaluations for progress tracking
        successful_evaluations = [e for e in all_evaluations if e.get('llm_decision') not in ['ERROR', 'EXCEPTION']]
        current_match = len(successful_evaluations)
        
        print(f"Starting from evaluation {current_match}/{total_matches}")
        print(f"  - Successful evaluations: {len(successful_evaluations)}")
        print(f"  - Quota errors (will retry): {len(all_evaluations) - len(successful_evaluations)}")
        
        for patient_match in self.matches_data['matches']:
            mrn = patient_match['MRN']
            patient_raw = self.patient_lookup.get(mrn)
            
            if not patient_raw:
                print(f"⚠️  Patient MRN {mrn} not found in patients data")
                continue
            
            patient_info = self.format_patient_data(patient_raw)
            patient_stats = {
                "mrn": mrn,
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "need_more_info": 0,
                "errors": 0
            }
            
            print(f"\n{'='*80}")
            print(f"Evaluating Patient MRN: {mrn}")
            print(f"{'='*80}")
            
            # Prepare trials for batch processing
            trials_to_evaluate = []
            trial_results_map = {}  # Map nct_id to original result
            
            for result in patient_match['results']:
                nct_id = result['trial_id']
                original_decision = result['decision']  # Should be "ELIGIBLE"
                
                # Skip if already evaluated
                eval_key = (mrn, nct_id)
                if eval_key in evaluated_set:
                    print(f"  ⏭️  Skipping {nct_id} (already evaluated)")
                    continue
                
                trial_raw = self.trial_lookup.get(nct_id)
                if not trial_raw:
                    print(f"  ⚠️  Trial {nct_id} not found in trial.json")
                    accuracy_stats["errors"] += 1
                    patient_stats["errors"] += 1
                    continue
                
                trial_info = self.format_trial_data(trial_raw)
                trials_to_evaluate.append(trial_info)
                trial_results_map[nct_id] = result
            
            # Process trials in batches
            if not trials_to_evaluate:
                print(f"  ⚠️  No trials to evaluate for patient {mrn}")
                continue
            
            total_batches = (len(trials_to_evaluate) + PATIENT_TRIAL_LLM_BATCH_SIZE - 1) // PATIENT_TRIAL_LLM_BATCH_SIZE
            print(f"  Processing {len(trials_to_evaluate)} trials in {total_batches} batch(es) of {PATIENT_TRIAL_LLM_BATCH_SIZE}...")
            print(f"  ⏳ Rate limiting: 5 seconds between batches to avoid hitting limits...")
            
            # Add initial delay before first batch to avoid immediate rate limit
            # Wait a bit to ensure we start fresh
            print(f"  ⏳ Initial delay: 10 seconds...")
            time.sleep(10.0)
            
            for batch_idx in range(0, len(trials_to_evaluate), PATIENT_TRIAL_LLM_BATCH_SIZE):
                try:
                    batch_trials = trials_to_evaluate[batch_idx:batch_idx + PATIENT_TRIAL_LLM_BATCH_SIZE]
                    batch_num = (batch_idx // PATIENT_TRIAL_LLM_BATCH_SIZE) + 1
                    batch_nct_ids = [t['trial_id'] for t in batch_trials]
                    
                    print(f"\n  {'='*60}")
                    print(f"  BATCH {batch_num}/{total_batches}: Evaluating {len(batch_trials)} trials")
                    print(f"  Trials: {', '.join(batch_nct_ids)}")
                    print(f"  {'='*60}")
                    
                    # Retry loop for handling 429 errors
                    max_retries = 3
                    retry_count = 0
                    batch_result = None
                    
                    while retry_count <= max_retries:
                        try:
                            # Call LLM to evaluate batch
                            batch_result = self.llm_utils.evaluate_patient_trial_matches_batch(batch_trials, patient_info)
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            error_str = str(e)
                            # Check if it's a quota exhaustion error
                            if "429" in error_str or "Resource exhausted" in error_str or "quota" in error_str.lower():
                                print(f"\n  ❌ QUOTA EXHAUSTED: {error_str}")
                                
                                # Try to extract retry delay from error message
                                retry_delay = None
                                delay_match = re.search(r'retry.*?in.*?(\d+(?:\.\d+)?).*?seconds?', error_str.lower())
                                if delay_match:
                                    retry_delay = float(delay_match.group(1))
                                
                                if retry_delay and retry_delay < 3600 and retry_count < max_retries:  # If less than 1 hour, wait and retry
                                    wait_minutes = int(retry_delay / 60) + 1
                                    print(f"     ⏳ Waiting {wait_minutes} minutes ({retry_delay:.0f} seconds) for quota reset...")
                                    print(f"     💾 Saving progress...")
                                    self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
                                    
                                    # Wait for the specified time
                                    time.sleep(retry_delay)
                                    print(f"     ✅ Wait complete. Retrying batch ({retry_count + 1}/{max_retries})...")
                                    retry_count += 1
                                    continue  # Retry
                                else:
                                    # Daily quota exhausted or max retries exceeded
                                    print(f"     💾 Saving progress and stopping. Please wait for quota reset (usually 24 hours).")
                                    print(f"     📝 You can resume later by running the script again - it will continue from checkpoint.")
                                    
                                    # Save progress before stopping
                                    self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
                                    
                                    # Return partial results
                                    return all_evaluations, accuracy_stats
                            else:
                                # Other error - don't retry
                                print(f"  ❌ EXCEPTION: {error_str}")
                                batch_result = {"error": error_str, "error_type": type(e).__name__}
                                break
                    
                    # Process batch results
                    if batch_result is None or "error" in batch_result:
                        error_msg = batch_result.get('error', 'Unknown error') if batch_result else 'No response'
                        error_type = batch_result.get('error_type', 'Unknown') if batch_result else 'NoResponse'
                        
                        # Check if it's a quota exhaustion error
                        is_quota_error = (
                            "429" in error_msg or 
                            "Resource exhausted" in error_msg or 
                            "quota" in error_msg.lower() or
                            error_type == "ResourceExhausted"
                        )
                        
                        if is_quota_error:
                            print(f"  ⚠️  QUOTA EXHAUSTED: {error_msg}")
                            print(f"     🔄 Switching to individual processing for this batch...")
                            # Process trials one by one instead of batch
                            all_evaluations, accuracy_stats, current_match = self._process_trials_individually(
                                batch_trials, patient_info, trial_results_map, 
                                all_evaluations, accuracy_stats, patient_stats, 
                                current_match, total_matches, mrn
                            )
                            # Continue with next batch after processing individually
                            continue
                        else:
                            print(f"  ❌ BATCH ERROR: {error_msg}")
                        
                        # Mark all trials in batch as errors (only for non-quota errors)
                        for trial_info in batch_trials:
                            nct_id = trial_info['trial_id']
                            current_match += 1
                            original_result = trial_results_map.get(nct_id, {})
                            original_decision = original_result.get('decision', 'ELIGIBLE')
                            
                            accuracy_stats["errors"] += 1
                            patient_stats["errors"] += 1
                            all_evaluations.append({
                                "mrn": mrn,
                                "nct_id": nct_id,
                                "original_decision": original_decision,
                                "llm_decision": "ERROR",
                                "match": False,
                                "error": error_msg,
                                "error_type": error_type,
                                "llm_evaluation": batch_result or {}
                            })
                    else:
                        # Process successful batch results
                        batch_evaluations = batch_result.get("evaluations", [])
                        print(f"  ✅ Batch {batch_num} completed: {len(batch_evaluations)} trials evaluated")
                        
                        # Process each evaluation in the batch
                        for eval_item in batch_evaluations:
                            nct_id = eval_item.get('trial_id', '')
                            if not nct_id:
                                continue
                            
                            original_result = trial_results_map.get(nct_id, {})
                            original_decision = original_result.get('decision', 'ELIGIBLE')
                            
                            llm_decision = eval_item.get('eligibility_status', 'UNKNOWN')
                            is_match = (llm_decision == original_decision)
                            
                            current_match += 1
                            accuracy_stats["total"] += 1
                            patient_stats["total"] += 1
                            
                            if is_match:
                                accuracy_stats["correct"] += 1
                                patient_stats["correct"] += 1
                                print(f"    ✅ {nct_id}: MATCH ({llm_decision}) [{current_match}/{total_matches}]")
                            else:
                                accuracy_stats["incorrect"] += 1
                                patient_stats["incorrect"] += 1
                                if llm_decision == "NEED_MORE_INFO":
                                    accuracy_stats["need_more_info"] += 1
                                    patient_stats["need_more_info"] += 1
                                print(f"    ❌ {nct_id}: MISMATCH (Original={original_decision}, LLM={llm_decision}) [{current_match}/{total_matches}]")
                            
                            all_evaluations.append({
                                "mrn": mrn,
                                "nct_id": nct_id,
                                "original_decision": original_decision,
                                "llm_decision": llm_decision,
                                "match": is_match,
                                "confidence_score": eval_item.get('confidence_score', 0),
                                "reasoning": eval_item.get('reasoning', ''),
                                "llm_evaluation": eval_item
                            })
                            
                            # Track by trial
                            if nct_id not in accuracy_stats["by_trial"]:
                                accuracy_stats["by_trial"][nct_id] = {"total": 0, "correct": 0, "incorrect": 0}
                            accuracy_stats["by_trial"][nct_id]["total"] += 1
                            if is_match:
                                accuracy_stats["by_trial"][nct_id]["correct"] += 1
                            else:
                                accuracy_stats["by_trial"][nct_id]["incorrect"] += 1
                    
                    # Rate limiting - delay between batches
                    # TPM (Tokens Per Minute) limits reset every minute
                    # Free tier: 1,000,000 TPM for gemini-2.0-flash
                    # Wait 60+ seconds between batches to allow TPM reset
                    print(f"  ⏳ Waiting 60 seconds for TPM (token) limit reset...")
                    time.sleep(60.0)  # Wait 60 seconds between batches for TPM reset
                    
                    # Save progress after each batch
                    self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
                    
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Interrupted by user. Saving progress...")
                    self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
                    print(f"   💾 Progress saved. Run the script again to resume from checkpoint.")
                    raise
                except Exception as e:
                    print(f"  ❌ BATCH EXCEPTION: {str(e)}")
                    # Mark all trials in batch as errors
                    for trial_info in batch_trials:
                        nct_id = trial_info['trial_id']
                        current_match += 1
                        original_result = trial_results_map.get(nct_id, {})
                        original_decision = original_result.get('decision', 'ELIGIBLE')
                        
                        accuracy_stats["errors"] += 1
                        patient_stats["errors"] += 1
                        all_evaluations.append({
                            "mrn": mrn,
                            "nct_id": nct_id,
                            "original_decision": original_decision,
                            "llm_decision": "EXCEPTION",
                            "match": False,
                            "error": str(e)
                        })
                    # Save progress on errors
                    self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
            
            accuracy_stats["by_patient"][mrn] = patient_stats
        
        return all_evaluations, accuracy_stats
    
    def _process_trials_individually(self, trials, patient_info, trial_results_map, 
                                     all_evaluations, accuracy_stats, patient_stats,
                                     current_match, total_matches, mrn):
        """Fallback: Process trials individually when batch fails"""
        print(f"  📋 Processing {len(trials)} trials individually...")
        
        for trial_info in trials:
            nct_id = trial_info['trial_id']
            original_result = trial_results_map.get(nct_id, {})
            original_decision = original_result.get('decision', 'ELIGIBLE')
            
            print(f"    [{current_match + 1}/{total_matches}] Processing {nct_id} individually...", end=" ", flush=True)
            
            try:
                # Process one trial at a time with delays
                llm_evaluation = self.llm_utils.evaluate_patient_trial_match(trial_info, patient_info)
                
                if "error" in llm_evaluation:
                    error_msg = llm_evaluation.get('error', 'Unknown error')
                    print(f"❌ ERROR: {error_msg}")
                    
                    accuracy_stats["errors"] += 1
                    patient_stats["errors"] += 1
                    all_evaluations.append({
                        "mrn": mrn,
                        "nct_id": nct_id,
                        "original_decision": original_decision,
                        "llm_decision": "ERROR",
                        "match": False,
                        "error": error_msg,
                        "error_type": llm_evaluation.get('error_type', 'Unknown'),
                        "llm_evaluation": llm_evaluation
                    })
                else:
                    llm_decision = llm_evaluation.get('eligibility_status', 'UNKNOWN')
                    is_match = (llm_decision == original_decision)
                    
                    current_match += 1
                    accuracy_stats["total"] += 1
                    patient_stats["total"] += 1
                    
                    if is_match:
                        accuracy_stats["correct"] += 1
                        patient_stats["correct"] += 1
                        print(f"✅ MATCH ({llm_decision})")
                    else:
                        accuracy_stats["incorrect"] += 1
                        patient_stats["incorrect"] += 1
                        if llm_decision == "NEED_MORE_INFO":
                            accuracy_stats["need_more_info"] += 1
                            patient_stats["need_more_info"] += 1
                        print(f"❌ MISMATCH (Original={original_decision}, LLM={llm_decision})")
                    
                    all_evaluations.append({
                        "mrn": mrn,
                        "nct_id": nct_id,
                        "original_decision": original_decision,
                        "llm_decision": llm_decision,
                        "match": is_match,
                        "confidence_score": llm_evaluation.get('confidence_score', 0),
                        "reasoning": llm_evaluation.get('reasoning', ''),
                        "llm_evaluation": llm_evaluation
                    })
                    
                    # Track by trial
                    if nct_id not in accuracy_stats["by_trial"]:
                        accuracy_stats["by_trial"][nct_id] = {"total": 0, "correct": 0, "incorrect": 0}
                    accuracy_stats["by_trial"][nct_id]["total"] += 1
                    if is_match:
                        accuracy_stats["by_trial"][nct_id]["correct"] += 1
                    else:
                        accuracy_stats["by_trial"][nct_id]["incorrect"] += 1
                
                # Wait 10 seconds between individual calls
                time.sleep(10.0)
                
            except Exception as e:
                print(f"❌ EXCEPTION: {str(e)}")
                accuracy_stats["errors"] += 1
                patient_stats["errors"] += 1
                all_evaluations.append({
                    "mrn": mrn,
                    "nct_id": nct_id,
                    "original_decision": original_decision,
                    "llm_decision": "EXCEPTION",
                    "match": False,
                    "error": str(e)
                })
                time.sleep(10.0)
            
            # Save progress every 5 evaluations
            if current_match % 5 == 0:
                self.save_progress(all_evaluations, accuracy_stats, current_match, total_matches)
        
        return all_evaluations, accuracy_stats, current_match
    
    def calculate_accuracy_metrics(self, accuracy_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics"""
        total = accuracy_stats.get("total", 0)
        if total == 0:
            return {
                "accuracy": 0,
                "error_rate": 0,
                "total_evaluations": 0,
                "correct": 0,
                "incorrect": 0,
                "need_more_info": 0,
                "errors": accuracy_stats.get("errors", 0)
            }
        
        accuracy = (accuracy_stats.get("correct", 0) / total) * 100
        total_with_errors = total + accuracy_stats.get("errors", 0)
        error_rate = (accuracy_stats.get("errors", 0) / total_with_errors) * 100 if total_with_errors > 0 else 0
        
        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "total_evaluations": total,
            "correct": accuracy_stats.get("correct", 0),
            "incorrect": accuracy_stats.get("incorrect", 0),
            "need_more_info": accuracy_stats.get("need_more_info", 0),
            "errors": accuracy_stats.get("errors", 0)
        }
    
    def save_progress(self, all_evaluations: List[Dict], accuracy_stats: Dict[str, Any], current: int, total: int):
        """Save progress incrementally"""
        try:
            progress_file = self.results_dir / "progress_checkpoint.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "evaluations": all_evaluations,
                    "accuracy_stats": accuracy_stats,
                    "progress": {
                        "current": current,
                        "total": total,
                        "timestamp": datetime.now().isoformat()
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  💾 Progress saved: {current}/{total} ({current/total*100:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Could not save progress: {e}")
    
    def save_results(self, all_evaluations: List[Dict], accuracy_stats: Dict[str, Any], metrics: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all evaluations
        evaluations_file = self.results_dir / f"all_evaluations_{timestamp}.json"
        with open(evaluations_file, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluations": all_evaluations,
                "metadata": {
                    "timestamp": timestamp,
                    "total_evaluations": len(all_evaluations),
                    "model": "gemini-2.0-flash-exp"
                }
            }, f, indent=2, ensure_ascii=False)
        
        # Save accuracy statistics
        stats_file = self.results_dir / f"accuracy_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "overall_metrics": metrics,
                "detailed_stats": accuracy_stats,
                "metadata": {
                    "timestamp": timestamp,
                    "model": "gemini-2.0-flash-exp"
                }
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        report_file = self.results_dir / f"accuracy_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LLM ACCURACY EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: gemini-2.0-flash-exp\n\n")
            f.write("OVERALL METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Evaluations: {metrics['total_evaluations']}\n")
            f.write(f"Correct Matches: {metrics['correct']}\n")
            f.write(f"Incorrect Matches: {metrics['incorrect']}\n")
            f.write(f"Need More Info: {metrics['need_more_info']}\n")
            f.write(f"Errors: {metrics['errors']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"Error Rate: {metrics['error_rate']:.2f}%\n\n")
            
            f.write("BY PATIENT:\n")
            f.write("-"*80 + "\n")
            for mrn, stats in accuracy_stats["by_patient"].items():
                if stats["total"] > 0:
                    patient_accuracy = (stats["correct"] / stats["total"]) * 100
                    f.write(f"MRN {mrn}: {stats['correct']}/{stats['total']} correct ({patient_accuracy:.2f}%)\n")
            
            f.write("\nBY TRIAL (Top 10):\n")
            f.write("-"*80 + "\n")
            trial_list = [(nct_id, stats) for nct_id, stats in accuracy_stats["by_trial"].items()]
            trial_list.sort(key=lambda x: x[1]["total"], reverse=True)
            for nct_id, stats in trial_list[:10]:
                if stats["total"] > 0:
                    trial_accuracy = (stats["correct"] / stats["total"]) * 100
                    f.write(f"{nct_id}: {stats['correct']}/{stats['total']} correct ({trial_accuracy:.2f}%)\n")
        
        print(f"\n✅ Results saved:")
        print(f"   - Evaluations: {evaluations_file}")
        print(f"   - Statistics: {stats_file}")
        print(f"   - Report: {report_file}")
        
        return evaluations_file, stats_file, report_file
    
    def load_checkpoint(self) -> tuple:
        """Load progress from checkpoint if exists"""
        checkpoint_file = self.results_dir / "progress_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                print(f"[OK] Found checkpoint: {checkpoint['progress']['current']}/{checkpoint['progress']['total']} evaluations")
                return checkpoint.get('evaluations', []), checkpoint.get('accuracy_stats', {})
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")
        return [], {}
    
    def run(self, resume: bool = True):
        """Run the complete evaluation"""
        try:
            # Load data
            self.load_data()
            
            # Try to resume from checkpoint
            all_evaluations = []
            accuracy_stats = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "need_more_info": 0,
                "errors": 0,
                "by_patient": {},
                "by_trial": {}
            }
            
            if resume:
                checkpoint_evaluations, checkpoint_stats = self.load_checkpoint()
                if checkpoint_evaluations:
                    print(f"Resuming from checkpoint: {len(checkpoint_evaluations)} evaluations already completed")
                    all_evaluations = checkpoint_evaluations
                    accuracy_stats = checkpoint_stats
            
            # Evaluate all matches (will skip already evaluated ones)
            all_evaluations, accuracy_stats = self.evaluate_all_matches(all_evaluations, accuracy_stats)
            
            # Calculate metrics
            metrics = self.calculate_accuracy_metrics(accuracy_stats)
            
            # Print summary
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            print(f"Total Evaluations: {metrics['total_evaluations']}")
            print(f"Correct Matches: {metrics['correct']}")
            print(f"Incorrect Matches: {metrics['incorrect']}")
            print(f"Need More Info: {metrics['need_more_info']}")
            print(f"Errors: {metrics['errors']}")
            print(f"\nAccuracy: {metrics['accuracy']:.2f}%")
            print(f"Error Rate: {metrics['error_rate']:.2f}%")
            
            # Save results
            if len(all_evaluations) > 0:
                self.save_results(all_evaluations, accuracy_stats, metrics)
                print("\n✅ Evaluation complete!")
            else:
                print("\n⚠️  No evaluations completed. Check for quota limits or errors.")
            
        except Exception as e:
            print(f"\n[ERROR] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    evaluator = LLMAccuracyEvaluator()
    evaluator.run()

