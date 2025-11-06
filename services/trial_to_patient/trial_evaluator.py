"""
Trial-to-Patient Evaluator
Evaluates patient matches for a specific clinical trial using LLM
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from services.shared.database_utils import DatabaseUtils
from services.shared.llm_utils import LLMUtils
from services.trial_to_patient.hybrid_matcher import HybridMatcher
from evaluation_results_db.utils.evaluation_results_db import EvaluationResultsDB
from config import TRIAL_PATIENT_LLM_BATCH_SIZE

class TrialEvaluator:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.llm_utils = LLMUtils()
        self.hybrid_matcher = HybridMatcher()
        self.eval_db = EvaluationResultsDB()  # Database for evaluation results
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_patient_for_trial(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single patient for a specific trial using LLM"""
        try:
            print(f"Evaluating patient MRN {patient_info['mrn']} for trial {trial_info.get('title', 'Unknown')}")
            
            evaluation = self.llm_utils.evaluate_trial_patient_match(trial_info, patient_info)
            
            # Add patient and trial info to evaluation
            evaluation['patient_info'] = {
                "patient_id": patient_info['patient_id'],
                "mrn": patient_info['mrn'],
                "age": patient_info['age'],
                "gender": patient_info['gender'],
                "oncologist": patient_info['oncologist']
            }
            
            evaluation['trial_info'] = {
                "trial_id": trial_info['trial_id'],
                "title": trial_info['title'],
                "condition": trial_info['condition'],
                "phase": trial_info['phase']
            }
            
            evaluation['hybrid_score'] = patient_info.get('hybrid_score', 0)
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating patient for trial: {e}")
            return {
                "eligibility_status": "NEED_MORE_INFO",
                "confidence_score": 0,
                "reasoning": f"Evaluation error: {str(e)}",
                "error": str(e)
            }

    def evaluate_top_patients_for_trial(self, trial_id: str, top_patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate top patients for a specific trial using LLM batch processing in chunks of 30"""
        print(f"EVALUATING Evaluating {len(top_patients)} patients for trial {trial_id}")
        print(f"BATCH_SIZE Processing in batches of {TRIAL_PATIENT_LLM_BATCH_SIZE} patients per LLM call")
        
        # Get detailed trial information
        trial_info = self.db_utils.get_trial_by_id(trial_id)
        if not trial_info:
            print(f"ERROR Trial {trial_id} not found")
            return {}
        
        # Get detailed patient information for all patients
        detailed_patients = []
        for patient in top_patients:
            # Try to get patient by ID first (more reliable)
            patient_id = patient.get('patient_id')
            detailed_patient_info = None
            
            if patient_id:
                # Convert patient_id to int if it's a string
                try:
                    patient_id_int = int(patient_id) if isinstance(patient_id, str) else patient_id
                    detailed_patient_info = self.db_utils.get_patient_by_id(patient_id_int)
                except (ValueError, TypeError) as e:
                    print(f"  WARNING Invalid patient_id {patient_id}: {e}")
            
            # Fall back to MRN lookup if patient_id lookup failed
            if not detailed_patient_info:
                mrn = patient.get('mrn')
                if mrn:
                    detailed_patient_info = self.db_utils.get_patient_by_mrn(mrn)
                    if not detailed_patient_info:
                        print(f"  ERROR Patient with ID {patient_id} / MRN {mrn} not found in database")
                        continue
                else:
                    print(f"  ERROR No patient_id or MRN found for patient")
                    continue
            
            # Merge hybrid score and other metadata from original patient
            detailed_patient_info['hybrid_score'] = patient.get('hybrid_score', 0)
            detailed_patient_info['embedding_score'] = patient.get('embedding_score', 0)
            detailed_patient_info['bm25_score'] = patient.get('bm25_score', 0)
            detailed_patients.append(detailed_patient_info)
        
        if not detailed_patients:
            print("ERROR No detailed patient information found")
            return {}
        
        print(f"TRIAL Trial: {trial_info['title']} (Phase: {trial_info['phase']}, Status: {trial_info['status']})")
        print(f"BATCH Processing {len(detailed_patients)} patients in batches of {TRIAL_PATIENT_LLM_BATCH_SIZE}...")
        
        # Split patients into batches of TRIAL_PATIENT_LLM_BATCH_SIZE
        all_evaluations = []
        all_batch_summaries = []
        total_batches = (len(detailed_patients) + TRIAL_PATIENT_LLM_BATCH_SIZE - 1) // TRIAL_PATIENT_LLM_BATCH_SIZE
        
        for batch_idx in range(0, len(detailed_patients), TRIAL_PATIENT_LLM_BATCH_SIZE):
            batch_patients = detailed_patients[batch_idx:batch_idx + TRIAL_PATIENT_LLM_BATCH_SIZE]
            batch_num = (batch_idx // TRIAL_PATIENT_LLM_BATCH_SIZE) + 1
            
            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}/{total_batches}: Processing {len(batch_patients)} patients (indices {batch_idx} to {batch_idx + len(batch_patients) - 1})")
            print(f"{'='*60}")
            
            # Process this batch
            batch_result = self.llm_utils.evaluate_trial_patient_matches_batch(batch_patients, trial_info)
            
            if "error" in batch_result:
                print(f"ERROR Batch {batch_num} evaluation failed: {batch_result['error']}")
                # Continue with other batches even if one fails
                continue
            
            batch_evaluations = batch_result.get("evaluations", [])
            batch_summary = batch_result.get("batch_summary", {})
            
            all_evaluations.extend(batch_evaluations)
            all_batch_summaries.append({
                "batch_number": batch_num,
                "batch_size": len(batch_patients),
                "summary": batch_summary
            })
            
            print(f"✓ Batch {batch_num} completed: {len(batch_evaluations)} patients evaluated")
            print(f"  - Eligible: {batch_summary.get('eligible_count', 0)}")
            print(f"  - Not Eligible: {batch_summary.get('not_eligible_count', 0)}")
            print(f"  - Need More Info: {batch_summary.get('need_more_info_count', 0)}")
        
        if not all_evaluations:
            print(f"ERROR All batch evaluations failed")
            return {
                "trial_id": trial_id,
                "trial_info": trial_info,
                "total_evaluations": 0,
                "evaluations": [],
                "error": "All batch evaluations failed",
                "summary": {
                    "eligible_count": 0,
                    "not_eligible_count": 0,
                    "need_more_info_count": 0,
                    "average_confidence": 0
                },
                "generated_at": datetime.now().isoformat(),
                "evaluation_method": "batched",
                "batch_summaries": all_batch_summaries
            }
        
        # Calculate overall summary
        eligible_count = len([e for e in all_evaluations if e.get('eligibility_status') == 'ELIGIBLE'])
        not_eligible_count = len([e for e in all_evaluations if e.get('eligibility_status') == 'NOT_ELIGIBLE'])
        need_more_info_count = len([e for e in all_evaluations if e.get('eligibility_status') == 'NEED_MORE_INFO'])
        average_confidence = sum(e.get('confidence_score', 0) for e in all_evaluations) / len(all_evaluations) if all_evaluations else 0
        
        print(f"\n{'='*60}")
        print(f"ALL BATCHES COMPLETED")
        print(f"{'='*60}")
        print(f"RESULTS Total: {len(all_evaluations)} patients evaluated across {total_batches} batches")
        print(f"   - Eligible: {eligible_count}")
        print(f"   - Not Eligible: {not_eligible_count}")
        print(f"   - Need More Info: {need_more_info_count}")
        print(f"   - Average Confidence: {average_confidence:.1f}%")
        
        # Sort evaluations by priority score and confidence
        all_evaluations.sort(key=lambda x: (
            x.get('priority_score', 0) * 0.7 + 
            x.get('confidence_score', 0) * 0.3
        ), reverse=True)
        
        return {
            "trial_id": trial_id,
            "trial_info": trial_info,
            "total_evaluations": len(all_evaluations),
            "evaluations": all_evaluations,
            "batch_summaries": all_batch_summaries,
            "summary": {
                "eligible_count": eligible_count,
                "not_eligible_count": not_eligible_count,
                "need_more_info_count": need_more_info_count,
                "average_confidence": average_confidence
            },
            "generated_at": datetime.now().isoformat(),
            "evaluation_method": "batched",
            "batch_size": TRIAL_PATIENT_LLM_BATCH_SIZE,
            "total_batches": total_batches
        }

    def run_complete_trial_matching(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None,
                                   max_distance_km: Optional[float] = None,
                                   location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run complete trial-to-patient matching with LLM evaluation"""
        print(f"Starting complete trial-to-patient matching for: {trial_id}")
        
        # Step 1: Hybrid matching to get top K patients
        print("Step 1: Running hybrid matching...")
        hybrid_results = self.hybrid_matcher.run_trial_to_patient_matching(
            trial_id=trial_id,
            age_range=age_range,
            gender=gender,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )
        
        if not hybrid_results['matching_patients']:
            print("No matching patients found in hybrid matching")
            # Return consistent structure with summary even when no patients found
            return {
                "trial_id": trial_id,
                "trial_info": hybrid_results['trial_info'],
                "hybrid_matching": hybrid_results,
                "llm_evaluation": {},
                "final_ranking": [],
                "summary": {
                    "total_patients_found": hybrid_results['total_matches'],
                    "patients_evaluated": 0,
                    "eligible_patients": 0,
                    "average_confidence": 0
                },
                "generated_at": datetime.now().isoformat()
            }
        
        # Step 2: LLM evaluation of top patients
        print("Step 2: Running LLM evaluation...")
        llm_evaluation = self.evaluate_top_patients_for_trial(
            trial_id=trial_id,
            top_patients=hybrid_results['matching_patients']
        )
        
        # Step 3: Combine results
        complete_results = {
            "trial_id": trial_id,
            "trial_info": hybrid_results['trial_info'],
            "hybrid_matching": hybrid_results,
            "llm_evaluation": llm_evaluation,
            "final_ranking": llm_evaluation.get('evaluations', []),
            "summary": {
                "total_patients_found": hybrid_results['total_matches'],
                "patients_evaluated": llm_evaluation.get('total_evaluations', 0),
                "eligible_patients": llm_evaluation.get('summary', {}).get('eligible_count', 0),
                "average_confidence": llm_evaluation.get('summary', {}).get('average_confidence', 0)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return complete_results

    def save_evaluation_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_evaluation_{results['trial_id']}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Use safe_json_dump to handle numpy types
        from services.shared.database_utils import safe_json_dump
        safe_json_dump(results, filepath, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to: {filepath}")
        return str(filepath)

    def run_trial_matching_pipeline(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None,
                                   max_distance_km: Optional[float] = None,
                                   location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete trial-to-patient matching pipeline"""
        print("=" * 80)
        print("TRIAL-TO-PATIENT MATCHING PIPELINE")
        print("=" * 80)
        
        # Run complete matching
        results = self.run_complete_trial_matching(
            trial_id, age_range, gender, max_distance_km, location_weight
        )
        
        # Save results to JSON file (existing functionality)
        filepath = self.save_evaluation_results(results)
        results['results_file'] = filepath
        
        # Note: Database saving is handled by the orchestrator to avoid duplicate saves
        # The orchestrator (trial_to_patient_pipeline.py) will save to the database
        
        # Print summary
        print(f"\nPipeline Summary:")
        print(f"Trial: {results['trial_info'].get('title', 'Unknown')}")
        print(f"Total patients found: {results['summary']['total_patients_found']}")
        print(f"Patients evaluated: {results['summary']['patients_evaluated']}")
        print(f"Eligible patients: {results['summary']['eligible_patients']}")
        print(f"Average confidence: {results['summary']['average_confidence']:.1f}%")
        print(f"Results saved to: {filepath}")
        if results.get('database_id'):
            print(f"Database ID: {results['database_id']}")
        
        return results

def main():
    """Main function to test the trial-to-patient pipeline"""
    evaluator = TrialEvaluator()
    
    # Test with a specific trial
    trial_id = "T001"  # Replace with actual trial ID
    results = evaluator.run_trial_matching_pipeline(
        trial_id=trial_id,
        age_range=(18, 75),
        gender="Male"
    )
    
    print(f"\nTop 5 Eligible Patients:")
    for i, evaluation in enumerate(results['final_ranking'][:5], 1):
        if evaluation.get('eligibility_status') == 'ELIGIBLE':
            patient = evaluation['patient_info']
            print(f"{i}. MRN: {patient['mrn']}")
            print(f"   Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"   Confidence: {evaluation.get('confidence_score', 0)}%")
            print(f"   Priority: {evaluation.get('priority_score', 0)}")
            print(f"   Reasoning: {evaluation.get('reasoning', 'N/A')[:100]}...")
            print()

if __name__ == "__main__":
    main()
