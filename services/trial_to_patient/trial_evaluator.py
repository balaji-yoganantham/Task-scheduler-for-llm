"""
Trial-to-Patient Evaluator
Evaluates patient matches for a specific clinical trial using LLM
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from services.shared.database_utils import DatabaseUtils
from services.shared.llm_utils import LLMUtils
from services.trial_to_patient.hybrid_matcher import HybridMatcher
from evaluation_results_db.utils.evaluation_results_db import EvaluationResultsDB

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
        """Evaluate top patients for a specific trial using LLM batch processing"""
        print(f"EVALUATING Evaluating {len(top_patients)} patients for trial {trial_id}")
        
        # Get detailed trial information
        trial_info = self.db_utils.get_trial_by_id(trial_id)
        if not trial_info:
            print(f"ERROR Trial {trial_id} not found")
            return {}
        
        # Get detailed patient information for all patients
        detailed_patients = []
        for patient in top_patients:
            mrn = patient.get('mrn')
            if not mrn:
                print(f"  ERROR No MRN found for patient")
                continue
                
            detailed_patient_info = self.db_utils.get_patient_by_mrn(mrn)
            if not detailed_patient_info:
                print(f"  ERROR Patient with MRN {mrn} not found in database")
                continue
            
            # Merge hybrid score and other metadata from original patient
            detailed_patient_info['hybrid_score'] = patient.get('hybrid_score', 0)
            detailed_patient_info['embedding_score'] = patient.get('embedding_score', 0)
            detailed_patient_info['bm25_score'] = patient.get('bm25_score', 0)
            detailed_patients.append(detailed_patient_info)
        
        if not detailed_patients:
            print("ERROR No detailed patient information found")
            return {}
        
        print(f"BATCH Running BATCH evaluation for {len(detailed_patients)} patients...")
        print(f"TRIAL Trial: {trial_info['title']} (Phase: {trial_info['phase']}, Status: {trial_info['status']})")
        
        # Use batch evaluation - this sends ALL patients to GPT-4o Mini in one API call
        batch_result = self.llm_utils.evaluate_trial_patient_matches_batch(detailed_patients, trial_info)
        
        if "error" in batch_result:
            print(f"ERROR Batch evaluation failed: {batch_result['error']}")
            return {
                "trial_id": trial_id,
                "trial_info": trial_info,
                "total_evaluations": 0,
                "evaluations": [],
                "error": batch_result["error"],
                "summary": {
                    "eligible_count": 0,
                    "not_eligible_count": 0,
                    "need_more_info_count": 0,
                    "average_confidence": 0
                },
                "generated_at": datetime.now().isoformat(),
                "evaluation_method": "batch_failed"
            }
        
        evaluations = batch_result.get("evaluations", [])
        batch_summary = batch_result.get("batch_summary", {})
        
        print(f"OK Batch evaluation completed successfully!")
        print(f"RESULTS Results: {len(evaluations)} patients evaluated")
        print(f"   - Eligible: {batch_summary.get('eligible_count', 0)}")
        print(f"   - Not Eligible: {batch_summary.get('not_eligible_count', 0)}")
        print(f"   - Need More Info: {batch_summary.get('need_more_info_count', 0)}")
        print(f"   - Average Confidence: {batch_summary.get('average_confidence', 0):.1f}%")
        
        # Sort evaluations by priority score and confidence
        evaluations.sort(key=lambda x: (
            x.get('priority_score', 0) * 0.7 + 
            x.get('confidence_score', 0) * 0.3
        ), reverse=True)
        
        return {
            "trial_id": trial_id,
            "trial_info": trial_info,
            "total_evaluations": len(evaluations),
            "evaluations": evaluations,
            "batch_summary": batch_summary,
            "summary": {
                "eligible_count": len([e for e in evaluations if e.get('eligibility_status') == 'ELIGIBLE']),
                "not_eligible_count": len([e for e in evaluations if e.get('eligibility_status') == 'NOT_ELIGIBLE']),
                "need_more_info_count": len([e for e in evaluations if e.get('eligibility_status') == 'NEED_MORE_INFO']),
                "average_confidence": sum(e.get('confidence_score', 0) for e in evaluations) / len(evaluations) if evaluations else 0
            },
            "generated_at": datetime.now().isoformat(),
            "evaluation_method": "batch"
        }

    def run_complete_trial_matching(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None) -> Dict[str, Any]:
        """Run complete trial-to-patient matching with LLM evaluation"""
        print(f"Starting complete trial-to-patient matching for: {trial_id}")
        
        # Step 1: Hybrid matching to get top 20 patients
        print("Step 1: Running hybrid matching...")
        hybrid_results = self.hybrid_matcher.run_trial_to_patient_matching(
            trial_id=trial_id,
            age_range=age_range,
            gender=gender
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

    def run_trial_matching_pipeline(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None) -> Dict[str, Any]:
        """Run the complete trial-to-patient matching pipeline"""
        print("=" * 80)
        print("TRIAL-TO-PATIENT MATCHING PIPELINE")
        print("=" * 80)
        
        # Run complete matching
        results = self.run_complete_trial_matching(trial_id, age_range, gender)
        
        # Save results to JSON file (existing functionality)
        filepath = self.save_evaluation_results(results)
        results['results_file'] = filepath
        
        # Save results to database (new functionality)
        print(f"\nSAVING Saving results to database...")
        try:
            db_id = self.eval_db.save_trial_to_patient_evaluation(results)
            if db_id:
                results['database_id'] = db_id
                print(f"OK Results saved to database with ID: {db_id}")
            else:
                print("WARNING Failed to save results to database")
        except Exception as e:
            print(f"WARNING Database save error (continuing with JSON): {e}")
        
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
