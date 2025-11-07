"""
Patient-to-Trial Evaluator
Evaluates trial matches for a specific patient using LLM
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from services.shared.database_utils import DatabaseUtils, safe_json_dump
from services.shared.llm_utils import LLMUtils
from services.patient_to_trial.patient_matcher import PatientMatcher
from evaluation_results_db.utils.evaluation_results_db import EvaluationResultsDB

class PatientEvaluator:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.llm_utils = LLMUtils()
        self.patient_matcher = PatientMatcher()
        self.eval_db = EvaluationResultsDB()  # Database for evaluation results
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_trial_for_patient(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single trial for a specific patient using LLM"""
        try:
            print(f"Evaluating trial {trial_info.get('title', 'Unknown')} for patient MRN {patient_info['mrn']}")
            
            evaluation = self.llm_utils.evaluate_patient_trial_match(trial_info, patient_info)
            
            # Add trial and patient info to evaluation
            evaluation['trial_info'] = {
                "trial_id": trial_info.get('trial_id', 'Unknown'),
                "title": trial_info.get('title', 'Unknown'),
                "condition": trial_info.get('condition', 'Unknown'),
                "phase": trial_info.get('phase', 'Unknown'),
                "status": trial_info.get('status', 'Unknown')
            }
            
            evaluation['patient_info'] = {
                "patient_id": patient_info['patient_id'],
                "mrn": patient_info['mrn'],
                "age": patient_info['age'],
                "gender": patient_info['gender'],
                "oncologist": patient_info['oncologist']
            }
            
            evaluation['hybrid_score'] = trial_info.get('hybrid_score', 0)
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating trial for patient: {e}")
            return {
                "eligibility_status": "NEED_MORE_INFO",
                "confidence_score": 0,
                "reasoning": f"Evaluation error: {str(e)}",
                "error": str(e)
            }

    def evaluate_top_trials_for_patient(self, patient_id: int, top_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate top trials for a specific patient using LLM batch processing"""
        print(f"🚀 Evaluating {len(top_trials)} trials for patient {patient_id}")
        
        # Get detailed patient information
        patient_info = self.db_utils.get_patient_by_id(patient_id)
        if not patient_info:
            print(f"❌ Patient {patient_id} not found")
            return {}
        
        # Get detailed trial information for all trials
        detailed_trials = []
        for trial in top_trials:
            trial_id = trial.get('trial_id')
            if trial_id:
                detailed_trial_info = self.db_utils.get_trial_by_id(trial_id)
                if detailed_trial_info:
                    # Merge hybrid score and other metadata from original trial
                    detailed_trial_info['hybrid_score'] = trial.get('hybrid_score', 0)
                    detailed_trial_info['embedding_score'] = trial.get('embedding_score', 0)
                    detailed_trial_info['bm25_score'] = trial.get('bm25_score', 0)
                    detailed_trials.append(detailed_trial_info)
        
        if not detailed_trials:
            print("❌ No detailed trial information found")
            return {}
        
        print(f"📊 Running BATCH evaluation for {len(detailed_trials)} trials...")
        print(f"📋 Patient: MRN {patient_info['mrn']} (Age: {patient_info['age']}, Gender: {patient_info['gender']})")
        
        # Use batch evaluation - this sends ALL trials to Gemini in one API call
        batch_result = self.llm_utils.evaluate_patient_trial_matches_batch(detailed_trials, patient_info)
        
        if "error" in batch_result:
            print(f"❌ Batch evaluation failed: {batch_result['error']}")
            return {
                "patient_id": patient_id,
                "patient_info": patient_info,
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
                "evaluation_method": "batch",
                "batch_summary": {"error": batch_result["error"]}
            }
        
        evaluations = batch_result.get("evaluations", [])
        batch_summary = batch_result.get("batch_summary", {})
        
        print(f"✅ Batch evaluation completed successfully!")
        print(f"📊 Results: {len(evaluations)} trials evaluated")
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
            "patient_id": patient_id,
            "patient_info": patient_info,
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

    def evaluate_top_trials_for_patient_individual(self, patient_id: int, top_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate top trials for a specific patient using individual LLM calls (fallback method)"""
        print(f"Evaluating {len(top_trials)} trials individually for patient {patient_id}")
        
        # Get detailed patient information
        patient_info = self.db_utils.get_patient_by_id(patient_id)
        if not patient_info:
            print(f"Patient {patient_id} not found")
            return {}
        
        evaluations = []
        
        for i, trial in enumerate(top_trials, 1):
            print(f"Evaluating trial {i}/{len(top_trials)}: {trial.get('title', 'Unknown')}")
            
            # Get detailed trial information
            trial_id = trial.get('trial_id')
            if not trial_id:
                continue
                
            detailed_trial_info = self.db_utils.get_trial_by_id(trial_id)
            if not detailed_trial_info:
                continue
            
            # Evaluate trial-patient match
            evaluation = self.evaluate_trial_for_patient(detailed_trial_info, patient_info)
            evaluations.append(evaluation)
        
        # Sort evaluations by priority score and confidence
        evaluations.sort(key=lambda x: (
            x.get('priority_score', 0) * 0.7 + 
            x.get('confidence_score', 0) * 0.3
        ), reverse=True)
        
        return {
            "patient_id": patient_id,
            "patient_info": patient_info,
            "total_evaluations": len(evaluations),
            "evaluations": evaluations,
            "summary": {
                "eligible_count": len([e for e in evaluations if e.get('eligibility_status') == 'ELIGIBLE']),
                "not_eligible_count": len([e for e in evaluations if e.get('eligibility_status') == 'NOT_ELIGIBLE']),
                "need_more_info_count": len([e for e in evaluations if e.get('eligibility_status') == 'NEED_MORE_INFO']),
                "average_confidence": sum(e.get('confidence_score', 0) for e in evaluations) / len(evaluations) if evaluations else 0
            },
            "generated_at": datetime.now().isoformat(),
            "evaluation_method": "individual"
        }

    def run_complete_patient_trial_matching(self, patient_id: int, age_range: Tuple[int, int] = None, 
                                          gender: str = None, phase_filter: List[str] = None,
                                          max_distance_km: Optional[float] = None,
                                          location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run complete patient-to-trial matching with LLM evaluation"""
        print(f"Starting complete patient-to-trial matching for patient: {patient_id}")
        
        # Step 1: Hybrid matching to get top trials
        print("Step 1: Running hybrid matching...")
        hybrid_results = self.patient_matcher.find_trials_for_patient(
            patient_id=patient_id,
            age_range=age_range,
            gender=gender,
            phase_filter=phase_filter,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )
        
        if not hybrid_results.get('matching_trials'):
            print("No matching trials found in hybrid matching")
            # Return consistent structure with summary even when no trials found
            return {
                "patient_id": patient_id,
                "patient_info": hybrid_results.get('patient_info', {}),
                "hybrid_matching": hybrid_results,
                "llm_evaluation": {},
                "final_ranking": [],
                "summary": {
                    "total_trials_found": hybrid_results.get('total_matches', 0),
                    "trials_evaluated": 0,
                    "eligible_trials": 0,
                    "average_confidence": 0
                },
                "generated_at": datetime.now().isoformat()
            }
        
        # Step 2: LLM evaluation of top trials
        print("Step 2: Running LLM evaluation...")
        llm_evaluation = self.evaluate_top_trials_for_patient(
            patient_id=patient_id,
            top_trials=hybrid_results['matching_trials']
        )
        
        # Step 3: Combine results
        complete_results = {
            "patient_id": patient_id,
            "patient_info": hybrid_results['patient_info'],
            "hybrid_matching": hybrid_results,
            "llm_evaluation": llm_evaluation,
            "final_ranking": llm_evaluation.get('evaluations', []),
            "summary": {
                "total_trials_found": hybrid_results['total_matches'],
                "trials_evaluated": llm_evaluation.get('total_evaluations', 0),
                "eligible_trials": llm_evaluation.get('summary', {}).get('eligible_count', 0),
                "average_confidence": llm_evaluation.get('summary', {}).get('average_confidence', 0)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return complete_results

    def save_evaluation_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patient_evaluation_{results['patient_id']}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        safe_json_dump(results, filepath, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to: {filepath}")
        return str(filepath)

    def run_patient_trial_pipeline(self, patient_id: int, age_range: Tuple[int, int] = None, 
                                 gender: str = None, phase_filter: List[str] = None,
                                 max_distance_km: Optional[float] = None,
                                 location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete patient-to-trial matching pipeline"""
        print("=" * 80)
        print("PATIENT-TO-TRIAL MATCHING PIPELINE")
        print("=" * 80)
        
        # Run complete matching
        results = self.run_complete_patient_trial_matching(
            patient_id, age_range, gender, phase_filter, max_distance_km, location_weight
        )
        
        # Save results to JSON file (existing functionality)
        filepath = self.save_evaluation_results(results)
        results['results_file'] = filepath
        
        # Save results to database (new functionality)
        print(f"\n[SAVE] Saving results to database...")
        try:
            # Save summary evaluation results
            db_id = self.eval_db.save_patient_to_trial_evaluation(results)
            if db_id:
                results['database_id'] = db_id
                print(f"✅ Results saved to database with ID: {db_id}")
            else:
                print("⚠️ Failed to save results to database")
            
            # Save individual patient-trial evaluations to normalized table
            print(f"\n[SAVE] Saving individual patient-trial evaluations...")
            try:
                saved_count = self.eval_db.save_patient_trial_evaluations(results)
                if saved_count > 0:
                    results['individual_evaluations_saved'] = saved_count
                    print(f"✅ Saved {saved_count} individual patient-trial evaluations")
                else:
                    print("⚠️ No individual evaluations were saved")
            except Exception as e:
                print(f"⚠️ Error saving individual evaluations: {e}")
        except Exception as e:
            print(f"⚠️ Database save error (continuing with JSON): {e}")
        
        # Print summary
        print(f"\nPipeline Summary:")
        print(f"Patient: {results.get('patient_info', {}).get('mrn', 'Unknown')}")
        summary = results.get('summary', {})
        print(f"Total trials found: {summary.get('total_trials_found', 0)}")
        print(f"Trials evaluated: {summary.get('trials_evaluated', 0)}")
        print(f"Eligible trials: {summary.get('eligible_trials', 0)}")
        print(f"Average confidence: {summary.get('average_confidence', 0):.1f}%")
        print(f"Results saved to: {filepath}")
        if results.get('database_id'):
            print(f"Database ID: {results['database_id']}")
        
        return results

def main():
    """Main function to test the patient-to-trial pipeline"""
    evaluator = PatientEvaluator()
    
    # Test with a specific patient
    patient_id = 1  # Replace with actual patient ID
    results = evaluator.run_patient_trial_pipeline(
        patient_id=patient_id,
        phase_filter=["Phase I", "Phase II", "Phase III"]
    )
    
    print(f"\n🏆 ALL EVALUATED TRIALS:")
    print("=" * 100)
    
    # Show summary first
    summary = results.get('summary', {})
    print(f"📊 EVALUATION SUMMARY:")
    print(f"  Total trials evaluated: {results.get('total_evaluations', 0)}")
    print(f"  Eligible trials: {summary.get('eligible_count', 0)}")
    print(f"  Not eligible trials: {summary.get('not_eligible_count', 0)}")
    print(f"  Need more info: {summary.get('need_more_info_count', 0)}")
    print(f"  Average confidence: {summary.get('average_confidence', 0):.1f}%")
    print()
    
    # Show top recommendations
    batch_summary = results.get('batch_summary', {})
    top_recommendations = batch_summary.get('top_recommendations', [])
    if top_recommendations:
        print("🎯 TOP RECOMMENDATIONS FROM GEMINI:")
        for i, rec in enumerate(top_recommendations[:3], 1):
            print(f"  {i}. {rec}")
        print()
    
    # Show all trials
    for i, evaluation in enumerate(results['final_ranking'], 1):
        trial = evaluation['trial_info']
        status = evaluation.get('eligibility_status', 'UNKNOWN')
        confidence = evaluation.get('confidence_score', 0)
        priority = evaluation.get('priority_score', 0)
        
        # Color coding for status
        status_emoji = "✅" if status == "ELIGIBLE" else "❌" if status == "NOT_ELIGIBLE" else "❓"
        
        print(f"{i:2d}. {status_emoji} {trial['title'][:65]}...")
        print(f"    Trial ID: {trial['trial_id']}")
        print(f"    Condition: {trial['condition'][:50]}...")
        print(f"    Phase: {trial['phase']}")
        print(f"    Status: {status} | Confidence: {confidence}% | Priority: {priority}")
        print(f"    Reasoning: {evaluation.get('reasoning', 'N/A')[:90]}...")
        print()

if __name__ == "__main__":
    main()
