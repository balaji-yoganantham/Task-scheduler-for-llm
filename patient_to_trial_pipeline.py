"""
Patient-to-Trial Pipeline Orchestrator
Coordinates patient-to-trial matching components
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from services.patient_to_trial.keyword_generator import PatientKeywordGenerator
from services.patient_to_trial.patient_embedding import PatientEmbeddingGenerator
from services.patient_to_trial.patient_matcher import PatientMatcher
from services.patient_to_trial.patient_evaluator import PatientEvaluator
from services.shared.database_utils import safe_json_dump
from evaluation_results_db.utils.evaluation_results_db import EvaluationResultsDB
from config import DEFAULT_PATIENT_LIMIT

class PatientToTrialOrchestrator:
    def __init__(self):
        self.keyword_generator = PatientKeywordGenerator()
        self.embedding_generator = PatientEmbeddingGenerator()
        self.patient_matcher = PatientMatcher()
        self.patient_evaluator = PatientEvaluator()
        self.eval_db = EvaluationResultsDB()
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def run_keyword_generation(self, patient_limit: int = None) -> Dict[str, Any]:
        """Run keyword generation for patients"""
        print("=" * 80)
        print("STEP 1: PATIENT KEYWORD GENERATION")
        print("=" * 80)
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        results = self.keyword_generator.run_keyword_generation(limit=patient_limit)
        
        if results:
            print("SUCCESS: Keyword generation completed successfully!")
            return results
        else:
            print("ERROR: Keyword generation failed!")
            return {}

    def run_embedding_generation(self, patient_limit: int = None) -> Dict[str, Any]:
        """Run embedding generation for patients"""
        print("=" * 80)
        print("STEP 2: PATIENT EMBEDDING GENERATION")
        print("=" * 80)
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        results = self.embedding_generator.run_patient_embedding_generation(limit=patient_limit)
        
        if results:
            print("SUCCESS: Embedding generation completed successfully!")
            return results
        else:
            print("ERROR: Embedding generation failed!")
            return {}

    def run_patient_trial_matching(self, patient_id: int, age_range: Tuple[int, int] = None, 
                                 gender: str = None, phase_filter: list = None,
                                 max_distance_km: Optional[float] = None,
                                 location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run patient-to-trial matching pipeline"""
        print("=" * 80)
        print("STEP 3: PATIENT-TO-TRIAL MATCHING")
        print("=" * 80)
        
        # Only apply filters if explicitly provided
        filters_applied = {}
        if age_range is not None:
            filters_applied["age_range"] = age_range
        if gender is not None:
            filters_applied["gender"] = gender
        if phase_filter is not None:
            filters_applied["phase_filter"] = phase_filter
        if max_distance_km is not None:
            filters_applied["max_distance_km"] = max_distance_km
        if location_weight is not None:
            filters_applied["location_weight"] = location_weight
            
        print(f"Filters applied: {filters_applied if filters_applied else 'None (processing all trials)'}")
        
        results = self.patient_evaluator.run_patient_trial_pipeline(
            patient_id=patient_id,
            age_range=age_range,
            gender=gender,
            phase_filter=phase_filter,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )
        
        if results:
            print("SUCCESS: Patient-to-trial matching completed successfully!")
            return results
        else:
            print("ERROR: Patient-to-trial matching failed!")
            return {}

    def run_complete_pipeline(self, patient_id: int, patient_limit: int = None, 
                            age_range: Tuple[int, int] = None, gender: str = None, 
                            phase_filter: list = None,
                            max_distance_km: Optional[float] = None,
                            location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete patient-to-trial matching pipeline"""
        print("STARTING PATIENT-TO-TRIAL MATCHING PIPELINE")
        print(f"Patient ID: {patient_id}")
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        print(f"Patient Limit: {patient_limit}")
        print(f"Age Range: {age_range if age_range else 'Not specified (processing all trials)'}")
        print(f"Gender Filter: {gender if gender else 'Not specified (processing all trials)'}")
        print(f"Phase Filter: {phase_filter if phase_filter else 'Not specified (processing all trials)'}")
        print("=" * 80)
        
        pipeline_results = {
            "pipeline_started_at": datetime.now().isoformat(),
            "patient_id": patient_id,
            "patient_limit": patient_limit,
            "filters": {
                "age_range": age_range,
                "gender": gender,
                "phase_filter": phase_filter,
                "max_distance_km": max_distance_km,
                "location_weight": location_weight
            },
            "steps": {}
        }
        
        try:
            # Step 1: Keyword Generation
            keyword_results = self.run_keyword_generation(patient_limit)
            pipeline_results["steps"]["keyword_generation"] = {
                "status": "completed" if keyword_results else "failed",
                "results": keyword_results
            }
            
            if not keyword_results:
                print("ERROR: Pipeline stopped due to keyword generation failure")
                return pipeline_results
            
            # Step 2: Embedding Generation
            embedding_results = self.run_embedding_generation(patient_limit)
            pipeline_results["steps"]["embedding_generation"] = {
                "status": "completed" if embedding_results else "failed",
                "results": embedding_results
            }
            
            if not embedding_results:
                print("ERROR: Pipeline stopped due to embedding generation failure")
                return pipeline_results
            
            # Step 3: Patient-to-Trial Matching
            matching_results = self.run_patient_trial_matching(
                patient_id, age_range, gender, phase_filter, max_distance_km, location_weight
            )
            pipeline_results["steps"]["patient_trial_matching"] = {
                "status": "completed" if matching_results else "failed",
                "results": matching_results
            }
            
            # Pipeline Summary
            pipeline_results["pipeline_completed_at"] = datetime.now().isoformat()
            pipeline_results["overall_status"] = "completed"
            
            # Save pipeline results to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_file = self.results_dir / f"patient_to_trial_pipeline_{patient_id}_{timestamp}.json"
            
            safe_json_dump(pipeline_results, pipeline_file, indent=2, ensure_ascii=False)
            
            # Save evaluation results to database
            print(f"\n💾 Saving evaluation results to database...")
            pipeline_results["database_save_status"] = "failed"
            pipeline_results["database_id"] = None
            pipeline_results["database_save_error"] = None
            
            try:
                if matching_results and matching_results.get('patient_info'):
                    db_id = self.eval_db.save_patient_to_trial_evaluation(matching_results)
                    if db_id:
                        pipeline_results["database_save_status"] = "success"
                        pipeline_results["database_id"] = db_id
                        print(f"✅ Evaluation results saved to database with ID: {db_id}")
                    else:
                        pipeline_results["database_save_error"] = "Database save returned None"
                        print("⚠️ Failed to save evaluation results to database")
                else:
                    pipeline_results["database_save_error"] = "No matching results to save"
                    print("⚠️ No matching results available to save to database")
            except Exception as e:
                pipeline_results["database_save_error"] = str(e)
                print(f"⚠️ Database save error (continuing with JSON): {e}")
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Pipeline results saved to: {pipeline_file}")
            if pipeline_results["database_save_status"] == "success":
                print(f"Database ID: {pipeline_results['database_id']}")
            else:
                print(f"Database save failed: {pipeline_results.get('database_save_error', 'Unknown error')}")
            
            # Print final summary
            if matching_results:
                summary = matching_results.get('summary', {})
                print(f"\nFinal Summary:")
                print(f"  Patient: {matching_results.get('patient_info', {}).get('mrn', 'Unknown')}")
                print(f"  Total trials found: {summary.get('total_trials_found', 0)}")
                print(f"  Trials evaluated: {summary.get('trials_evaluated', 0)}")
                print(f"  Eligible trials: {summary.get('eligible_trials', 0)}")
                print(f"  Average confidence: {summary.get('average_confidence', 0):.1f}%")
                
                # Show all evaluated trials with detailed results
                final_ranking = matching_results.get('final_ranking', [])
                if final_ranking:
                    print(f"\n🏆 ALL {len(final_ranking)} TRIALS EVALUATED BY GEMINI:")
                    print("=" * 100)
                    
                    # Show top recommendations first
                    batch_summary = matching_results.get('batch_summary', {})
                    top_recommendations = batch_summary.get('top_recommendations', [])
                    if top_recommendations:
                        print("🎯 TOP RECOMMENDATIONS FROM GEMINI:")
                        for i, rec in enumerate(top_recommendations[:3], 1):
                            print(f"  {i}. {rec}")
                        print()
                    
                    # Show detailed results for all trials
                    for i, evaluation in enumerate(final_ranking, 1):
                        trial_info = evaluation.get('trial_info', {})
                        status = evaluation.get('eligibility_status', 'UNKNOWN')
                        confidence = evaluation.get('confidence_score', 0)
                        priority = evaluation.get('priority_score', 0)
                        
                        # Color coding for status
                        status_emoji = "✅" if status == "ELIGIBLE" else "❌" if status == "NOT_ELIGIBLE" else "❓"
                        
                        print(f"{i:2d}. {status_emoji} {trial_info.get('title', 'Unknown')[:65]}...")
                        print(f"    Trial ID: {trial_info.get('trial_id', 'Unknown')}")
                        print(f"    Condition: {trial_info.get('condition', 'Unknown')[:50]}...")
                        print(f"    Phase: {trial_info.get('phase', 'Unknown')}")
                        print(f"    Status: {status} | Confidence: {confidence}% | Priority: {priority}")
                        print(f"    Reasoning: {evaluation.get('reasoning', 'N/A')[:90]}...")
                        print()
            
            return pipeline_results
            
        except Exception as e:
            print(f"ERROR: Pipeline failed with error: {e}")
            pipeline_results["pipeline_failed_at"] = datetime.now().isoformat()
            pipeline_results["overall_status"] = "failed"
            pipeline_results["error"] = str(e)
            return pipeline_results

    def run_individual_step(self, step: str, **kwargs) -> Dict[str, Any]:
        """Run individual pipeline steps"""
        if step == "keywords":
            return self.run_keyword_generation(kwargs.get('patient_limit'))
        elif step == "embeddings":
            return self.run_embedding_generation(kwargs.get('patient_limit'))
        elif step == "matching":
            return self.run_patient_trial_matching(
                kwargs.get('patient_id'),
                kwargs.get('age_range'),
                kwargs.get('gender'),
                kwargs.get('phase_filter')
            )
        else:
            print(f"Unknown step: {step}")
            return {}

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Patient-to-Trial Matching Pipeline")
    
    # Pipeline options
    parser.add_argument("--patient-id", type=int, required=True, help="Patient ID to find trials for")
    parser.add_argument("--patient-limit", type=int, help=f"Number of patients to process for embeddings (default: {DEFAULT_PATIENT_LIMIT})")
    parser.add_argument("--age-min", type=int, help="Minimum age filter (optional - if not specified, processes all trials)")
    parser.add_argument("--age-max", type=int, help="Maximum age filter (optional - if not specified, processes all trials)")
    parser.add_argument("--gender", choices=['Male', 'Female'], help="Gender filter (optional - if not specified, processes all trials)")
    parser.add_argument("--phase-filter", nargs='+', help="Phase filter (optional - e.g., Phase I Phase II)")
    
    # Individual step options
    parser.add_argument("--step", choices=['keywords', 'embeddings', 'matching'], 
                       help="Run only a specific step")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PatientToTrialOrchestrator()
    
    # Prepare age range
    age_range = None
    if args.age_min and args.age_max:
        age_range = (args.age_min, args.age_max)
    
    # Run pipeline or individual step
    if args.step:
        print(f"Running individual step: {args.step}")
        results = orchestrator.run_individual_step(
            args.step,
            patient_id=args.patient_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender,
            phase_filter=args.phase_filter
        )
    else:
        print("Running complete patient-to-trial pipeline")
        results = orchestrator.run_complete_pipeline(
            patient_id=args.patient_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender,
            phase_filter=args.phase_filter
        )
    
    if results:
        print("\nSUCCESS: Operation completed successfully!")
    else:
        print("\nERROR: Operation failed!")

if __name__ == "__main__":
    main()
