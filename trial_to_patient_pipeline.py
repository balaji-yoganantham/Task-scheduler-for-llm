"""
Trial-to-Patient Pipeline Orchestrator
Coordinates trial-to-patient matching components following the same pattern as patient-to-trial pipeline
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from services.patient_to_trial.keyword_generator import PatientKeywordGenerator
from services.patient_to_trial.patient_embedding import PatientEmbeddingGenerator
from services.trial_to_patient.trial_evaluator import TrialEvaluator
from services.shared.database_utils import safe_json_dump
from evaluation_results_db.utils.evaluation_results_db import EvaluationResultsDB
from config import DEFAULT_PATIENT_LIMIT

class TrialToPatientOrchestrator:
    def __init__(self):
        # Reuse patient components for keyword and embedding generation
        self.keyword_generator = PatientKeywordGenerator()
        self.embedding_generator = PatientEmbeddingGenerator()
        self.trial_evaluator = TrialEvaluator()
        self.eval_db = EvaluationResultsDB()
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def run_patient_keyword_generation(self, patient_limit: int = None) -> Dict[str, Any]:
        """Run keyword generation for patients (Step 1)"""
        print("=" * 80)
        print("STEP 1: PATIENT KEYWORD GENERATION")
        print("=" * 80)
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        results = self.keyword_generator.run_keyword_generation(limit=patient_limit)
        
        if results:
            print("SUCCESS: Patient keyword generation completed successfully!")
            return results
        else:
            print("ERROR: Patient keyword generation failed!")
            return {}

    def run_patient_embedding_generation(self, patient_limit: int = None) -> Dict[str, Any]:
        """Run embedding generation for patients (Step 2)"""
        print("=" * 80)
        print("STEP 2: PATIENT EMBEDDING GENERATION")
        print("=" * 80)
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        results = self.embedding_generator.run_patient_embedding_generation(limit=patient_limit)
        
        if results:
            print("SUCCESS: Patient embedding generation completed successfully!")
            return results
        else:
            print("ERROR: Patient embedding generation failed!")
            return {}

    def run_trial_patient_matching(self, trial_id: str, age_range: Tuple[int, int] = None, 
                                  gender: str = None,
                                  max_distance_km: Optional[float] = None,
                                  location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run trial-to-patient matching pipeline (Step 3)"""
        print("=" * 80)
        print("STEP 3: TRIAL-TO-PATIENT MATCHING")
        print("=" * 80)
        
        # Only apply filters if explicitly provided
        filters_applied = {}
        if age_range is not None:
            filters_applied["age_range"] = age_range
        if gender is not None:
            filters_applied["gender"] = gender
        if max_distance_km is not None:
            filters_applied["max_distance_km"] = max_distance_km
        if location_weight is not None:
            filters_applied["location_weight"] = location_weight
            
        print(f"Filters applied: {filters_applied if filters_applied else 'None (processing all patients)'}")
        
        results = self.trial_evaluator.run_trial_matching_pipeline(
            trial_id=trial_id,
            age_range=age_range,
            gender=gender,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )
        
        if results:
            print("SUCCESS: Trial-to-patient matching completed successfully!")
            return results
        else:
            print("ERROR: Trial-to-patient matching failed!")
            return {}

    def run_complete_pipeline(self, trial_id: str, patient_limit: int = None, 
                            age_range: Tuple[int, int] = None, gender: str = None,
                            max_distance_km: Optional[float] = None,
                            location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete trial-to-patient matching pipeline"""
        print("STARTING TRIAL-TO-PATIENT MATCHING PIPELINE")
        print(f"Trial ID: {trial_id}")
        
        # Use config default if not specified
        if patient_limit is None:
            patient_limit = DEFAULT_PATIENT_LIMIT
            
        print(f"Patient Limit: {patient_limit}")
        print(f"Age Range: {age_range if age_range else 'Not specified (processing all patients)'}")
        print(f"Gender Filter: {gender if gender else 'Not specified (processing all patients)'}")
        print("=" * 80)
        
        pipeline_results = {
            "pipeline_started_at": datetime.now().isoformat(),
            "trial_id": trial_id,
            "patient_limit": patient_limit,
            "filters": {
                "age_range": age_range,
                "gender": gender,
                "max_distance_km": max_distance_km,
                "location_weight": location_weight
            },
            "steps": {}
        }
        
        try:
            # Step 1: Patient Keyword Generation
            keyword_results = self.run_patient_keyword_generation(patient_limit)
            pipeline_results["steps"]["patient_keyword_generation"] = {
                "status": "completed" if keyword_results else "failed",
                "results": keyword_results
            }
            
            if not keyword_results:
                print("ERROR: Pipeline stopped due to patient keyword generation failure")
                return pipeline_results
            
            # Step 2: Patient Embedding Generation
            embedding_results = self.run_patient_embedding_generation(patient_limit)
            pipeline_results["steps"]["patient_embedding_generation"] = {
                "status": "completed" if embedding_results else "failed",
                "results": embedding_results
            }
            
            if not embedding_results:
                print("ERROR: Pipeline stopped due to patient embedding generation failure")
                return pipeline_results
            
            # Step 3: Trial-to-Patient Matching
            matching_results = self.run_trial_patient_matching(
                trial_id, age_range, gender, max_distance_km, location_weight
            )
            pipeline_results["steps"]["trial_patient_matching"] = {
                "status": "completed" if matching_results else "failed",
                "results": matching_results
            }
            
            # Pipeline Summary
            pipeline_results["pipeline_completed_at"] = datetime.now().isoformat()
            pipeline_results["overall_status"] = "completed"
            
            # Save pipeline results to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_file = self.results_dir / f"trial_to_patient_pipeline_{trial_id}_{timestamp}.json"
            
            safe_json_dump(pipeline_results, pipeline_file, indent=2, ensure_ascii=False)
            
            # Save evaluation results to database
            print(f"\n💾 Saving evaluation results to database...")
            pipeline_results["database_save_status"] = "failed"
            pipeline_results["database_id"] = None
            pipeline_results["database_save_error"] = None
            
            try:
                if matching_results and matching_results.get('trial_info'):
                    db_id = self.eval_db.save_trial_to_patient_evaluation(matching_results)
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
            
            # Save individual trial-patient evaluations to normalized table
            print(f"\n💾 Saving individual trial-patient evaluations...")
            try:
                saved_count = self.eval_db.save_trial_patient_evaluations(matching_results)
                if saved_count > 0:
                    pipeline_results["individual_evaluations_saved"] = saved_count
                    print(f"✅ Saved {saved_count} individual trial-patient evaluations")
                else:
                    print("⚠️ No individual evaluations were saved")
            except Exception as e:
                print(f"⚠️ Error saving individual evaluations: {e}")
                pipeline_results["individual_evaluations_error"] = str(e)
            
            print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Pipeline results saved to: {pipeline_file}")
            if pipeline_results["database_save_status"] == "success":
                print(f"Database ID: {pipeline_results['database_id']}")
            else:
                print(f"Database save failed: {pipeline_results.get('database_save_error', 'Unknown error')}")
            
            # Print final summary
            if matching_results:
                summary = matching_results.get('summary', {})
                print(f"\nFinal Summary:")
                print(f"  Trial: {matching_results.get('trial_info', {}).get('title', 'Unknown')}")
                print(f"  Total patients found: {summary.get('total_patients_found', 0)}")
                print(f"  Patients evaluated: {summary.get('patients_evaluated', 0)}")
                print(f"  Eligible patients: {summary.get('eligible_patients', 0)}")
                print(f"  Average confidence: {summary.get('average_confidence', 0):.1f}%")
                
                # Show all evaluated patients with detailed results
                final_ranking = matching_results.get('final_ranking', [])
                if final_ranking:
                    print(f"\nALL {len(final_ranking)} PATIENTS EVALUATED BY GEMINI:")
                    print("=" * 100)
                    
                    # Show detailed results for all patients
                    for i, evaluation in enumerate(final_ranking, 1):
                        patient_info = evaluation.get('patient_info', {})
                        status = evaluation.get('eligibility_status', 'UNKNOWN')
                        confidence = evaluation.get('confidence_score', 0)
                        priority = evaluation.get('priority_score', 0)
                        
                        # Color coding for status
                        status_emoji = "OK" if status == "ELIGIBLE" else "NO" if status == "NOT_ELIGIBLE" else "?"
                        
                        print(f"{i:2d}. {status_emoji} MRN: {patient_info.get('mrn', 'Unknown')}")
                        print(f"    Age: {patient_info.get('age', 'Unknown')}, Gender: {patient_info.get('gender', 'Unknown')}")
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
            return self.run_patient_keyword_generation(kwargs.get('patient_limit'))
        elif step == "embeddings":
            return self.run_patient_embedding_generation(kwargs.get('patient_limit'))
        elif step == "matching":
            return self.run_trial_patient_matching(
                kwargs.get('trial_id'),
                kwargs.get('age_range'),
                kwargs.get('gender')
            )
        else:
            print(f"Unknown step: {step}")
            return {}

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Trial-to-Patient Matching Pipeline")
    
    # Pipeline options
    parser.add_argument("--trial-id", type=str, required=True, help="Trial ID to find patients for")
    parser.add_argument("--patient-limit", type=int, help=f"Number of patients to process for embeddings (default: {DEFAULT_PATIENT_LIMIT})")
    parser.add_argument("--age-min", type=int, help="Minimum age filter (optional - if not specified, processes all patients)")
    parser.add_argument("--age-max", type=int, help="Maximum age filter (optional - if not specified, processes all patients)")
    parser.add_argument("--gender", choices=['Male', 'Female'], help="Gender filter (optional - if not specified, processes all patients)")
    
    # Individual step options
    parser.add_argument("--step", choices=['keywords', 'embeddings', 'matching'], 
                       help="Run only a specific step")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TrialToPatientOrchestrator()
    
    # Prepare age range
    age_range = None
    if args.age_min and args.age_max:
        age_range = (args.age_min, args.age_max)
    
    # Run pipeline or individual step
    if args.step:
        print(f"Running individual step: {args.step}")
        results = orchestrator.run_individual_step(
            args.step,
            trial_id=args.trial_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender
        )
    else:
        print("Running complete trial-to-patient pipeline")
        results = orchestrator.run_complete_pipeline(
            trial_id=args.trial_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender
        )
    
    if results:
        print("\nSUCCESS: Operation completed successfully!")
    else:
        print("\nERROR: Operation failed!")

if __name__ == "__main__":
    main()