"""
Main Orchestrator for Clinical Trial Matching System
This script coordinates both patient-to-trial and trial-to-patient matching pipelines
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from patient_to_trial_pipeline import PatientToTrialOrchestrator
from trial_to_patient_pipeline import TrialToPatientOrchestrator

class ClinicalTrialMatchingOrchestrator:
    def __init__(self):
        self.patient_to_trial_orchestrator = PatientToTrialOrchestrator()
        self.trial_to_patient_orchestrator = TrialToPatientOrchestrator()
        
        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def run_patient_to_trial_pipeline(self, patient_id: int, patient_limit: int = 50, 
                                    age_range: Tuple[int, int] = None, gender: str = None, 
                                    phase_filter: list = None,
                                    max_distance_km: Optional[float] = None,
                                    location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run patient-to-trial matching pipeline"""
        print("=" * 80)
        print("RUNNING PATIENT-TO-TRIAL PIPELINE")
        print("=" * 80)
        
        return self.patient_to_trial_orchestrator.run_complete_pipeline(
            patient_id=patient_id,
            patient_limit=patient_limit,
            age_range=age_range,
            gender=gender,
            phase_filter=phase_filter,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )

    def run_trial_to_patient_pipeline(self, trial_id: str, age_range: Tuple[int, int] = None, 
                                    gender: str = None,
                                    max_distance_km: Optional[float] = None,
                                    location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Run trial-to-patient matching pipeline"""
        print("=" * 80)
        print("RUNNING TRIAL-TO-PATIENT PIPELINE")
        print("=" * 80)
        
        return self.trial_to_patient_orchestrator.run_complete_pipeline(
            trial_id=trial_id,
            age_range=age_range,
            gender=gender,
            max_distance_km=max_distance_km,
            location_weight=location_weight
        )

    def run_both_pipelines(self, patient_id: int, trial_id: str, patient_limit: int = 50,
                          age_range: Tuple[int, int] = None, gender: str = None, 
                          phase_filter: list = None) -> Dict[str, Any]:
        """Run both patient-to-trial and trial-to-patient pipelines"""
        print("🚀 STARTING COMPLETE CLINICAL TRIAL MATCHING SYSTEM")
        print(f"Patient ID: {patient_id}")
        print(f"Trial ID: {trial_id}")
        print(f"Patient Limit: {patient_limit}")
        print(f"Age Range: {age_range}")
        print(f"Gender Filter: {gender}")
        print(f"Phase Filter: {phase_filter}")
        print("=" * 80)
        
        pipeline_results = {
            "pipeline_started_at": datetime.now().isoformat(),
            "patient_id": patient_id,
            "trial_id": trial_id,
            "patient_limit": patient_limit,
            "filters": {
                "age_range": age_range,
                "gender": gender,
                "phase_filter": phase_filter
            },
            "pipelines": {}
        }
        
        try:
            # Run Patient-to-Trial Pipeline
            patient_to_trial_results = self.run_patient_to_trial_pipeline(
                patient_id=patient_id,
                patient_limit=patient_limit,
                age_range=age_range,
                gender=gender,
                phase_filter=phase_filter
            )
            
            pipeline_results["pipelines"]["patient_to_trial"] = {
                "status": "completed" if patient_to_trial_results else "failed",
                "results": patient_to_trial_results
            }
            
            # Run Trial-to-Patient Pipeline
            trial_to_patient_results = self.run_trial_to_patient_pipeline(
                trial_id=trial_id,
                age_range=age_range,
                gender=gender
            )
            
            pipeline_results["pipelines"]["trial_to_patient"] = {
                "status": "completed" if trial_to_patient_results else "failed",
                "results": trial_to_patient_results
            }
            
            # Pipeline Summary
            pipeline_results["pipeline_completed_at"] = datetime.now().isoformat()
            pipeline_results["overall_status"] = "completed"
            
            # Save pipeline results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_file = self.results_dir / f"complete_matching_pipeline_{patient_id}_{trial_id}_{timestamp}.json"
            
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n🎉 COMPLETE PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Pipeline results saved to: {pipeline_file}")
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            pipeline_results["pipeline_failed_at"] = datetime.now().isoformat()
            pipeline_results["overall_status"] = "failed"
            pipeline_results["error"] = str(e)
            return pipeline_results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Clinical Trial Matching Pipeline")
    
    # Pipeline options
    parser.add_argument("--patient-id", type=int, help="Patient ID to find trials for")
    parser.add_argument("--trial-id", help="Trial ID to find patients for")
    parser.add_argument("--patient-limit", type=int, default=50, help="Number of patients to process")
    parser.add_argument("--age-min", type=int, help="Minimum age filter")
    parser.add_argument("--age-max", type=int, help="Maximum age filter")
    parser.add_argument("--gender", choices=['Male', 'Female'], help="Gender filter")
    parser.add_argument("--phase-filter", nargs='+', help="Phase filter (e.g., Phase I Phase II)")
    
    # Pipeline type
    parser.add_argument("--pipeline-type", choices=['patient-to-trial', 'trial-to-patient', 'both'], 
                       default='both', help="Type of pipeline to run")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ClinicalTrialMatchingOrchestrator()
    
    # Prepare age range
    age_range = None
    if args.age_min and args.age_max:
        age_range = (args.age_min, args.age_max)
    
    # Run appropriate pipeline
    if args.pipeline_type == 'patient-to-trial':
        if not args.patient_id:
            print("Error: --patient-id is required for patient-to-trial pipeline")
            return
        
        print("Running patient-to-trial pipeline")
        results = orchestrator.run_patient_to_trial_pipeline(
            patient_id=args.patient_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender,
            phase_filter=args.phase_filter
        )
        
    elif args.pipeline_type == 'trial-to-patient':
        if not args.trial_id:
            print("Error: --trial-id is required for trial-to-patient pipeline")
            return
        
        print("Running trial-to-patient pipeline")
        results = orchestrator.run_trial_to_patient_pipeline(
            trial_id=args.trial_id,
            age_range=age_range,
            gender=args.gender
        )
        
    else:  # both
        if not args.patient_id or not args.trial_id:
            print("Error: Both --patient-id and --trial-id are required for complete pipeline")
            return
        
        print("Running both patient-to-trial and trial-to-patient pipelines")
        results = orchestrator.run_both_pipelines(
            patient_id=args.patient_id,
            trial_id=args.trial_id,
            patient_limit=args.patient_limit,
            age_range=age_range,
            gender=args.gender,
            phase_filter=args.phase_filter
        )
    
    if results:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")

if __name__ == "__main__":
    main()
