"""
Run LLM Evaluation for Patient-to-Trial Matching Results
Evaluates all patient-trial matches using LLM and saves results
"""

import json
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from test_accuracy_with_json_data import JSONDataLoader
from services.shared.llm_utils import LLMUtils


class PatientToTrialLLMEvaluator:
    """Run LLM evaluation for patient-to-trial matches"""
    
    def __init__(self, data_dir: Path, results_dir: Path):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.llm_utils = LLMUtils()
        self.loader = JSONDataLoader(data_dir)
        
        # Load data
        patients_list = self.loader.load_patients()
        self.patients = {p['mrn']: p for p in patients_list}
        self.trials = {t['trial_id']: t for t in self.loader.load_trials()}
        
        # Add missing fields to patients for LLM evaluation
        for mrn, patient in self.patients.items():
            if 'raw_data' in patient:
                raw = patient['raw_data']
                if 'oncologist' not in patient and 'Oncologist' in raw:
                    patient['oncologist'] = raw['Oncologist']
                if 'date_of_visit' not in patient and 'Date_of_Visit' in raw:
                    patient['date_of_visit'] = raw['Date_of_Visit']
            if not patient.get('age'):
                patient['age'] = 65
            if not patient.get('gender'):
                patient['gender'] = 'Unknown'
            if not patient.get('oncologist'):
                patient['oncologist'] = 'Unknown'
            if not patient.get('date_of_visit'):
                patient['date_of_visit'] = 'Unknown'
    
    def evaluate_all_matches(self, matches_file: Path) -> Dict[str, Any]:
        """Evaluate all patient-trial matches using LLM"""
        print("\n" + "=" * 80)
        print("LLM Evaluation for Patient-to-Trial Matches")
        print("=" * 80)
        
        # Load matching results
        with open(matches_file, 'r', encoding='utf-8') as f:
            matching_results = json.load(f)
        
        all_evaluations = {}
        summary = {
            'total_evaluations': 0,
            'eligible_count': 0,
            'not_eligible_count': 0,
            'need_more_info_count': 0,
            'error_count': 0,
            'patients_processed': 0
        }
        
        results = matching_results.get('results', {})
        
        # Process each patient
        for mrn, patient_result in results.items():
            patient = self.patients.get(mrn)
            if not patient:
                print(f"Warning: Patient {mrn} not found")
                continue
            
            matches = patient_result.get('matches', [])
            if not matches:
                print(f"\nPatient {mrn}: No matches to evaluate")
                continue
            
            print(f"\n{'='*80}")
            print(f"Patient {mrn} - Evaluating {len(matches)} trial matches...")
            print(f"{'='*80}")
            
            patient_evaluations = []
            
            for i, trial_data in enumerate(matches, 1):
                trial_id = trial_data.get('trial_id', '')
                if not trial_id:
                    continue
                
                trial = self.trials.get(trial_id)
                if not trial:
                    print(f"  [{i}/{len(matches)}] Warning: Trial {trial_id} not found")
                    continue
                
                # Merge trial data
                full_trial = trial.copy()
                full_trial.update(trial_data)
                if 'raw_data' in trial:
                    full_trial['raw_data'] = trial['raw_data']
                    raw = trial['raw_data']
                    if 'inclusion_criteria' not in full_trial and 'inclusion_criteria' in raw:
                        full_trial['inclusion_criteria'] = raw['inclusion_criteria']
                    if 'exclusion_criteria' not in full_trial and 'exclusion_criteria' in raw:
                        full_trial['exclusion_criteria'] = raw['exclusion_criteria']
                    if 'brief_summary' not in full_trial and 'brief_summary' in raw:
                        full_trial['brief_summary'] = raw['brief_summary']
                    if 'detailed_description' not in full_trial and 'detailed_description' in raw:
                        full_trial['detailed_description'] = raw['detailed_description']
                    if 'minimum_age' not in full_trial and 'minimum_age' in raw:
                        full_trial['minimum_age'] = raw['minimum_age']
                    if 'maximum_age' not in full_trial and 'maximum_age' in raw:
                        full_trial['maximum_age'] = raw['maximum_age']
                    if 'sex' not in full_trial and 'sex' in raw:
                        full_trial['sex'] = raw['sex']
                
                print(f"\n  [{i}/{len(matches)}] Evaluating trial {trial_id}...")
                print(f"      Title: {trial.get('title', 'Unknown')[:60]}...")
                
                try:
                    # Get LLM evaluation
                    llm_result = self.llm_utils.evaluate_patient_trial_match(
                        full_trial,
                        patient
                    )
                    
                    eligibility_status = llm_result.get('eligibility_status', 'UNKNOWN')
                    confidence = llm_result.get('confidence_score', 0)
                    reasoning = llm_result.get('reasoning', '')[:200] + "..." if len(llm_result.get('reasoning', '')) > 200 else llm_result.get('reasoning', '')
                    
                    print(f"      Status: {eligibility_status}")
                    print(f"      Confidence: {confidence}")
                    print(f"      Reasoning: {reasoning}")
                    
                    # Update summary
                    summary['total_evaluations'] += 1
                    if eligibility_status == 'ELIGIBLE':
                        summary['eligible_count'] += 1
                    elif eligibility_status == 'NOT_ELIGIBLE':
                        summary['not_eligible_count'] += 1
                    elif eligibility_status == 'NEED_MORE_INFO':
                        summary['need_more_info_count'] += 1
                    else:
                        summary['error_count'] += 1
                    
                    # Store evaluation
                    evaluation = {
                        'trial_id': trial_id,
                        'trial_title': trial.get('title', ''),
                        'hybrid_score': trial_data.get('hybrid_score', 0),
                        'embedding_score': trial_data.get('embedding_score', 0),
                        'bm25_score': trial_data.get('bm25_score', 0),
                        'llm_evaluation': {
                            'eligibility_status': eligibility_status,
                            'confidence_score': confidence,
                            'reasoning': llm_result.get('reasoning', ''),
                            'inclusion_criteria_met': llm_result.get('inclusion_criteria_met', []),
                            'exclusion_criteria_violated': llm_result.get('exclusion_criteria_violated', []),
                            'inclusion_criteria_met_count': llm_result.get('inclusion_criteria_met_count', 0),
                            'exclusion_criteria_violated_count': llm_result.get('exclusion_criteria_violated_count', 0),
                            'recommendations': llm_result.get('recommendations', '')
                        }
                    }
                    
                    patient_evaluations.append(evaluation)
                    
                except Exception as e:
                    error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                    print(f"      ERROR: {error_msg}")
                    summary['error_count'] += 1
                    summary['total_evaluations'] += 1
                    
                    evaluation = {
                        'trial_id': trial_id,
                        'trial_title': trial.get('title', ''),
                        'hybrid_score': trial_data.get('hybrid_score', 0),
                        'error': error_msg,
                        'llm_evaluation': None
                    }
                    patient_evaluations.append(evaluation)
            
            all_evaluations[mrn] = {
                'patient_info': {
                    'mrn': mrn,
                    'patient_id': patient['patient_id'],
                    'age': patient.get('age'),
                    'gender': patient.get('gender')
                },
                'evaluations': patient_evaluations,
                'total_trials_evaluated': len(patient_evaluations),
                'eligible_trials': len([e for e in patient_evaluations if e.get('llm_evaluation', {}).get('eligibility_status') == 'ELIGIBLE']),
                'not_eligible_trials': len([e for e in patient_evaluations if e.get('llm_evaluation', {}).get('eligibility_status') == 'NOT_ELIGIBLE']),
                'need_more_info_trials': len([e for e in patient_evaluations if e.get('llm_evaluation', {}).get('eligibility_status') == 'NEED_MORE_INFO'])
            }
            
            summary['patients_processed'] += 1
            
            # Print patient summary
            eligible = all_evaluations[mrn]['eligible_trials']
            not_eligible = all_evaluations[mrn]['not_eligible_trials']
            need_info = all_evaluations[mrn]['need_more_info_trials']
            print(f"\n  Patient {mrn} Summary:")
            print(f"    ELIGIBLE: {eligible}")
            print(f"    NOT_ELIGIBLE: {not_eligible}")
            print(f"    NEED_MORE_INFO: {need_info}")
        
        return {
            'summary': summary,
            'evaluations': all_evaluations,
            'generated_at': datetime.now().isoformat()
        }


def main():
    """Main function"""
    print("=" * 80)
    print("LLM Evaluation for Patient-to-Trial Matching")
    print("=" * 80)
    
    # Setup paths
    data_dir = Path("PATient data")
    results_dir = Path("test_results")
    
    # Find the most recent patient-to-trial results
    test_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("patient_to_trial_all_")], reverse=True)
    if not test_dirs:
        print("Error: No patient-to-trial matching results found.")
        print("Please run run_patient_to_trial_all.py first.")
        return
    
    latest_results_dir = test_dirs[0]
    matches_file = latest_results_dir / "patient_to_trial_results.json"
    
    print(f"\nUsing matching results from: {latest_results_dir}")
    
    if not matches_file.exists():
        print(f"Error: Results file not found: {matches_file}")
        return
    
    # Create LLM evaluation results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_results_dir = results_dir / f"llm_evaluation_p2t_{timestamp}"
    llm_results_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = PatientToTrialLLMEvaluator(data_dir, llm_results_dir)
    
    # Run evaluations
    print("\n" + "=" * 80)
    print("Starting LLM Evaluations...")
    print("=" * 80)
    print("Note: This will make LLM API calls and may take some time.")
    print("=" * 80)
    
    results = evaluator.evaluate_all_matches(matches_file)
    
    # Save results
    results_file = llm_results_dir / "llm_evaluations.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("LLM EVALUATION SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    print(f"\nTotal Evaluations: {summary['total_evaluations']}")
    print(f"Patients Processed: {summary['patients_processed']}")
    print(f"\nEligibility Decisions:")
    print(f"  ELIGIBLE: {summary['eligible_count']} ({summary['eligible_count']/summary['total_evaluations']*100:.1f}%)" if summary['total_evaluations'] > 0 else "  ELIGIBLE: 0")
    print(f"  NOT_ELIGIBLE: {summary['not_eligible_count']} ({summary['not_eligible_count']/summary['total_evaluations']*100:.1f}%)" if summary['total_evaluations'] > 0 else "  NOT_ELIGIBLE: 0")
    print(f"  NEED_MORE_INFO: {summary['need_more_info_count']} ({summary['need_more_info_count']/summary['total_evaluations']*100:.1f}%)" if summary['total_evaluations'] > 0 else "  NEED_MORE_INFO: 0")
    print(f"  ERRORS: {summary['error_count']}")
    
    # Show per-patient summary
    print("\n" + "=" * 80)
    print("PER-PATIENT SUMMARY")
    print("=" * 80)
    for mrn, patient_data in results['evaluations'].items():
        patient_info = patient_data['patient_info']
        print(f"\nPatient {mrn} (Age: {patient_info.get('age', 'Unknown')}):")
        print(f"  Total Trials Evaluated: {patient_data['total_trials_evaluated']}")
        print(f"  ELIGIBLE: {patient_data['eligible_trials']}")
        print(f"  NOT_ELIGIBLE: {patient_data['not_eligible_trials']}")
        print(f"  NEED_MORE_INFO: {patient_data['need_more_info_trials']}")
        
        # Show eligible trials
        eligible_trials = [e for e in patient_data['evaluations'] if e.get('llm_evaluation', {}).get('eligibility_status') == 'ELIGIBLE']
        if eligible_trials:
            print(f"  Eligible Trials:")
            for trial in eligible_trials:
                trial_id = trial['trial_id']
                confidence = trial.get('llm_evaluation', {}).get('confidence_score', 0)
                print(f"    - {trial_id} (Confidence: {confidence})")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

