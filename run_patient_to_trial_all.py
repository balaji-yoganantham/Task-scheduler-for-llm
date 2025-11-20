"""
Run Patient-to-Trial Matching for All Patients
Loads all patients from JSON and runs matching, saves results
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

from test_accuracy_with_json_data import JSONDataLoader, FileBasedMatcher


def main():
    """Run patient-to-trial matching for all patients"""
    print("=" * 80)
    print("Patient-to-Trial Matching for All Patients")
    print("=" * 80)
    
    # Setup paths
    data_dir = Path("PATient data")
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_results_dir = results_dir / f"patient_to_trial_all_{timestamp}"
    test_results_dir.mkdir(exist_ok=True)
    
    print(f"\n1. Loading data from {data_dir}...")
    loader = JSONDataLoader(data_dir)
    patients = loader.load_patients()
    trials = loader.load_trials()
    
    print(f"   Loaded {len(patients)} patients")
    print(f"   Loaded {len(trials)} trials")
    
    print(f"\n2. Building matching indices...")
    matcher = FileBasedMatcher(patients, trials)
    
    print(f"\n3. Running Patient-to-Trial matching for all patients...")
    print("=" * 80)
    
    all_results = {}
    summary = {
        'total_patients': len(patients),
        'total_trials': len(trials),
        'patients_processed': 0,
        'total_matches_found': 0,
        'patients_with_matches': 0,
        'patients_without_matches': 0
    }
    
    for i, patient in enumerate(patients, 1):
        mrn = patient['mrn']
        print(f"\n[{i}/{len(patients)}] Processing patient {mrn}...")
        
        try:
            # Suppress print statements that might have encoding issues
            import contextlib
            import io
            
            # Capture stdout to avoid encoding errors
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                matches = matcher.find_trials_for_patient_from_data(patient, top_k=50)
            
            all_results[mrn] = {
                'patient_info': {
                    'mrn': mrn,
                    'patient_id': patient['patient_id'],
                    'age': patient.get('age'),
                    'gender': patient.get('gender')
                },
                'matches': matches,
                'match_count': len(matches)
            }
            
            summary['patients_processed'] += 1
            summary['total_matches_found'] += len(matches)
            
            if len(matches) > 0:
                summary['patients_with_matches'] += 1
                print(f"   [OK] Found {len(matches)} matching trials")
                # Show top 3 matches
                for j, match in enumerate(matches[:3], 1):
                    trial_id = match.get('trial_id', 'Unknown')
                    score = match.get('hybrid_score', 0)
                    print(f"      {j}. {trial_id} - Score: {score:.4f}")
            else:
                summary['patients_without_matches'] += 1
                print(f"   [NO MATCHES] No matches found")
                
        except Exception as e:
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"   ERROR: {error_msg}")
            all_results[mrn] = {
                'patient_info': {
                    'mrn': mrn,
                    'patient_id': patient['patient_id']
                },
                'matches': [],
                'match_count': 0,
                'error': error_msg
            }
    
    # Save results
    results_file = test_results_dir / "patient_to_trial_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': all_results,
            'generated_at': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Patients Processed: {summary['patients_processed']}")
    print(f"Total Matches Found: {summary['total_matches_found']}")
    print(f"Patients with Matches: {summary['patients_with_matches']}")
    print(f"Patients without Matches: {summary['patients_without_matches']}")
    print(f"Average Matches per Patient: {summary['total_matches_found'] / summary['patients_processed']:.2f}" if summary['patients_processed'] > 0 else "N/A")
    
    # Show detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY PATIENT")
    print("=" * 80)
    for mrn, result in all_results.items():
        match_count = result['match_count']
        patient_info = result['patient_info']
        print(f"\nPatient {mrn} (Age: {patient_info.get('age', 'Unknown')}, Gender: {patient_info.get('gender', 'Unknown')}):")
        print(f"  Matches: {match_count}")
        if match_count > 0:
            print(f"  Top 5 Trials:")
            for j, match in enumerate(result['matches'][:5], 1):
                trial_id = match.get('trial_id', 'Unknown')
                score = match.get('hybrid_score', 0)
                title = match.get('title', 'Unknown')[:60] + "..." if match.get('title') else 'Unknown'
                print(f"    {j}. {trial_id} - {title}")
                print(f"       Score: {score:.4f} (Embedding: {match.get('embedding_score', 0):.4f}, BM25: {match.get('bm25_score', 0):.4f})")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

