"""
Quick check script to verify hybrid score calculation
Shows detailed breakdown for a single patient or trial match
"""

import sys
from services.patient_to_trial.patient_matcher import PatientMatcher
from services.trial_to_patient.trial_matcher import TrialMatcher
from services.shared.database_utils import DatabaseUtils


def quick_validate_patient(patient_id: int, alpha: float = 0.7, top_n: int = 5):
    """Quick validation for patient-to-trial matching"""
    matcher = PatientMatcher()
    db_utils = DatabaseUtils()
    
    patient_data = db_utils.get_patient_by_id(patient_id)
    if not patient_data:
        print(f"❌ Patient {patient_id} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"QUICK HYBRID SCORE VALIDATION - Patient {patient_id}")
    print(f"{'='*80}")
    print(f"Patient: MRN {patient_data['mrn']}")
    print(f"Alpha (embedding weight): {alpha}")
    print(f"Formula: hybrid = {alpha} * embedding + {1-alpha} * normalized_bm25")
    print(f"{'='*80}\n")
    
    # Get matching trials
    matching_trials = matcher.hybrid_search_trials_for_patient(patient_data, alpha=alpha)
    
    if not matching_trials:
        print("❌ No matching trials found")
        return
    
    print(f"Found {len(matching_trials)} matching trials\n")
    print(f"{'Rank':<5} {'Trial ID':<15} {'Embedding':<12} {'BM25 Raw':<12} {'BM25 Norm':<12} {'Hybrid':<12} {'Calc':<12} {'Status':<10}")
    print("-" * 100)
    
    for i, trial in enumerate(matching_trials[:top_n], 1):
        trial_id = trial.get('trial_id', 'Unknown')
        embedding = trial.get('embedding_score', 0.0)
        bm25_raw = trial.get('bm25_score', 0.0)
        hybrid_stored = trial.get('hybrid_score', 0.0)
        
        # Recalculate
        bm25_norm = min(bm25_raw / 10.0, 1.0)
        hybrid_calc = alpha * embedding + (1 - alpha) * bm25_norm
        
        # Check accuracy
        diff = abs(hybrid_stored - hybrid_calc)
        status = "✓" if diff < 0.0001 else f"✗ ({diff:.6f})"
        
        print(f"{i:<5} {trial_id:<15} {embedding:<12.6f} {bm25_raw:<12.6f} {bm25_norm:<12.6f} "
              f"{hybrid_stored:<12.6f} {hybrid_calc:<12.6f} {status:<10}")


def quick_validate_trial(trial_id: str, alpha: float = 0.7, top_n: int = 5):
    """Quick validation for trial-to-patient matching"""
    matcher = TrialMatcher()
    db_utils = DatabaseUtils()
    
    trial_data = db_utils.get_trial_by_id(trial_id)
    if not trial_data:
        print(f"❌ Trial {trial_id} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"QUICK HYBRID SCORE VALIDATION - Trial {trial_id}")
    print(f"{'='*80}")
    print(f"Trial: {trial_data.get('title', 'Unknown')[:60]}")
    print(f"Alpha (embedding weight): {alpha}")
    print(f"Formula: hybrid = {alpha} * embedding + {1-alpha} * normalized_bm25")
    print(f"{'='*80}\n")
    
    # Get matching patients
    matching_patients = matcher.hybrid_search_patients_for_trial(trial_data, alpha=alpha)
    
    if not matching_patients:
        print("❌ No matching patients found")
        return
    
    print(f"Found {len(matching_patients)} matching patients\n")
    print(f"{'Rank':<5} {'Patient ID':<12} {'MRN':<15} {'Embedding':<12} {'BM25 Raw':<12} {'BM25 Norm':<12} {'Hybrid':<12} {'Calc':<12} {'Status':<10}")
    print("-" * 110)
    
    for i, patient in enumerate(matching_patients[:top_n], 1):
        patient_id = patient.get('patient_id', 'Unknown')
        mrn = patient.get('mrn', 'Unknown')
        embedding = patient.get('embedding_score', 0.0)
        bm25_raw = patient.get('bm25_score', 0.0)
        hybrid_stored = patient.get('hybrid_score', 0.0)
        
        # Recalculate
        bm25_norm = min(bm25_raw / 10.0, 1.0)
        hybrid_calc = alpha * embedding + (1 - alpha) * bm25_norm
        
        # Check accuracy
        diff = abs(hybrid_stored - hybrid_calc)
        status = "✓" if diff < 0.0001 else f"✗ ({diff:.6f})"
        
        print(f"{i:<5} {patient_id:<12} {mrn:<15} {embedding:<12.6f} {bm25_raw:<12.6f} {bm25_norm:<12.6f} "
              f"{hybrid_stored:<12.6f} {hybrid_calc:<12.6f} {status:<10}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_check_hybrid_scores.py patient <patient_id> [alpha] [top_n]")
        print("  python quick_check_hybrid_scores.py trial <trial_id> [alpha] [top_n]")
        print("\nExamples:")
        print("  python quick_check_hybrid_scores.py patient 1")
        print("  python quick_check_hybrid_scores.py patient 1 0.7 10")
        print("  python quick_check_hybrid_scores.py trial NCT04929223")
        return
    
    match_type = sys.argv[1].lower()
    
    if match_type == "patient":
        if len(sys.argv) < 3:
            print("❌ Please provide patient_id")
            return
        patient_id = int(sys.argv[2])
        alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        quick_validate_patient(patient_id, alpha, top_n)
    
    elif match_type == "trial":
        if len(sys.argv) < 3:
            print("❌ Please provide trial_id")
            return
        trial_id = sys.argv[2]
        alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        quick_validate_trial(trial_id, alpha, top_n)
    
    else:
        print(f"❌ Unknown match type: {match_type}")
        print("Use 'patient' or 'trial'")


if __name__ == "__main__":
    main()

