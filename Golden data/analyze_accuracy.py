"""
Analyze test results and calculate accuracy metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def load_golden_data() -> Dict[str, List[Dict]]:
    """Load golden data matches"""
    matches_file = Path(__file__).parent / "matches.json"
    with open(matches_file, 'r', encoding='utf-8') as f:
        matches_data = json.load(f)
    
    # Convert to dictionary keyed by MRN
    matches_dict = {}
    for match in matches_data['matches']:
        matches_dict[match['MRN']] = match['results']
    
    return matches_dict

def load_test_results() -> Dict[str, Any]:
    """Load test results"""
    results_file = Path(__file__).parent / "test_patient_to_trial_results_20251109_194301.json"
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_decision(decision: str) -> str:
    """Normalize decision strings for comparison"""
    if not decision:
        return "NOT_EVALUATED"
    
    decision = decision.upper().strip()
    
    # Map various formats to standard format
    if decision in ["ELIGIBLE", "ELIGIBLE"]:
        return "ELIGIBLE"
    elif decision in ["NOT_ELIGIBLE", "NOT ELIGIBLE", "NOTELIGIBLE"]:
        return "NOT ELIGIBLE"
    elif decision in ["NEED_MORE_INFO", "NEED MORE INFO", "NEEDMOREINFO"]:
        return "NEED_MORE_INFO"
    else:
        return "NOT_EVALUATED"

def extract_llm_decisions(evaluation_result: Dict) -> Dict[str, str]:
    """Extract LLM decisions from evaluation result"""
    llm_decisions = {}
    
    # Try to get from final_ranking first
    final_ranking = evaluation_result.get('final_ranking', [])
    if final_ranking:
        for eval_item in final_ranking:
            trial_id = eval_item.get('trial_info', {}).get('trial_id') or eval_item.get('trial_id')
            if trial_id:
                status = eval_item.get('eligibility_status', 'UNKNOWN')
                llm_decisions[trial_id] = normalize_decision(status)
    
    # If no final_ranking, try evaluations
    if not llm_decisions:
        evaluations = evaluation_result.get('evaluations', [])
        for eval_item in evaluations:
            trial_id = eval_item.get('trial_id')
            if trial_id:
                status = eval_item.get('eligibility_status', 'UNKNOWN')
                llm_decisions[trial_id] = normalize_decision(status)
    
    return llm_decisions

def calculate_accuracy():
    """Calculate accuracy metrics"""
    golden_data = load_golden_data()
    test_results = load_test_results()
    
    # Overall metrics
    total_matches = 0
    correct_matches = 0
    incorrect_matches = 0
    not_evaluated = 0
    
    # Per-patient metrics
    patient_metrics = {}
    
    # Per-decision type metrics
    decision_metrics = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    
    # Detailed comparison
    detailed_comparisons = []
    
    for patient_result in test_results.get('patients_tested', []):
        mrn = patient_result.get('mrn')
        if not mrn:
            continue
        
        # Get golden matches for this patient
        golden_matches = golden_data.get(mrn, [])
        if not golden_matches:
            continue
        
        # Get LLM evaluation result
        llm_eval = patient_result.get('steps', {}).get('llm_evaluation', {}).get('result', {})
        llm_decisions = extract_llm_decisions(llm_eval)
        
        # Patient-level metrics
        patient_correct = 0
        patient_incorrect = 0
        patient_not_evaluated = 0
        patient_total = len(golden_matches)
        
        # Compare each golden match
        for golden_match in golden_matches:
            trial_id = golden_match['trial_id']
            golden_decision = normalize_decision(golden_match['decision'])
            
            llm_decision = llm_decisions.get(trial_id, "NOT_EVALUATED")
            
            total_matches += 1
            patient_total += 1
            
            # Track decision type metrics
            decision_metrics[golden_decision]["total"] += 1
            
            if llm_decision == "NOT_EVALUATED":
                not_evaluated += 1
                patient_not_evaluated += 1
                decision_metrics[golden_decision]["incorrect"] += 1  # Count as incorrect
            elif llm_decision == golden_decision:
                correct_matches += 1
                patient_correct += 1
                decision_metrics[golden_decision]["correct"] += 1
            else:
                incorrect_matches += 1
                patient_incorrect += 1
                decision_metrics[golden_decision]["incorrect"] += 1
            
            # Store detailed comparison
            detailed_comparisons.append({
                "mrn": mrn,
                "trial_id": trial_id,
                "golden_decision": golden_decision,
                "llm_decision": llm_decision,
                "is_correct": llm_decision == golden_decision and llm_decision != "NOT_EVALUATED",
                "golden_reasons": golden_match.get('reasons', [])
            })
        
        # Store patient metrics
        patient_accuracy = (patient_correct / patient_total * 100) if patient_total > 0 else 0.0
        patient_metrics[mrn] = {
            "total": patient_total,
            "correct": patient_correct,
            "incorrect": patient_incorrect,
            "not_evaluated": patient_not_evaluated,
            "accuracy": patient_accuracy
        }
    
    # Calculate overall accuracy
    evaluated_matches = correct_matches + incorrect_matches
    overall_accuracy = (correct_matches / evaluated_matches * 100) if evaluated_matches > 0 else 0.0
    
    # Calculate precision, recall, F1 for ELIGIBLE decisions
    eligible_tp = sum(1 for c in detailed_comparisons 
                     if c['golden_decision'] == 'ELIGIBLE' and c['llm_decision'] == 'ELIGIBLE')
    eligible_fp = sum(1 for c in detailed_comparisons 
                     if c['golden_decision'] != 'ELIGIBLE' and c['llm_decision'] == 'ELIGIBLE')
    eligible_fn = sum(1 for c in detailed_comparisons 
                     if c['golden_decision'] == 'ELIGIBLE' and c['llm_decision'] != 'ELIGIBLE' and c['llm_decision'] != 'NOT_EVALUATED')
    
    eligible_precision = (eligible_tp / (eligible_tp + eligible_fp)) if (eligible_tp + eligible_fp) > 0 else 0.0
    eligible_recall = (eligible_tp / (eligible_tp + eligible_fn)) if (eligible_tp + eligible_fn) > 0 else 0.0
    eligible_f1 = (2 * eligible_precision * eligible_recall / (eligible_precision + eligible_recall)) if (eligible_precision + eligible_recall) > 0 else 0.0
    
    # Print results
    print("=" * 100)
    print("ACCURACY ANALYSIS RESULTS")
    print("=" * 100)
    print(f"\nOverall Metrics:")
    print(f"  Total Matches: {total_matches}")
    print(f"  Correct: {correct_matches}")
    print(f"  Incorrect: {incorrect_matches}")
    print(f"  Not Evaluated: {not_evaluated}")
    print(f"  Evaluated Matches: {evaluated_matches}")
    print(f"\n  Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"  (Accuracy = Correct / Evaluated)")
    
    print(f"\nPer-Patient Metrics:")
    for mrn, metrics in patient_metrics.items():
        print(f"  {mrn}:")
        print(f"    Total: {metrics['total']}")
        print(f"    Correct: {metrics['correct']}")
        print(f"    Incorrect: {metrics['incorrect']}")
        print(f"    Not Evaluated: {metrics['not_evaluated']}")
        print(f"    Accuracy: {metrics['accuracy']:.2f}%")
    
    print(f"\nPer-Decision Type Metrics:")
    for decision_type, metrics in decision_metrics.items():
        if metrics['total'] > 0:
            accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0.0
            print(f"  {decision_type}:")
            print(f"    Total: {metrics['total']}")
            print(f"    Correct: {metrics['correct']}")
            print(f"    Incorrect: {metrics['incorrect']}")
            print(f"    Accuracy: {accuracy:.2f}%")
    
    print(f"\nELIGIBLE Decision Metrics (Precision/Recall/F1):")
    print(f"  True Positives (TP): {eligible_tp}")
    print(f"  False Positives (FP): {eligible_fp}")
    print(f"  False Negatives (FN): {eligible_fn}")
    print(f"  Precision: {eligible_precision * 100:.2f}%")
    print(f"  Recall: {eligible_recall * 100:.2f}%")
    print(f"  F1 Score: {eligible_f1 * 100:.2f}%")
    
    print(f"\nDetailed Comparisons (First 10):")
    for i, comp in enumerate(detailed_comparisons[:10]):
        status = "✓" if comp['is_correct'] else "✗"
        print(f"  {status} {comp['mrn']} - {comp['trial_id']}: "
              f"Golden={comp['golden_decision']}, LLM={comp['llm_decision']}")
    
    # Save detailed results
    results_summary = {
        "overall_metrics": {
            "total_matches": total_matches,
            "correct": correct_matches,
            "incorrect": incorrect_matches,
            "not_evaluated": not_evaluated,
            "evaluated_matches": evaluated_matches,
            "overall_accuracy": overall_accuracy
        },
        "patient_metrics": patient_metrics,
        "decision_type_metrics": dict(decision_metrics),
        "eligible_metrics": {
            "true_positives": eligible_tp,
            "false_positives": eligible_fp,
            "false_negatives": eligible_fn,
            "precision": eligible_precision * 100,
            "recall": eligible_recall * 100,
            "f1_score": eligible_f1 * 100
        },
        "detailed_comparisons": detailed_comparisons
    }
    
    output_file = Path(__file__).parent / "accuracy_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 100)

if __name__ == "__main__":
    calculate_accuracy()

