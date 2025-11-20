"""
Summarize partial results from LLM accuracy evaluation
Shows what was completed before hitting quota limits
"""

import json
from pathlib import Path
from collections import defaultdict

def summarize_checkpoint():
    """Summarize results from checkpoint file"""
    checkpoint_file = Path("llm_accuracy_results/progress_checkpoint.json")
    
    if not checkpoint_file.exists():
        print("No checkpoint file found. Run the evaluation script first.")
        return
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    evaluations = checkpoint.get('evaluations', [])
    stats = checkpoint.get('accuracy_stats', {})
    progress = checkpoint.get('progress', {})
    
    print("="*80)
    print("PARTIAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nProgress: {progress.get('current', 0)}/{progress.get('total', 0)} evaluations")
    print(f"Timestamp: {progress.get('timestamp', 'Unknown')}")
    
    if stats:
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        incorrect = stats.get('incorrect', 0)
        need_more_info = stats.get('need_more_info', 0)
        errors = stats.get('errors', 0)
        
        print(f"\nCompleted Evaluations: {total}")
        print(f"  - Correct Matches: {correct}")
        print(f"  - Incorrect Matches: {incorrect}")
        print(f"  - Need More Info: {need_more_info}")
        print(f"  - Errors: {errors}")
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"\nAccuracy: {accuracy:.2f}%")
    
    # Count by decision type
    decision_counts = defaultdict(int)
    for eval_item in evaluations:
        llm_decision = eval_item.get('llm_decision', 'UNKNOWN')
        decision_counts[llm_decision] += 1
    
    print(f"\nLLM Decisions:")
    for decision, count in sorted(decision_counts.items()):
        print(f"  - {decision}: {count}")
    
    # Show mismatches
    mismatches = [e for e in evaluations if not e.get('match', False) and e.get('llm_decision') != 'ERROR']
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  - MRN {mismatch.get('mrn')}, Trial {mismatch.get('nct_id')}: "
                  f"Original={mismatch.get('original_decision')}, LLM={mismatch.get('llm_decision')}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    
    print("\n" + "="*80)
    print("To continue evaluation, run: python llm_accuracy_evaluation.py")
    print("The script will automatically resume from checkpoint.")
    print("="*80)

if __name__ == "__main__":
    summarize_checkpoint()




