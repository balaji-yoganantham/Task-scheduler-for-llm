"""
Clear quota errors from checkpoint to allow retry when quota resets
"""

import json
from pathlib import Path

def clear_quota_errors():
    """Remove quota-related errors from checkpoint so they can be retried"""
    checkpoint_file = Path("llm_accuracy_results/progress_checkpoint.json")
    
    if not checkpoint_file.exists():
        print("No checkpoint file found.")
        return
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    evaluations = checkpoint.get('evaluations', [])
    accuracy_stats = checkpoint.get('accuracy_stats', {})
    
    # Filter out quota errors
    quota_error_count = 0
    valid_evaluations = []
    
    for eval_item in evaluations:
        error_type = eval_item.get('error_type', '')
        error_msg = str(eval_item.get('error', '')).lower()
        
        # Check if it's a quota error
        is_quota_error = (
            error_type == 'ResourceExhausted' or
            '429' in error_msg or
            'quota' in error_msg or
            'resource exhausted' in error_msg
        )
        
        if is_quota_error:
            quota_error_count += 1
            # Don't add to valid_evaluations - will be retried
        else:
            valid_evaluations.append(eval_item)
    
    # Update checkpoint with only valid evaluations
    checkpoint['evaluations'] = valid_evaluations
    
    # Reset stats to only include valid evaluations
    if accuracy_stats:
        accuracy_stats['errors'] = accuracy_stats.get('errors', 0) - quota_error_count
    
    # Save updated checkpoint
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Removed {quota_error_count} quota errors from checkpoint")
    print(f"   Valid evaluations remaining: {len(valid_evaluations)}")
    print(f"   These evaluations will be retried when quota resets")

if __name__ == "__main__":
    clear_quota_errors()



