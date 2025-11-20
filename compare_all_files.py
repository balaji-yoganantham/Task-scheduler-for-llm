import json
from pathlib import Path

# Load all three files
files = [
    'results/patient_evaluation_41_20251120_074546.json',
    'results/patient_evaluation_41_20251120_075350.json',
    'results/patient_evaluation_41_20251120_081254.json'
]

data = []
for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data.append(json.load(f))

print("=" * 80)
print("COMPARISON OF ALL THREE EVALUATION FILES")
print("=" * 80)

# Compare summaries
print("\nSUMMARY COMPARISON:")
print("-" * 80)
for i, (filepath, d) in enumerate(zip(files, data), 1):
    filename = Path(filepath).name
    summary = d.get('summary', {})
    gen_time = d.get('generated_at', 'N/A')
    print(f"\nFile {i}: {filename}")
    print(f"  Generated at: {gen_time}")
    print(f"  Eligible trials: {summary.get('eligible_trials', 0)}")
    print(f"  Average confidence: {summary.get('average_confidence', 0):.2f}%")
    print(f"  Total trials found: {summary.get('total_trials_found', 0)}")
    print(f"  Trials evaluated: {summary.get('trials_evaluated', 0)}")

# Compare evaluations
print("\n" + "=" * 80)
print("EVALUATION STATUS COMPARISON")
print("=" * 80)

# Get all evaluations from each file
evals = []
for d in data:
    evals_dict = {}
    if 'llm_evaluation' in d and 'evaluations' in d['llm_evaluation']:
        for e in d['llm_evaluation']['evaluations']:
            trial_id = e.get('trial_id')
            if trial_id:
                evals_dict[trial_id] = {
                    'status': e.get('eligibility_status'),
                    'confidence': e.get('confidence_score', 0),
                    'inclusion_met': e.get('inclusion_criteria_met_count', 0),
                    'total_inclusion': e.get('total_inclusion_criteria', 0)
                }
    evals.append(evals_dict)

# Find all unique trial IDs
all_trial_ids = set()
for evals_dict in evals:
    all_trial_ids.update(evals_dict.keys())

# Compare status changes
status_changes = []
confidence_changes = []

for trial_id in sorted(all_trial_ids):
    statuses = [e.get(trial_id, {}).get('status') for e in evals]
    confidences = [e.get(trial_id, {}).get('confidence', 0) for e in evals]
    
    # Check if status changed
    if len(set([s for s in statuses if s])) > 1:
        status_changes.append({
            'trial_id': trial_id,
            'statuses': statuses,
            'confidences': confidences
        })
    
    # Check if confidence changed significantly (>1 point)
    if len(set(confidences)) > 1:
        max_conf = max(confidences)
        min_conf = min(confidences)
        if max_conf - min_conf > 1:
            confidence_changes.append({
                'trial_id': trial_id,
                'confidences': confidences,
                'diff': max_conf - min_conf
            })

print(f"\nTrials with status changes: {len(status_changes)}")
if status_changes:
    print("\nStatus Changes:")
    for change in status_changes[:10]:  # Show first 10
        print(f"  {change['trial_id']}: {change['statuses']} (confidence: {change['confidences']})")

print(f"\nTrials with confidence changes >1 point: {len(confidence_changes)}")
if confidence_changes:
    print("\nSignificant Confidence Changes:")
    for change in sorted(confidence_changes, key=lambda x: x['diff'], reverse=True)[:10]:
        print(f"  {change['trial_id']}: {change['confidences']} (diff: {change['diff']:.1f})")

# Check consistency between last two files (after temperature change)
print("\n" + "=" * 80)
print("CONSISTENCY CHECK: Last 2 files (after temperature=0.0)")
print("=" * 80)

file2_evals = evals[1]  # 075350
file3_evals = evals[2]  # 081254

consistent_status = 0
inconsistent_status = 0
consistent_confidence = 0
inconsistent_confidence = 0

for trial_id in all_trial_ids:
    status2 = file2_evals.get(trial_id, {}).get('status')
    status3 = file3_evals.get(trial_id, {}).get('status')
    conf2 = file2_evals.get(trial_id, {}).get('confidence', 0)
    conf3 = file3_evals.get(trial_id, {}).get('confidence', 0)
    
    if status2 and status3:
        if status2 == status3:
            consistent_status += 1
        else:
            inconsistent_status += 1
            print(f"  Status mismatch: {trial_id} - {status2} vs {status3}")
    
    if conf2 and conf3:
        if abs(conf2 - conf3) <= 1:
            consistent_confidence += 1
        else:
            inconsistent_confidence += 1
            if abs(conf2 - conf3) > 1:
                print(f"  Confidence mismatch: {trial_id} - {conf2} vs {conf3} (diff: {abs(conf2 - conf3):.1f})")

print(f"\nStatus consistency: {consistent_status} consistent, {inconsistent_status} inconsistent")
print(f"Confidence consistency: {consistent_confidence} consistent (within 1 point), {inconsistent_confidence} inconsistent")

