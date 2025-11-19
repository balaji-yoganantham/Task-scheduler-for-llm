import json
from collections import defaultdict

# Load the matches file
with open('patients_5_realistic_final_matches.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize counters
total_matches = 0
eligible_matches = 0
not_eligible_matches = 0
need_more_info_matches = 0

# Track eligible NCT IDs per patient and overall
eligible_nct_ids_by_patient = {}
all_eligible_nct_ids = set()
all_nct_ids = set()

# Process each patient's matches
for patient_match in data['matches']:
    mrn = patient_match['MRN']
    results = patient_match['results']
    eligible_nct_ids_by_patient[mrn] = []
    
    for result in results:
        trial_id = result['trial_id']
        decision = result['decision']
        
        all_nct_ids.add(trial_id)
        total_matches += 1
        
        if decision == 'ELIGIBLE':
            eligible_matches += 1
            eligible_nct_ids_by_patient[mrn].append(trial_id)
            all_eligible_nct_ids.add(trial_id)
        elif decision == 'NOT_ELIGIBLE':
            not_eligible_matches += 1
        elif decision == 'NEED_MORE_INFO':
            need_more_info_matches += 1

# Print summary
print("=" * 80)
print("MATCH ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nTotal Patients: {len(data['matches'])}")
print(f"Total Matches: {total_matches}")
print(f"\nDecision Breakdown:")
print(f"  - ELIGIBLE: {eligible_matches} ({eligible_matches/total_matches*100:.1f}%)")
print(f"  - NOT_ELIGIBLE: {not_eligible_matches} ({not_eligible_matches/total_matches*100:.1f}%)")
print(f"  - NEED_MORE_INFO: {need_more_info_matches} ({need_more_info_matches/total_matches*100:.1f}%)")

print(f"\nUnique NCT IDs:")
print(f"  - Total unique trials: {len(all_nct_ids)}")
print(f"  - Unique eligible trials: {len(all_eligible_nct_ids)}")

print("\n" + "=" * 80)
print("ELIGIBLE MATCHES BY PATIENT")
print("=" * 80)

for patient_match in data['matches']:
    mrn = patient_match['MRN']
    eligible_count = patient_match['eligible_count']
    eligible_trials = eligible_nct_ids_by_patient[mrn]
    
    print(f"\nPatient MRN: {mrn}")
    print(f"  Eligible Count: {eligible_count}")
    print(f"  Eligible NCT IDs ({len(eligible_trials)}):")
    for nct_id in sorted(eligible_trials):
        print(f"    - {nct_id}")

print("\n" + "=" * 80)
print("ALL UNIQUE ELIGIBLE NCT IDs (Sorted)")
print("=" * 80)
for nct_id in sorted(all_eligible_nct_ids):
    print(f"  - {nct_id}")

print(f"\nTotal unique eligible NCT IDs: {len(all_eligible_nct_ids)}")



