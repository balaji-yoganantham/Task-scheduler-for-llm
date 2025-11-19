import json

# Load all three files
with open('trial.json', 'r', encoding='utf-8') as f:
    trials_data = json.load(f)

with open('patients_5_realistic_final_matches.json', 'r', encoding='utf-8') as f:
    matches_data = json.load(f)

with open('patients_5_realistic_final.json', 'r', encoding='utf-8') as f:
    patients_data = json.load(f)

# Extract NCT IDs from trials.json
trial_nct_ids = set(trial['nct_id'] for trial in trials_data)
print("=" * 80)
print("NCT ID CROSS-CHECK ANALYSIS")
print("=" * 80)

# Extract NCT IDs from matches
match_nct_ids = set()
for patient_match in matches_data['matches']:
    for result in patient_match['results']:
        match_nct_ids.add(result['trial_id'])

print(f"\n1. NCT ID Comparison:")
print(f"   - NCT IDs in trial.json: {len(trial_nct_ids)}")
print(f"   - NCT IDs in matches: {len(match_nct_ids)}")

# Find missing NCT IDs
missing_in_trials = match_nct_ids - trial_nct_ids
missing_in_matches = trial_nct_ids - match_nct_ids

if missing_in_trials:
    print(f"\n   ⚠️  NCT IDs in matches but NOT in trial.json ({len(missing_in_trials)}):")
    for nct_id in sorted(missing_in_trials):
        print(f"      - {nct_id}")

if missing_in_matches:
    print(f"\n   ℹ️  NCT IDs in trial.json but NOT in matches ({len(missing_in_matches)}):")
    for nct_id in sorted(missing_in_matches):
        print(f"      - {nct_id}")

if not missing_in_trials and not missing_in_matches:
    print(f"\n   ✅ All NCT IDs match perfectly!")

# Create a lookup for trials
trial_lookup = {trial['nct_id']: trial for trial in trials_data}

# Create a lookup for patients
patient_lookup = {}
for patient in patients_data['patients']:
    mrn = patient['Patient Information']['MRN']
    patient_lookup[mrn] = patient

print("\n" + "=" * 80)
print("ELIGIBILITY CROSS-CHECK")
print("=" * 80)

# Check each patient's eligible matches
for patient_match in matches_data['matches']:
    mrn = patient_match['MRN']
    patient = patient_lookup.get(mrn)
    
    if not patient:
        print(f"\n⚠️  Patient MRN {mrn} not found in patients data")
        continue
    
    # Get patient info
    patient_info = patient['Patient Information']
    hpi = patient['History of Present Illness']
    lab_results = patient.get('Laboratory & Imaging Results', {})
    molecular_markers = lab_results.get('Molecular_markers', '')
    performance_status = hpi.get('Performance status', '')
    prior_treatments = hpi.get('Summary of prior treatments', '')
    
    print(f"\n{'='*80}")
    print(f"Patient MRN: {mrn}")
    print(f"  - Performance Status: {performance_status}")
    print(f"  - Molecular Markers: {molecular_markers}")
    print(f"  - Prior Treatments: {prior_treatments}")
    print(f"  - Eligible Count: {patient_match['eligible_count']}")
    
    # Check each eligible match
    eligible_trials = [r for r in patient_match['results'] if r['decision'] == 'ELIGIBLE']
    print(f"\n  Eligible Trials ({len(eligible_trials)}):")
    
    for result in eligible_trials:
        nct_id = result['trial_id']
        trial = trial_lookup.get(nct_id)
        
        if not trial:
            print(f"    ⚠️  {nct_id}: Trial not found in trial.json")
            continue
        
        # Check eligibility criteria
        issues = []
        
        # Check age
        min_age = trial.get('minimum_age')
        max_age = trial.get('maximum_age')
        if min_age or max_age:
            # Age check would require patient age - skipping for now
            pass
        
        # Check sex/gender
        trial_sex = trial.get('sex')
        if trial_sex and trial_sex != 'All':
            # Gender check would require patient gender - skipping for now
            pass
        
        # Check molecular markers
        conditions = trial.get('conditions', [])
        inclusion = trial.get('inclusion_criteria', '')
        exclusion = trial.get('exclusion_criteria', '')
        
        # Check if trial requires specific molecular markers
        if 'KRAS G12D' in str(inclusion) or 'KRAS G12D' in str(conditions):
            if 'KRAS G12D' not in molecular_markers:
                issues.append(f"Trial requires KRAS G12D but patient has: {molecular_markers}")
        
        if 'HER2' in str(inclusion) or 'HER2' in str(conditions):
            if 'HER2' not in molecular_markers:
                issues.append(f"Trial requires HER2 but patient has: {molecular_markers}")
        
        if 'RAS/BRAF wild-type' in str(inclusion) or 'RAS/BRAF wild-type' in str(exclusion):
            if 'wild-type' not in molecular_markers and 'RAS' in molecular_markers:
                issues.append(f"Trial requires RAS/BRAF wild-type but patient has: {molecular_markers}")
        
        if 'MSI-H' in str(inclusion) or 'dMMR' in str(inclusion):
            if 'MSI-H' not in molecular_markers and 'dMMR' not in molecular_markers:
                issues.append(f"Trial requires MSI-H/dMMR but patient has: {molecular_markers}")
        
        # Check prior treatment requirements
        if 'prior' in str(inclusion).lower() or 'previously treated' in str(inclusion).lower():
            if 'treatment-naïve' in prior_treatments.lower() or 'newly diagnosed' in prior_treatments.lower():
                issues.append(f"Trial may require prior treatment but patient is treatment-naïve")
        
        status = trial.get('overall_status', '')
        if status not in ['Recruiting', 'Active, not recruiting', 'Enrolling by invitation', 'Available']:
            issues.append(f"Trial status is '{status}' (may not be actively recruiting)")
        
        if issues:
            print(f"    ⚠️  {nct_id}: {trial.get('study_title', 'N/A')[:60]}")
            for issue in issues:
                print(f"         - {issue}")
        else:
            print(f"    ✅ {nct_id}: {trial.get('study_title', 'N/A')[:60]}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total trials in trial.json: {len(trial_nct_ids)}")
print(f"Total unique NCT IDs in matches: {len(match_nct_ids)}")
print(f"Matching NCT IDs: {len(trial_nct_ids & match_nct_ids)}")
print(f"Missing in trial.json: {len(missing_in_trials)}")
print(f"Extra in trial.json: {len(missing_in_matches)}")



