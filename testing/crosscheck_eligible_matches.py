import json

# Load all files
with open('trial.json', 'r', encoding='utf-8') as f:
    trials_data = json.load(f)

with open('patients_5_realistic_final_matches_eligible_only.json', 'r', encoding='utf-8') as f:
    eligible_matches = json.load(f)

with open('patients_5_realistic_final.json', 'r', encoding='utf-8') as f:
    patients_data = json.load(f)

# Create lookups
trial_lookup = {trial['nct_id']: trial for trial in trials_data}
patient_lookup = {}
for patient in patients_data['patients']:
    mrn = patient['Patient Information']['MRN']
    patient_lookup[mrn] = patient

print("=" * 80)
print("CROSS-CHECK: ELIGIBLE MATCHES vs TRIAL.JSON & PATIENT DATA")
print("=" * 80)

# Track statistics
total_eligible = 0
missing_trials = set()
validation_issues = []

for patient_match in eligible_matches['matches']:
    mrn = patient_match['MRN']
    patient = patient_lookup.get(mrn)
    
    if not patient:
        print(f"\n⚠️  Patient MRN {mrn} NOT FOUND in patients_5_realistic_final.json")
        continue
    
    # Get patient info
    hpi = patient['History of Present Illness']
    lab_results = patient.get('Laboratory & Imaging Results', {})
    molecular_markers = lab_results.get('Molecular_markers', '')
    performance_status = hpi.get('Performance status', '')
    prior_treatments = hpi.get('Summary of prior treatments', '')
    diagnosis = patient.get('Assessment', {}).get('Diagnosis', '')
    
    print(f"\n{'='*80}")
    print(f"Patient MRN: {mrn}")
    print(f"  - Performance Status: {performance_status}")
    print(f"  - Molecular Markers: {molecular_markers}")
    print(f"  - Prior Treatments: {prior_treatments}")
    print(f"  - Diagnosis: {diagnosis}")
    print(f"  - Eligible Matches: {patient_match['eligible_count']}")
    
    patient_issues = []
    
    for result in patient_match['results']:
        nct_id = result['trial_id']
        total_eligible += 1
        
        # Check if trial exists in trial.json
        trial = trial_lookup.get(nct_id)
        if not trial:
            missing_trials.add(nct_id)
            patient_issues.append(f"  ⚠️  {nct_id}: NOT FOUND in trial.json")
            continue
        
        # Validate eligibility
        issues = []
        
        # Check molecular markers
        conditions = trial.get('conditions', [])
        inclusion = trial.get('inclusion_criteria', '')
        exclusion = trial.get('exclusion_criteria', '')
        trial_text = f"{inclusion} {exclusion} {' '.join(conditions)}"
        
        # Check KRAS G12D requirement
        if 'KRAS G12D' in trial_text and 'KRAS G12D' not in molecular_markers:
            issues.append(f"Trial requires KRAS G12D but patient has: {molecular_markers}")
        
        # Check HER2 requirement
        if ('HER2' in trial_text or 'HER2' in str(conditions)) and 'HER2' not in molecular_markers:
            issues.append(f"Trial requires HER2 but patient has: {molecular_markers}")
        
        # Check RAS/BRAF wild-type requirement
        if 'RAS/BRAF wild-type' in trial_text and 'wild-type' not in molecular_markers:
            if 'RAS' in molecular_markers or 'BRAF' in molecular_markers:
                issues.append(f"Trial requires RAS/BRAF wild-type but patient has: {molecular_markers}")
        
        # Check MSI-H/dMMR requirement
        if ('MSI-H' in trial_text or 'dMMR' in trial_text) and 'MSI-H' not in molecular_markers and 'dMMR' not in molecular_markers:
            issues.append(f"Trial requires MSI-H/dMMR but patient has: {molecular_markers}")
        
        # Check prior treatment requirement
        if 'prior' in inclusion.lower() and 'previously treated' in inclusion.lower():
            if 'treatment-naïve' in prior_treatments.lower() or 'newly diagnosed' in prior_treatments.lower():
                issues.append(f"Trial requires prior treatment but patient is treatment-naïve")
        
        # Check trial status
        status = trial.get('overall_status', '')
        if status not in ['Recruiting', 'Active, not recruiting', 'Enrolling by invitation', 'Available']:
            issues.append(f"Trial status is '{status}' (may not be actively recruiting)")
        
        # Check age (if specified)
        min_age = trial.get('minimum_age')
        max_age = trial.get('maximum_age')
        # Note: Patient age not in the JSON structure, so we skip this check
        
        # Check sex/gender
        trial_sex = trial.get('sex')
        # Note: Patient gender not easily accessible in current structure
        
        if issues:
            patient_issues.append(f"  ⚠️  {nct_id}: {trial.get('study_title', 'N/A')[:60]}")
            for issue in issues:
                patient_issues.append(f"      - {issue}")
            validation_issues.append({
                'mrn': mrn,
                'nct_id': nct_id,
                'issues': issues
            })
        else:
            patient_issues.append(f"  ✅ {nct_id}: {trial.get('study_title', 'N/A')[:60]}")
    
    if patient_issues:
        for issue in patient_issues:
            print(issue)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total eligible matches checked: {total_eligible}")
print(f"Total patients: {len(eligible_matches['matches'])}")
print(f"\nNCT ID Validation:")
print(f"  - Trials found in trial.json: {total_eligible - len(missing_trials)}")
print(f"  - Trials missing from trial.json: {len(missing_trials)}")

if missing_trials:
    print(f"\n  Missing NCT IDs ({len(missing_trials)}):")
    for nct_id in sorted(missing_trials):
        print(f"    - {nct_id}")

print(f"\nEligibility Validation:")
print(f"  - Matches with potential issues: {len(validation_issues)}")
print(f"  - Matches validated correctly: {total_eligible - len(validation_issues) - len(missing_trials)}")

if validation_issues:
    print(f"\n  Issues found:")
    for issue in validation_issues[:10]:  # Show first 10
        print(f"    - Patient {issue['mrn']}, Trial {issue['nct_id']}: {issue['issues'][0]}")
    if len(validation_issues) > 10:
        print(f"    ... and {len(validation_issues) - 10} more")

# Check if all patient MRNs match
print(f"\nPatient MRN Validation:")
all_mrns_match = True
for patient_match in eligible_matches['matches']:
    mrn = patient_match['MRN']
    if mrn not in patient_lookup:
        print(f"  ⚠️  MRN {mrn} not found in patients_5_realistic_final.json")
        all_mrns_match = False

if all_mrns_match:
    print(f"  ✅ All patient MRNs match")

print("\n" + "=" * 80)




