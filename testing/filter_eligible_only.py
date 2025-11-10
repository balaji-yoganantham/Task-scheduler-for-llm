import json

# Load the matches file
with open('patients_5_realistic_final_matches.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter to only eligible matches
filtered_data = {
    "matches": [],
    "summary": []
}

for patient_match in data['matches']:
    mrn = patient_match['MRN']
    
    # Filter results to only eligible ones
    eligible_results = [
        result for result in patient_match['results']
        if result['decision'] == 'ELIGIBLE'
    ]
    
    # Only add patient if they have eligible matches
    if eligible_results:
        filtered_patient = {
            "MRN": mrn,
            "results": eligible_results,
            "eligible_count": len(eligible_results)
        }
        filtered_data['matches'].append(filtered_patient)
        filtered_data['summary'].append([mrn, len(eligible_results)])

# Save to new file
output_file = 'patients_5_realistic_final_matches_eligible_only.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"✅ Created {output_file}")
print(f"   Total patients: {len(filtered_data['matches'])}")
print(f"   Total eligible matches: {sum(len(m['results']) for m in filtered_data['matches'])}")

# Print summary by patient
print("\nEligible matches per patient:")
for patient_match in filtered_data['matches']:
    print(f"  MRN {patient_match['MRN']}: {patient_match['eligible_count']} eligible matches")

