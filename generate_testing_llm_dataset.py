"""
Generate Testing LLM Dataset
Creates a comprehensive test dataset with:
- Patient JSON files (matching project structure)
- 20 Trial JSON files
- Golden matches JSON (100% accuracy matches)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Create testing_llm folder
testing_dir = Path("testing_llm")
testing_dir.mkdir(exist_ok=True)

# Subdirectories
patients_dir = testing_dir / "patients"
trials_dir = testing_dir / "trials"
patients_dir.mkdir(exist_ok=True)
trials_dir.mkdir(exist_ok=True)

print("=" * 100)
print("GENERATING TESTING LLM DATASET")
print("=" * 100)

# Trial templates - different cancer types and stages
trial_templates = [
    # BRAF V600E Mutant mCRC - First Line
    {
        "condition_key": "BRAF_V600E_FIRSTLINE",
        "title": "First-Line Encorafenib Plus Cetuximab for Metastatic BRAF V600E-Mutant Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 3",
        "status": "Active, not recruiting",
        "inclusion_criteria": "Metastatic Colorectal Cancer (mCRC); Locally confirmed BRAF V600E-mutant; Previously untreated (0 prior regimens) in metastatic setting; ECOG PS 0-1; Measurable disease",
        "exclusion_criteria": "Locally confirmed or unknown MSI-H or dMMR; Symptomatic brain metastases; Active bacterial or viral infections",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": ["BRAF V600E"],
        "prior_lines": [0],
        "disease_stage": ["Stage IV"],
        "msi_status": ["pMMR", "MSS"]
    },
    # RAS Mutant mCRC - Refractory (2+ prior lines)
    {
        "condition_key": "RAS_MUTANT_REFRACTORY",
        "title": "SX-682 and Nivolumab for RAS-Mutated MSS Metastatic Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 1b/2",
        "status": "Active, not recruiting",
        "inclusion_criteria": "Histologically confirmed adenocarcinoma of colon or rectum that is metastatic; Tumor is RAS-mutated (KRAS or NRAS) and MSS/pMMR; Received at least two prior regimens; ECOG PS 0-1",
        "exclusion_criteria": "Major surgery within 4 weeks; Prior treatment with IDO inhibitor or PD-1/PD-L1/CTLA-4 inhibitor; Untreated or symptomatic brain metastases",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": ["KRAS Mutant", "NRAS Mutant"],
        "prior_lines": [2, 3, 4],
        "disease_stage": ["Stage IV"],
        "msi_status": ["pMMR", "MSS"]
    },
    # Early-Stage Rectal Cancer
    {
        "condition_key": "EARLY_STAGE_RECTAL",
        "title": "Neoadjuvant Chemotherapy, Excision and Observation for Early Rectal Cancer",
        "condition": "Early-Stage Rectal Adenocarcinoma",
        "phase": "Phase 2",
        "status": "Recruiting",
        "inclusion_criteria": "Histologically confirmed invasive well-moderately differentiated rectal adenocarcinoma; Tumor stage cT1-T3abN0 (early-stage); M0 stage (no evidence of metastatic disease); ECOG PS 0-1",
        "exclusion_criteria": "History of other malignancy; Synchronous cancer; Prior treatment for rectal cancer",
        "min_age": 18,
        "max_age": 80,
        "sex": "All",
        "molecular_markers": [],
        "prior_lines": [0],
        "disease_stage": ["Early/Localized", "cT2N0M0"],
        "msi_status": []
    },
    # HER2 Positive mCRC
    {
        "condition_key": "HER2_POSITIVE_MCRC",
        "title": "Trastuzumab and Pertuzumab for HER2-Positive Metastatic Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 2",
        "status": "Recruiting",
        "inclusion_criteria": "Histologically confirmed mCRC with HER2 overexpression (IHC 3+ or IHC 2+ with FISH+); At least one prior line of therapy; ECOG PS 0-1; Measurable disease",
        "exclusion_criteria": "Prior HER2-targeted therapy; Symptomatic brain metastases; Severe cardiac dysfunction",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": ["HER2 Positive"],
        "prior_lines": [1, 2, 3],
        "disease_stage": ["Stage IV"],
        "msi_status": []
    },
    # MSI-H/dMMR mCRC
    {
        "condition_key": "MSI_HIGH_MCRC",
        "title": "Pembrolizumab for MSI-H/dMMR Metastatic Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 2",
        "status": "Active, not recruiting",
        "inclusion_criteria": "Histologically confirmed mCRC with MSI-H or dMMR status; Any number of prior lines; ECOG PS 0-1; Measurable disease",
        "exclusion_criteria": "Prior PD-1/PD-L1 inhibitor; Active autoimmune disease; Organ transplant",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": [],
        "prior_lines": [0, 1, 2, 3],
        "disease_stage": ["Stage IV"],
        "msi_status": ["MSI-H", "dMMR"]
    },
    # EGFR Wild-Type Left-Sided mCRC - First Line
    {
        "condition_key": "EGFR_WILD_LEFT_FIRSTLINE",
        "title": "Cetuximab Plus Chemotherapy for EGFR Wild-Type Left-Sided Metastatic Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 3",
        "status": "Recruiting",
        "inclusion_criteria": "Left-sided mCRC (splenic flexure to rectum); RAS/BRAF wild-type; 0 prior lines for metastatic disease; ECOG PS 0-1",
        "exclusion_criteria": "Right-sided colon cancer; Prior anti-EGFR therapy; Symptomatic brain metastases",
        "min_age": 18,
        "max_age": 75,
        "sex": "All",
        "molecular_markers": ["RAS Wild-Type", "BRAF Wild-Type"],
        "prior_lines": [0],
        "disease_stage": ["Stage IV"],
        "msi_status": []
    },
    # BRAF Mutant Refractory
    {
        "condition_key": "BRAF_REFRACTORY",
        "title": "Autophagy Modulation with Hydroxychloroquine in BRAF-Mutated Refractory Colorectal Cancer",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 2",
        "status": "Active, not recruiting",
        "inclusion_criteria": "Metastatic BRAF V600E-mutated Colorectal Cancer; Progression on at least 1 prior line (refractory); ECOG PS 0-1; Adequate organ function",
        "exclusion_criteria": "Prior chemotherapy ≤14 days; Symptomatic leptomeningeal disease; Recent MI or coronary syndromes",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": ["BRAF V600E"],
        "prior_lines": [1, 2, 3],
        "disease_stage": ["Stage IV"],
        "msi_status": []
    },
    # RAS G12 Mutant
    {
        "condition_key": "RAS_G12_MUTANT",
        "title": "RMC-6236 for Patients with RAS G12 Mutations in Advanced Solid Tumors",
        "condition": "Metastatic Colorectal Adenocarcinoma",
        "phase": "Phase 1/1b",
        "status": "Recruiting",
        "inclusion_criteria": "Advanced solid tumor with specific KRAS G12 mutations; Received prior standard therapy; ECOG PS 0-1; Adequate organ function",
        "exclusion_criteria": "Active or untreated brain metastases; History of pneumonitis or ILD; Prior RAS-MULTI(ON) inhibitor",
        "min_age": 18,
        "max_age": None,
        "sex": "All",
        "molecular_markers": ["KRAS G12"],
        "prior_lines": [1, 2, 3],
        "disease_stage": ["Stage IV"],
        "msi_status": []
    }
]

# Generate 20 trials with variations
trials = []
trial_id_counter = 100000

for i in range(20):
    # Select template with rotation
    template = trial_templates[i % len(trial_templates)]
    
    # Create unique NCT ID
    nct_id = f"NCT{trial_id_counter + i:08d}"
    
    # Create trial with variations
    trial = {
        "trial_id": nct_id,
        "title": template["title"],
        "condition": template["condition"],
        "phase": template["phase"],
        "status": template["status"],
        "investigator": f"Dr. Test Investigator {i+1}",
        "brief_summary": f"Clinical trial {i+1} for {template['condition']}. This is a {template['phase']} study.",
        "detailed_description": f"This clinical trial evaluates the efficacy and safety of investigational therapy for {template['condition']}. Participants will be assessed for response, progression-free survival, and overall survival.",
        "inclusion_criteria": template["inclusion_criteria"],
        "exclusion_criteria": template["exclusion_criteria"],
        "eligibility_criteria": f"{template['inclusion_criteria']}. Exclusion: {template['exclusion_criteria']}",
        "minimum_age": str(template["min_age"]) if template["min_age"] else None,
        "maximum_age": str(template["max_age"]) if template["max_age"] else None,
        "sex": template["sex"],
        "combined_trial_text": f"Trial ID: {nct_id}\nTitle: {template['title']}\nCondition: {template['condition']}\nPhase: {template['phase']}\nStatus: {template['status']}\nInvestigator: Dr. Test Investigator {i+1}\n\nInclusion: {template['inclusion_criteria']}\nExclusion: {template['exclusion_criteria']}",
        "_metadata": {
            "condition_key": template["condition_key"],
            "molecular_markers": template["molecular_markers"],
            "prior_lines": template["prior_lines"],
            "disease_stage": template["disease_stage"],
            "msi_status": template["msi_status"]
        }
    }
    
    trials.append(trial)
    
    # Save individual trial JSON
    trial_file = trials_dir / f"trial_{nct_id}.json"
    with open(trial_file, 'w', encoding='utf-8') as f:
        json.dump(trial, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  Created trial: {nct_id} - {template['title'][:50]}...")

# Save all trials
all_trials_file = testing_dir / "all_trials.json"
with open(all_trials_file, 'w', encoding='utf-8') as f:
    json.dump(trials, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✓ Created {len(trials)} trials")
print(f"✓ Saved all trials to: {all_trials_file.name}")

# Generate patients matching trials perfectly
print(f"\n{'='*100}")
print("GENERATING PATIENTS (PERFECT MATCHES)")
print(f"{'='*100}")

patients = []
golden_matches = []

patient_id_counter = 1

# Create patients that perfectly match trials
for i, trial in enumerate(trials):
    metadata = trial["_metadata"]
    
    # Generate patient matching this trial perfectly
    min_age = int(trial["minimum_age"]) if trial["minimum_age"] else 30
    max_age = int(trial["maximum_age"]) if trial["maximum_age"] else 75
    age = random.randint(min_age, max_age)
    
    gender = random.choice(["Male", "Female"])
    
    # Determine molecular marker status based on trial requirements
    braf_status = "Wild-Type"
    kras_status = "Wild-Type"
    nras_status = "Wild-Type"
    msi_status = "pMMR/MSS"
    
    if "BRAF V600E" in metadata["molecular_markers"]:
        braf_status = "V600E Mutant"
    if "BRAF" in str(metadata["molecular_markers"]) and "V600E" not in str(metadata["molecular_markers"]):
        braf_status = "Mutant"
    
    if "KRAS Mutant" in metadata["molecular_markers"] or "KRAS G12" in str(metadata["molecular_markers"]):
        kras_status = "Mutant"
    if "NRAS Mutant" in metadata["molecular_markers"]:
        nras_status = "Mutant"
    
    if "MSI-H" in metadata["msi_status"] or "dMMR" in metadata["msi_status"]:
        msi_status = "MSI-H/dMMR"
    
    prior_lines = random.choice(metadata["prior_lines"]) if metadata["prior_lines"] else 0
    
    disease_stage = random.choice(metadata["disease_stage"]) if metadata["disease_stage"] else "Stage IV"
    
    ecog = random.choice([0, 1])
    
    # Generate patient MRN
    mrn = 1000000 + patient_id_counter * 1000
    
    # Create combined_text matching project structure
    combined_text = f"""Patient MRN: {mrn}
Age: {age}, Gender: {gender}
Date of Visit: {(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')}
Oncologist: Dr. Test Oncologist {patient_id_counter}

Chief Complaint: {trial['condition']}, seeking clinical trial participation

History of Present Illness: Patient diagnosed with {disease_stage} {trial['condition'].lower()}. Molecular testing shows BRAF: {braf_status}, KRAS: {kras_status}, NRAS: {nras_status}. MSI/MMR Status: {msi_status}. Previously treated with {prior_lines} prior line(s) of therapy for metastatic disease.

Past Medical History: No significant comorbidities. Performance status ECOG {ecog}.

Family History: Maternal grandmother with colorectal cancer at age 65.

Social History: Non-smoker, occasional alcohol use.

Review of Systems: Fatigue, occasional abdominal discomfort. No fever, chills, or weight loss.

Medications and Allergies: Current medications include supportive care. No known drug allergies.

Physical Examination: Vital signs stable. ECOG performance status {ecog}. Abdominal examination reveals mild tenderness. No signs of organ dysfunction.

Laboratory and Imaging Results: Complete blood count within normal limits. Liver and kidney function tests adequate. Recent CT scan shows {disease_stage} disease with measurable lesions.

Imaging: CT chest/abdomen/pelvis demonstrates metastatic disease. MRI brain shows no evidence of brain metastases.

Assessment: {disease_stage} {trial['condition']} with {braf_status} BRAF, {kras_status} KRAS, {nras_status} NRAS status. {msi_status} molecular profile. {prior_lines} prior line(s) of therapy. ECOG {ecog}. Eligible for clinical trial consideration."""

    patient = {
        "patient_id": patient_id_counter,
        "mrn": str(mrn),
        "age": age,
        "gender": gender,
        "combined_text": combined_text,
        "oncologist": f"Dr. Test Oncologist {patient_id_counter}",
        "date_of_visit": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
        "created_at": datetime.now().isoformat(),
        "_metadata": {
            "matched_trial_id": trial["trial_id"],
            "disease_stage": disease_stage,
            "braf_status": braf_status,
            "kras_status": kras_status,
            "nras_status": nras_status,
            "msi_status": msi_status,
            "prior_lines": prior_lines,
            "ecog": ecog
        }
    }
    
    patients.append(patient)
    
    # Save individual patient JSON
    patient_file = patients_dir / f"patient_{patient_id_counter}.json"
    with open(patient_file, 'w', encoding='utf-8') as f:
        json.dump(patient, f, indent=2, ensure_ascii=False, default=str)
    
    # Add to golden matches
    golden_matches.append({
        "patient_id": patient_id_counter,
        "patient_mrn": str(mrn),
        "matched_trial_id": trial["trial_id"],
        "match_confidence": 100.0,
        "match_reason": f"Perfect match: Patient characteristics align perfectly with trial {trial['trial_id']} inclusion criteria"
    })
    
    print(f"  Created patient {patient_id_counter} (MRN: {mrn}) -> Matches trial {trial['trial_id']}")
    
    patient_id_counter += 1

# Save all patients
all_patients_file = testing_dir / "all_patients.json"
with open(all_patients_file, 'w', encoding='utf-8') as f:
    json.dump(patients, f, indent=2, ensure_ascii=False, default=str)

# Save golden matches
golden_matches_file = testing_dir / "golden_matches.json"
with open(golden_matches_file, 'w', encoding='utf-8') as f:
    json.dump(golden_matches, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✓ Created {len(patients)} patients")
print(f"✓ Saved all patients to: {all_patients_file.name}")
print(f"✓ Saved golden matches to: {golden_matches_file.name}")

# Create dataset summary
summary = {
    "dataset_info": {
        "created_at": datetime.now().isoformat(),
        "total_patients": len(patients),
        "total_trials": len(trials),
        "total_golden_matches": len(golden_matches),
        "match_accuracy_target": "100%",
        "description": "Test dataset for LLM evaluation. Each patient has a perfect match to one trial."
    },
    "directories": {
        "patients": str(patients_dir),
        "trials": str(trials_dir),
        "root": str(testing_dir)
    },
    "files": {
        "all_patients": "all_patients.json",
        "all_trials": "all_trials.json",
        "golden_matches": "golden_matches.json"
    }
}

summary_file = testing_dir / "dataset_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"✓ Created dataset summary: {summary_file.name}")

print(f"\n{'='*100}")
print("DATASET GENERATION COMPLETE")
print(f"{'='*100}")
print(f"\nDataset location: {testing_dir}")
print(f"\nStructure:")
print(f"  {testing_dir}/")
print(f"    patients/          ({len(patients)} individual patient JSON files)")
print(f"    trials/            ({len(trials)} individual trial JSON files)")
print(f"    all_patients.json  (all patients in one file)")
print(f"    all_trials.json    (all trials in one file)")
print(f"    golden_matches.json (100% accuracy matches)")
print(f"    dataset_summary.json")
print(f"\nReady for LLM performance testing!")

