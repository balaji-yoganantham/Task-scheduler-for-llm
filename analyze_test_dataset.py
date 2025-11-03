"""
Analyze Testing LLM Dataset
Reads and analyzes the Excel files in testing_llm _dataset folder
"""
import pandas as pd
import json
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Path to testing dataset folder
dataset_dir = Path("testing_llm _dataset")

# Read all Excel files
files = {
    "patient_data": "First give me a Excel of those patients. First gi....xlsx",
    "golden_matches": "matched golden dataset.xlsx",
    "trial_data": "NCT04607421 NCT05576896 NCT04599140 NCT05379985 N....xlsx"
}

results = {}

print("=" * 100)
print("TESTING LLM DATASET ANALYSIS")
print("=" * 100)

# 1. Read Patient Data
print("\n" + "="*100)
print("1. PATIENT DATA")
print("="*100)
try:
    patient_df = pd.read_excel(dataset_dir / files["patient_data"])
    results["patient_data"] = patient_df
    print(f"\nShape: {patient_df.shape}")
    print(f"\nColumns: {list(patient_df.columns)}")
    print(f"\nFirst few rows:")
    print(patient_df.head().to_string())
    print(f"\nData Types:")
    print(patient_df.dtypes)
    print(f"\nPatient IDs/MRNs:")
    if 'ID' in patient_df.columns:
        print(patient_df['ID'].tolist())
    if 'MRN' in patient_df.columns:
        print(f"MRNs: {patient_df['MRN'].tolist()}")
except Exception as e:
    print(f"Error reading patient data: {e}")

# 2. Read Golden Matches
print("\n" + "="*100)
print("2. GOLDEN MATCHES (GROUND TRUTH)")
print("="*100)
try:
    golden_df = pd.read_excel(dataset_dir / files["golden_matches"])
    results["golden_matches"] = golden_df
    print(f"\nShape: {golden_df.shape}")
    print(f"\nColumns: {list(golden_df.columns)}")
    print(f"\nContent:")
    print(golden_df.to_string())
    print(f"\nMatch Summary:")
    print(f"  Total matches: {len(golden_df)}")
    if 'patient id ' in golden_df.columns:
        print(f"  Patient IDs: {golden_df['patient id '].tolist()}")
    if 'Optimized Trial Match (NCT ID)' in golden_df.columns:
        print(f"  Trial NCT IDs: {golden_df['Optimized Trial Match (NCT ID)'].tolist()}")
except Exception as e:
    print(f"Error reading golden matches: {e}")

# 3. Read Trial Data
print("\n" + "="*100)
print("3. TRIAL DATA")
print("="*100)
try:
    trial_df = pd.read_excel(dataset_dir / files["trial_data"])
    results["trial_data"] = trial_df
    print(f"\nShape: {trial_df.shape}")
    print(f"\nColumns: {list(trial_df.columns)}")
    print(f"\nFirst few rows (showing column names and sample data):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    # Try to print with error handling
    try:
        print(trial_df.head().to_string())
    except UnicodeEncodeError:
        print("(Content contains special characters - showing summary only)")
        for col in trial_df.columns:
            print(f"  {col}: {trial_df[col].iloc[0][:50] if pd.notna(trial_df[col].iloc[0]) else 'N/A'}...")
    if 'NCT ID' in trial_df.columns:
        print(f"\nTrial NCT IDs: {trial_df['NCT ID'].unique().tolist()}")
except Exception as e:
    print(f"Error reading trial data: {e}")
    import traceback
    traceback.print_exc()

# 4. Cross-Reference Analysis
print("\n" + "="*100)
print("4. CROSS-REFERENCE ANALYSIS")
print("="*100)
try:
    if "golden_matches" in results and "patient_data" in results:
        golden = results["golden_matches"]
        patients = results["patient_data"]
        
        print("\nGolden Match to Patient Mapping:")
        for idx, row in golden.iterrows():
            patient_id_col = 'patient id ' if 'patient id ' in golden.columns else None
            trial_col = 'Optimized Trial Match (NCT ID)' if 'Optimized Trial Match (NCT ID)' in golden.columns else None
            
            if patient_id_col and trial_col:
                patient_id = row[patient_id_col]
                trial_id = row[trial_col]
                
                print(f"\n  Patient ID: {patient_id} -> Trial: {trial_id}")
                
                # Try to find patient in patient_data
                if 'ID' in patients.columns:
                    patient_info = patients[patients['ID'] == patient_id]
                    if not patient_info.empty:
                        print(f"    Patient Info: {patient_info.iloc[0].to_dict()}")
                elif 'MRN' in patients.columns:
                    # Try MRN if available
                    mrn_col = patients['MRN'] if 'MRN' in patients.columns else None
                    if mrn_col is not None:
                        print(f"    Available MRNs: {patients['MRN'].tolist()}")
except Exception as e:
    print(f"Error in cross-reference: {e}")
    import traceback
    traceback.print_exc()

# 5. Save Analysis to JSON
print("\n" + "="*100)
print("5. SAVING ANALYSIS")
print("="*100)
try:
    analysis_output = {
        "patient_count": len(results.get("patient_data", pd.DataFrame())),
        "golden_match_count": len(results.get("golden_matches", pd.DataFrame())),
        "trial_count": len(results.get("trial_data", pd.DataFrame())),
        "patient_ids": results["patient_data"]['MRN'].tolist() if "patient_data" in results and 'MRN' in results["patient_data"].columns else [],
        "golden_trials": results["golden_matches"]['Optimized Trial Match (NCT ID)'].tolist() if "golden_matches" in results and 'Optimized Trial Match (NCT ID)' in results["golden_matches"].columns else []
    }
    
    output_file = dataset_dir / "analysis_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_output, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to: {output_file}")
except Exception as e:
    print(f"Error saving analysis: {e}")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
