"""
Convert Testing LLM Dataset Excel files to JSON
Converts all Excel files in testing_llm _dataset folder to JSON format
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

print("=" * 100)
print("CONVERTING EXCEL DATASETS TO JSON")
print("=" * 100)

# Find all Excel files
excel_files = list(dataset_dir.glob("*.xlsx"))

if not excel_files:
    print("No Excel files found in testing_llm _dataset folder")
    sys.exit(1)

print(f"\nFound {len(excel_files)} Excel file(s):")
for f in excel_files:
    print(f"  - {f.name}")

# Convert each Excel file to JSON
converted_files = []

for excel_file in excel_files:
    try:
        print(f"\n{'='*100}")
        print(f"Processing: {excel_file.name}")
        print(f"{'='*100}")
        
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Convert to JSON
        # Use records orientation for better readability
        json_data = df.to_dict(orient='records')
        
        # Create output filename
        json_filename = excel_file.stem + ".json"
        json_path = dataset_dir / json_filename
        
        # Save as JSON with proper formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  ✓ Converted to: {json_filename}")
        print(f"  ✓ File size: {json_path.stat().st_size / 1024:.2f} KB")
        
        converted_files.append(json_path)
        
        # Also create a summary file with metadata
        summary_data = {
            "source_file": excel_file.name,
            "rows": len(df),
            "columns": list(df.columns),
            "data_preview": json_data[:3] if len(json_data) > 3 else json_data
        }
        
        summary_filename = excel_file.stem + "_summary.json"
        summary_path = dataset_dir / summary_filename
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  ✓ Summary saved to: {summary_filename}")
        
    except Exception as e:
        print(f"  ✗ Error converting {excel_file.name}: {e}")
        import traceback
        traceback.print_exc()

# Create a master index file
print(f"\n{'='*100}")
print("Creating Master Index")
print(f"{'='*100}")

master_index = {
    "dataset_info": {
        "total_files": len(converted_files),
        "converted_at": pd.Timestamp.now().isoformat(),
        "files": []
    }
}

for json_file in converted_files:
    # Read the JSON to get structure info
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    master_index["dataset_info"]["files"].append({
        "filename": json_file.name,
        "rows": len(data) if isinstance(data, list) else 1,
        "file_size_kb": round(json_file.stat().st_size / 1024, 2)
    })

master_index_path = dataset_dir / "dataset_index.json"
with open(master_index_path, 'w', encoding='utf-8') as f:
    json.dump(master_index, f, indent=2, ensure_ascii=False)

print(f"✓ Master index created: dataset_index.json")

print(f"\n{'='*100}")
print("CONVERSION COMPLETE")
print(f"{'='*100}")
print(f"\nConverted {len(converted_files)} file(s) to JSON format")
print(f"\nJSON files location: {dataset_dir}")
print("\nFiles created:")
for json_file in converted_files:
    print(f"  - {json_file.name}")
print(f"  - dataset_index.json")

