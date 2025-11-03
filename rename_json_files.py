"""
Rename JSON files to cleaner names
Creates copies with better names for easier use
"""
import shutil
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

dataset_dir = Path("testing_llm _dataset")

# Mapping of original files to clean names
file_mapping = {
    "First give me a Excel of those patients. First gi....json": "patients.json",
    "matched golden dataset.json": "golden_matches.json",
    "NCT04607421 NCT05576896 NCT04599140 NCT05379985 N....json": "trials.json"
}

print("=" * 100)
print("CREATING CLEANLY NAMED JSON FILES")
print("=" * 100)

for original_name, clean_name in file_mapping.items():
    original_path = dataset_dir / original_name
    clean_path = dataset_dir / clean_name
    
    if original_path.exists():
        shutil.copy2(original_path, clean_path)
        print(f"[OK] {original_name}")
        print(f"     -> {clean_name}")
    else:
        print(f"[FAIL] {original_name} not found")

print(f"\n{'='*100}")
print("All files renamed successfully!")
print(f"{'='*100}")

