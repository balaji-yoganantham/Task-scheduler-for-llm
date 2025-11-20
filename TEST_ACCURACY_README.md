# Accuracy Testing Script

This script tests the accuracy of both patient-to-trial and trial-to-patient matching flows using JSON data files instead of a database.

## Overview

The script:
1. Loads patient and trial data from JSON files in the `PATient data` folder
2. Generates embeddings for patients and trials (persisted to `persist/embeddings/`)
3. Builds FAISS and BM25 indices for fast similarity search
4. Runs patient-to-trial matching for all patients
5. Runs trial-to-patient matching for all trials
6. Compares results against ground truth matches from `matches.json`
7. Calculates accuracy metrics (precision, recall, F1 score)
8. Saves all results to `test_results/accuracy_test_[timestamp]/`

## Usage

```bash
python test_accuracy_with_json_data.py
```

## Data Requirements

The script expects the following files in the `PATient data` folder:
- `patient.json` - Array of patient records
- `trial.json` - Array of trial records
- `matches.json` - Ground truth matches with format:
  ```json
  {
    "matches": [
      {
        "MRN": "patient_mrn",
        "results": [
          {
            "trial_id": "NCT...",
            "decision": "ELIGIBLE" | "NOT_ELIGIBLE" | "NEED_MORE_INFORMATION",
            "reasons": [...]
          }
        ]
      }
    ]
  }
  ```

## Output

Results are saved to `test_results/accuracy_test_[timestamp]/`:
- `loaded_patients.json` - Formatted patient data
- `loaded_trials.json` - Formatted trial data
- `patient_to_trial_predictions.json` - Patient-to-trial matching results
- `trial_to_patient_predictions.json` - Trial-to-patient matching results
- `patient_to_trial_accuracy.json` - Accuracy metrics for patient-to-trial
- `trial_to_patient_accuracy.json` - Accuracy metrics for trial-to-patient

## Accuracy Metrics

For each matching flow, the script calculates:
- **Precision**: Correct matches / Total predicted matches
- **Recall**: Correct matches / Total ground truth matches
- **F1 Score**: Harmonic mean of precision and recall
- **Correct Matches**: Number of patients/trials with at least one correct match
- **Incorrect Matches**: Number of patients/trials with incorrect predictions
- **Missing Matches**: Number of patients/trials with missing ground truth matches

## Notes

- Embeddings are persisted to `persist/embeddings/` and reused if they already exist
- The script does NOT use the database - all data comes from JSON files
- Ground truth matches are based on `decision: "ELIGIBLE"` in the matches.json file

