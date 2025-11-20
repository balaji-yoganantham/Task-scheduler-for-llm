# LLM Accuracy Test Script

This script evaluates the accuracy of LLM-based eligibility evaluation by comparing LLM decisions against ground truth matches.

## Overview

The script:
1. Loads predictions from the similarity matching test results
2. Uses LLM to evaluate each patient-trial match
3. Compares LLM decisions (ELIGIBLE/NOT_ELIGIBLE/NEED_MORE_INFO) against ground truth
4. Calculates accuracy metrics (precision, recall, F1 score, confusion matrix)
5. Saves detailed results to `test_results/llm_accuracy_test_[timestamp]/`

## Usage

```bash
python test_llm_accuracy.py
```

**Prerequisites:**
- Must run `test_accuracy_with_json_data.py` first to generate predictions
- Requires OpenAI API key configured in environment variables
- Uses the most recent test results automatically

## What It Tests

### Patient-to-Trial LLM Accuracy
- For each patient and their predicted trials, the LLM evaluates eligibility
- Compares LLM decision against ground truth from `matches.json`
- Calculates:
  - **Accuracy**: Overall correctness of LLM decisions
  - **Precision**: Of trials LLM says ELIGIBLE, how many are actually eligible
  - **Recall**: Of actually eligible trials, how many did LLM find
  - **F1 Score**: Harmonic mean of precision and recall
  - **Confusion Matrix**: TP, TN, FP, FN breakdown

### Trial-to-Patient LLM Accuracy
- For each trial and their predicted patients, the LLM evaluates eligibility
- Compares LLM decision against ground truth
- Same metrics as above

## Output

Results are saved to `test_results/llm_accuracy_test_[timestamp]/`:
- `patient_to_trial_llm_accuracy.json` - LLM accuracy metrics and detailed results
- `trial_to_patient_llm_accuracy.json` - LLM accuracy metrics and detailed results

Each file contains:
- **metrics**: Overall accuracy statistics
- **evaluations**: Detailed per-match evaluation results with:
  - LLM decision vs ground truth
  - Whether LLM was correct
  - LLM reasoning
  - Confidence scores

## Metrics Explained

- **Accuracy**: (Correct Predictions) / (Total Evaluations)
- **Precision**: (True Positives) / (True Positives + False Positives)
  - Of all trials/patients LLM marked as ELIGIBLE, how many were actually eligible?
- **Recall**: (True Positives) / (True Positives + False Negatives)
  - Of all actually eligible trials/patients, how many did LLM find?
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Balanced measure of precision and recall

## Confusion Matrix

- **True Positives (TP)**: LLM correctly identified as ELIGIBLE
- **True Negatives (TN)**: LLM correctly identified as NOT_ELIGIBLE
- **False Positives (FP)**: LLM incorrectly identified as ELIGIBLE
- **False Negatives (FN)**: LLM missed eligible matches (said NOT_ELIGIBLE)

## Notes

- This test uses the actual LLM API, so it will make API calls and may take time
- Results depend on LLM model quality and prompt engineering
- Ground truth is based on `decision: "ELIGIBLE"` in `matches.json`
- The script processes all matches from the similarity matching results

## Expected Results

Good LLM accuracy would show:
- **Accuracy**: > 80%
- **Precision**: > 70% (few false positives)
- **Recall**: > 70% (few false negatives)
- **F1 Score**: > 70%

If accuracy is lower, consider:
- Improving LLM prompts
- Adding more context to patient/trial data
- Fine-tuning evaluation criteria

