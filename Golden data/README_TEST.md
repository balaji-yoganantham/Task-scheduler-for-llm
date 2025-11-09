# Golden Data Test Pipelines

These test scripts evaluate both patient-to-trial and trial-to-patient matching pipelines using Golden data (ground truth) without connecting to PostgreSQL database.

## Overview

### Patient-to-Trial Pipeline (`test_patient_to_trial_pipeline.py`)

The test script runs the complete patient-to-trial matching pipeline:
1. **Keyword Generation** - Generates keywords from patient medical records
2. **Embedding Generation** - Creates embeddings for patients and trials
3. **Hybrid Matching** - Uses FAISS (embeddings) + BM25 (keywords) to find matching trials
4. **LLM Evaluation** - Uses Google Gemini to evaluate patient-trial eligibility
5. **Accuracy Calculation** - Compares LLM decisions with Golden data ground truth

### Trial-to-Patient Pipeline (`test_trial_to_patient_pipeline.py`)

The test script runs the complete trial-to-patient matching pipeline:
1. **Patient Keyword Generation** - Generates keywords from patient medical records
2. **Patient Embedding Generation** - Creates embeddings for patients using MedCPT
3. **Trial-to-Patient Hybrid Matching** - Uses FAISS (embeddings) + BM25 (keywords) to find matching patients
4. **LLM Evaluation** - Uses Google Gemini to evaluate patient eligibility (batch processing: 20 patients per batch)
5. **Accuracy Calculation** - Compares LLM decisions with Golden data ground truth

**Key Features:**
- Uses **MedCPT embeddings** from NCBI via Hugging Face
- **Chunking method** for trials exceeding 512 tokens (max 450 tokens per chunk)
- Batch processing with 20 patients per LLM batch

## Files

- `test_patient_to_trial_pipeline.py` - Patient-to-trial test script
- `test_trial_to_patient_pipeline.py` - Trial-to-patient test script
- `patient.json` - Golden patient data (4 patients)
- `matches.json` - Golden match results (ground truth)
- `trials.json` - Empty (trials loaded from `testing/trial.json`)

## Test Persist Folders

All test data (embeddings, keywords, FAISS indices) are saved to separate folders:
- `Golden data/test_persist/` - Patient-to-trial test data (separate from production)
- `Golden data/test_persist_trial_to_patient/` - Trial-to-patient test data (separate from production)

## Usage

### Patient-to-Trial Test

```bash
# Activate virtual environment first
cd "Golden data"
python test_patient_to_trial_pipeline.py
```

### Trial-to-Patient Test

```bash
# Activate virtual environment first
cd "Golden data"
python test_trial_to_patient_pipeline.py
```

**Note:** Always use virtual environment (venv) to execute the code.

## What It Does

### Patient-to-Trial Pipeline

1. **Loads Golden Data**
   - Patients from `patient.json`
   - Ground truth matches from `matches.json`
   - Trials from `testing/trial.json`

2. **Runs Pipeline for Each Patient**
   - Generates keywords (saved to `test_persist/`)
   - Generates embeddings (saved to `test_persist/embeddings/`)
   - Performs hybrid matching to find top trials
   - Evaluates trials using LLM (Google Gemini)

3. **Compares with Golden Data**
   - For each patient, compares LLM decisions with Golden data
   - Calculates accuracy metrics:
     - Overall accuracy
     - Per-patient accuracy
     - Correct/Incorrect/Not Evaluated counts

4. **Saves Results**
   - Test results saved to `test_patient_to_trial_results_YYYYMMDD_HHMMSS.json`
   - Includes detailed comparison for each patient-trial match

### Trial-to-Patient Pipeline

1. **Loads Golden Data**
   - Patients from `patient.json`
   - Ground truth matches from `matches.json` (extracts unique trials)
   - Trials from `testing/trial.json`

2. **Runs Pipeline for Each Trial**
   - Generates patient keywords (saved to `test_persist_trial_to_patient/`)
   - Generates patient embeddings using **MedCPT** (saved to `test_persist_trial_to_patient/embeddings/`)
   - Performs hybrid matching to find top patients
   - Evaluates patients using LLM (Google Gemini) in **batches of 20**
   - Uses **chunking method** for trials exceeding 512 tokens (max 450 tokens per chunk)

3. **Compares with Golden Data**
   - For each trial, compares LLM decisions with Golden data
   - Calculates accuracy metrics:
     - Overall accuracy
     - Per-trial accuracy
     - Correct/Incorrect/Not Evaluated counts

4. **Saves Results**
   - Test results saved to `test_trial_to_patient_results_YYYYMMDD_HHMMSS.json`
   - Includes detailed comparison for each trial-patient match

## Output

### Patient-to-Trial Test

The script prints:
- Progress for each step
- Per-patient accuracy
- Overall accuracy across all patients
- Detailed comparison for each match

Results are saved to JSON file with:
- Patient-by-patient results
- LLM decisions vs Golden data
- Accuracy metrics
- Detailed reasoning comparisons

### Trial-to-Patient Test

The script prints:
- Progress for each step (including chunking info for trials > 512 tokens)
- Per-trial accuracy
- Overall accuracy across all trials
- Batch processing information (20 patients per batch)
- Detailed comparison for each match

Results are saved to JSON file with:
- Trial-by-trial results
- LLM decisions vs Golden data
- Accuracy metrics
- Detailed reasoning comparisons

## Requirements

- Python 3.11+
- **Virtual Environment (venv)** - Always use venv to execute code
- All dependencies from `requirements.txt`
- Google Gemini API key (in `.env` or `config.py`)
- Golden data files in `Golden data/` folder
- Trial data in `testing/trial.json`
- MedCPT models (automatically downloaded from Hugging Face)

## Notes

- **No Database Connection**: All data is loaded from JSON files
- **Separate Persist Folders**: Test data doesn't interfere with production
- **Mock Database Utils**: Uses in-memory data instead of PostgreSQL
- **Full Pipeline**: Tests complete flow from keywords to LLM evaluation
- **MedCPT Embeddings**: Uses NCBI MedCPT models via Hugging Face
- **Chunking Support**: Automatically chunks trials exceeding 512 tokens (max 450 per chunk)
- **Batch Processing**: Trial-to-patient test processes patients in batches of 20 for LLM evaluation

