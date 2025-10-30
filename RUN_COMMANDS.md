# How to Run the Clinical Trial Matching System

This document provides all the commands to run different components of the project.

## Prerequisites

1. **Activate Virtual Environment** (Required - Always activate before running)
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies** (First time setup only)
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   - Create a `.env` file from `env_template.txt` if not already present
   - Update configuration values in `.env` file (or use defaults from `config.py`)

---

## Main Ways to Run the Project

### 1. **Task Scheduler Service** (Background Service)
Runs scheduled tasks automatically in the background (Patient-to-Trial, Trial-to-Patient matching, etc.)

```bash
python main.py
```

**What it does:**
- Starts a background service that runs scheduled tasks
- Patient-to-Trial matching (every 30 minutes)
- Trial-to-Patient matching (every 30 minutes)
- Trial eligibility updates (hourly)
- Cleanup tasks (daily at 2 AM)

**To stop:** Press `Ctrl+C`

---

### 2. **Patient-to-Trial Pipeline** (Manual Execution)
Find matching trials for a specific patient

#### Run Complete Pipeline:
```bash
python patient_to_trial_pipeline.py --patient-id 1
```

#### With Filters (Age, Gender, Phase):
```bash
python patient_to_trial_pipeline.py --patient-id 1 --age-min 18 --age-max 65 --gender Male --phase-filter "Phase I" "Phase II"
```

#### With Patient Limit (for embedding generation):
```bash
python patient_to_trial_pipeline.py --patient-id 1 --patient-limit 100
```

#### Run Individual Steps:
```bash
# Only keyword generation
python patient_to_trial_pipeline.py --patient-id 1 --step keywords

# Only embedding generation
python patient_to_trial_pipeline.py --patient-id 1 --step embeddings

# Only matching
python patient_to_trial_pipeline.py --patient-id 1 --step matching --age-min 18 --age-max 65
```

---

### 3. **Trial-to-Patient Pipeline** (Manual Execution)
Find matching patients for a specific trial

```bash
python trial_to_patient_pipeline.py --trial-id NCT04929223
```

#### With Filters:
```bash
python trial_to_patient_pipeline.py --trial-id NCT04929223 --age-min 18 --age-max 65 --gender Female
```

---

### 4. **Clinical Trial Matching Pipeline** (Combined)
Run both Patient-to-Trial and Trial-to-Patient pipelines together

#### Run Both Pipelines:
```bash
python clinical_trial_matching_pipeline.py --patient-id 1 --trial-id NCT04929223 --pipeline-type both
```

#### Run Only Patient-to-Trial:
```bash
python clinical_trial_matching_pipeline.py --patient-id 1 --pipeline-type patient-to-trial --patient-limit 50
```

#### Run Only Trial-to-Patient:
```bash
python clinical_trial_matching_pipeline.py --trial-id NCT04929223 --pipeline-type trial-to-patient
```

#### With Filters:
```bash
python clinical_trial_matching_pipeline.py --patient-id 1 --trial-id NCT04929223 --pipeline-type both --age-min 18 --age-max 65 --gender Male --phase-filter "Phase I"
```

---

## Quick Reference: Common Commands

### Start Background Service
```bash
# Activate venv first
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac

# Then run
python main.py
```

### Run Patient Matching (Most Common)
```bash
python patient_to_trial_pipeline.py --patient-id 1
```

### Run Trial Matching
```bash
python trial_to_patient_pipeline.py --trial-id NCT04929223
```

### Run Combined Pipeline
```bash
python clinical_trial_matching_pipeline.py --patient-id 1 --trial-id NCT04929223
```

---

## Command Line Arguments Reference

### Patient-to-Trial Pipeline
| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--patient-id` | Yes | Patient ID to find trials for | `--patient-id 1` |
| `--patient-limit` | No | Number of patients for embedding generation | `--patient-limit 100` |
| `--age-min` | No | Minimum age filter | `--age-min 18` |
| `--age-max` | No | Maximum age filter | `--age-max 65` |
| `--gender` | No | Gender filter (Male/Female) | `--gender Male` |
| `--phase-filter` | No | Phase filter (space-separated) | `--phase-filter "Phase I" "Phase II"` |
| `--step` | No | Run specific step (keywords/embeddings/matching) | `--step matching` |

### Trial-to-Patient Pipeline
| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--trial-id` | Yes | Trial ID (NCT number) to find patients for | `--trial-id NCT04929223` |
| `--age-min` | No | Minimum age filter | `--age-min 18` |
| `--age-max` | No | Maximum age filter | `--age-max 65` |
| `--gender` | No | Gender filter | `--gender Female` |

### Clinical Trial Matching Pipeline
| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--patient-id` | Conditional | Required for patient-to-trial | `--patient-id 1` |
| `--trial-id` | Conditional | Required for trial-to-patient | `--trial-id NCT04929223` |
| `--pipeline-type` | No | Type: patient-to-trial/trial-to-patient/both | `--pipeline-type both` |
| `--patient-limit` | No | Number of patients to process | `--patient-limit 50` |
| `--age-min` | No | Minimum age filter | `--age-min 18` |
| `--age-max` | No | Maximum age filter | `--age-max 65` |
| `--gender` | No | Gender filter | `--gender Male` |
| `--phase-filter` | No | Phase filter | `--phase-filter "Phase I"` |

---

## Environment Setup Checklist

Before running any command, ensure:

- [ ] Virtual environment is activated
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] `.env` file is configured (or use defaults in `config.py`)
- [ ] Database is accessible (check `DATABASE_URL` in config)
- [ ] Gemini API key is set (check `GEMINI_API_KEY` in config)

---

## Troubleshooting

### Error: Module not found
**Solution:** Make sure virtual environment is activated and dependencies are installed
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Error: Database connection failed
**Solution:** Check `DATABASE_URL` in `.env` or `config.py`

### Error: API key not found
**Solution:** Check `GEMINI_API_KEY` in `.env` or `config.py`

### Error: Patient/Trial not found
**Solution:** Verify the patient_id or trial_id exists in the database

---

## Additional Notes

- **Logs:** Check `task_scheduler.log` for service logs
- **Results:** Check `results/` directory for pipeline execution results
- **Embeddings:** Pre-computed embeddings are stored in `persist/embeddings/`
- **Location Cache:** Location data is cached in `persist/location_cache.json`

---

## Example Workflow

1. **Start the background service:**
   ```bash
   python main.py
   ```

2. **Or run a one-time patient matching:**
   ```bash
   python patient_to_trial_pipeline.py --patient-id 1 --age-min 18 --age-max 65
   ```

3. **Or run a one-time trial matching:**
   ```bash
   python trial_to_patient_pipeline.py --trial-id NCT04929223
   ```


