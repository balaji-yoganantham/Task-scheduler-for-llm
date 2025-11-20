"""
Configuration for Task Scheduler LLM Service
"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Configuration flags
USE_DUMMY_DATA = False
USE_LLM_PROCESSING = os.getenv("USE_LLM_PROCESSING", "true").lower() == "true"
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # GPT-4o Mini model
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))  # Set to 0.0 for deterministic outputs
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "8192"))  # Max output tokens
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))  # Increased to 5 minutes for batch operations
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

# Legacy Gemini Configuration (kept for backward compatibility, but not used)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://Admin:NeXtUrN%40123@13.60.219.182:5432/Insightedgedb")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Task Scheduler Configuration
SCHEDULER_INTERVAL_MINUTES = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", "30"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "300"))

# Fixed ID execution (explicit run lists)
ENABLE_FIXED_IDS = os.getenv("ENABLE_FIXED_IDS", "true").lower() == "true"
FIXED_TRIAL_IDS = os.getenv("FIXED_TRIAL_IDS", "NCT06298916").split(",")
FIXED_TRIAL_IDS = [t.strip() for t in FIXED_TRIAL_IDS if t.strip()]

FIXED_PATIENT_IDS = os.getenv("FIXED_PATIENT_IDS", "1,2").split(",")
FIXED_PATIENT_IDS = [p.strip() for p in FIXED_PATIENT_IDS if p.strip()]

# Run interval jobs once immediately on scheduler start
RUN_JOBS_ON_START = os.getenv("RUN_JOBS_ON_START", "true").lower() == "true"

# Run both jobs once and exit (no scheduler loop)
RUN_ONCE_AND_EXIT = os.getenv("RUN_ONCE_AND_EXIT", "false").lower() == "true"

# Enable/Disable specific flows
ENABLE_PATIENT_TO_TRIAL_FLOW = os.getenv("ENABLE_PATIENT_TO_TRIAL_FLOW", "false").lower() == "true"  # Set to "true" to enable patient-to-trial matching
ENABLE_TRIAL_TO_PATIENT_FLOW = os.getenv("ENABLE_TRIAL_TO_PATIENT_FLOW", "true").lower() == "true"  # Set to "true" to enable trial-to-patient matching

# Patient Processing Configuration
DEFAULT_PATIENT_LIMIT = int(os.getenv("DEFAULT_PATIENT_LIMIT", "50"))
# Keyword Generation Configuration - Only 50 patients for keyword generation
KEYWORD_GENERATION_PATIENT_LIMIT = int(os.getenv("KEYWORD_GENERATION_PATIENT_LIMIT", "50"))  # Only 50 patients for keyword generation
MAX_KEYWORD_BATCH_SIZE = int(os.getenv("MAX_KEYWORD_BATCH_SIZE", "20"))  # Max patients per LLM batch call (default: 20)

# Trial-to-Patient Matching Configuration
TOP_K_PATIENTS = int(os.getenv("TOP_K_PATIENTS", "50"))  # Number of top patients to retrieve from hybrid matching
TRIAL_PATIENT_LLM_BATCH_SIZE = int(os.getenv("TRIAL_PATIENT_LLM_BATCH_SIZE", "1"))  # Patients per LLM batch evaluation

# Patient-to-Trial Matching Configuration
TOP_K_TRIALS = int(os.getenv("TOP_K_TRIALS", "50"))  # Number of top trials to retrieve from hybrid matching (same as TOP_K_PATIENTS)
PATIENT_TRIAL_LLM_BATCH_SIZE = int(os.getenv("PATIENT_TRIAL_LLM_BATCH_SIZE", "20"))  # Trials per LLM batch evaluation

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "task_scheduler.log")

# Location-based Filtering Configuration
# Set to True to enable location filtering, False to disable
# You can also override via environment variable: LOCATION_ENABLED=true or LOCATION_ENABLED=false
LOCATION_ENABLED_ENV = os.getenv("LOCATION_ENABLED", "").lower()
if LOCATION_ENABLED_ENV:
    # If environment variable is set, use it
    LOCATION_ENABLED = LOCATION_ENABLED_ENV == "true"
else:
    # Otherwise, set directly here: True = ON, False = OFF
    LOCATION_ENABLED = False  # Change to True to enable location filtering
MAX_DEFAULT_DISTANCE_KM = float(os.getenv("MAX_DEFAULT_DISTANCE_KM", "600"))
LOCATION_WEIGHT = float(os.getenv("LOCATION_WEIGHT", "0.1"))  # Weight for location in hybrid score (0-1)
GEOCODING_CACHE_ENABLED = os.getenv("GEOCODING_CACHE_ENABLED", "true").lower() == "true"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "clinical-trial-matcher")

# Embedding Chunking Configuration
USE_CHUNKED_TRIAL_EMBEDDINGS = os.getenv("USE_CHUNKED_TRIAL_EMBEDDINGS", "true").lower() == "true"
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "450"))  # Leave buffer below 512
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))  # Overlap between chunks

# Chunk weights for weighted average aggregation (must sum to 1.0)
TRIAL_CHUNK_WEIGHTS = {
    "basic_info_summary": float(os.getenv("CHUNK_WEIGHT_BASIC", "0.15")),
    "detailed_description": float(os.getenv("CHUNK_WEIGHT_DESCRIPTION", "0.20")),
    "inclusion_criteria": float(os.getenv("CHUNK_WEIGHT_INCLUSION", "0.25")),
    "exclusion_criteria": float(os.getenv("CHUNK_WEIGHT_EXCLUSION", "0.25")),
    "eligibility_criteria": float(os.getenv("CHUNK_WEIGHT_ELIGIBILITY", "0.15"))
}

# Auto Evaluation Monitor Configuration
# Polling interval in seconds - how often to check for new unevaluated items
AUTO_EVAL_POLL_INTERVAL = int(os.getenv("AUTO_EVAL_POLL_INTERVAL", "5"))  # Check every 5 seconds

# Batch size limits - how many items to process per polling cycle
# Set to 1 to process one at a time, or "all" to process all available items
AUTO_EVAL_PATIENT_BATCH_SIZE = os.getenv("AUTO_EVAL_PATIENT_BATCH_SIZE", "1")  # "1" or "all" - patients per cycle
AUTO_EVAL_TRIAL_BATCH_SIZE = os.getenv("AUTO_EVAL_TRIAL_BATCH_SIZE", "1")  # "1" or "all" - trials per cycle

# Convert "all" to None (unlimited), otherwise convert to int
AUTO_EVAL_PATIENT_BATCH_SIZE_INT = None if AUTO_EVAL_PATIENT_BATCH_SIZE.lower() == "all" else int(AUTO_EVAL_PATIENT_BATCH_SIZE)
AUTO_EVAL_TRIAL_BATCH_SIZE_INT = None if AUTO_EVAL_TRIAL_BATCH_SIZE.lower() == "all" else int(AUTO_EVAL_TRIAL_BATCH_SIZE)

# Enable/disable flows
AUTO_EVAL_ENABLE_PATIENTS = os.getenv("AUTO_EVAL_ENABLE_PATIENTS", "true").lower() == "true"  # Process patients
AUTO_EVAL_ENABLE_TRIALS = os.getenv("AUTO_EVAL_ENABLE_TRIALS", "true").lower() == "true"  # Process trials

# Timeout settings (seconds) - max time per pipeline run
AUTO_EVAL_PATIENT_TIMEOUT = int(os.getenv("AUTO_EVAL_PATIENT_TIMEOUT", "1000"))  # 1000 seconds (~16.7 minutes) per patient
AUTO_EVAL_TRIAL_TIMEOUT = int(os.getenv("AUTO_EVAL_TRIAL_TIMEOUT", "600"))  # 10 minutes per trial

# LLM Sent Text Saving Configuration
# If True: Save LLM prompts/responses to llm_sent/ folder AND database (for patient-to-trial and trial-to-patient pipelines)
# If False: Save only to database, skip folder saving (for patient-to-trial and trial-to-patient pipelines)
# Note: This only affects evaluation calls (trial_evaluation_*, patient_evaluation_*), not keyword generation
SAVE_LLM_SENT_TO_FOLDER = os.getenv("SAVE_LLM_SENT_TO_FOLDER", "true").lower() == "true"

# API Configuration
API_TITLE = "Task Scheduler LLM Service"
API_DESCRIPTION = "Standalone task scheduler for LLM processing"

if USE_DUMMY_DATA:
    API_DESCRIPTION += " (Dummy Data Mode)"
else:
    API_DESCRIPTION += " (Real Data Mode)"

if USE_LLM_PROCESSING:
    API_DESCRIPTION += " with LLM Processing"
else:
    API_DESCRIPTION += " with Static Data"

print(f"Task Scheduler Configuration loaded:")
print(f"  - USE_DUMMY_DATA: {USE_DUMMY_DATA}")
print(f"  - USE_LLM_PROCESSING: {USE_LLM_PROCESSING}")
print(f"  - SCHEDULER_INTERVAL_MINUTES: {SCHEDULER_INTERVAL_MINUTES}")
print(f"  - MAX_CONCURRENT_TASKS: {MAX_CONCURRENT_TASKS}")
print(f"  - LOG_LEVEL: {LOG_LEVEL}")
print(f"  - ENABLE_PATIENT_TO_TRIAL_FLOW: {ENABLE_PATIENT_TO_TRIAL_FLOW} (Patient-to-Trial matching: {'ON' if ENABLE_PATIENT_TO_TRIAL_FLOW else 'OFF'})")
print(f"  - ENABLE_TRIAL_TO_PATIENT_FLOW: {ENABLE_TRIAL_TO_PATIENT_FLOW} (Trial-to-Patient matching: {'ON' if ENABLE_TRIAL_TO_PATIENT_FLOW else 'OFF'})")
print(f"  - LOCATION_ENABLED: {LOCATION_ENABLED} (Location-based filtering: {'ON' if LOCATION_ENABLED else 'OFF'})")
print(f"  - TOP_K_PATIENTS: {TOP_K_PATIENTS} (Top K patients from hybrid matching)")
print(f"  - TOP_K_TRIALS: {TOP_K_TRIALS} (Top K trials from hybrid matching)")
print(f"  - PATIENT_TRIAL_LLM_BATCH_SIZE: {PATIENT_TRIAL_LLM_BATCH_SIZE} (Trials per LLM batch)")
print(f"  - TRIAL_PATIENT_LLM_BATCH_SIZE: {TRIAL_PATIENT_LLM_BATCH_SIZE} (Patients per LLM batch)")
print(f"\nAuto Evaluation Monitor Configuration:")
print(f"  - AUTO_EVAL_POLL_INTERVAL: {AUTO_EVAL_POLL_INTERVAL} seconds")
print(f"  - AUTO_EVAL_ENABLE_PATIENTS: {AUTO_EVAL_ENABLE_PATIENTS} (Patient processing: {'ON' if AUTO_EVAL_ENABLE_PATIENTS else 'OFF'})")
print(f"  - AUTO_EVAL_ENABLE_TRIALS: {AUTO_EVAL_ENABLE_TRIALS} (Trial processing: {'ON' if AUTO_EVAL_ENABLE_TRIALS else 'OFF'})")
print(f"  - AUTO_EVAL_PATIENT_BATCH_SIZE: {AUTO_EVAL_PATIENT_BATCH_SIZE} ({'ALL items' if AUTO_EVAL_PATIENT_BATCH_SIZE_INT is None else f'{AUTO_EVAL_PATIENT_BATCH_SIZE_INT} item(s)'} per cycle)")
print(f"  - AUTO_EVAL_TRIAL_BATCH_SIZE: {AUTO_EVAL_TRIAL_BATCH_SIZE} ({'ALL items' if AUTO_EVAL_TRIAL_BATCH_SIZE_INT is None else f'{AUTO_EVAL_TRIAL_BATCH_SIZE_INT} item(s)'} per cycle)")
print(f"  - AUTO_EVAL_PATIENT_TIMEOUT: {AUTO_EVAL_PATIENT_TIMEOUT}s")
print(f"  - AUTO_EVAL_TRIAL_TIMEOUT: {AUTO_EVAL_TRIAL_TIMEOUT}s")
print(f"\nLLM Sent Text Saving Configuration:")
print(f"  - SAVE_LLM_SENT_TO_FOLDER: {SAVE_LLM_SENT_TO_FOLDER} (Save to folder: {'ON' if SAVE_LLM_SENT_TO_FOLDER else 'OFF'} - only affects patient-to-trial and trial-to-patient evaluation calls)")
