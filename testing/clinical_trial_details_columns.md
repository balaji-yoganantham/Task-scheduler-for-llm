# Clinical Trial Details Table - Column Reference

## Overview
This document lists all columns in the `insightsedge.clinical_trial_details` table based on the stored procedures and queries used in the system.

## Column List

### Primary Identifiers
1. **nct_id** (VARCHAR/TEXT)
   - Clinical trial identifier (NCT number)
   - Primary key/unique identifier
   - Example: "NCT03093116"

### Basic Trial Information
2. **study_title** (TEXT)
   - Full title of the clinical trial
   - Used in matching and display

3. **conditions** (TEXT)
   - Medical conditions/diseases the trial addresses
   - Used for matching with patient conditions

4. **study_phase** (VARCHAR/TEXT)
   - Clinical trial phase (e.g., "Phase 1", "Phase 2", "Phase 3", "Phase 4")
   - Used for filtering and matching

5. **overall_status** (VARCHAR/TEXT)
   - Current status of the trial
   - Common values: 'Recruiting', 'Active, not recruiting', 'Enrolling by invitation', 'RECRUITING', 'ENROLLING_BY_INVITATION', 'AVAILABLE'
   - Used to filter active trials

6. **lead_sponsor_name** (TEXT)
   - Name of the principal investigator or lead sponsor
   - Used for display and reference

7. **start_date** (DATE/TIMESTAMP)
   - Trial start date
   - Converted to VARCHAR/TEXT in queries for display

### Eligibility Criteria
8. **minimum_age** (VARCHAR/TEXT)
   - Minimum age requirement for participants
   - Can be NULL (defaults to 'Not specified')

9. **maximum_age** (VARCHAR/TEXT)
   - Maximum age requirement for participants
   - Can be NULL (defaults to 'Not specified')

10. **sex** (VARCHAR/TEXT)
    - Gender eligibility (e.g., 'Male', 'Female', 'All')
    - Can be NULL (defaults to 'Not specified')

### Trial Descriptions
11. **brief_summary** (TEXT)
    - Short summary of the trial
    - Used in matching and display
    - Can be NULL (defaults to 'Not available')

12. **detailed_description** (TEXT)
    - Comprehensive description of the trial
    - Used in embedding generation and matching
    - Can be NULL (defaults to 'Not available')

### Eligibility Details
13. **inclusion_criteria** (TEXT)
    - Criteria that participants must meet to be included
    - Critical for eligibility matching
    - Can be NULL (defaults to 'Not available')

14. **exclusion_criteria** (TEXT)
    - Criteria that exclude participants from the trial
    - Critical for eligibility matching
    - Can be NULL (defaults to 'Not available')

15. **eligibility_criteria** (TEXT)
    - Combined or general eligibility criteria
    - Used in matching and evaluation
    - Can be NULL (defaults to 'Not available')

### System/Status Fields
16. **is_evaluated** (INTEGER)
    - Flag indicating if the trial has been evaluated
    - 0 = not evaluated, 1 = evaluated
    - Used to filter trials for processing

17. **created_at** (TIMESTAMP)
    - Timestamp when the record was created
    - Used for ordering results (newest first)

## Usage in Stored Procedures

### `get_trial_data_for_embeddings()`
Returns columns: nct_id, study_title, conditions, study_phase, overall_status, lead_sponsor_name, start_date, inclusion_criteria, exclusion_criteria, eligibility_criteria, brief_summary, detailed_description, minimum_age, maximum_age, sex, is_evaluated

### `get_detailed_trial_data()`
Returns all above columns plus a computed `combined_trial_text` field that concatenates all trial information into a single text field for LLM processing.

## Notes
- All TEXT fields can be NULL and are handled with COALESCE in queries
- The table is in the `insightsedge` schema
- Only trials with status 'Recruiting', 'Active, not recruiting', or 'Enrolling by invitation' are typically included in matching
- The `combined_trial_text` field is computed, not stored in the table




