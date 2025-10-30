-- Stored Procedure to get patient medical history data for keyword generation
-- This procedure retrieves patient data and formats it for LLM processing

CREATE OR REPLACE FUNCTION insightsedge.get_patient_data_for_keywords(
    limit_count INTEGER DEFAULT 50
)
RETURNS TABLE (
    patient_id INTEGER,
    mrn VARCHAR,
    age INTEGER,
    gender VARCHAR(20),
    combined_text TEXT,
    oncologist TEXT,
    date_of_visit DATE,
    created_at TIMESTAMP
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pmh.id as patient_id,
        pmh.mrn,
        pmh.age,
        pmh.gender,
        CONCAT(
            'Patient MRN: ', pmh.mrn, E'\n',
            'Age: ', COALESCE(pmh.age::TEXT, 'Not specified'), ', Gender: ', COALESCE(pmh.gender, 'Not specified'), E'\n',
            'Date of Visit: ', COALESCE(pmh.date_of_visit::TEXT, 'Not specified'), E'\n',
            'Oncologist: ', COALESCE(pmh.oncologist, 'Not specified'), E'\n\n',
            'Chief Complaint: ', COALESCE(pmh.chief_complaint, 'Not specified'), E'\n\n',
            'History of Present Illness: ', COALESCE(pmh.history_of_present_illness, 'Not specified'), E'\n\n',
            'Past Medical History: ', COALESCE(pmh.past_medical_history, 'Not specified'), E'\n\n',
            'Family History: ', COALESCE(pmh.family_history, 'Not specified'), E'\n\n',
            'Social History: ', COALESCE(pmh.social_history, 'Not specified'), E'\n\n',
            'Review of Systems: ', COALESCE(pmh.review_of_systems, 'Not specified'), E'\n\n',
            'Medications and Allergies: ', COALESCE(pmh.medications_allergies, 'Not specified'), E'\n\n',
            'Physical Examination: ', COALESCE(pmh.physical_examination, 'Not specified'), E'\n\n',
            'Laboratory and Imaging Results: ', COALESCE(pmh.laboratory_imaging_results, 'Not specified'), E'\n\n',
            'Imaging: ', COALESCE(pmh.imaging, 'Not specified'), E'\n\n',
            'Assessment: ', COALESCE(pmh.assessment, 'Not specified')
        ) as combined_text,
        pmh.oncologist,
        pmh.date_of_visit,
        pmh.created_at
    FROM insightsedge.patient_medical_history_temp pmh
    WHERE pmh.age IS NOT NULL 
        AND pmh.gender IS NOT NULL
        AND pmh.gender IN ('Male', 'Female')
    ORDER BY pmh.created_at DESC
    LIMIT limit_count;
END;
$$;

-- Stored Procedure to get clinical trial details for embedding generation
CREATE OR REPLACE FUNCTION insightsedge.get_trial_data_for_embeddings()
RETURNS TABLE (
    trial_id VARCHAR,
    title TEXT,
    condition TEXT,
    phase VARCHAR,
    status VARCHAR,
    investigator TEXT,
    created_date VARCHAR,
    patients_matched INTEGER,
    matching_status VARCHAR,
    inclusion_criteria TEXT,
    exclusion_criteria TEXT,
    eligibility_criteria TEXT,
    brief_summary TEXT,
    detailed_description TEXT,
    minimum_age VARCHAR,
    maximum_age VARCHAR,
    sex VARCHAR
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ctd.nct_id as trial_id,
        ctd.study_title as title,
        ctd.conditions as condition,
        ctd.study_phase as phase,
        ctd.overall_status as status,
        ctd.lead_sponsor_name as investigator,
        ctd.start_date::VARCHAR as created_date,
        0 as patients_matched,  -- Default value
        'pending' as matching_status,  -- Default value
        ctd.inclusion_criteria,
        ctd.exclusion_criteria,
        ctd.eligibility_criteria,
        ctd.brief_summary,
        ctd.detailed_description,
        ctd.minimum_age,
        ctd.maximum_age,
        ctd.sex
    FROM insightsedge.clinical_trial_details ctd
    WHERE ctd.overall_status IN ('Recruiting', 'Active, not recruiting', 'Enrolling by invitation')
    ORDER BY ctd.created_at DESC;
END;
$$;

-- Stored Procedure to get detailed clinical trial information
CREATE OR REPLACE FUNCTION insightsedge.get_detailed_trial_data()
RETURNS TABLE (
    trial_id TEXT,
    title TEXT,
    condition TEXT,
    phase TEXT,
    status TEXT,
    investigator TEXT,
    created_date TEXT,
    patients_matched INTEGER,
    matching_status TEXT,
    inclusion_criteria TEXT,
    exclusion_criteria TEXT,
    eligibility_criteria TEXT,
    brief_summary TEXT,
    detailed_description TEXT,
    minimum_age TEXT,
    maximum_age TEXT,
    sex TEXT,
    combined_trial_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ctd.nct_id::TEXT as trial_id,
        ctd.study_title::TEXT as title,
        ctd.conditions::TEXT as condition,
        ctd.study_phase::TEXT as phase,
        ctd.overall_status::TEXT as status,
        ctd.lead_sponsor_name::TEXT as investigator,
        ctd.start_date::TEXT as created_date,
        0 as patients_matched,  -- Default value
        'pending'::TEXT as matching_status,  -- Default value
        ctd.inclusion_criteria::TEXT,
        ctd.exclusion_criteria::TEXT,
        ctd.eligibility_criteria::TEXT,
        ctd.brief_summary::TEXT,
        ctd.detailed_description::TEXT,
        ctd.minimum_age::TEXT,
        ctd.maximum_age::TEXT,
        ctd.sex::TEXT,
        CONCAT(
            'Trial ID: ', ctd.nct_id, E'\n',
            'Title: ', ctd.study_title, E'\n',
            'Condition: ', ctd.conditions, E'\n',
            'Phase: ', ctd.study_phase, E'\n',
            'Status: ', ctd.overall_status, E'\n',
            'Investigator: ', ctd.lead_sponsor_name, E'\n',
            'Start Date: ', ctd.start_date::TEXT, E'\n',
            'Age Range: ', COALESCE(ctd.minimum_age, 'Not specified'), ' - ', COALESCE(ctd.maximum_age, 'Not specified'), E'\n',
            'Gender: ', COALESCE(ctd.sex, 'Not specified'), E'\n\n',
            'Brief Summary: ', COALESCE(ctd.brief_summary, 'Not available'), E'\n\n',
            'Detailed Description: ', COALESCE(ctd.detailed_description, 'Not available'), E'\n\n',
            'Inclusion Criteria: ', COALESCE(ctd.inclusion_criteria, 'Not available'), E'\n\n',
            'Exclusion Criteria: ', COALESCE(ctd.exclusion_criteria, 'Not available'), E'\n\n',
            'Eligibility Criteria: ', COALESCE(ctd.eligibility_criteria, 'Not available')
        ) as combined_trial_text
    FROM insightsedge.clinical_trial_details ctd
    WHERE ctd.overall_status IN ('Recruiting', 'Active, not recruiting', 'Enrolling by invitation')
    ORDER BY ctd.created_at DESC;
END;
$$;

-- Stored Procedure to get detailed trial data by trial ID
CREATE OR REPLACE FUNCTION insightsedge.get_detailed_trial_data_by_id(
    p_trial_id VARCHAR(100)
)
RETURNS TABLE (
    trial_id TEXT,
    title TEXT,
    condition TEXT,
    phase TEXT,
    status TEXT,
    investigator TEXT,
    created_date TEXT,
    patients_matched INTEGER,
    matching_status TEXT,
    inclusion_criteria TEXT,
    exclusion_criteria TEXT,
    eligibility_criteria TEXT,
    brief_summary TEXT,
    detailed_description TEXT,
    minimum_age TEXT,
    maximum_age TEXT,
    sex TEXT,
    combined_trial_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ctd.nct_id::TEXT as trial_id,
        ctd.study_title::TEXT as title,
        ctd.conditions::TEXT as condition,
        ctd.study_phase::TEXT as phase,
        ctd.overall_status::TEXT as status,
        ctd.lead_sponsor_name::TEXT as investigator,
        ctd.start_date::TEXT as created_date,
        0 as patients_matched,  -- Default value
        'pending'::TEXT as matching_status,  -- Default value
        ctd.inclusion_criteria::TEXT,
        ctd.exclusion_criteria::TEXT,
        ctd.eligibility_criteria::TEXT,
        ctd.brief_summary::TEXT,
        ctd.detailed_description::TEXT,
        ctd.minimum_age::TEXT,
        ctd.maximum_age::TEXT,
        ctd.sex::TEXT,
        CONCAT(
            'Trial ID: ', ctd.nct_id, E'\n',
            'Title: ', ctd.study_title, E'\n',
            'Condition: ', ctd.conditions, E'\n',
            'Phase: ', ctd.study_phase, E'\n',
            'Status: ', ctd.overall_status, E'\n',
            'Investigator: ', ctd.lead_sponsor_name, E'\n',
            'Start Date: ', ctd.start_date::TEXT, E'\n',
            'Age Range: ', COALESCE(ctd.minimum_age, 'Not specified'), ' - ', COALESCE(ctd.maximum_age, 'Not specified'), E'\n',
            'Gender: ', COALESCE(ctd.sex, 'Not specified'), E'\n\n',
            'Brief Summary: ', COALESCE(ctd.brief_summary, 'Not available'), E'\n\n',
            'Detailed Description: ', COALESCE(ctd.detailed_description, 'Not available'), E'\n\n',
            'Inclusion Criteria: ', COALESCE(ctd.inclusion_criteria, 'Not available'), E'\n\n',
            'Exclusion Criteria: ', COALESCE(ctd.exclusion_criteria, 'Not available'), E'\n\n',
            'Eligibility Criteria: ', COALESCE(ctd.eligibility_criteria, 'Not available')
        ) as combined_trial_text
    FROM insightsedge.clinical_trial_details ctd
    WHERE ctd.nct_id = p_trial_id
        AND ctd.overall_status IN ('Recruiting', 'Active, not recruiting', 'Enrolling by invitation');
END;
$$;
