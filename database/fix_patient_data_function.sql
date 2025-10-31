-- Fix for get_patient_data_for_keywords stored procedure
-- This updates the function to match the actual table schema (date_of_visit as VARCHAR)

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
    date_of_visit VARCHAR(50),
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
    FROM insightsedge.patient_medical_history pmh
    WHERE pmh.age IS NOT NULL 
        AND pmh.gender IS NOT NULL
        AND pmh.gender IN ('Male', 'Female')
    ORDER BY pmh.created_at DESC
    LIMIT limit_count;
END;
$$;

-- Verify the function was updated
SELECT 
    proname as function_name,
    pg_get_function_result(oid) as return_type
FROM pg_proc
WHERE proname = 'get_patient_data_for_keywords'
    AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'insightsedge');

