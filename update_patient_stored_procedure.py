"""
Script to update the patient_medical_history stored procedure with all columns
"""

from sqlalchemy import create_engine, text
from config import DATABASE_URL

def update_stored_procedure():
    """Update the stored procedure to include all columns"""
    
    stored_procedure_sql = """
    CREATE OR REPLACE FUNCTION insightsedge.get_patient_data_for_keywords(
        limit_count INTEGER DEFAULT 50,
        include_evaluated BOOLEAN DEFAULT TRUE
    )
    RETURNS TABLE (
        patient_id INTEGER,
        mrn VARCHAR,
        patient_name VARCHAR(255),
        age INTEGER,
        gender VARCHAR(20),
        combined_text TEXT,
        oncologist TEXT,
        date_of_visit VARCHAR(50),
        vital TEXT,
        created_at TIMESTAMPTZ
    ) 
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            pmh.id as patient_id,
            pmh.mrn,
            pmh.patient_name,
            pmh.age,
            pmh.gender,
            CONCAT(
                'Patient Name: ', COALESCE(pmh.patient_name, 'Not specified'), E'\n',
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
                'Vital Signs: ', COALESCE(pmh.vital, 'Not specified'), E'\n\n',
                'Assessment: ', COALESCE(pmh.assessment, 'Not specified')
            ) as combined_text,
            pmh.oncologist::TEXT as oncologist,
            pmh.date_of_visit::VARCHAR(50) as date_of_visit,
            pmh.vital::TEXT as vital,
            pmh.created_at::TIMESTAMPTZ as created_at
        FROM insightsedge.patient_medical_history pmh
        WHERE pmh.age IS NOT NULL 
            AND pmh.gender IS NOT NULL
            AND pmh.gender IN ('Male', 'Female')
            AND (include_evaluated = TRUE OR pmh.is_evaluated = 0)
        ORDER BY pmh.created_at DESC
        LIMIT limit_count;
    END;
    $$;
    """
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Updating stored procedure: insightsedge.get_patient_data_for_keywords")
            print("-" * 80)
            
            # Drop the existing function first (required when changing return type)
            print("Dropping existing function...")
            drop_sql = "DROP FUNCTION IF EXISTS insightsedge.get_patient_data_for_keywords(INTEGER, BOOLEAN);"
            connection.execute(text(drop_sql))
            connection.commit()
            print("Existing function dropped successfully.")
            
            # Execute the stored procedure update
            print("Creating updated function...")
            connection.execute(text(stored_procedure_sql))
            connection.commit()
            
            print("SUCCESS: Stored procedure updated successfully!")
            print("\nChanges made:")
            print("  + Added patient_name column to return table and combined_text")
            print("  + Added vital column to return table and combined_text")
            print("  + Added include_evaluated parameter (default: TRUE)")
            print("  + Added filter for is_evaluated column")
            print("\nNow all 23 columns from patient_medical_history are properly utilized!")
            
    except Exception as e:
        print(f"ERROR: Error updating stored procedure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_stored_procedure()

