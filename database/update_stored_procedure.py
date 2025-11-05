"""
Script to update the get_patient_data_for_keywords stored procedure
This fixes the datatype mismatch error between date_of_visit (VARCHAR) and expected DATE
"""

import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config import DATABASE_URL

def update_stored_procedure():
    """Update the stored procedure to fix the datatype mismatch"""
    
    # First drop the existing function
    drop_sql = """
    DROP FUNCTION IF EXISTS insightsedge.get_patient_data_for_keywords(integer);
    """
    
    # Then create the new function with corrected return type
    create_sql = """
    CREATE FUNCTION insightsedge.get_patient_data_for_keywords(
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
                'Patient MRN: ', pmh.mrn, E'\\n',
                'Age: ', COALESCE(pmh.age::TEXT, 'Not specified'), ', Gender: ', COALESCE(pmh.gender, 'Not specified'), E'\\n',
                'Date of Visit: ', COALESCE(pmh.date_of_visit::TEXT, 'Not specified'), E'\\n',
                'Oncologist: ', COALESCE(pmh.oncologist, 'Not specified'), E'\\n\\n',
                'Chief Complaint: ', COALESCE(pmh.chief_complaint, 'Not specified'), E'\\n\\n',
                'History of Present Illness: ', COALESCE(pmh.history_of_present_illness, 'Not specified'), E'\\n\\n',
                'Past Medical History: ', COALESCE(pmh.past_medical_history, 'Not specified'), E'\\n\\n',
                'Family History: ', COALESCE(pmh.family_history, 'Not specified'), E'\\n\\n',
                'Social History: ', COALESCE(pmh.social_history, 'Not specified'), E'\\n\\n',
                'Review of Systems: ', COALESCE(pmh.review_of_systems, 'Not specified'), E'\\n\\n',
                'Medications and Allergies: ', COALESCE(pmh.medications_allergies, 'Not specified'), E'\\n\\n',
                'Physical Examination: ', COALESCE(pmh.physical_examination, 'Not specified'), E'\\n\\n',
                'Laboratory and Imaging Results: ', COALESCE(pmh.laboratory_imaging_results, 'Not specified'), E'\\n\\n',
                'Imaging: ', COALESCE(pmh.imaging, 'Not specified'), E'\\n\\n',
                'Assessment: ', COALESCE(pmh.assessment, 'Not specified')
            ) as combined_text,
            pmh.oncologist,
            pmh.date_of_visit::VARCHAR(50) as date_of_visit,
            pmh.created_at
        FROM insightsedge.patient_medical_history pmh
        WHERE pmh.age IS NOT NULL 
            AND pmh.gender IS NOT NULL
            AND pmh.gender IN ('Male', 'Female')
        ORDER BY pmh.created_at DESC
        LIMIT limit_count;
    END;
    $$;
    """
    
    try:
        engine = create_engine(DATABASE_URL)
        print("Connecting to database...")
        
        with engine.connect() as connection:
            print("Dropping old function definition...")
            connection.execute(text(drop_sql))
            connection.commit()
            print("[OK] Old function dropped")
            
            print("Creating new function with corrected return type...")
            connection.execute(text(create_sql))
            connection.commit()
            print("[OK] Stored procedure updated successfully!")
            
            # Verify the update
            verify_sql = text("""
                SELECT 
                    proname as function_name,
                    pg_get_function_result(oid) as return_type
                FROM pg_proc
                WHERE proname = 'get_patient_data_for_keywords'
                    AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'insightsedge')
            """)
            
            result = connection.execute(verify_sql)
            row = result.fetchone()
            if row:
                print(f"\n[OK] Verification: Function '{row[0]}' exists with return type:")
                print(f"   {row[1]}")
            
    except Exception as e:
        print(f"[ERROR] Error updating stored procedure: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Updating get_patient_data_for_keywords stored procedure")
    print("=" * 60)
    
    success = update_stored_procedure()
    
    if success:
        print("\n[SUCCESS] Stored procedure has been updated!")
        print("You can now run your pipeline again.")
    else:
        print("\n[FAILED] Could not update stored procedure.")
        print("Please check the error message above and try again.")
        sys.exit(1)

