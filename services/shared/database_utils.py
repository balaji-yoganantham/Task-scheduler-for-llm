"""
Shared Database Utilities
Common database operations and stored procedures for both patient-to-trial and trial-to-patient matching
"""

import json
import numpy as np
from sqlalchemy import create_engine, text
from config import DATABASE_URL
from typing import List, Dict, Any, Optional

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def safe_json_dump(data, file_path, **kwargs):
    """Safely dump data to JSON file handling numpy types"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=JSONEncoder, **kwargs)

class DatabaseUtils:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
    
    def _safe_json_load(self, data):
        """Safely load JSON data, handling both string and list formats"""
        if data is None:
            return []
        elif isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return []
        elif isinstance(data, (list, dict)):
            return data
        else:
            return []
    
    def get_connection(self):
        """Get database connection"""
        return self.engine.connect()
    
    def get_patient_data_for_keywords(self, limit: int = 50, include_evaluated: bool = True) -> List[Dict[str, Any]]:
        """Get patient data from database using stored procedure"""
        try:
            with self.get_connection() as connection:
                query = text("SELECT * FROM insightsedge.get_patient_data_for_keywords(:limit, :include_evaluated)")
                result = connection.execute(query, {"limit": limit, "include_evaluated": include_evaluated})
                
                patients = []
                for row in result:
                    # date_of_visit is now VARCHAR, so use it directly or convert if needed
                    date_of_visit = row[7]
                    if date_of_visit and hasattr(date_of_visit, 'isoformat'):
                        date_of_visit = date_of_visit.isoformat()
                    elif date_of_visit is None:
                        date_of_visit = None
                    else:
                        # Already a string, use as is
                        date_of_visit = str(date_of_visit) if date_of_visit else None
                    
                    patients.append({
                        "patient_id": row[0],
                        "mrn": row[1],
                        "patient_name": row[2],
                        "age": row[3],
                        "gender": row[4],
                        "combined_text": row[5],
                        "oncologist": row[6],
                        "date_of_visit": date_of_visit,
                        "vital": row[8],
                        "created_at": row[9].isoformat() if row[9] else None,
                        "is_shortlisted": row[10] if len(row) > 10 else 0
                    })
                
                return patients
        except Exception as e:
            print(f"Error getting patient data: {e}")
            return []
    
    def get_trial_data_for_embeddings(self) -> List[Dict[str, Any]]:
        """Get clinical trial data from database"""
        try:
            with self.get_connection() as connection:
                query = text("SELECT * FROM insightsedge.get_trial_data_for_embeddings()")
                result = connection.execute(query)
                
                trials = []
                for row in result:
                    trials.append({
                        "trial_id": row[0],
                        "title": row[1],
                        "condition": row[2],
                        "phase": row[3],
                        "status": row[4],
                        "investigator": row[5],
                        "created_date": row[6],
                        "patients_matched": row[7],
                        "matching_status": row[8],
                        "inclusion_criteria": row[9] if len(row) > 9 else None,
                        "exclusion_criteria": row[10] if len(row) > 10 else None,
                        "eligibility_criteria": row[11] if len(row) > 11 else None,
                        "brief_summary": row[12] if len(row) > 12 else None,
                        "detailed_description": row[13] if len(row) > 13 else None,
                        "minimum_age": row[14] if len(row) > 14 else None,
                        "maximum_age": row[15] if len(row) > 15 else None,
                        "sex": row[16] if len(row) > 16 else None,
                        "is_evaluated": row[17] if len(row) > 17 else 0
                    })
                
                return trials
        except Exception as e:
            print(f"Error getting trial data: {e}")
            return []
    
    def get_detailed_trial_data(self) -> List[Dict[str, Any]]:
        """Get detailed clinical trial information"""
        try:
            with self.get_connection() as connection:
                # Use direct query instead of stored procedure to avoid data type issues
                query = text("""
                    SELECT 
                        ctd.nct_id::TEXT as trial_id,
                        ctd.study_title::TEXT as title,
                        ctd.conditions::TEXT as condition,
                        ctd.study_phase::TEXT as phase,
                        ctd.overall_status::TEXT as status,
                        ctd.lead_sponsor_name::TEXT as investigator,
                        ctd.start_date::TEXT as created_date,
                        0 as patients_matched,
                        'pending'::TEXT as matching_status,
                        COALESCE(ctd.minimum_age, 'Not specified')::TEXT as minimum_age,
                        COALESCE(ctd.maximum_age, 'Not specified')::TEXT as maximum_age,
                        COALESCE(ctd.sex, 'Not specified')::TEXT as sex,
                        COALESCE(ctd.brief_summary, 'Not available')::TEXT as brief_summary,
                        COALESCE(ctd.detailed_description, 'Not available')::TEXT as detailed_description,
                        COALESCE(ctd.inclusion_criteria, 'Not available')::TEXT as inclusion_criteria,
                        COALESCE(ctd.exclusion_criteria, 'Not available')::TEXT as exclusion_criteria,
                        COALESCE(ctd.eligibility_criteria, 'Not available')::TEXT as eligibility_criteria,
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
                        ) as combined_trial_text,
                        COALESCE(ctd.is_evaluated, 0) as is_evaluated
                    FROM insightsedge.clinical_trial_details ctd
                    WHERE ctd.overall_status IN ('RECRUITING', 'ENROLLING_BY_INVITATION', 'AVAILABLE')
                    ORDER BY ctd.created_at DESC
                    LIMIT 100
                """)
                result = connection.execute(query)
                
                trials = []
                for row in result:
                    trials.append({
                        "trial_id": str(row[0]),
                        "title": str(row[1]),
                        "condition": str(row[2]),
                        "phase": str(row[3]),
                        "status": str(row[4]),
                        "investigator": str(row[5]),
                        "created_date": str(row[6]),
                        "patients_matched": int(row[7]),
                        "matching_status": str(row[8]),
                        "minimum_age": str(row[9]),
                        "maximum_age": str(row[10]),
                        "sex": str(row[11]),
                        "brief_summary": str(row[12]),
                        "detailed_description": str(row[13]),
                        "inclusion_criteria": str(row[14]),
                        "exclusion_criteria": str(row[15]),
                        "eligibility_criteria": str(row[16]),
                        "combined_trial_text": str(row[17]),
                        "is_evaluated": int(row[18]) if len(row) > 18 else 0
                    })
                
                print(f"Successfully retrieved {len(trials)} trials from database")
                return trials
        except Exception as e:
            print(f"Error getting detailed trial data: {e}")
            print("Returning empty list to continue processing")
            return []
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get specific patient by ID from patient_medical_history table"""
        try:
            with self.get_connection() as connection:
                query = text("""
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
                        pmh.oncologist,
                        pmh.date_of_visit,
                        pmh.vital,
                        pmh.created_at,
                        COALESCE(pmh.is_shortlisted, 0) as is_shortlisted
                    FROM insightsedge.patient_medical_history pmh
                    WHERE pmh.id = :patient_id
                """)
                result = connection.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "patient_id": row[0],
                        "mrn": row[1],
                        "patient_name": row[2],
                        "age": row[3],
                        "gender": row[4],
                        "combined_text": row[5],
                        "oncologist": row[6],
                        "date_of_visit": row[7].isoformat() if row[7] and hasattr(row[7], 'isoformat') else str(row[7]) if row[7] else None,
                        "vital": row[8],
                        "created_at": row[9].isoformat() if row[9] else None,
                        "is_shortlisted": int(row[10]) if len(row) > 10 else 0
                    }
                return None
        except Exception as e:
            print(f"Error getting patient by ID: {e}")
            return None

    def get_patient_by_mrn(self, mrn: str) -> Optional[Dict[str, Any]]:
        """Get specific patient by MRN"""
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT * FROM insightsedge.get_patient_data_for_keywords(1000, TRUE)
                    WHERE mrn = :mrn
                """)
                result = connection.execute(query, {"mrn": mrn})
                row = result.fetchone()
                
                if row:
                    date_of_visit = row[7]
                    if date_of_visit and hasattr(date_of_visit, 'isoformat'):
                        date_of_visit = date_of_visit.isoformat()
                    elif date_of_visit is None:
                        date_of_visit = None
                    else:
                        date_of_visit = str(date_of_visit) if date_of_visit else None
                    
                    return {
                        "patient_id": row[0],
                        "mrn": row[1],
                        "patient_name": row[2],
                        "age": row[3],
                        "gender": row[4],
                        "combined_text": row[5],
                        "oncologist": row[6],
                        "date_of_visit": date_of_visit,
                        "vital": row[8],
                        "created_at": row[9].isoformat() if row[9] else None,
                        "is_shortlisted": row[10] if len(row) > 10 else 0
                    }
                return None
        except Exception as e:
            print(f"Error getting patient by MRN: {e}")
            return None
    
    def get_trial_by_id(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trial by ID with all detailed fields"""
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT 
                        ctd.nct_id::TEXT as trial_id,
                        ctd.study_title::TEXT as title,
                        ctd.conditions::TEXT as condition,
                        ctd.study_phase::TEXT as phase,
                        ctd.overall_status::TEXT as status,
                        ctd.lead_sponsor_name::TEXT as investigator,
                        ctd.start_date::TEXT as created_date,
                        0 as patients_matched,
                        'pending'::TEXT as matching_status,
                        COALESCE(ctd.minimum_age, 'Not specified')::TEXT as minimum_age,
                        COALESCE(ctd.maximum_age, 'Not specified')::TEXT as maximum_age,
                        COALESCE(ctd.sex, 'Not specified')::TEXT as sex,
                        COALESCE(ctd.brief_summary, 'Not available')::TEXT as brief_summary,
                        COALESCE(ctd.detailed_description, 'Not available')::TEXT as detailed_description,
                        COALESCE(ctd.inclusion_criteria, 'Not available')::TEXT as inclusion_criteria,
                        COALESCE(ctd.exclusion_criteria, 'Not available')::TEXT as exclusion_criteria,
                        COALESCE(ctd.eligibility_criteria, 'Not available')::TEXT as eligibility_criteria,
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
                        ) as combined_trial_text,
                        COALESCE(ctd.is_evaluated, 0) as is_evaluated
                    FROM insightsedge.clinical_trial_details ctd
                    WHERE ctd.nct_id = :trial_id
                        AND ctd.overall_status IN ('RECRUITING', 'ENROLLING_BY_INVITATION', 'AVAILABLE')
                """)
                result = connection.execute(query, {"trial_id": trial_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "trial_id": str(row[0]),
                        "title": str(row[1]),
                        "condition": str(row[2]),
                        "phase": str(row[3]),
                        "status": str(row[4]),
                        "investigator": str(row[5]),
                        "created_date": str(row[6]),
                        "patients_matched": int(row[7]),
                        "matching_status": str(row[8]),
                        "minimum_age": str(row[9]),
                        "maximum_age": str(row[10]),
                        "sex": str(row[11]),
                        "brief_summary": str(row[12]),
                        "detailed_description": str(row[13]),
                        "inclusion_criteria": str(row[14]),
                        "exclusion_criteria": str(row[15]),
                        "eligibility_criteria": str(row[16]),
                        "combined_trial_text": str(row[17]),
                        "is_evaluated": int(row[18]) if len(row) > 18 else 0
                    }
                return None
        except Exception as e:
            print(f"Error getting trial by ID {trial_id}: {e}")
            return None
    
    def get_patient_location(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Get patient location information from database
        
        Returns:
            Dict with 'location' (address string), 'latitude', 'longitude' fields, or None
        """
        try:
            with self.get_connection() as connection:
                # Try to get location from patient_medical_history table
                # Only location column exists (latitude, longitude, address, and city columns don't exist)
                query = text("""
                    SELECT 
                        COALESCE(pmh.location, '') as location
                    FROM insightsedge.patient_medical_history pmh
                    WHERE pmh.id = :patient_id
                """)
                result = connection.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                if row:
                    location_str = str(row[0]) if row[0] and str(row[0]).strip() else None
                    if not location_str:
                        return None
                    location_data = {
                        "location": location_str,
                        "latitude": None,  # Will be geocoded from location string if needed
                        "longitude": None  # Will be geocoded from location string if needed
                    }
                    return location_data
                return None
        except Exception as e:
            # If columns don't exist, silently return None (graceful degradation)
            print(f"Error getting patient location (may not exist in schema): {e}")
            return None
    
    def get_trial_location(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trial location information from database
        The locations column contains a JSON array of location objects with city, state, country
        
        Returns:
            Dict with:
            - 'locations': List of location objects (parsed from JSON)
            - Or None if no location data found
        """
        try:
            with self.get_connection() as connection:
                # Try to get location from clinical_trial_details table
                # Adjust column names based on actual database schema
                query = text("""
                    SELECT 
                        ctd.locations
                    FROM insightsedge.clinical_trial_details ctd
                    WHERE ctd.nct_id = :trial_id
                """)
                result = connection.execute(query, {"trial_id": trial_id})
                row = result.fetchone()
                
                if not row or not row[0]:
                    return None
                
                locations_json = row[0]  # This is the JSON array
                
                # Parse locations JSON array
                location_objects = []
                if locations_json:
                    try:
                        # If it's already a list (from psycopg2's JSON handling)
                        if isinstance(locations_json, list):
                            location_objects = locations_json
                        # If it's a string, parse it as JSON
                        elif isinstance(locations_json, str):
                            location_objects = json.loads(locations_json)
                        # If it's already a dict (single location), wrap it in a list
                        elif isinstance(locations_json, dict):
                            location_objects = [locations_json]
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Error parsing locations JSON for trial {trial_id}: {e}")
                        location_objects = []
                
                # Return location objects if we have any
                if location_objects and len(location_objects) > 0:
                    return {
                        "locations": location_objects
                    }
                
                return None
        except Exception as e:
            # If columns don't exist, silently return None (graceful degradation)
            print(f"Error getting trial location (may not exist in schema): {e}")
            return None
    
    def save_patient_keywords(self, keywords_data: Dict[str, Any]) -> bool:
        """Save patient keywords to database"""
        try:
            with self.get_connection() as connection:
                # Check if keywords already exist for this patient
                check_query = text("""
                    SELECT id FROM insightsedge.patient_keywords 
                    WHERE patient_id = :patient_id
                """)
                result = connection.execute(check_query, {"patient_id": keywords_data["patient_id"]})
                existing_record = result.fetchone()
                
                if existing_record:
                    # Update existing record
                    update_query = text("""
                        UPDATE insightsedge.patient_keywords SET
                            mrn = :mrn,
                            age = :age,
                            gender = :gender,
                            summary = :summary,
                            primary_diagnosis = :primary_diagnosis,
                            stage = :stage,
                            metastatic_sites = :metastatic_sites,
                            molecular_markers = :molecular_markers,
                            comorbidities = :comorbidities,
                            medications = :medications,
                            allergies = :allergies,
                            performance_status = :performance_status,
                            family_history = :family_history,
                            keywords = :keywords,
                            keywords_text = :keywords_text,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE patient_id = :patient_id
                    """)
                    
                    connection.execute(update_query, {
                        "patient_id": keywords_data["patient_id"],
                        "mrn": keywords_data.get("mrn"),
                        "age": keywords_data.get("age"),
                        "gender": keywords_data.get("gender"),
                        "summary": keywords_data.get("summary"),
                        "primary_diagnosis": keywords_data.get("primary_diagnosis"),
                        "stage": keywords_data.get("stage"),
                        "metastatic_sites": json.dumps(keywords_data.get("metastatic_sites", [])),
                        "molecular_markers": json.dumps(keywords_data.get("molecular_markers", [])),
                        "comorbidities": json.dumps(keywords_data.get("comorbidities", [])),
                        "medications": json.dumps(keywords_data.get("medications", [])),
                        "allergies": json.dumps(keywords_data.get("allergies", [])),
                        "performance_status": keywords_data.get("performance_status"),
                        "family_history": json.dumps(keywords_data.get("family_history", [])),
                        "keywords": json.dumps(keywords_data.get("keywords", [])),
                        "keywords_text": keywords_data.get("keywords_text")
                    })
                    print(f"Updated keywords for patient {keywords_data['patient_id']}")
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO insightsedge.patient_keywords (
                            patient_id, mrn, age, gender, summary, primary_diagnosis, stage,
                            metastatic_sites, molecular_markers, comorbidities, medications,
                            allergies, performance_status, family_history, keywords, keywords_text
                        ) VALUES (
                            :patient_id, :mrn, :age, :gender, :summary, :primary_diagnosis, :stage,
                            :metastatic_sites, :molecular_markers, :comorbidities, :medications,
                            :allergies, :performance_status, :family_history, :keywords, :keywords_text
                        )
                    """)
                    
                    connection.execute(insert_query, {
                        "patient_id": keywords_data["patient_id"],
                        "mrn": keywords_data.get("mrn"),
                        "age": keywords_data.get("age"),
                        "gender": keywords_data.get("gender"),
                        "summary": keywords_data.get("summary"),
                        "primary_diagnosis": keywords_data.get("primary_diagnosis"),
                        "stage": keywords_data.get("stage"),
                        "metastatic_sites": json.dumps(keywords_data.get("metastatic_sites", [])),
                        "molecular_markers": json.dumps(keywords_data.get("molecular_markers", [])),
                        "comorbidities": json.dumps(keywords_data.get("comorbidities", [])),
                        "medications": json.dumps(keywords_data.get("medications", [])),
                        "allergies": json.dumps(keywords_data.get("allergies", [])),
                        "performance_status": keywords_data.get("performance_status"),
                        "family_history": json.dumps(keywords_data.get("family_history", [])),
                        "keywords": json.dumps(keywords_data.get("keywords", [])),
                        "keywords_text": keywords_data.get("keywords_text")
                    })
                    print(f"Inserted keywords for patient {keywords_data['patient_id']}")
                
                connection.commit()
                return True
                
        except Exception as e:
            print(f"Error saving patient keywords: {e}")
            return False
    
    def get_patient_keywords(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get patient keywords from database"""
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT * FROM insightsedge.patient_keywords 
                    WHERE patient_id = :patient_id
                """)
                result = connection.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "id": row[0],
                        "patient_id": row[1],
                        "mrn": row[2],
                        "age": row[3],
                        "gender": row[4],
                        "summary": row[5],
                        "primary_diagnosis": row[6],
                        "stage": row[7],
                        "metastatic_sites": self._safe_json_load(row[8]),
                        "molecular_markers": self._safe_json_load(row[9]),
                        "comorbidities": self._safe_json_load(row[10]),
                        "medications": self._safe_json_load(row[11]),
                        "allergies": self._safe_json_load(row[12]),
                        "performance_status": row[13],
                        "family_history": self._safe_json_load(row[14]),
                        "keywords": self._safe_json_load(row[15]),
                        "keywords_text": row[16],
                        "generated_at": row[17].isoformat() if row[17] else None,
                        "created_at": row[18].isoformat() if row[18] else None,
                        "updated_at": row[19].isoformat() if row[19] else None
                    }
                return None
        except Exception as e:
            # Check if it's a table not found error
            if "does not exist" in str(e) or "UndefinedTable" in str(type(e).__name__):
                print(f"WARNING: patient_keywords table does not exist. Run create_patient_keywords_table.py to create it.")
            else:
                print(f"Error getting patient keywords: {e}")
            return None
    
    def get_all_patient_keywords(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all patient keywords from database"""
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT * FROM insightsedge.patient_keywords 
                    ORDER BY created_at DESC 
                    LIMIT :limit
                """)
                result = connection.execute(query, {"limit": limit})
                
                keywords_list = []
                for row in result:
                    keywords_list.append({
                        "id": row[0],
                        "patient_id": row[1],
                        "mrn": row[2],
                        "age": row[3],
                        "gender": row[4],
                        "summary": row[5],
                        "primary_diagnosis": row[6],
                        "stage": row[7],
                        "metastatic_sites": self._safe_json_load(row[8]),
                        "molecular_markers": self._safe_json_load(row[9]),
                        "comorbidities": self._safe_json_load(row[10]),
                        "medications": self._safe_json_load(row[11]),
                        "allergies": self._safe_json_load(row[12]),
                        "performance_status": row[13],
                        "family_history": self._safe_json_load(row[14]),
                        "keywords": self._safe_json_load(row[15]),
                        "keywords_text": row[16],
                        "generated_at": row[17].isoformat() if row[17] else None,
                        "created_at": row[18].isoformat() if row[18] else None,
                        "updated_at": row[19].isoformat() if row[19] else None
                    })
                
                return keywords_list
        except Exception as e:
            print(f"Error getting all patient keywords: {e}")
            return []
    
    def delete_patient_keywords(self, patient_id: int) -> bool:
        """Delete patient keywords from database"""
        try:
            with self.get_connection() as connection:
                query = text("""
                    DELETE FROM insightsedge.patient_keywords 
                    WHERE patient_id = :patient_id
                """)
                result = connection.execute(query, {"patient_id": patient_id})
                connection.commit()
                
                if result.rowcount > 0:
                    print(f"Deleted keywords for patient {patient_id}")
                    return True
                else:
                    print(f"No keywords found for patient {patient_id}")
                    return False
                    
        except Exception as e:
            print(f"Error deleting patient keywords: {e}")
            return False
    
    def update_patients_evaluated(self, patient_ids: List[int]) -> int:
        """
        Update is_evaluated = 1 for a list of patient IDs
        
        Args:
            patient_ids: List of patient IDs to update
            
        Returns:
            Number of patients successfully updated
        """
        if not patient_ids:
            return 0
            
        try:
            with self.get_connection() as connection:
                # Use IN clause for PostgreSQL compatibility
                query = text("""
                    UPDATE insightsedge.patient_medical_history 
                    SET is_evaluated = 1
                    WHERE id = ANY(:patient_ids::int[])
                """)
                result = connection.execute(query, {"patient_ids": patient_ids})
                connection.commit()
                
                updated_count = result.rowcount
                if updated_count > 0:
                    print(f"✅ Updated is_evaluated=1 for {updated_count} patients: {patient_ids}")
                return updated_count
        except Exception as e:
            # Fallback to IN clause if array casting fails
            try:
                with self.get_connection() as connection:
                    placeholders = ','.join([f':id{i}' for i in range(len(patient_ids))])
                    query = text(f"""
                        UPDATE insightsedge.patient_medical_history 
                        SET is_evaluated = 1
                        WHERE id IN ({placeholders})
                    """)
                    params = {f'id{i}': pid for i, pid in enumerate(patient_ids)}
                    result = connection.execute(query, params)
                    connection.commit()
                    
                    updated_count = result.rowcount
                    if updated_count > 0:
                        print(f"✅ Updated is_evaluated=1 for {updated_count} patients: {patient_ids}")
                    return updated_count
            except Exception as e2:
                print(f"Error updating patients evaluated status (fallback also failed): {e2}")
                return 0
    
    def get_unevaluated_patient_ids(self, limit: int = 50) -> List[int]:
        """
        Get list of patient IDs where is_evaluated = 0
        
        Args:
            limit: Maximum number of patient IDs to return
            
        Returns:
            List of patient IDs
        """
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT id 
                    FROM insightsedge.patient_medical_history 
                    WHERE is_evaluated = 0 
                        AND age IS NOT NULL 
                        AND gender IS NOT NULL
                        AND gender IN ('Male', 'Female')
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                result = connection.execute(query, {"limit": limit})
                
                patient_ids = [row[0] for row in result]
                return patient_ids
        except Exception as e:
            print(f"Error getting unevaluated patient IDs: {e}")
            return []
    
    def get_patients_evaluated_status(self, patient_ids: List[int]) -> Dict[int, int]:
        """
        Get is_evaluated status for multiple patient IDs
        
        Args:
            patient_ids: List of patient IDs to check
            
        Returns:
            Dictionary mapping patient_id to is_evaluated status (0 or 1)
        """
        if not patient_ids:
            return {}
        
        try:
            with self.get_connection() as connection:
                # Use ANY for PostgreSQL array compatibility
                query = text("""
                    SELECT id, is_evaluated
                    FROM insightsedge.patient_medical_history 
                    WHERE id = ANY(:patient_ids::int[])
                """)
                result = connection.execute(query, {"patient_ids": patient_ids})
                
                status_dict = {row[0]: row[1] for row in result}
                return status_dict
        except Exception as e:
            # Fallback to IN clause if array casting fails
            try:
                with self.get_connection() as connection:
                    placeholders = ','.join([f':id{i}' for i in range(len(patient_ids))])
                    query = text(f"""
                        SELECT id, is_evaluated
                        FROM insightsedge.patient_medical_history 
                        WHERE id IN ({placeholders})
                    """)
                    params = {f'id{i}': pid for i, pid in enumerate(patient_ids)}
                    result = connection.execute(query, params)
                    
                    status_dict = {row[0]: row[1] for row in result}
                    return status_dict
            except Exception as e2:
                print(f"Error getting patients evaluated status (fallback also failed): {e2}")
                return {}
    
    def get_patients_evaluated_and_shortlisted_status(self, patient_ids: List[int]) -> Dict[int, Dict[str, int]]:
        """
        Get is_evaluated and is_shortlisted status for multiple patient IDs
        
        Args:
            patient_ids: List of patient IDs to check
            
        Returns:
            Dictionary mapping patient_id to {'is_evaluated': int, 'is_shortlisted': int}
        """
        if not patient_ids:
            return {}
        
        try:
            with self.get_connection() as connection:
                # Use ANY for PostgreSQL array compatibility
                query = text("""
                    SELECT id, COALESCE(is_evaluated, 0) as is_evaluated, COALESCE(is_shortlisted, 0) as is_shortlisted
                    FROM insightsedge.patient_medical_history 
                    WHERE id = ANY(:patient_ids::int[])
                """)
                result = connection.execute(query, {"patient_ids": patient_ids})
                
                status_dict = {
                    row[0]: {
                        'is_evaluated': row[1],
                        'is_shortlisted': row[2]
                    } for row in result
                }
                return status_dict
        except Exception as e:
            # Fallback to IN clause if array casting fails
            try:
                with self.get_connection() as connection:
                    placeholders = ','.join([f':id{i}' for i in range(len(patient_ids))])
                    query = text(f"""
                        SELECT id, COALESCE(is_evaluated, 0) as is_evaluated, COALESCE(is_shortlisted, 0) as is_shortlisted
                        FROM insightsedge.patient_medical_history 
                        WHERE id IN ({placeholders})
                    """)
                    params = {f'id{i}': pid for i, pid in enumerate(patient_ids)}
                    result = connection.execute(query, params)
                    
                    status_dict = {
                        row[0]: {
                            'is_evaluated': row[1],
                            'is_shortlisted': row[2]
                        } for row in result
                    }
                    return status_dict
            except Exception as e2:
                print(f"Error getting patients evaluated and shortlisted status (fallback also failed): {e2}")
                return {}
    
    def update_trial_evaluated(self, trial_id: str) -> bool:
        """
        Update is_evaluated = 1 for a specific trial
        
        Args:
            trial_id: Trial ID (NCT ID) to update
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                query = text("""
                    UPDATE insightsedge.clinical_trial_details 
                    SET is_evaluated = 1
                    WHERE nct_id = :trial_id
                """)
                result = connection.execute(query, {"trial_id": trial_id})
                connection.commit()
                
                updated_count = result.rowcount
                if updated_count > 0:
                    print(f"✅ Updated is_evaluated=1 for trial {trial_id}")
                    return True
                else:
                    print(f"⚠️ No trial found with ID {trial_id} to update")
                    return False
        except Exception as e:
            print(f"Error updating trial evaluated status for {trial_id}: {e}")
            return False
    
    def get_unevaluated_trial_ids(self, limit: int = 50) -> List[str]:
        """
        Get list of trial IDs where is_evaluated = 0
        
        Args:
            limit: Maximum number of trial IDs to return
            
        Returns:
            List of trial IDs (NCT IDs)
        """
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT nct_id 
                    FROM insightsedge.clinical_trial_details 
                    WHERE COALESCE(is_evaluated, 0) = 0
                        AND overall_status IN ('RECRUITING', 'ENROLLING_BY_INVITATION', 'AVAILABLE', 
                                               'Recruiting', 'Active, not recruiting', 'Enrolling by invitation')
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                result = connection.execute(query, {"limit": limit})
                
                trial_ids = [row[0] for row in result]
                return trial_ids
        except Exception as e:
            print(f"Error getting unevaluated trial IDs: {e}")
            return []
    
    def update_scheduler_timestamp(self, scheduler_id: int = 2) -> bool:
        """
        Update the last run timestamp for a scheduler in trial_scheduler_details table
        
        Args:
            scheduler_id: ID of the scheduler to update (default: 2 for "LLM Scheduler")
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                # Try common column names for timestamp
                # The table might have: timestamp, last_run, last_run_timestamp, updated_at, etc.
                # Try to get the actual column name first
                column_query = text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'insightsedge' 
                      AND table_name = 'trial_scheduler_details'
                      AND column_name IN ('timestamp', 'last_run', 'last_run_timestamp', 'updated_at', 'last_updated')
                    ORDER BY column_name
                    LIMIT 1
                """)
                result = connection.execute(column_query)
                row = result.fetchone()
                
                if row:
                    column_name = row[0]
                else:
                    # If no matching column found, try to get any timestamp-like column
                    fallback_query = text("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = 'insightsedge' 
                          AND table_name = 'trial_scheduler_details'
                          AND data_type IN ('timestamp without time zone', 'timestamp with time zone', 'timestamp')
                        LIMIT 1
                    """)
                    result = connection.execute(fallback_query)
                    row = result.fetchone()
                    if row:
                        column_name = row[0]
                    else:
                        print(f"⚠️ No timestamp column found in trial_scheduler_details table")
                        return False
                
                # Update using the found column name
                query = text(f"""
                    UPDATE insightsedge.trial_scheduler_details 
                    SET {column_name} = CURRENT_TIMESTAMP
                    WHERE id = :scheduler_id
                """)
                result = connection.execute(query, {"scheduler_id": scheduler_id})
                connection.commit()
                
                updated_count = result.rowcount
                if updated_count > 0:
                    print(f"✅ Updated scheduler timestamp ({column_name}) for scheduler ID {scheduler_id}")
                    return True
                else:
                    print(f"⚠️ No scheduler found with ID {scheduler_id} to update")
                    return False
        except Exception as e:
            print(f"Error updating scheduler timestamp for scheduler ID {scheduler_id}: {e}")
            return False
    
    def get_evaluated_patient_ids_for_trial(self, trial_id: str) -> List[int]:
        """
        Get list of patient IDs that were evaluated for a specific trial.
        These patient IDs come from patient_medical_history table and were evaluated for the trial.
        The trial_to_patient table stores the evaluation results linking trial_id to patient_id.
        
        Args:
            trial_id: Trial ID (NCT ID) to get evaluated patients for
            
        Returns:
            List of patient IDs from patient_medical_history that were evaluated for this trial
        """
        try:
            with self.get_connection() as connection:
                # Get patient IDs from trial_to_patient table
                # These patient_ids reference patient_medical_history.id
                query = text("""
                    SELECT DISTINCT ttp.patient_id 
                    FROM insightsedge.trial_to_patient ttp
                    WHERE ttp.trial_id = :trial_id
                      AND ttp.patient_id IS NOT NULL
                """)
                result = connection.execute(query, {"trial_id": trial_id})
                
                patient_ids = [int(row[0]) for row in result if row[0] is not None]
                return patient_ids
        except Exception as e:
            print(f"Error getting evaluated patient IDs from patient_medical_history for trial {trial_id}: {e}")
            return []