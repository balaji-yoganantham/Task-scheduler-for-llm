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
    
    def get_patient_data_for_keywords(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get patient data from database using stored procedure"""
        try:
            with self.get_connection() as connection:
                query = text("SELECT * FROM insightsedge.get_patient_data_for_keywords(:limit)")
                result = connection.execute(query, {"limit": limit})
                
                patients = []
                for row in result:
                    patients.append({
                        "patient_id": row[0],
                        "mrn": row[1],
                        "age": row[2],
                        "gender": row[3],
                        "combined_text": row[4],
                        "oncologist": row[5],
                        "date_of_visit": row[6].isoformat() if row[6] else None,
                        "created_at": row[7].isoformat() if row[7] else None
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
                        "matching_status": row[8]
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
                        "combined_trial_text": str(row[9])
                    })
                
                print(f"Successfully retrieved {len(trials)} trials from database")
                return trials
        except Exception as e:
            print(f"Error getting detailed trial data: {e}")
            print("Returning empty list to continue processing")
            return []
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get specific patient by ID from patient_medical_history_temp table"""
        try:
            with self.get_connection() as connection:
                query = text("""
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
                    WHERE pmh.id = :patient_id
                """)
                result = connection.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "patient_id": row[0],
                        "mrn": row[1],
                        "age": row[2],
                        "gender": row[3],
                        "combined_text": row[4],
                        "oncologist": row[5],
                        "date_of_visit": row[6].isoformat() if row[6] else None,
                        "created_at": row[7].isoformat() if row[7] else None
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
                    SELECT * FROM insightsedge.get_patient_data_for_keywords(1000)
                    WHERE mrn = :mrn
                """)
                result = connection.execute(query, {"mrn": mrn})
                row = result.fetchone()
                
                if row:
                    return {
                        "patient_id": row[0],
                        "mrn": row[1],
                        "age": row[2],
                        "gender": row[3],
                        "combined_text": row[4],
                        "oncologist": row[5],
                        "date_of_visit": row[6].isoformat() if row[6] else None,
                        "created_at": row[7].isoformat() if row[7] else None
                    }
                return None
        except Exception as e:
            print(f"Error getting patient by MRN: {e}")
            return None
    
    def get_trial_by_id(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trial by ID"""
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
                        "combined_trial_text": str(row[9])
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
                # Try to get location from patient_medical_history_temp table
                # Only location column exists (latitude, longitude, address, and city columns don't exist)
                query = text("""
                    SELECT 
                        COALESCE(pmh.location, '') as location
                    FROM insightsedge.patient_medical_history_temp pmh
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