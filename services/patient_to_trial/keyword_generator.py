"""
Patient-to-Trial Keyword Generator
Generates keywords from patient medical records for finding suitable clinical trials
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from services.shared.database_utils import DatabaseUtils
from services.shared.llm_utils import LLMUtils

class PatientKeywordGenerator:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.llm_utils = LLMUtils()

    def check_existing_keywords(self, patient_id: int) -> bool:
        """Check if keywords already exist for a patient in database only"""
        try:
            # Check database
            db_keywords = self.db_utils.get_patient_keywords(patient_id)
            return db_keywords is not None
            
        except Exception as e:
            print(f"Error checking existing keywords for patient {patient_id}: {e}")
            return False

    def load_existing_keywords(self, patient_id: int) -> Dict[str, Any]:
        """Load existing keywords for a patient from database only"""
        try:
            # Load from database
            db_keywords = self.db_utils.get_patient_keywords(patient_id)
            if db_keywords:
                print(f"OK Loaded keywords from database for patient {patient_id}")
                return db_keywords
            
            print(f"INFO No existing keywords found for patient {patient_id}")
            return {}
            
        except Exception as e:
            print(f"Error loading existing keywords for patient {patient_id}: {e}")
            return {}

    def save_keywords(self, keywords_data: Dict[str, Any]) -> str:
        """Save keywords to database only"""
        try:
            # Save to database
            success = self.db_utils.save_patient_keywords(keywords_data)
            
            if success:
                print(f"OK Keywords saved to database for patient {keywords_data.get('patient_id', 'unknown')}")
                return f"Database saved for patient {keywords_data.get('patient_id', 'unknown')}"
            else:
                print(f"ERROR Failed to save keywords to database for patient {keywords_data.get('patient_id', 'unknown')}")
                return "Database save failed"
                
        except Exception as e:
            print(f"Error saving keywords: {e}")
            return f"Error: {str(e)}"

    def generate_keywords_for_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate keywords for a single patient using LLM (only if not already exists)"""
        patient_id = patient_data['patient_id']
        
        # Check if keywords already exist
        if self.check_existing_keywords(patient_id):
            print(f"OK Patient {patient_id} (MRN: {patient_data['mrn']}) already has keywords - loading existing")
            return self.load_existing_keywords(patient_id)
        
        try:
            print(f"NEW Generating NEW keywords for patient MRN: {patient_data['mrn']}")
            
            keywords_data = self.llm_utils.generate_keywords_for_patient(patient_data)
            
            if "error" in keywords_data:
                print(f"Failed to generate keywords for patient {patient_id}")
                return {
                    "patient_id": patient_id,
                    "error": keywords_data["error"],
                    "raw_response": keywords_data.get("raw_response", "")
                }
            
            # Save the generated keywords
            self.save_keywords(keywords_data)
            print(f"SAVED Keywords saved for patient {patient_id}")
            
            return keywords_data
                
        except Exception as e:
            print(f"Error generating keywords for patient {patient_id}: {e}")
            return {
                "patient_id": patient_id,
                "error": str(e)
            }

    def generate_keywords_batch(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate keywords for multiple patients in one batch (only new patients)"""
        results = {
            "successful": {},
            "failed": {},
            "skipped": {},
            "metadata": {
                "total_patients": len(patients),
                "generated_at": datetime.now().isoformat(),
                "model": "gemini-2.0-flash-exp",
                "matching_direction": "patient_to_trial",
                "batch_processing": "optimized_new_patients_only"
            }
        }
        
        # Separate new patients from existing ones
        new_patients = []
        existing_patients = []
        
        for patient in patients:
            patient_id = patient['patient_id']
            if self.check_existing_keywords(patient_id):
                existing_patients.append(patient)
                # Load existing keywords
                existing_keywords = self.load_existing_keywords(patient_id)
                results["skipped"][patient_id] = existing_keywords
            else:
                new_patients.append(patient)
        
        print(f"SUMMARY Keyword Generation Summary:")
        print(f"   Total patients: {len(patients)}")
        print(f"   Existing patients (skipped): {len(existing_patients)}")
        print(f"   New patients (to process): {len(new_patients)}")
        
        if not new_patients:
            print("SUCCESS All patients already have keywords - no new generation needed!")
            results["metadata"]["batch_status"] = "all_existing"
            return results
        
        print(f"NEW Generating keywords for {len(new_patients)} NEW patients...")
        
        # Process only new patients in batches
        try:
            from config import MAX_KEYWORD_BATCH_SIZE
            
            # Process patients in batches
            total_processed = 0
            total_successful = 0
            total_failed = 0
            
            for batch_start in range(0, len(new_patients), MAX_KEYWORD_BATCH_SIZE):
                batch_end = min(batch_start + MAX_KEYWORD_BATCH_SIZE, len(new_patients))
                batch_patients = new_patients[batch_start:batch_end]
                batch_num = (batch_start // MAX_KEYWORD_BATCH_SIZE) + 1
                total_batches = (len(new_patients) + MAX_KEYWORD_BATCH_SIZE - 1) // MAX_KEYWORD_BATCH_SIZE
                
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch_patients)} patients)...")
                
                batch_keywords_data = self.llm_utils.generate_keywords_for_patient_batch(batch_patients)
                
                if "error" in batch_keywords_data:
                    print(f"ERROR Batch {batch_num} processing failed: {batch_keywords_data['error']}")
                    print(f"FALLBACK Falling back to individual patient processing for batch {batch_num}...")
                    
                    # Fallback to individual processing for this batch
                    individual_results = self._process_patients_individually(batch_patients)
                    for patient_id, keywords_data in individual_results.get("successful", {}).items():
                        results["successful"][patient_id] = keywords_data
                        self.save_keywords(keywords_data)
                        total_successful += 1
                    
                    for patient_id, error_data in individual_results.get("failed", {}).items():
                        results["failed"][patient_id] = error_data
                        total_failed += 1
                else:
                    # Process successful results from batch
                    for patient_id, keywords_data in batch_keywords_data.get("successful", {}).items():
                        results["successful"][patient_id] = keywords_data
                        self.save_keywords(keywords_data)
                        total_successful += 1
                    
                    # Process failed results from batch
                    for patient_id, error_data in batch_keywords_data.get("failed", {}).items():
                        results["failed"][patient_id] = error_data
                        total_failed += 1
                
                total_processed += len(batch_patients)
            
            results["metadata"]["batch_status"] = "success"
            results["metadata"]["batches_processed"] = total_batches
            results["metadata"]["total_processed"] = total_processed
            
            print(f"OK All batches completed:")
            print(f"   Total batches: {total_batches}")
            print(f"   Total processed: {total_processed}")
            print(f"   Successful: {total_successful}")
            print(f"   Failed: {total_failed}")
            print(f"   Skipped: {len(results['skipped'])}")
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            results["metadata"]["batch_status"] = "error"
            results["metadata"]["error"] = str(e)
            for patient in new_patients:
                results["failed"][patient['patient_id']] = {
                    "error": str(e)
                }
        
        return results

    def _process_patients_individually(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process patients individually as fallback when batch processing fails"""
        print(f"PROCESSING Processing {len(patients)} patients individually...")
        
        results = {
            "successful": {},
            "failed": {}
        }
        
        for i, patient in enumerate(patients):
            patient_id = patient['patient_id']
            print(f"Processing patient {i+1}/{len(patients)}: MRN {patient['mrn']}")
            
            try:
                keywords_data = self.generate_keywords_for_patient(patient)
                
                if "error" in keywords_data:
                    results["failed"][patient_id] = keywords_data
                    print(f"ERROR Failed to generate keywords for patient {patient_id}")
                else:
                    results["successful"][patient_id] = keywords_data
                    print(f"OK Generated keywords for patient {patient_id}")
                    
            except Exception as e:
                results["failed"][patient_id] = {
                    "patient_id": patient_id,
                    "error": str(e)
                }
                print(f"ERROR Error processing patient {patient_id}: {e}")
        
        print(f"Individual processing completed:")
        print(f"   Successful: {len(results['successful'])}")
        print(f"   Failed: {len(results['failed'])}")
        
        return results

    def run_keyword_generation(self, limit: int = 50) -> Dict[str, Any]:
        """Main method to run keyword generation process - OPTIMIZED for existing patients"""
        print("Starting patient keyword generation for trial matching...")
        
        # Get patient data (only unevaluated patients - is_evaluated = 0)
        patients = self.db_utils.get_patient_data_for_keywords(limit, include_evaluated=False)
        if not patients:
            print("No patient data found!")
            return {}
        
        print(f"Found {len(patients)} patients")
        
        # Check which patients already have keywords in database
        new_patients = []
        existing_count = 0
        
        for patient in patients:
            patient_id = patient['patient_id']
            if not self.check_existing_keywords(patient_id):
                new_patients.append(patient)
            else:
                existing_count += 1
                print(f"OK Patient {patient_id} (MRN: {patient['mrn']}) already has keywords - skipping")
        
        if not new_patients:
            print("SUCCESS All patients already have keywords - no new generation needed!")
            return {
                "successful": {},
                "failed": {},
                "metadata": {
                    "total_patients": len(patients),
                    "new_patients": 0,
                    "existing_patients": len(patients),
                    "generated_at": datetime.now().isoformat(),
                    "note": "All patients already had keywords in database"
                }
            }
        
        print(f"NEW Found {len(new_patients)} NEW patients needing keyword generation")
        
        # Generate keywords only for new patients
        print("Generating keywords for NEW patients only...")
        results = self.generate_keywords_batch(new_patients)
        
        # Update results with metadata
        results["metadata"]["total_patients"] = len(patients)
        results["metadata"]["new_patients"] = len(new_patients)
        results["metadata"]["existing_patients"] = existing_count
        results["metadata"]["note"] = f"Generated keywords for {len(new_patients)} new patients"
        
        # Print summary
        successful_count = len(results["successful"])
        failed_count = len(results["failed"])
        
        print(f"\nPatient Keyword Generation Summary:")
        print(f"Total patients: {len(patients)}")
        print(f"New patients processed: {len(new_patients)}")
        print(f"Existing patients skipped: {existing_count}")
        print(f"Total successful: {successful_count}")
        print(f"Failed: {failed_count}")
        
        return results

def main():
    """Main function to run patient keyword generation"""
    generator = PatientKeywordGenerator()
    results = generator.run_keyword_generation(limit=50)
    
    if results:
        print("\nPatient keyword generation completed successfully!")
    else:
        print("\nPatient keyword generation failed!")

if __name__ == "__main__":
    main()
