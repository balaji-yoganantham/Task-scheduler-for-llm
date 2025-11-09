"""
Patient-to-Trial Embedding Generator
Generates and persists embeddings for patients to enable trial matching
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from services.shared.database_utils import DatabaseUtils, safe_json_dump
from services.shared.embedding_utils import EmbeddingUtils

class PatientEmbeddingGenerator:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.embedding_utils = EmbeddingUtils()
        
        # Patient-specific directories
        self.patients_dir = self.embedding_utils.patients_dir
        
        # Create embeddings results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.embeddings_dir = self.results_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

    def _load_existing_embeddings(self) -> Dict[str, Any]:
        """Load existing embeddings metadata"""
        try:
            metadata_file = self.patients_dir / "metadata.json"
            if not metadata_file.exists():
                return {}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
            return {}

    def _check_patient_embedding_exists(self, patient_id: str) -> bool:
        """Check if patient embedding already exists"""
        try:
            embedding_file = self.patients_dir / f"patient_{patient_id}.npy"
            return embedding_file.exists()
        except Exception as e:
            print(f"Error checking embedding for patient {patient_id}: {e}")
            return False

    def generate_patient_embeddings(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for multiple patients using keywords from database (only new patients)"""
        print(f"SUMMARY Embedding Generation Summary:")
        print(f"   Total patients: {len(patients)}")
        
        # Separate new patients from existing ones
        new_patients = []
        existing_patients = []
        
        for patient in patients:
            patient_id = str(patient['patient_id'])
            if self._check_patient_embedding_exists(patient_id):
                existing_patients.append(patient)
                print(f"OK Patient {patient_id} (MRN: {patient['mrn']}) already has embeddings - skipping")
            else:
                new_patients.append(patient)
        
        print(f"   Existing patients (skipped): {len(existing_patients)}")
        print(f"   New patients (to process): {len(new_patients)}")
        
        if not new_patients:
            print("SUCCESS All patients already have embeddings - no new generation needed!")
            return {
                "embeddings": [],
                "metadata": {},
                "total_patients": len(patients),
                "new_patients": 0,
                "existing_patients": len(existing_patients),
                "status": "all_existing"
            }
        
        print(f"NEW Generating embeddings for {len(new_patients)} NEW patients...")
        
        patient_embeddings = []
        patient_metadata = {}
        
        for i, patient in enumerate(new_patients):
            print(f"Processing patient {i+1}/{len(new_patients)}: MRN {patient['mrn']}")
            
            # Get keywords from database for this patient
            keywords_data = self.db_utils.get_patient_keywords(patient['patient_id'])
            
            if not keywords_data:
                print(f"ERROR No keywords found for patient {patient['patient_id']} - skipping embedding generation")
                continue
            
            # Use keywords_text for embedding generation instead of full medical history
            keywords_text = keywords_data.get('keywords_text', '')
            if not keywords_text:
                # Fallback to keywords array if keywords_text is empty
                keywords_list = keywords_data.get('keywords', [])
                keywords_text = ', '.join(keywords_list) if keywords_list else ''
            
            if not keywords_text:
                print(f"ERROR No keywords text found for patient {patient['patient_id']} - skipping")
                continue
            
            print(f"   Using keywords: {keywords_text[:100]}...")
            
            # Generate embedding for patient keywords
            embedding = self.embedding_utils.generate_embedding(
                keywords_text, 
                task_type="retrieval_document"
            )
            patient_embeddings.append(embedding)
            
            # Store metadata
            patient_metadata[patient['patient_id']] = {
                "mrn": patient.get('mrn', ''),
                "age": patient.get('age', None),
                "gender": patient.get('gender', ''),
                "oncologist": patient.get('oncologist', ''),
                "date_of_visit": patient.get('date_of_visit', ''),
                "created_at": patient.get('created_at', None),  # Optional field
                "keywords_count": len(keywords_data.get('keywords', [])),
                "keywords_text_length": len(keywords_text),
                "embedding_index": i
            }
            
            # Save individual patient embedding
            patient_file = self.patients_dir / f"patient_{patient['patient_id']}.npy"
            np.save(patient_file, embedding)
        
        # Save results summary
        results = {
            "embeddings": patient_embeddings,
            "metadata": patient_metadata,
            "total_patients": len(patients),
            "new_patients": len(new_patients),
            "existing_patients": len(existing_patients),
            "status": "completed"
        }
        
        # Save results to embeddings directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.embeddings_dir / f"patient_embeddings_{timestamp}.json"
        safe_json_dump(results, results_file, indent=2, ensure_ascii=False)
        
        print(f"OK Embedding generation completed:")
        print(f"   New embeddings generated: {len(patient_embeddings)}")
        print(f"   Existing embeddings skipped: {len(existing_patients)}")
        print(f"Results saved to: {results_file}")
        
        return results

    def save_patient_embeddings(self, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save patient embeddings and create FAISS index with ALL patients (existing + new)"""
        try:
            new_embeddings = embedding_data["embeddings"]
            new_metadata = embedding_data["metadata"]
            
            # Load ALL existing embeddings and metadata
            existing_metadata = self._load_existing_embeddings()
            all_embeddings = []
            all_metadata = {}
            
            # First, load all existing embeddings
            if existing_metadata:
                print(f"Loading {len(existing_metadata)} existing patient embeddings...")
                for patient_id_str, existing_meta in existing_metadata.items():
                    embedding_file = self.patients_dir / f"patient_{patient_id_str}.npy"
                    if embedding_file.exists():
                        existing_embedding = np.load(embedding_file)
                        all_embeddings.append(existing_embedding)
                        
                        # Update embedding_index for existing patients
                        existing_meta['embedding_index'] = len(all_embeddings) - 1
                        all_metadata[patient_id_str] = existing_meta
            
            # Then add new embeddings
            if new_embeddings:
                print(f"Adding {len(new_embeddings)} new patient embeddings...")
                start_index = len(all_embeddings)
                for i, new_embedding in enumerate(new_embeddings):
                    all_embeddings.append(new_embedding)
                    
                # Update embedding_index for new patients in metadata
                for patient_id_str, new_meta in new_metadata.items():
                    new_meta['embedding_index'] = start_index + list(new_metadata.keys()).index(patient_id_str)
                    all_metadata[patient_id_str] = new_meta
            
            if not all_embeddings:
                print("WARNING No embeddings to save!")
                return {}
            
            # Save embeddings matrix (all patients)
            matrix_file = self.embedding_utils.save_embeddings_matrix(
                all_embeddings, self.patients_dir, "patient_embeddings_matrix.npy"
            )
            
            # Save individual embeddings and metadata (overwrites with all data)
            saved_files = self.embedding_utils.save_individual_embeddings(
                all_embeddings, all_metadata, self.patients_dir, "patient"
            )
            
            # Create FAISS index with ALL embeddings
            patient_index = self.embedding_utils.create_faiss_index(all_embeddings, "cosine")
            
            # Save FAISS index
            index_file = self.embedding_utils.save_faiss_index(
                patient_index, self.patients_dir, "faiss_index.pkl"
            )
            
            print(f"Patient embeddings saved to: {self.patients_dir}")
            print(f"Total patients in index: {len(all_embeddings)} (existing: {len(existing_metadata)}, new: {len(new_embeddings)})")
            
            return {
                "total_patients": len(all_embeddings),
                "embedding_dimension": self.embedding_utils.dimension,
                "matrix_shape": np.vstack(all_embeddings).shape,
                "metadata_file": saved_files.get('metadata', ''),
                "matrix_file": matrix_file,
                "faiss_index_file": index_file,
                "index_size": patient_index.ntotal if patient_index else 0
            }
            
        except Exception as e:
            print(f"Error saving patient embeddings: {e}")
            return {}

    def run_patient_embedding_generation(self, limit: int = 50) -> Dict[str, Any]:
        """Main method to run patient embedding generation using keywords from database"""
        print("Starting patient embedding generation for trial matching...")
        
        # Get patient data (only unevaluated patients - is_evaluated = 0)
        patients = self.db_utils.get_patient_data_for_keywords(limit, include_evaluated=False)
        if not patients:
            print("No patient data found!")
            return {}
        
        print(f"Found {len(patients)} patients")
        
        # Check which patients already have embeddings and keywords
        existing_metadata = self._load_existing_embeddings()
        new_patients = []
        skipped_patients = []
        
        for patient in patients:
            patient_id = str(patient['patient_id'])
            
            # Check if embedding already exists
            if patient_id in existing_metadata or self._check_patient_embedding_exists(patient_id):
                skipped_patients.append(patient)
                print(f"OK Patient {patient_id} (MRN: {patient['mrn']}) already has embeddings - skipping")
                continue
            
            # Check if keywords exist in database
            keywords_data = self.db_utils.get_patient_keywords(patient['patient_id'])
            if not keywords_data:
                skipped_patients.append(patient)
                print(f"WARNING Patient {patient_id} (MRN: {patient['mrn']}) has no keywords - skipping embedding generation")
                continue
            
            new_patients.append(patient)
        
        if not new_patients:
            print("SUCCESS All patients already have embeddings or no keywords - no new generation needed!")
            return {
                "total_patients": len(patients),
                "new_patients": 0,
                "existing_patients": len(skipped_patients),
                "embedding_dimension": self.embedding_utils.dimension,
                "note": "All patients already had embeddings or no keywords"
            }
        
        print(f"NEW Found {len(new_patients)} NEW patients needing embedding generation")
        
        # Generate embeddings only for new patients with keywords
        print("Generating embeddings for NEW patients with keywords...")
        embedding_data = self.generate_patient_embeddings(new_patients)
        
        # Update embedding data with combined info
        embedding_data["total_patients"] = len(patients)
        embedding_data["new_patients"] = len(new_patients)
        embedding_data["existing_patients"] = len(skipped_patients)
        embedding_data["note"] = f"Generated embeddings for {len(new_patients)} new patients using keywords"
        
        # Save embeddings
        save_info = self.save_patient_embeddings(embedding_data)
        
        if save_info:
            print(f"\nPatient Embedding Generation Summary:")
            print(f"Total patients: {len(patients)}")
            print(f"New patients processed: {len(new_patients)}")
            print(f"Existing/skipped patients: {len(skipped_patients)}")
            print(f"Embedding dimension: {save_info['embedding_dimension']}")
            print(f"Matrix shape: {save_info['matrix_shape']}")
            print(f"FAISS index size: {save_info['index_size']}")
            print(f"Files saved to: {self.patients_dir}")
        
        return save_info

def main():
    """Main function to run patient embedding generation"""
    generator = PatientEmbeddingGenerator()
    results = generator.run_patient_embedding_generation(limit=50)
    
    if results:
        print("\nPatient embedding generation completed successfully!")
    else:
        print("\nPatient embedding generation failed!")

if __name__ == "__main__":
    main()
