"""
Trial-to-Patient Hybrid Matcher
Combines MedCPT embeddings with BM25 for optimal patient matching
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from services.shared.database_utils import DatabaseUtils
from services.shared.embedding_utils import EmbeddingUtils
from rank_bm25 import BM25Okapi
import re

class HybridMatcher:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.embedding_utils = EmbeddingUtils()
        
        # Load existing embeddings and indices
        self.trial_index = None
        self.patient_index = None
        self.trial_metadata = {}
        self.patient_metadata = {}
        self.trial_texts = []
        self.patient_texts = []
        self.bm25_trials = None
        self.bm25_patients = None
        
        self.load_existing_embeddings()

    def load_existing_embeddings(self):
        """Load existing embeddings and metadata"""
        try:
            # Load trial FAISS index
            trial_faiss_file = self.embedding_utils.trials_dir / "faiss_index.pkl"
            if trial_faiss_file.exists():
                self.trial_index = self.embedding_utils.load_faiss_index(trial_faiss_file)
                print(f"Loaded trial FAISS index with {self.trial_index.ntotal} vectors")
            
            # Load patient FAISS index
            patient_faiss_file = self.embedding_utils.patients_dir / "faiss_index.pkl"
            if patient_faiss_file.exists():
                self.patient_index = self.embedding_utils.load_faiss_index(patient_faiss_file)
                print(f"Loaded patient FAISS index with {self.patient_index.ntotal} vectors")
            
            # Load trial metadata
            trial_metadata_file = self.embedding_utils.trials_dir / "metadata.json"
            if trial_metadata_file.exists():
                self.trial_metadata = self.embedding_utils.load_metadata(trial_metadata_file)
                print(f"Loaded metadata for {len(self.trial_metadata)} trials")
            
            # Load patient metadata
            patient_metadata_file = self.embedding_utils.patients_dir / "metadata.json"
            if patient_metadata_file.exists():
                self.patient_metadata = self.embedding_utils.load_metadata(patient_metadata_file)
                print(f"Loaded metadata for {len(self.patient_metadata)} patients")
            
            # Load trial texts for BM25
            self.load_trial_texts()
            
            # Load patient texts for BM25
            self.load_patient_texts()
            
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")

    def load_trial_texts(self):
        """Load trial texts for BM25 indexing"""
        try:
            trials = self.db_utils.get_detailed_trial_data()
            
            self.trial_texts = []
            for trial in trials:
                trial_text = f"{trial['title']} {trial['condition']} {trial['phase']} {trial['status']} {trial['investigator']}"
                self.trial_texts.append(self.tokenize_text(trial_text))
            
            # Create BM25 index for trials
            if self.trial_texts:
                self.bm25_trials = BM25Okapi(self.trial_texts)
                print(f"Created BM25 index for {len(self.trial_texts)} trials")
            
        except Exception as e:
            print(f"Error loading trial texts: {e}")

    def load_patient_texts(self):
        """Load patient texts for BM25 indexing"""
        try:
            patients = self.db_utils.get_patient_data_for_keywords(1000)
            
            self.patient_texts = []
            for patient in patients:
                self.patient_texts.append(self.tokenize_text(patient['combined_text']))
            
            # Create BM25 index for patients
            if self.patient_texts:
                self.bm25_patients = BM25Okapi(self.patient_texts)
                print(f"Created BM25 index for {len(self.patient_texts)} patients")
            
        except Exception as e:
            print(f"Error loading patient texts: {e}")

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        if not text:
            return []
        
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using GPT-4o Mini"""
        return self.embedding_utils.generate_embedding(text)

    def get_embedding_similarity(self, query_embedding: np.ndarray, index_type: str = "trial") -> List[Tuple[int, float]]:
        """Get similarity scores using FAISS index"""
        try:
            if index_type == "trial" and self.trial_index:
                scores, indices = self.trial_index.search(query_embedding.reshape(1, -1), k=min(50, self.trial_index.ntotal))
                return list(zip(indices[0], scores[0]))
            elif index_type == "patient" and self.patient_index:
                scores, indices = self.patient_index.search(query_embedding.reshape(1, -1), k=min(50, self.patient_index.ntotal))
                return list(zip(indices[0], scores[0]))
            else:
                return []
        except Exception as e:
            print(f"Error getting embedding similarity: {e}")
            return []

    def get_bm25_similarity(self, query_text: str, index_type: str = "trial") -> List[Tuple[int, float]]:
        """Get similarity scores using BM25"""
        try:
            query_tokens = self.tokenize_text(query_text)
            
            if index_type == "trial" and self.bm25_trials:
                scores = self.bm25_trials.get_scores(query_tokens)
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                return indexed_scores[:50]
            elif index_type == "patient" and self.bm25_patients:
                scores = self.bm25_patients.get_scores(query_tokens)
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                return indexed_scores[:50]
            else:
                return []
        except Exception as e:
            print(f"Error getting BM25 similarity: {e}")
            return []

    def hybrid_search(self, query_text: str, index_type: str = "trial", alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Combine embedding and BM25 scores for hybrid search"""
        try:
            # Get embedding similarity
            query_embedding = self.generate_embedding(query_text)
            embedding_scores = self.get_embedding_similarity(query_embedding, index_type)
            
            # Get BM25 similarity
            bm25_scores = self.get_bm25_similarity(query_text, index_type)
            
            # Normalize scores
            embedding_scores_dict = {idx: score for idx, score in embedding_scores}
            bm25_scores_dict = {idx: score for idx, score in bm25_scores}
            
            # Combine scores
            combined_scores = {}
            all_indices = set(embedding_scores_dict.keys()) | set(bm25_scores_dict.keys())
            
            for idx in all_indices:
                embedding_score = embedding_scores_dict.get(idx, 0.0)
                bm25_score = bm25_scores_dict.get(idx, 0.0)
                
                # Normalize BM25 score (assuming max BM25 score is around 10)
                normalized_bm25 = min(bm25_score / 10.0, 1.0)
                
                # Weighted combination
                combined_score = alpha * embedding_score + (1 - alpha) * normalized_bm25
                combined_scores[idx] = combined_score
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get metadata for top results
            results = []
            metadata = self.trial_metadata if index_type == "trial" else self.patient_metadata
            
            for idx, score in sorted_results[:20]:  # Top 20 results
                # Find patient ID that corresponds to this FAISS index
                patient_id = None
                for pid, patient_data in metadata.items():
                    if patient_data.get('embedding_index') == idx:
                        patient_id = pid
                        break
                
                if patient_id:
                    result = metadata[patient_id].copy()
                    result['patient_id'] = patient_id
                    result['index'] = idx
                    result['hybrid_score'] = score
                    result['embedding_score'] = embedding_scores_dict.get(idx, 0.0)
                    result['bm25_score'] = bm25_scores_dict.get(idx, 0.0)
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def find_matching_patients_for_trial(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None) -> List[Dict[str, Any]]:
        """Find most eligible patients for a specific trial using hybrid matching"""
        try:
            # Get trial information
            trial_info = self.trial_metadata.get(trial_id)
            if not trial_info:
                print(f"Trial {trial_id} not found in metadata")
                return []
            
            # Create query text for the trial
            query_text = f"{trial_info['title']} {trial_info['condition']} {trial_info['phase']}"
            
            print(f"Finding patients for trial: {trial_info['title']}")
            print(f"Condition: {trial_info['condition']}")
            print(f"Phase: {trial_info['phase']}")
            
            # Perform hybrid search
            matching_patients = self.hybrid_search(query_text, index_type="patient")
            
            # Apply age and gender filters
            filtered_patients = []
            for patient in matching_patients:
                # Check age filter
                if age_range:
                    patient_age = patient.get('age')
                    if patient_age and not (age_range[0] <= patient_age <= age_range[1]):
                        continue
                
                # Check gender filter
                if gender:
                    patient_gender = patient.get('gender')
                    if patient_gender and patient_gender.lower() != gender.lower():
                        continue
                
                filtered_patients.append(patient)
            
            print(f"Found {len(matching_patients)} patients before filtering")
            print(f"Found {len(filtered_patients)} patients after age/gender filtering")
            
            return filtered_patients[:20]  # Return top 20
            
        except Exception as e:
            print(f"Error finding matching patients: {e}")
            return []

    def run_trial_to_patient_matching(self, trial_id: str, age_range: Tuple[int, int] = None, gender: str = None) -> Dict[str, Any]:
        """Run complete trial-to-patient matching process"""
        print(f"Starting trial-to-patient matching for trial: {trial_id}")
        
        # Find matching patients
        matching_patients = self.find_matching_patients_for_trial(trial_id, age_range, gender)
        
        # Prepare results
        results = {
            "trial_id": trial_id,
            "trial_info": self.trial_metadata.get(trial_id, {}),
            "matching_patients": matching_patients,
            "total_matches": len(matching_patients),
            "filters_applied": {
                "age_range": age_range,
                "gender": gender
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return results

def main():
    """Main function to test hybrid matching"""
    hybrid_matcher = HybridMatcher()
    
    # Test with a specific trial
    trial_id = "T001"  # Replace with actual trial ID
    results = hybrid_matcher.run_trial_to_patient_matching(
        trial_id=trial_id,
        age_range=(18, 75),
        gender="Male"
    )
    
    print(f"\nHybrid Matching Results:")
    print(f"Trial: {results['trial_info'].get('title', 'Unknown')}")
    print(f"Total matches: {results['total_matches']}")
    
    for i, patient in enumerate(results['matching_patients'][:5], 1):
        print(f"{i}. Patient MRN: {patient.get('mrn', 'Unknown')}")
        print(f"   Age: {patient.get('age', 'Unknown')}, Gender: {patient.get('gender', 'Unknown')}")
        print(f"   Hybrid Score: {patient.get('hybrid_score', 0):.4f}")
        print()

if __name__ == "__main__":
    main()
