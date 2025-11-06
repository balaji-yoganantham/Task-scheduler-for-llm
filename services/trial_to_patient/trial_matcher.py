"""
Trial-to-Patient Matcher
Finds suitable patients for a specific clinical trial using hybrid matching
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

class TrialMatcher:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.embedding_utils = EmbeddingUtils()
        
        # Load existing embeddings and indices
        self.patient_index = None
        self.patient_metadata = {}
        self.patient_texts = []
        self.bm25_patients = None
        
        self.load_patient_data()

    def load_patient_data(self):
        """Load patient embeddings and metadata, including is_evaluated status"""
        try:
            # Load patient FAISS index
            patient_faiss_file = self.embedding_utils.patients_dir / "faiss_index.pkl"
            if patient_faiss_file.exists():
                self.patient_index = self.embedding_utils.load_faiss_index(patient_faiss_file)
                print(f"Loaded patient FAISS index with {self.patient_index.ntotal} vectors")
            
            # Load patient metadata
            patient_metadata_file = self.embedding_utils.patients_dir / "metadata.json"
            if patient_metadata_file.exists():
                self.patient_metadata = self.embedding_utils.load_metadata(patient_metadata_file)
                print(f"Loaded metadata for {len(self.patient_metadata)} patients")
                
                # Fetch is_evaluated status from database and add to metadata
                patient_ids = [int(pid) for pid in self.patient_metadata.keys() if pid.isdigit()]
                if patient_ids:
                    evaluated_status = self.db_utils.get_patients_evaluated_status(patient_ids)
                    for patient_id_str, metadata in self.patient_metadata.items():
                        patient_id = int(patient_id_str) if patient_id_str.isdigit() else None
                        if patient_id and patient_id in evaluated_status:
                            metadata['is_evaluated'] = evaluated_status[patient_id]
                        else:
                            # Default to 0 if not found (shouldn't happen, but safe fallback)
                            metadata['is_evaluated'] = 0
                    print(f"Added is_evaluated status to metadata for {len(evaluated_status)} patients")
            
            # Load patient texts for BM25
            self.load_patient_texts()
            
        except Exception as e:
            print(f"Error loading patient data: {e}")

    def load_patient_texts(self):
        """Load patient texts for BM25 indexing"""
        try:
            # Only get unevaluated patients (is_evaluated = 0) for BM25
            patients = self.db_utils.get_patient_data_for_keywords(1000, include_evaluated=False)
            
            self.patient_texts = []
            for patient in patients:
                # Use combined text for BM25
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
        
        # Simple tokenization - can be improved with medical tokenizer
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens

    def get_embedding_similarity(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """Get similarity scores using FAISS index"""
        try:
            if self.patient_index:
                scores, indices = self.patient_index.search(query_embedding.reshape(1, -1), k=min(50, self.patient_index.ntotal))
                return list(zip(indices[0], scores[0]))
            else:
                return []
        except Exception as e:
            print(f"Error getting embedding similarity: {e}")
            return []

    def get_bm25_similarity(self, query_text: str) -> List[Tuple[int, float]]:
        """Get similarity scores using BM25"""
        try:
            query_tokens = self.tokenize_text(query_text)
            
            if self.bm25_patients:
                scores = self.bm25_patients.get_scores(query_tokens)
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                return indexed_scores[:50]
            else:
                return []
        except Exception as e:
            print(f"Error getting BM25 similarity: {e}")
            return []

    def hybrid_search_patients_for_trial(self, trial_data: Dict[str, Any], alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Find suitable patients for a trial using hybrid matching (only patients with is_evaluated = 0)"""
        try:
            print(f"Finding patients for trial: {trial_data['title']}")
            
            # Generate embedding for trial
            trial_embedding = self.embedding_utils.generate_embedding(
                trial_data['combined_trial_text'], 
                task_type="retrieval_query"
            )
            
            # Get embedding similarity
            embedding_scores = self.get_embedding_similarity(trial_embedding)
            
            # Get BM25 similarity
            bm25_scores = self.get_bm25_similarity(trial_data['combined_trial_text'])
            
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
            
            # Get metadata for top results, filtering to only include patients with is_evaluated = 0
            # FAISS returns index positions, need to find corresponding patient_id by matching embedding_index
            results = []
            for idx, score in sorted_results[:50]:  # Check top 50, then filter
                # Find patient with matching embedding_index
                patient_id = None
                patient_meta = None
                for pid, metadata in self.patient_metadata.items():
                    if metadata.get('embedding_index') == idx:
                        patient_id = pid
                        patient_meta = metadata
                        break
                
                if not patient_meta:
                    continue  # Skip if no matching patient found
                
                # Filter: Only include patients with is_evaluated = 0
                is_evaluated = patient_meta.get('is_evaluated', 0)
                if is_evaluated != 0:
                    continue  # Skip evaluated patients
                
                result = patient_meta.copy()
                result['patient_id'] = patient_id
                result['index'] = idx
                result['hybrid_score'] = score
                result['embedding_score'] = embedding_scores_dict.get(idx, 0.0)
                result['bm25_score'] = bm25_scores_dict.get(idx, 0.0)
                results.append(result)
                
                # Stop after getting top 20 unevaluated patients
                if len(results) >= 20:
                    break
            
            print(f"Filtered to {len(results)} patients with is_evaluated = 0")
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def find_patients_for_trial(self, trial_id: str, age_range: Tuple[int, int] = None, 
                               gender: str = None) -> Dict[str, Any]:
        """Find suitable patients for a specific trial"""
        try:
            # Get trial information
            trial_data = self.db_utils.get_trial_by_id(trial_id)
            if not trial_data:
                print(f"Trial {trial_id} not found")
                return {}
            
            print(f"Finding patients for trial: {trial_data['title']}")
            print(f"Condition: {trial_data['condition']}")
            print(f"Phase: {trial_data['phase']}")
            
            # Perform hybrid search
            matching_patients = self.hybrid_search_patients_for_trial(trial_data)
            
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
            
            return {
                "trial_id": trial_id,
                "trial_info": trial_data,
                "matching_patients": filtered_patients[:20],  # Top 20 patients
                "total_matches": len(filtered_patients),
                "filters_applied": {
                    "age_range": age_range,
                    "gender": gender
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error finding patients for trial: {e}")
            return {}

def main():
    """Main function to test trial matching"""
    matcher = TrialMatcher()
    
    # Test with a specific trial
    trial_id = "T001"  # Replace with actual trial ID
    results = matcher.find_patients_for_trial(
        trial_id=trial_id,
        age_range=(18, 75),
        gender="Male"
    )
    
    print(f"\nTrial Patient Matching Results:")
    print(f"Trial: {results.get('trial_info', {}).get('title', 'Unknown')}")
    print(f"Total matches: {results.get('total_matches', 0)}")
    
    for i, patient in enumerate(results.get('matching_patients', [])[:5], 1):
        print(f"{i}. Patient MRN: {patient.get('mrn', 'Unknown')}")
        print(f"   Age: {patient.get('age', 'Unknown')}, Gender: {patient.get('gender', 'Unknown')}")
        print(f"   Hybrid Score: {patient.get('hybrid_score', 0):.4f}")
        print()

if __name__ == "__main__":
    main()
