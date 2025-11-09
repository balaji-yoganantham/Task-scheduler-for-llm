"""
Patient-to-Trial Matcher
Finds suitable clinical trials for a specific patient using hybrid matching
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from services.shared.database_utils import DatabaseUtils
from services.shared.embedding_utils import EmbeddingUtils
from services.shared.location_utils import LocationUtils
from services.trial_to_patient.trial_embedding import TrialEmbeddingGenerator
from rank_bm25 import BM25Okapi
from config import LOCATION_ENABLED, MAX_DEFAULT_DISTANCE_KM, LOCATION_WEIGHT, TOP_K_TRIALS
import re

class PatientMatcher:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.embedding_utils = EmbeddingUtils()
        self.location_utils = LocationUtils() if LOCATION_ENABLED else None
        self.trial_embedding_generator = TrialEmbeddingGenerator()
        
        # Load existing embeddings and indices
        self.trial_index = None
        self.trial_metadata = {}
        self.trial_texts = []
        self.bm25_trials = None
        
        self.load_trial_data()

    def _check_trial_embeddings_exist(self) -> bool:
        """Check if trial embeddings and FAISS index exist"""
        trial_faiss_file = self.embedding_utils.trials_dir / "faiss_index.pkl"
        trial_metadata_file = self.embedding_utils.trials_dir / "metadata.json"
        return trial_faiss_file.exists() and trial_metadata_file.exists()

    def _ensure_trial_embeddings(self):
        """Ensure trial embeddings exist, create them if missing"""
        if not self._check_trial_embeddings_exist():
            print("Trial embeddings not found. Generating trial embeddings...")
            try:
                # Generate trial embeddings using TrialEmbeddingGenerator
                result = self.trial_embedding_generator.run_trial_embedding_generation()
                if result and result.get('status') != 'all_existing':
                    print("Trial embeddings generated successfully")
                else:
                    print("Trial embeddings already exist")
            except Exception as e:
                print(f"Error generating trial embeddings: {e}")
                raise

    def load_trial_data(self):
        """Load trial embeddings and metadata, create if missing"""
        try:
            # Ensure embeddings exist before loading
            self._ensure_trial_embeddings()
            
            # Load trial FAISS index
            trial_faiss_file = self.embedding_utils.trials_dir / "faiss_index.pkl"
            if trial_faiss_file.exists():
                self.trial_index = self.embedding_utils.load_faiss_index(trial_faiss_file)
                print(f"Loaded trial FAISS index with {self.trial_index.ntotal} vectors")
            else:
                print("Warning: FAISS index file not found after generation attempt")
            
            # Load trial metadata
            trial_metadata_file = self.embedding_utils.trials_dir / "metadata.json"
            if trial_metadata_file.exists():
                self.trial_metadata = self.embedding_utils.load_metadata(trial_metadata_file)
                print(f"Loaded metadata for {len(self.trial_metadata)} trials")
                
                # Fetch is_evaluated status from database and add to metadata
                trial_ids = list(self.trial_metadata.keys())
                if trial_ids:
                    # Get trial is_evaluated status from database
                    trials_data = self.db_utils.get_detailed_trial_data()
                    trial_status_dict = {str(trial['trial_id']): trial.get('is_evaluated', 0) for trial in trials_data}
                    
                    for trial_id_str, metadata in self.trial_metadata.items():
                        if trial_id_str in trial_status_dict:
                            metadata['is_evaluated'] = trial_status_dict[trial_id_str]
                        else:
                            # Default to 0 if not found (shouldn't happen, but safe fallback)
                            metadata['is_evaluated'] = metadata.get('is_evaluated', 0)
                    print(f"Added is_evaluated status to metadata for {len(trial_status_dict)} trials")
            else:
                print("Warning: Metadata file not found after generation attempt")
            
            # Load trial texts for BM25
            self.load_trial_texts()
            
        except Exception as e:
            print(f"Error loading trial data: {e}")
            import traceback
            traceback.print_exc()

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
            if self.trial_index:
                scores, indices = self.trial_index.search(query_embedding.reshape(1, -1), k=min(50, self.trial_index.ntotal))
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
            
            if self.bm25_trials:
                scores = self.bm25_trials.get_scores(query_tokens)
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                return indexed_scores[:50]
            else:
                return []
        except Exception as e:
            print(f"Error getting BM25 similarity: {e}")
            return []

    def hybrid_search_trials_for_patient(self, patient_data: Dict[str, Any], alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Find suitable trials for a patient using hybrid matching"""
        try:
            print(f"Finding trials for patient MRN: {patient_data['mrn']}")
            
            # Generate embedding for patient
            patient_embedding = self.embedding_utils.generate_embedding(
                patient_data['combined_text'], 
                task_type="retrieval_query"
            )
            
            # Get embedding similarity
            embedding_scores = self.get_embedding_similarity(patient_embedding)
            
            # Get BM25 similarity
            bm25_scores = self.get_bm25_similarity(patient_data['combined_text'])
            
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
            
            # Get metadata for top results (no filtering)
            results = []
            for idx, score in sorted_results:
                # Stop after getting top K matching trials (configurable)
                if len(results) >= TOP_K_TRIALS:
                    break
                
                # Find trial with matching embedding_index
                trial_id = None
                trial_meta = None
                for tid, metadata in self.trial_metadata.items():
                    if metadata.get('embedding_index') == idx:
                        trial_id = tid
                        trial_meta = metadata
                        break
                
                if not trial_meta:
                    continue  # Skip if no matching trial found
                
                # No filtering - return all matching trials
                
                result = trial_meta.copy()
                result['trial_id'] = trial_id
                result['index'] = idx
                result['hybrid_score'] = score
                result['embedding_score'] = embedding_scores_dict.get(idx, 0.0)
                result['bm25_score'] = bm25_scores_dict.get(idx, 0.0)
                results.append(result)
            
            print(f"Found {len(results)} trials from hybrid search")
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def find_trials_for_patient(self, patient_id: int, age_range: Tuple[int, int] = None, 
                              gender: str = None, phase_filter: List[str] = None,
                              max_distance_km: Optional[float] = None,
                              location_weight: Optional[float] = None) -> Dict[str, Any]:
        """Find suitable trials for a specific patient"""
        try:
            # Get patient information
            patient_data = self.db_utils.get_patient_by_id(patient_id)
            if not patient_data:
                print(f"Patient {patient_id} not found")
                # Return structure with patient_id even if patient not found
                return {
                    "patient_id": patient_id,
                    "patient_info": {
                        "patient_id": patient_id,
                        "mrn": None,
                        "age": None,
                        "gender": None,
                        "is_evaluated": 0
                    },
                    "matching_trials": [],
                    "total_matches": 0,
                    "filters_applied": {
                        "age_range": age_range,
                        "gender": gender,
                        "phase_filter": phase_filter,
                        "max_distance_km": max_distance_km if LOCATION_ENABLED else None,
                        "location_weight": location_weight if LOCATION_ENABLED else None
                    },
                    "generated_at": datetime.now().isoformat(),
                    "error": f"Patient {patient_id} not found in database"
                }
            
            print(f"Finding trials for patient: MRN {patient_data['mrn']}")
            print(f"Age: {patient_data['age']}, Gender: {patient_data['gender']}")
            
            # Get patient location if location filtering is enabled
            patient_lat = None
            patient_lon = None
            identified_location = False
            
            if LOCATION_ENABLED and self.location_utils:
                patient_location_data = self.db_utils.get_patient_location(patient_id)
                if patient_location_data:
                    # Try to get coordinates directly
                    if patient_location_data.get('latitude') and patient_location_data.get('longitude'):
                        patient_lat = float(patient_location_data['latitude'])
                        patient_lon = float(patient_location_data['longitude'])
                        identified_location = True
                    # Otherwise try to geocode the location string
                    elif patient_location_data.get('location'):
                        coords = self.location_utils.geocode_location(patient_location_data['location'])
                        if coords:
                            patient_lat, patient_lon = coords
                            identified_location = True
                
                if identified_location:
                    print(f"Patient location identified: ({patient_lat}, {patient_lon})")
                else:
                    print("Patient location not available - location filtering will be skipped")
            
            # Perform hybrid search
            matching_trials = self.hybrid_search_trials_for_patient(patient_data)
            
            # Apply filters
            filtered_trials = []
            for trial in matching_trials:
                # Check phase filter
                if phase_filter:
                    trial_phase = trial.get('phase', '')
                    if trial_phase and trial_phase not in phase_filter:
                        continue
                
                filtered_trials.append(trial)
            
            print(f"Found {len(matching_trials)} trials before filtering")
            print(f"Found {len(filtered_trials)} trials after phase filtering")
            
            # Apply location filtering if enabled and patient location is available
            if LOCATION_ENABLED and self.location_utils and identified_location and patient_lat and patient_lon:
                # Use default max distance if not specified
                max_dist = max_distance_km if max_distance_km is not None else MAX_DEFAULT_DISTANCE_KM
                location_wt = location_weight if location_weight is not None else LOCATION_WEIGHT
                
                print(f"\nApplying location filtering with max distance: {max_dist} km")
                print(f"   Patient location: ({patient_lat}, {patient_lon})")
                print(f"   Processing {len(filtered_trials)} trials for location filtering...")
                
                # Get trial locations and calculate distances
                location_filtered_trials = []
                trials_with_location = 0
                trials_without_location = 0
                trials_within_distance = 0
                trials_outside_distance = 0
                trials_before_location_filter = len(filtered_trials)
                
                for trial in filtered_trials:
                    trial_id = trial.get('trial_id')
                    if not trial_id:
                        continue
                    
                    # Get trial location(s) - now returns array of location objects
                    trial_location_data = self.db_utils.get_trial_location(trial_id)
                    trial_lat = None
                    trial_lon = None
                    closest_distance = None
                    closest_location = None
                    all_locations_with_coords = []
                    
                    if trial_location_data:
                        # Check if we have multiple locations (new format)
                        if trial_location_data.get('locations') and isinstance(trial_location_data['locations'], list) and len(trial_location_data['locations']) > 0:
                            locations_list = trial_location_data['locations']
                            trials_with_location += 1
                            
                            # Process each location: geocode and calculate distance
                            for loc_obj in locations_list:
                                # Build address string from location object
                                address_str = self.location_utils.build_address_string(loc_obj)
                                if not address_str:
                                    continue
                                
                                # Geocode this location
                                coords = self.location_utils.geocode_location(address_str)
                                if not coords:
                                    continue
                                
                                loc_lat, loc_lon = coords
                                
                                # Calculate distance from patient to this location
                                distance = self.location_utils.calculate_distance(
                                    patient_lat, patient_lon, loc_lat, loc_lon
                                )
                                
                                if distance is not None:
                                    # Store location with coordinates and distance
                                    loc_with_data = {
                                        **loc_obj,  # Include original fields (city, state, country, name)
                                        'address_string': address_str,
                                        'latitude': loc_lat,
                                        'longitude': loc_lon,
                                        'distance_km': round(distance, 2)
                                    }
                                    all_locations_with_coords.append(loc_with_data)
                                    
                                    # Track closest location
                                    if closest_distance is None or distance < closest_distance:
                                        closest_distance = distance
                                        closest_location = loc_with_data
                                        trial_lat = loc_lat
                                        trial_lon = loc_lon
                            
                            # If we found at least one location with coordinates
                            if closest_distance is not None:
                                # Apply distance filter using closest location
                                if max_dist is not None and closest_distance > max_dist:
                                    trials_outside_distance += 1
                                    continue
                                
                                trials_within_distance += 1
                                # Calculate location score based on closest distance
                                location_score = self.location_utils.calculate_location_score(closest_distance, max_dist)
                                
                                # Combine hybrid score with location score
                                hybrid_score = trial.get('hybrid_score', 0.0)
                                final_score = (1.0 - location_wt) * hybrid_score + location_wt * location_score
                                
                                trial['distance_km'] = round(closest_distance, 2)  # Closest location distance
                                trial['location_score'] = round(location_score, 4)
                                trial['final_score'] = round(final_score, 4)
                                trial['latitude'] = trial_lat  # Closest location coordinates
                                trial['longitude'] = trial_lon
                                trial['all_locations'] = all_locations_with_coords  # All locations with distances
                                trial['closest_location'] = closest_location  # Reference to closest location
                            else:
                                # Locations found but geocoding failed for all
                                trial['distance_km'] = None
                                trial['location_score'] = 0.0
                                trial['final_score'] = trial.get('hybrid_score', 0.0)
                                trial['all_locations'] = []
                        
                        else:
                            # No valid trial location data
                            trials_without_location += 1
                            trial['distance_km'] = None
                            trial['location_score'] = 0.0
                            trial['final_score'] = trial.get('hybrid_score', 0.0)
                    else:
                        # No trial location data at all
                        trials_without_location += 1
                        trial['distance_km'] = None
                        trial['location_score'] = 0.0
                        trial['final_score'] = trial.get('hybrid_score', 0.0)
                    
                    location_filtered_trials.append(trial)
                
                filtered_trials = location_filtered_trials
                
                print(f"\nLocation Filtering Results:")
                print(f"   Total trials from hybrid search: {len(matching_trials)}")
                print(f"   Trials after phase filtering: {trials_before_location_filter}")
                print(f"   ────────────────────────────────────────────")
                print(f"   Trials with location data: {trials_with_location}")
                print(f"   Trials without location data: {trials_without_location}")
                print(f"   ────────────────────────────────────────────")
                print(f"   Trials within {max_dist} km: {trials_within_distance}")
                print(f"   Trials outside {max_dist} km (filtered out): {trials_outside_distance}")
                print(f"   ────────────────────────────────────────────")
                print(f"   Final trials after location filter: {len(filtered_trials)}")
                print(f"   Distance threshold: {max_dist} km")
                
                # Sort by final score (or hybrid_score if location not available)
                filtered_trials.sort(key=lambda x: x.get('final_score', x.get('hybrid_score', 0.0)), reverse=True)
            
            return {
                "patient_id": patient_id,
                "patient_info": patient_data,
                "matching_trials": filtered_trials[:TOP_K_TRIALS],  # Top K trials (configurable)
                "total_matches": len(filtered_trials),
                "filters_applied": {
                    "age_range": age_range,
                    "gender": gender,
                    "phase_filter": phase_filter,
                    "max_distance_km": max_distance_km if LOCATION_ENABLED else None,
                    "location_weight": location_weight if LOCATION_ENABLED else None
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error finding trials for patient: {e}")
            return {}

def main():
    """Main function to test patient matching"""
    matcher = PatientMatcher()
    
    # Test with a specific patient
    patient_id = 1  # Replace with actual patient ID
    results = matcher.find_trials_for_patient(
        patient_id=patient_id,
        phase_filter=["Phase I", "Phase II", "Phase III"]
    )
    
    print(f"\nPatient Trial Matching Results:")
    print(f"Patient: {results.get('patient_info', {}).get('mrn', 'Unknown')}")
    print(f"Total matches: {results.get('total_matches', 0)}")
    
    for i, trial in enumerate(results.get('matching_trials', [])[:5], 1):
        print(f"{i}. Trial: {trial.get('title', 'Unknown')}")
        print(f"   Condition: {trial.get('condition', 'Unknown')}")
        print(f"   Phase: {trial.get('phase', 'Unknown')}")
        print(f"   Hybrid Score: {trial.get('hybrid_score', 0):.4f}")
        print()

if __name__ == "__main__":
    main()
