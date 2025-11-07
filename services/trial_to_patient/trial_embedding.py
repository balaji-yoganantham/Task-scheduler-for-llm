"""
Trial-to-Patient Embedding Generator
Generates and persists embeddings for clinical trials to enable patient matching
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from services.shared.database_utils import DatabaseUtils
from services.shared.embedding_utils import EmbeddingUtils
from config import USE_CHUNKED_TRIAL_EMBEDDINGS

class TrialEmbeddingGenerator:
    def __init__(self):
        self.db_utils = DatabaseUtils()
        self.embedding_utils = EmbeddingUtils()
        
        # Trial-specific directories
        self.trials_dir = self.embedding_utils.trials_dir

    def _check_trial_embedding_exists(self, trial_id: str) -> bool:
        """Check if trial embedding already exists"""
        try:
            embedding_file = self.trials_dir / f"trial_{trial_id}.npy"
            return embedding_file.exists()
        except Exception as e:
            print(f"Error checking embedding for trial {trial_id}: {e}")
            return False

    def generate_trial_embeddings(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for multiple trials (only new trials)"""
        print(f"📊 Trial Embedding Generation Summary:")
        print(f"   Total trials: {len(trials)}")
        
        # Separate new trials from existing ones
        new_trials = []
        existing_trials = []
        
        for trial in trials:
            trial_id = trial['trial_id']
            if self._check_trial_embedding_exists(trial_id):
                existing_trials.append(trial)
                print(f"[OK] Trial {trial_id} already has embeddings - skipping")
            else:
                new_trials.append(trial)
        
        print(f"   Existing trials (skipped): {len(existing_trials)}")
        print(f"   New trials (to process): {len(new_trials)}")
        
        if not new_trials:
            print("[SUCCESS] All trials already have embeddings - no new generation needed!")
            return {
                "embeddings": [],
                "metadata": {},
                "total_trials": len(trials),
                "new_trials": 0,
                "existing_trials": len(existing_trials),
                "status": "all_existing"
            }
        
        print(f"🆕 Generating embeddings for {len(new_trials)} NEW trials...")
        
        trial_embeddings = []
        trial_metadata = {}
        
        for i, trial in enumerate(new_trials):
            print(f"Processing trial {i+1}/{len(new_trials)}: {trial['trial_id']}")
            
            # Generate embedding using chunking if enabled, otherwise use original method
            if USE_CHUNKED_TRIAL_EMBEDDINGS:
                embedding, chunk_metadata = self.embedding_utils.generate_chunked_embedding(
                    trial,
                    task_type="retrieval_document"
                )
                print(f"   Generated chunked embedding with {chunk_metadata.get('total_chunks', 0)} chunks")
            else:
                embedding = self.embedding_utils.generate_embedding(
                    trial['combined_trial_text'], 
                    task_type="retrieval_document"
                )
                chunk_metadata = {}
            
            trial_embeddings.append(embedding)
            
            # Store metadata
            trial_metadata[trial['trial_id']] = {
                "title": trial['title'],
                "condition": trial['condition'],
                "phase": trial['phase'],
                "status": trial['status'],
                "investigator": trial['investigator'],
                "created_date": trial['created_date'],
                "patients_matched": trial['patients_matched'],
                "matching_status": trial['matching_status'],
                "embedding_index": i,
                "embedding_method": "chunked" if USE_CHUNKED_TRIAL_EMBEDDINGS else "single",
                "chunk_metadata": chunk_metadata,
                "is_evaluated": trial.get('is_evaluated', 0)
            }
            
            # Save individual trial embedding
            trial_file = self.trials_dir / f"trial_{trial['trial_id']}.npy"
            np.save(trial_file, embedding)
        
        return {
            "embeddings": trial_embeddings,
            "metadata": trial_metadata,
            "total_trials": len(trials),
            "new_trials": len(new_trials),
            "existing_trials": len(existing_trials),
            "status": "new_generated"
        }

    def save_trial_embeddings(self, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save trial embeddings and create FAISS index"""
        try:
            embeddings = embedding_data["embeddings"]
            metadata = embedding_data["metadata"]
            
            # If no new embeddings were generated, return existing info
            if not embeddings:
                print("No new embeddings to save - using existing embeddings")
                return {
                    "total_trials": embedding_data["total_trials"],
                    "new_trials": embedding_data.get("new_trials", 0),
                    "existing_trials": embedding_data.get("existing_trials", 0),
                    "status": embedding_data.get("status", "no_new_embeddings"),
                    "message": "All trials already have embeddings"
                }
            
            # Save embeddings matrix
            matrix_file = self.embedding_utils.save_embeddings_matrix(
                embeddings, self.trials_dir, "trial_embeddings_matrix.npy"
            )
            
            # Save individual embeddings and metadata
            saved_files = self.embedding_utils.save_individual_embeddings(
                embeddings, metadata, self.trials_dir, "trial"
            )
            
            # Create FAISS index
            trial_index = self.embedding_utils.create_faiss_index(embeddings, "cosine")
            
            # Save FAISS index
            index_file = self.embedding_utils.save_faiss_index(
                trial_index, self.trials_dir, "faiss_index.pkl"
            )
            
            print(f"Trial embeddings saved to: {self.trials_dir}")
            
            return {
                "total_trials": embedding_data["total_trials"],
                "new_trials": embedding_data.get("new_trials", 0),
                "existing_trials": embedding_data.get("existing_trials", 0),
                "embedding_dimension": self.embedding_utils.dimension,
                "matrix_shape": np.vstack(embeddings).shape,
                "metadata_file": saved_files.get('metadata', ''),
                "matrix_file": matrix_file,
                "faiss_index_file": index_file,
                "index_size": trial_index.ntotal if trial_index else 0,
                "status": embedding_data.get("status", "completed")
            }
            
        except Exception as e:
            print(f"Error saving trial embeddings: {e}")
            return {}

    def run_trial_embedding_generation(self) -> Dict[str, Any]:
        """Main method to run trial embedding generation"""
        print("Starting trial embedding generation for patient matching...")
        
        # Get trial data
        trials = self.db_utils.get_detailed_trial_data()
        if not trials:
            print("No trial data found!")
            return {}
        
        print(f"Found {len(trials)} trials")
        
        # Generate embeddings
        embedding_data = self.generate_trial_embeddings(trials)
        
        # Save embeddings
        save_info = self.save_trial_embeddings(embedding_data)
        
        if save_info:
            print(f"\nTrial Embedding Generation Summary:")
            print(f"Total trials: {save_info['total_trials']}")
            print(f"New trials processed: {save_info.get('new_trials', 0)}")
            print(f"Existing trials (skipped): {save_info.get('existing_trials', 0)}")
            
            if save_info.get('status') == 'all_existing':
                print(f"Status: All trials already have embeddings - no new generation needed!")
            else:
                print(f"Embedding dimension: {save_info.get('embedding_dimension', 'N/A')}")
                print(f"Matrix shape: {save_info.get('matrix_shape', 'N/A')}")
                print(f"FAISS index size: {save_info.get('index_size', 'N/A')}")
            
            print(f"Files saved to: {self.trials_dir}")
        
        return save_info

def main():
    """Main function to run trial embedding generation"""
    generator = TrialEmbeddingGenerator()
    results = generator.run_trial_embedding_generation()
    
    if results:
        print("\nTrial embedding generation completed successfully!")
    else:
        print("\nTrial embedding generation failed!")

if __name__ == "__main__":
    main()
