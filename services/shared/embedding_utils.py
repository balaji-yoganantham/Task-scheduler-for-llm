"""
Shared Embedding Utilities
Common embedding generation and persistence utilities for both patient-to-trial and trial-to-patient matching
Uses MedCPT embeddings from NCBI via Hugging Face
"""

import json
import numpy as np
import pickle
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
from datetime import datetime

class EmbeddingUtils:
    def __init__(self):
        # Initialize MedCPT models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load MedCPT models
        self.query_model = None
        self.query_tokenizer = None
        self.document_model = None
        self.document_tokenizer = None
        
        # Directory structure
        self.persist_dir = Path("persist")
        self.embeddings_dir = self.persist_dir / "embeddings"
        self.trials_dir = self.embeddings_dir / "trials"
        self.patients_dir = self.embeddings_dir / "patients"
        
        # Create directories
        for dir_path in [self.persist_dir, self.embeddings_dir, self.trials_dir, self.patients_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # MedCPT embedding dimension
        self.dimension = 768

    def load_query_model(self):
        """Load MedCPT Query Encoder model"""
        if self.query_model is None:
            print("Loading MedCPT Query Encoder...")
            self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
            self.query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            print("OK MedCPT Query Encoder loaded successfully")
        return self.query_model, self.query_tokenizer

    def load_document_model(self):
        """Load MedCPT Article Encoder model"""
        if self.document_model is None:
            print("Loading MedCPT Article Encoder...")
            self.document_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(self.device)
            self.document_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
            print("OK MedCPT Article Encoder loaded successfully")
        return self.document_model, self.document_tokenizer

    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """Generate embedding for text using MedCPT"""
        try:
            if task_type == "retrieval_query":
                # Use query encoder for patient queries
                model, tokenizer = self.load_query_model()
            else:
                # Use document encoder for trials and patient documents
                model, tokenizer = self.load_document_model()
            
            # Tokenize and encode
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating MedCPT embedding: {e}")
            # Return zero vector if embedding fails
            return np.zeros(self.dimension, dtype=np.float32)

    def save_embeddings_matrix(self, embeddings: List[np.ndarray], target_dir: Path, filename: str) -> str:
        """Save embeddings matrix to file"""
        try:
            embeddings_matrix = np.vstack(embeddings)
            matrix_file = target_dir / filename
            np.save(matrix_file, embeddings_matrix)
            return str(matrix_file)
        except Exception as e:
            print(f"Error saving embeddings matrix: {e}")
            return ""

    def save_individual_embeddings(self, embeddings: List[np.ndarray], metadata: Dict[str, Any], 
                                 target_dir: Path, prefix: str) -> Dict[str, str]:
        """Save individual embeddings and metadata"""
        try:
            saved_files = {}
            
            # Save individual embeddings
            for i, embedding in enumerate(embeddings):
                entity_id = list(metadata.keys())[i] if i < len(metadata) else f"{prefix}_{i}"
                embedding_file = target_dir / f"{prefix}_{entity_id}.npy"
                np.save(embedding_file, embedding)
                saved_files[entity_id] = str(embedding_file)
            
            # Save metadata
            metadata_file = target_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
            
            return saved_files
        except Exception as e:
            print(f"Error saving individual embeddings: {e}")
            return {}

    def create_faiss_index(self, embeddings: List[np.ndarray], index_type: str = "cosine") -> Any:
        """Create FAISS index for embeddings"""
        try:
            embeddings_matrix = np.vstack(embeddings)
            
            if index_type == "cosine":
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            else:
                index = faiss.IndexFlatL2(self.dimension)  # L2 distance
            
            index.add(embeddings_matrix)
            return index
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return None

    def save_faiss_index(self, index: Any, target_dir: Path, filename: str) -> str:
        """Save FAISS index to file"""
        try:
            index_file = target_dir / filename
            with open(index_file, 'wb') as f:
                pickle.dump(index, f)
            return str(index_file)
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return ""

    def load_faiss_index(self, index_file: Path) -> Any:
        """Load FAISS index from file"""
        try:
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    def load_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """Load metadata from file"""
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def search_similar(self, query_embedding: np.ndarray, index: Any, 
                      metadata: Dict[str, Any], k: int = 20) -> List[Dict[str, Any]]:
        """Search for similar embeddings using FAISS index"""
        try:
            scores, indices = index.search(query_embedding.reshape(1, -1), k=min(k, index.ntotal))
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if str(idx) in metadata:
                    result = metadata[str(idx)].copy()
                    result['index'] = int(idx)
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Error searching similar embeddings: {e}")
            return []

    def create_global_cache_index(self, trial_info: Dict, patient_info: Dict) -> str:
        """Create global cache index for all embeddings"""
        try:
            global_index = {
                "created_at": datetime.now().isoformat(),
                "trials": trial_info,
                "patients": patient_info
            }
            
            cache_index_file = self.embeddings_dir / "cache_index.json"
            with open(cache_index_file, 'w') as f:
                json.dump(global_index, f, indent=2)
            
            return str(cache_index_file)
        except Exception as e:
            print(f"Error creating global cache index: {e}")
            return ""
