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
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from config import USE_CHUNKED_TRIAL_EMBEDDINGS, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP_TOKENS, TRIAL_CHUNK_WEIGHTS

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

    def count_tokens(self, text: str, task_type: str = "retrieval_document") -> int:
        """Count tokens in text using MedCPT tokenizer"""
        try:
            if task_type == "retrieval_query":
                _, tokenizer = self.load_query_model()
            else:
                _, tokenizer = self.load_document_model()
            
            tokens = tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback: approximate token count (rough estimate: 1 token ≈ 4 characters)
            return len(text) // 4

    def split_text_into_chunks(self, text: str, max_tokens: int, overlap_tokens: int, 
                               task_type: str = "retrieval_document") -> List[str]:
        """Split long text into overlapping chunks based on token count"""
        try:
            if task_type == "retrieval_query":
                _, tokenizer = self.load_query_model()
            else:
                _, tokenizer = self.load_document_model()
            
            # Tokenize the entire text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= max_tokens:
                return [text]
            
            chunks = []
            start_idx = 0
            
            while start_idx < len(tokens):
                end_idx = start_idx + max_tokens
                chunk_tokens = tokens[start_idx:end_idx]
                
                # Decode chunk tokens back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                
                # Move start index forward with overlap
                start_idx += max_tokens - overlap_tokens
                
                # Avoid infinite loop
                if start_idx >= len(tokens):
                    break
            
            return chunks
        except Exception as e:
            print(f"Error splitting text into chunks: {e}")
            # Fallback: simple character-based splitting
            chunk_size = max_tokens * 4  # Approximate 4 chars per token
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap_tokens * 4)]

    def chunk_trial_text(self, trial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Chunk trial text into semantic sections
        
        Returns:
            List of dicts with 'text', 'chunk_type', and 'weight' keys
        """
        chunks = []
        
        # Extract trial fields
        trial_id = str(trial_data.get('trial_id', ''))
        title = str(trial_data.get('title', ''))
        condition = str(trial_data.get('condition', ''))
        phase = str(trial_data.get('phase', ''))
        status = str(trial_data.get('status', ''))
        investigator = str(trial_data.get('investigator', ''))
        minimum_age = str(trial_data.get('minimum_age', 'Not specified'))
        maximum_age = str(trial_data.get('maximum_age', 'Not specified'))
        sex = str(trial_data.get('sex', 'Not specified'))
        brief_summary = str(trial_data.get('brief_summary', ''))
        detailed_description = str(trial_data.get('detailed_description', ''))
        inclusion_criteria = str(trial_data.get('inclusion_criteria', ''))
        exclusion_criteria = str(trial_data.get('exclusion_criteria', ''))
        eligibility_criteria = str(trial_data.get('eligibility_criteria', ''))
        
        # Chunk 1: Basic Info + Brief Summary (high priority for context)
        basic_info_text = f"""Trial ID: {trial_id}
Title: {title}
Condition: {condition}
Phase: {phase}
Status: {status}
Investigator: {investigator}
Age Range: {minimum_age} - {maximum_age}
Gender: {sex}

Brief Summary: {brief_summary}"""
        
        chunks.append({
            'text': basic_info_text,
            'chunk_type': 'basic_info_summary',
            'weight': TRIAL_CHUNK_WEIGHTS.get('basic_info_summary', 0.15)
        })
        
        # Chunk 2: Detailed Description (split if too long)
        if detailed_description and detailed_description.lower() != 'not available':
            desc_tokens = self.count_tokens(detailed_description)
            if desc_tokens > MAX_TOKENS_PER_CHUNK:
                # Split into multiple chunks
                desc_chunks = self.split_text_into_chunks(
                    detailed_description, 
                    MAX_TOKENS_PER_CHUNK, 
                    CHUNK_OVERLAP_TOKENS
                )
                weight_per_chunk = TRIAL_CHUNK_WEIGHTS.get('detailed_description', 0.20) / len(desc_chunks)
                for i, desc_chunk in enumerate(desc_chunks):
                    chunks.append({
                        'text': f"Detailed Description (Part {i+1}): {desc_chunk}",
                        'chunk_type': 'detailed_description',
                        'weight': weight_per_chunk
                    })
            else:
                chunks.append({
                    'text': f"Detailed Description: {detailed_description}",
                    'chunk_type': 'detailed_description',
                    'weight': TRIAL_CHUNK_WEIGHTS.get('detailed_description', 0.20)
                })
        
        # Chunk 3: Inclusion Criteria (high priority - split if too long)
        if inclusion_criteria and inclusion_criteria.lower() not in ['not available', '']:
            incl_tokens = self.count_tokens(inclusion_criteria)
            if incl_tokens > MAX_TOKENS_PER_CHUNK:
                incl_chunks = self.split_text_into_chunks(
                    inclusion_criteria,
                    MAX_TOKENS_PER_CHUNK,
                    CHUNK_OVERLAP_TOKENS
                )
                weight_per_chunk = TRIAL_CHUNK_WEIGHTS.get('inclusion_criteria', 0.25) / len(incl_chunks)
                for i, incl_chunk in enumerate(incl_chunks):
                    chunks.append({
                        'text': f"Inclusion Criteria (Part {i+1}): {incl_chunk}",
                        'chunk_type': 'inclusion_criteria',
                        'weight': weight_per_chunk
                    })
            else:
                chunks.append({
                    'text': f"Inclusion Criteria: {inclusion_criteria}",
                    'chunk_type': 'inclusion_criteria',
                    'weight': TRIAL_CHUNK_WEIGHTS.get('inclusion_criteria', 0.25)
                })
        
        # Chunk 4: Exclusion Criteria (high priority - split if too long)
        if exclusion_criteria and exclusion_criteria.lower() not in ['not available', '']:
            excl_tokens = self.count_tokens(exclusion_criteria)
            if excl_tokens > MAX_TOKENS_PER_CHUNK:
                excl_chunks = self.split_text_into_chunks(
                    exclusion_criteria,
                    MAX_TOKENS_PER_CHUNK,
                    CHUNK_OVERLAP_TOKENS
                )
                weight_per_chunk = TRIAL_CHUNK_WEIGHTS.get('exclusion_criteria', 0.25) / len(excl_chunks)
                for i, excl_chunk in enumerate(excl_chunks):
                    chunks.append({
                        'text': f"Exclusion Criteria (Part {i+1}): {excl_chunk}",
                        'chunk_type': 'exclusion_criteria',
                        'weight': weight_per_chunk
                    })
            else:
                chunks.append({
                    'text': f"Exclusion Criteria: {exclusion_criteria}",
                    'chunk_type': 'exclusion_criteria',
                    'weight': TRIAL_CHUNK_WEIGHTS.get('exclusion_criteria', 0.25)
                })
        
        # Chunk 5: Eligibility Criteria (if separate from inclusion/exclusion)
        if eligibility_criteria and eligibility_criteria.lower() not in ['not available', '']:
            # Check if eligibility is different from inclusion/exclusion
            if eligibility_criteria != inclusion_criteria and eligibility_criteria != exclusion_criteria:
                elig_tokens = self.count_tokens(eligibility_criteria)
                if elig_tokens > MAX_TOKENS_PER_CHUNK:
                    elig_chunks = self.split_text_into_chunks(
                        eligibility_criteria,
                        MAX_TOKENS_PER_CHUNK,
                        CHUNK_OVERLAP_TOKENS
                    )
                    weight_per_chunk = TRIAL_CHUNK_WEIGHTS.get('eligibility_criteria', 0.15) / len(elig_chunks)
                    for i, elig_chunk in enumerate(elig_chunks):
                        chunks.append({
                            'text': f"Eligibility Criteria (Part {i+1}): {elig_chunk}",
                            'chunk_type': 'eligibility_criteria',
                            'weight': weight_per_chunk
                        })
                else:
                    chunks.append({
                        'text': f"Eligibility Criteria: {eligibility_criteria}",
                        'chunk_type': 'eligibility_criteria',
                        'weight': TRIAL_CHUNK_WEIGHTS.get('eligibility_criteria', 0.15)
                    })
        
        return chunks

    def generate_chunked_embedding(self, trial_data: Dict[str, Any], 
                                   task_type: str = "retrieval_document") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate embedding for trial using chunking strategy with weighted aggregation
        
        Returns:
            Tuple of (aggregated_embedding, chunk_metadata)
        """
        try:
            # Get chunks
            chunks = self.chunk_trial_text(trial_data)
            
            if not chunks:
                print(f"Warning: No chunks generated for trial {trial_data.get('trial_id', 'unknown')}")
                return self.generate_embedding(trial_data.get('combined_trial_text', ''), task_type), {}
            
            # Generate embeddings for each chunk
            chunk_embeddings = []
            chunk_weights = []
            chunk_metadata = {
                'total_chunks': len(chunks),
                'chunks': []
            }
            
            for i, chunk in enumerate(chunks):
                chunk_embedding = self.generate_embedding(chunk['text'], task_type)
                chunk_embeddings.append(chunk_embedding)
                chunk_weights.append(chunk['weight'])
                
                chunk_metadata['chunks'].append({
                    'index': i,
                    'chunk_type': chunk['chunk_type'],
                    'weight': chunk['weight'],
                    'token_count': self.count_tokens(chunk['text'], task_type)
                })
            
            # Normalize weights to sum to 1.0
            total_weight = sum(chunk_weights)
            if total_weight > 0:
                chunk_weights = [w / total_weight for w in chunk_weights]
            else:
                # Equal weights if all weights are zero
                chunk_weights = [1.0 / len(chunk_embeddings)] * len(chunk_embeddings)
            
            # Weighted average aggregation
            chunk_embeddings_array = np.array(chunk_embeddings)
            chunk_weights_array = np.array(chunk_weights).reshape(-1, 1)
            
            # Element-wise weighted average
            aggregated_embedding = np.sum(chunk_embeddings_array * chunk_weights_array, axis=0)
            
            # Normalize the aggregated embedding
            norm = np.linalg.norm(aggregated_embedding)
            if norm > 0:
                aggregated_embedding = aggregated_embedding / norm
            
            chunk_metadata['aggregation_method'] = 'weighted_average'
            chunk_metadata['normalized'] = True
            
            return aggregated_embedding.astype(np.float32), chunk_metadata
            
        except Exception as e:
            print(f"Error generating chunked embedding: {e}")
            # Fallback to regular embedding
            return self.generate_embedding(trial_data.get('combined_trial_text', ''), task_type), {}

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
