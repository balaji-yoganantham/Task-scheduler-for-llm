"""
Evaluate System Performance on Testing LLM Dataset
Tests the matching system against golden matches without saving to database
"""
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Import project services
from services.shared.embedding_utils import EmbeddingUtils
from services.patient_to_trial.patient_matcher import PatientMatcher
from services.patient_to_trial.patient_evaluator import PatientEvaluator

# Disable location filtering for testing
import config
original_location_enabled = config.LOCATION_ENABLED
config.LOCATION_ENABLED = False

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 100)
print("SYSTEM ACCURACY EVALUATION - Testing LLM Dataset")
print("=" * 100)

# Load testing dataset
testing_dir = Path("testing_llm")
patients_file = testing_dir / "all_patients.json"
trials_file = testing_dir / "all_trials.json"
golden_matches_file = testing_dir / "golden_matches.json"

print(f"\nLoading test dataset from: {testing_dir}")

# Load data
with open(patients_file, 'r', encoding='utf-8') as f:
    test_patients = json.load(f)

with open(trials_file, 'r', encoding='utf-8') as f:
    test_trials = json.load(f)

with open(golden_matches_file, 'r', encoding='utf-8') as f:
    golden_matches = json.load(f)

print(f"✓ Loaded {len(test_patients)} patients")
print(f"✓ Loaded {len(test_trials)} trials")
print(f"✓ Loaded {len(golden_matches)} golden matches")

# Initialize services
print(f"\n{'='*100}")
print("INITIALIZING SYSTEM SERVICES")
print(f"{'='*100}")

embedding_utils = EmbeddingUtils()
patient_matcher = PatientMatcher()
patient_evaluator = PatientEvaluator()

# Create mapping for easy lookup
golden_match_map = {gm['patient_id']: gm['matched_trial_id'] for gm in golden_matches}
trial_map = {trial['trial_id']: trial for trial in test_trials}
patient_map = {p['patient_id']: p for p in test_patients}

print(f"\n✓ Golden matches mapping: {len(golden_match_map)} patients")
print(f"✓ Trial mapping: {len(trial_map)} trials")
print(f"✓ Patient mapping: {len(patient_map)} patients")

# Step 1: Generate Trial Embeddings
print(f"\n{'='*100}")
print("STEP 1: GENERATING TRIAL EMBEDDINGS")
print(f"{'='*100}")

trial_embeddings = []
trial_metadata = {}

print("  Initializing embedding models (this may take 1-2 minutes on first run)...")
# Pre-load models to avoid repeated loading
embedding_utils.load_document_model()
print("  ✓ Models loaded, starting embedding generation...\n")

for i, trial in enumerate(test_trials):
    print(f"  [{i+1}/{len(test_trials)}] Processing trial {trial['trial_id']}: {trial['title'][:60]}...")
    sys.stdout.flush()  # Force output
    
    # Generate embedding using project's method
    if hasattr(embedding_utils, 'generate_chunked_embedding'):
        from config import USE_CHUNKED_TRIAL_EMBEDDINGS
        if USE_CHUNKED_TRIAL_EMBEDDINGS:
            embedding, chunk_metadata = embedding_utils.generate_chunked_embedding(
                trial,
                task_type="retrieval_document"
            )
        else:
            embedding = embedding_utils.generate_embedding(
                trial.get('combined_trial_text', ''),
                task_type="retrieval_document"
            )
            chunk_metadata = {}
    else:
        embedding = embedding_utils.generate_embedding(
            trial.get('combined_trial_text', ''),
            task_type="retrieval_document"
        )
        chunk_metadata = {}
    
    trial_embeddings.append(embedding)
    trial_metadata[trial['trial_id']] = {
        'trial_id': trial['trial_id'],
        'embedding_index': i,
        'title': trial.get('title', ''),
        'condition': trial.get('condition', ''),
        'phase': trial.get('phase', ''),
        'status': trial.get('status', ''),
        'investigator': trial.get('investigator', ''),
        # Add other fields that might be needed
        'brief_summary': trial.get('brief_summary', ''),
        'detailed_description': trial.get('detailed_description', ''),
        'inclusion_criteria': trial.get('inclusion_criteria', ''),
        'exclusion_criteria': trial.get('exclusion_criteria', ''),
        'eligibility_criteria': trial.get('eligibility_criteria', ''),
        'minimum_age': trial.get('minimum_age', ''),
        'maximum_age': trial.get('maximum_age', ''),
        'sex': trial.get('sex', '')
    }
    
    # Save embedding for patient_matcher to use
    trial_emb_file = embedding_utils.trials_dir / f"trial_{trial['trial_id']}.npy"
    np.save(trial_emb_file, embedding)

# Create FAISS index for trials
print(f"\n  Creating FAISS index for {len(trial_embeddings)} trials...")
trial_index = embedding_utils.create_faiss_index(trial_embeddings, "cosine")
trial_index_file = embedding_utils.trials_dir / "faiss_index.pkl"
with open(trial_index_file, 'wb') as f:
    import pickle
    pickle.dump(trial_index, f)

# Save trial metadata
trial_metadata_file = embedding_utils.trials_dir / "metadata.json"
with open(trial_metadata_file, 'w', encoding='utf-8') as f:
    json.dump(trial_metadata, f, indent=2, ensure_ascii=False)

# Create trial texts for BM25 indexing (needed for hybrid search)
print(f"\n  Creating trial texts for BM25 indexing...")
trial_texts = []
for trial in test_trials:
    # Create searchable text from trial data (same format as patient_matcher)
    trial_text = f"{trial.get('title', '')} {trial.get('condition', '')} {trial.get('phase', '')} {trial.get('status', '')} {trial.get('investigator', '')}"
    # Tokenize text (simple tokenization)
    import re
    text = re.sub(r'[^\w\s]', ' ', trial_text.lower())
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 2]
    trial_texts.append(tokens)

# Create BM25 index
from rank_bm25 import BM25Okapi
bm25_trials = BM25Okapi(trial_texts)
print(f"✓ Created BM25 index for {len(trial_texts)} trials")

print(f"✓ Generated {len(trial_embeddings)} trial embeddings")
print(f"✓ Created FAISS index with {trial_index.ntotal} vectors")
print(f"✓ Saved trial embeddings and metadata")

# Step 2: Generate Patient Embeddings
print(f"\n{'='*100}")
print("STEP 2: GENERATING PATIENT EMBEDDINGS")
print(f"{'='*100}")

patient_embeddings = []
patient_metadata = {}

print("  Loading query encoder for patients...")
embedding_utils.load_query_model()
print("  ✓ Query encoder loaded, starting patient embedding generation...\n")

for i, patient in enumerate(test_patients):
    print(f"  [{i+1}/{len(test_patients)}] Processing patient {patient['patient_id']} (MRN: {patient['mrn']})...")
    sys.stdout.flush()  # Force output
    
    # Generate embedding using patient's combined_text (same as project)
    patient_embedding = embedding_utils.generate_embedding(
        patient['combined_text'],
        task_type="retrieval_query"  # Query encoder for patients
    )
    
    patient_embeddings.append(patient_embedding)
    patient_metadata[str(patient['patient_id'])] = {
        'patient_id': patient['patient_id'],
        'mrn': patient['mrn'],
        'embedding_index': i
    }
    
    # Save embedding
    patient_emb_file = embedding_utils.patients_dir / f"patient_{patient['patient_id']}.npy"
    np.save(patient_emb_file, patient_embedding)

# Save patient metadata
patient_metadata_file = embedding_utils.patients_dir / "metadata.json"
with open(patient_metadata_file, 'w', encoding='utf-8') as f:
    json.dump(patient_metadata, f, indent=2, ensure_ascii=False)

print(f"✓ Generated {len(patient_embeddings)} patient embeddings")
print(f"✓ Saved patient embeddings and metadata")

# Reload patient_matcher to pick up new embeddings
patient_matcher.load_trial_data()

# Manually set BM25 index and trial texts (since we created them from JSON)
patient_matcher.trial_texts = trial_texts
patient_matcher.bm25_trials = bm25_trials

# Step 3: Run Hybrid Matching for Each Patient
print(f"\n{'='*100}")
print("STEP 3: RUNNING HYBRID MATCHING SYSTEM")
print(f"{'='*100}")

hybrid_results = []
for patient in test_patients:
    patient_id = patient['patient_id']
    print(f"\n  Testing patient {patient_id} (MRN: {patient['mrn']})...")
    
    # Use hybrid_search_trials_for_patient directly with patient data (skip database lookup)
    try:
        matching_trials = patient_matcher.hybrid_search_trials_for_patient(patient)
        
        if not matching_trials:
            print(f"    ⚠ No trials found for patient {patient_id}")
            hybrid_results.append({
                'patient_id': patient_id,
                'patient_mrn': patient['mrn'],
                'matching_trials': [],
                'top_trial_id': None,
                'top_trial_score': 0
            })
            continue
        
        # Sort by hybrid_score
        matching_trials.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        top_trial = matching_trials[0] if matching_trials else None
        
        hybrid_results.append({
            'patient_id': patient_id,
            'patient_mrn': patient['mrn'],
            'patient_data': patient,
            'matching_trials': matching_trials[:20],  # Top 20 for LLM evaluation
            'top_trial_id': top_trial.get('trial_id') if top_trial else None,
            'top_trial_score': top_trial.get('hybrid_score', 0) if top_trial else 0,
            'total_matches': len(matching_trials)
        })
        
        print(f"    Hybrid matching: Found {len(matching_trials)} trials, top: {top_trial.get('trial_id') if top_trial else 'N/A'}")
        
    except Exception as e:
        print(f"    ✗ Error matching patient {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        hybrid_results.append({
            'patient_id': patient_id,
            'patient_mrn': patient['mrn'],
            'matching_trials': [],
            'top_trial_id': None,
            'top_trial_score': 0
        })

# Step 4: Run LLM Evaluation
print(f"\n{'='*100}")
print("STEP 4: RUNNING LLM EVALUATION")
print(f"{'='*100}")

results = []
for hybrid_result in hybrid_results:
    patient_id = hybrid_result['patient_id']
    patient = hybrid_result.get('patient_data')
    matching_trials = hybrid_result.get('matching_trials', [])
    
    if not patient or not matching_trials:
        results.append({
            'patient_id': patient_id,
            'patient_mrn': hybrid_result['patient_mrn'],
            'golden_trial_id': golden_match_map.get(patient_id),
            'hybrid_top_trial_id': None,
            'hybrid_top_trial_rank': None,
            'llm_top_trial_id': None,
            'llm_golden_trial_rank': None,
            'llm_accuracy': False,
            'hybrid_accuracy': False,
            'error': 'No matching trials'
        })
        continue
    
    print(f"\n  Evaluating patient {patient_id} (MRN: {patient['mrn']})...")
    print(f"    Hybrid matches: {len(matching_trials)} trials")
    
    # Get detailed trial information for LLM evaluation
    detailed_trials = []
    for trial in matching_trials:
        trial_id = trial.get('trial_id')
        if trial_id and trial_id in trial_map:
            trial_info = trial_map[trial_id].copy()
            trial_info['hybrid_score'] = trial.get('hybrid_score', 0)
            trial_info['embedding_score'] = trial.get('embedding_score', 0)
            trial_info['bm25_score'] = trial.get('bm25_score', 0)
            detailed_trials.append(trial_info)
    
    if not detailed_trials:
        print(f"    ⚠ No detailed trial information available")
        results.append({
            'patient_id': patient_id,
            'patient_mrn': hybrid_result['patient_mrn'],
            'golden_trial_id': golden_match_map.get(patient_id),
            'hybrid_top_trial_id': hybrid_result.get('top_trial_id'),
            'hybrid_top_trial_rank': None,
            'llm_top_trial_id': None,
            'llm_golden_trial_rank': None,
            'llm_accuracy': False,
            'hybrid_accuracy': False,
            'error': 'No detailed trial info'
        })
        continue
    
    # Run LLM batch evaluation
    try:
        print(f"    Running LLM batch evaluation for {len(detailed_trials)} trials...")
        sys.stdout.flush()  # Force output before potentially long API call
        llm_evaluation = patient_evaluator.llm_utils.evaluate_patient_trial_matches_batch(
            detailed_trials, 
            patient
        )
        
        if "error" in llm_evaluation:
            print(f"    ✗ LLM evaluation failed: {llm_evaluation['error']}")
            results.append({
                'patient_id': patient_id,
                'patient_mrn': hybrid_result['patient_mrn'],
                'golden_trial_id': golden_match_map.get(patient_id),
                'hybrid_top_trial_id': hybrid_result.get('top_trial_id'),
                'llm_error': llm_evaluation['error']
            })
            continue
        
        evaluations = llm_evaluation.get('evaluations', [])
        if not evaluations:
            print(f"    ⚠ No LLM evaluations returned")
            continue
        
        # Sort evaluations by LLM priority score
        evaluations.sort(key=lambda x: (
            x.get('priority_score', 0) * 0.7 + 
            x.get('confidence_score', 0) * 0.3
        ), reverse=True)
        
        # Find golden match in hybrid results (before LLM)
        golden_trial_id = golden_match_map.get(patient_id)
        hybrid_golden_rank = None
        for rank, trial in enumerate(matching_trials, 1):
            if trial.get('trial_id') == golden_trial_id:
                hybrid_golden_rank = rank
                break
        
        # Find golden match in LLM evaluations (after LLM)
        llm_golden_rank = None
        llm_golden_evaluation = None
        for rank, eval_result in enumerate(evaluations, 1):
            trial_info = eval_result.get('trial_info', {})
            if trial_info.get('trial_id') == golden_trial_id:
                llm_golden_rank = rank
                llm_golden_evaluation = eval_result
                break
        
        llm_top_evaluation = evaluations[0] if evaluations else None
        llm_top_trial_id = llm_top_evaluation.get('trial_info', {}).get('trial_id') if llm_top_evaluation else None
        
        accuracy_status_hybrid = "✓" if hybrid_golden_rank == 1 else ("✓" if hybrid_golden_rank and hybrid_golden_rank <= 5 else "✗")
        accuracy_status_llm = "✓" if llm_golden_rank == 1 else ("✓" if llm_golden_rank and llm_golden_rank <= 5 else "✗")
        
        print(f"    Hybrid: Golden rank {hybrid_golden_rank or 'NOT FOUND'} | Top: {hybrid_result.get('top_trial_id')}")
        print(f"    LLM:    Golden rank {llm_golden_rank or 'NOT FOUND'} | Top: {llm_top_trial_id}")
        print(f"            {accuracy_status_hybrid} Hybrid | {accuracy_status_llm} LLM")
        
        results.append({
            'patient_id': patient_id,
            'patient_mrn': hybrid_result['patient_mrn'],
            'golden_trial_id': golden_trial_id,
            'hybrid_top_trial_id': hybrid_result.get('top_trial_id'),
            'hybrid_top_trial_score': hybrid_result.get('top_trial_score', 0),
            'hybrid_golden_trial_rank': hybrid_golden_rank,
            'hybrid_golden_trial_score': next((t.get('hybrid_score') for t in matching_trials if t.get('trial_id') == golden_trial_id), None) if golden_trial_id else None,
            'llm_top_trial_id': llm_top_trial_id,
            'llm_top_priority_score': llm_top_evaluation.get('priority_score', 0) if llm_top_evaluation else None,
            'llm_top_confidence_score': llm_top_evaluation.get('confidence_score', 0) if llm_top_evaluation else None,
            'llm_golden_trial_rank': llm_golden_rank,
            'llm_golden_priority_score': llm_golden_evaluation.get('priority_score', 0) if llm_golden_evaluation else None,
            'llm_golden_confidence_score': llm_golden_evaluation.get('confidence_score', 0) if llm_golden_evaluation else None,
            'llm_golden_eligibility_status': llm_golden_evaluation.get('eligibility_status', 'UNKNOWN') if llm_golden_evaluation else None,
            'llm_accuracy': llm_golden_rank == 1 if llm_golden_rank else False,
            'hybrid_accuracy': hybrid_golden_rank == 1 if hybrid_golden_rank else False,
            'total_trials_evaluated': len(evaluations),
            'llm_evaluations': evaluations[:5]  # Top 5 for detailed analysis
        })
        
    except Exception as e:
        print(f"    ✗ Error in LLM evaluation: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'patient_id': patient_id,
            'patient_mrn': hybrid_result['patient_mrn'],
            'golden_trial_id': golden_match_map.get(patient_id),
            'error': str(e)
        })

# Step 5: Calculate Accuracy Metrics
print(f"\n{'='*100}")
print("STEP 5: CALCULATING ACCURACY METRICS")
print(f"{'='*100}")

# Calculate HYBRID matching metrics
total_patients = len(results)
hybrid_top1_correct = sum(1 for r in results if r.get('hybrid_golden_trial_rank') == 1)
hybrid_top3_correct = sum(1 for r in results if r.get('hybrid_golden_trial_rank') and r.get('hybrid_golden_trial_rank') <= 3)
hybrid_top5_correct = sum(1 for r in results if r.get('hybrid_golden_trial_rank') and r.get('hybrid_golden_trial_rank') <= 5)
hybrid_golden_found = sum(1 for r in results if r.get('hybrid_golden_trial_rank') is not None)

hybrid_top1_accuracy = (hybrid_top1_correct / total_patients) * 100 if total_patients > 0 else 0
hybrid_top3_accuracy = (hybrid_top3_correct / total_patients) * 100 if total_patients > 0 else 0
hybrid_top5_accuracy = (hybrid_top5_correct / total_patients) * 100 if total_patients > 0 else 0
hybrid_recall = (hybrid_golden_found / total_patients) * 100 if total_patients > 0 else 0

hybrid_ranks = [r['hybrid_golden_trial_rank'] for r in results if r.get('hybrid_golden_trial_rank') is not None]
hybrid_avg_rank = sum(hybrid_ranks) / len(hybrid_ranks) if hybrid_ranks else None

# Calculate LLM EVALUATION metrics
llm_top1_correct = sum(1 for r in results if r.get('llm_golden_trial_rank') == 1)
llm_top3_correct = sum(1 for r in results if r.get('llm_golden_trial_rank') and r.get('llm_golden_trial_rank') <= 3)
llm_top5_correct = sum(1 for r in results if r.get('llm_golden_trial_rank') and r.get('llm_golden_trial_rank') <= 5)
llm_golden_found = sum(1 for r in results if r.get('llm_golden_trial_rank') is not None)

llm_top1_accuracy = (llm_top1_correct / total_patients) * 100 if total_patients > 0 else 0
llm_top3_accuracy = (llm_top3_correct / total_patients) * 100 if total_patients > 0 else 0
llm_top5_accuracy = (llm_top5_correct / total_patients) * 100 if total_patients > 0 else 0
llm_recall = (llm_golden_found / total_patients) * 100 if total_patients > 0 else 0

llm_ranks = [r['llm_golden_trial_rank'] for r in results if r.get('llm_golden_trial_rank') is not None]
llm_avg_rank = sum(llm_ranks) / len(llm_ranks) if llm_ranks else None

# Calculate LLM score statistics
llm_golden_priority_scores = [r['llm_golden_priority_score'] for r in results if r.get('llm_golden_priority_score') is not None]
llm_golden_confidence_scores = [r['llm_golden_confidence_score'] for r in results if r.get('llm_golden_confidence_score') is not None]
llm_top_priority_scores = [r['llm_top_priority_score'] for r in results if r.get('llm_top_priority_score') is not None]
llm_top_confidence_scores = [r['llm_top_confidence_score'] for r in results if r.get('llm_top_confidence_score') is not None]

avg_llm_golden_priority = sum(llm_golden_priority_scores) / len(llm_golden_priority_scores) if llm_golden_priority_scores else None
avg_llm_golden_confidence = sum(llm_golden_confidence_scores) / len(llm_golden_confidence_scores) if llm_golden_confidence_scores else None
avg_llm_top_priority = sum(llm_top_priority_scores) / len(llm_top_priority_scores) if llm_top_priority_scores else None
avg_llm_top_confidence = sum(llm_top_confidence_scores) / len(llm_top_confidence_scores) if llm_top_confidence_scores else None

# Count eligibility statuses
eligible_count = sum(1 for r in results if r.get('llm_golden_eligibility_status') == 'ELIGIBLE')
not_eligible_count = sum(1 for r in results if r.get('llm_golden_eligibility_status') == 'NOT_ELIGIBLE')
need_info_count = sum(1 for r in results if r.get('llm_golden_eligibility_status') == 'NEED_MORE_INFO')

print(f"\n{'='*80}")
print("HYBRID MATCHING METRICS (Before LLM)")
print(f"{'='*80}")
print(f"  Total patients tested: {total_patients}")
print(f"  Top-1 Accuracy: {hybrid_top1_accuracy:.2f}% ({hybrid_top1_correct}/{total_patients})")
print(f"  Top-3 Accuracy: {hybrid_top3_accuracy:.2f}% ({hybrid_top3_correct}/{total_patients})")
print(f"  Top-5 Accuracy: {hybrid_top5_accuracy:.2f}% ({hybrid_top5_correct}/{total_patients})")
print(f"  Recall: {hybrid_recall:.2f}% ({hybrid_golden_found}/{total_patients})")
print(f"  Average rank: {hybrid_avg_rank:.2f}" if hybrid_avg_rank else "  N/A")

print(f"\n{'='*80}")
print("LLM EVALUATION METRICS (After LLM)")
print(f"{'='*80}")
print(f"  Total patients evaluated: {total_patients}")
print(f"  Top-1 Accuracy: {llm_top1_accuracy:.2f}% ({llm_top1_correct}/{total_patients})")
print(f"  Top-3 Accuracy: {llm_top3_accuracy:.2f}% ({llm_top3_correct}/{total_patients})")
print(f"  Top-5 Accuracy: {llm_top5_accuracy:.2f}% ({llm_top5_correct}/{total_patients})")
print(f"  Recall: {llm_recall:.2f}% ({llm_golden_found}/{total_patients})")
print(f"  Average rank: {llm_avg_rank:.2f}" if llm_avg_rank else "  N/A")

print(f"\n  LLM Score Statistics:")
print(f"    Average golden priority score: {avg_llm_golden_priority:.2f}" if avg_llm_golden_priority else "    N/A")
print(f"    Average golden confidence score: {avg_llm_golden_confidence:.2f}" if avg_llm_golden_confidence else "    N/A")
print(f"    Average top match priority score: {avg_llm_top_priority:.2f}" if avg_llm_top_priority else "    N/A")
print(f"    Average top match confidence score: {avg_llm_top_confidence:.2f}" if avg_llm_top_confidence else "    N/A")

print(f"\n  LLM Eligibility Assessment:")
print(f"    Golden matches marked ELIGIBLE: {eligible_count}/{total_patients}")
print(f"    Golden matches marked NOT_ELIGIBLE: {not_eligible_count}/{total_patients}")
print(f"    Golden matches marked NEED_MORE_INFO: {need_info_count}/{total_patients}")

print(f"\n{'='*80}")
print("IMPROVEMENT COMPARISON")
print(f"{'='*80}")
improvement_top1 = llm_top1_accuracy - hybrid_top1_accuracy
improvement_top3 = llm_top3_accuracy - hybrid_top3_accuracy
improvement_top5 = llm_top5_accuracy - hybrid_top5_accuracy
improvement_avg_rank = (hybrid_avg_rank - llm_avg_rank) if (hybrid_avg_rank and llm_avg_rank) else None

print(f"  Top-1 Accuracy improvement: {improvement_top1:+.2f}%")
print(f"  Top-3 Accuracy improvement: {improvement_top3:+.2f}%")
print(f"  Top-5 Accuracy improvement: {improvement_top5:+.2f}%")
if improvement_avg_rank:
    print(f"  Average rank improvement: {improvement_avg_rank:.2f} positions better")

# Step 6: Generate Detailed Report
print(f"\n{'='*100}")
print("STEP 6: GENERATING DETAILED REPORT")
print(f"{'='*100}")

report = {
    'evaluation_info': {
        'evaluation_date': datetime.now().isoformat(),
        'dataset': 'testing_llm',
        'total_patients': total_patients,
        'total_trials': len(test_trials),
        'evaluation_method': 'Patient-to-Trial Matching with Hybrid Search + LLM Evaluation'
    },
    'hybrid_matching_metrics': {
        'top1_accuracy': round(hybrid_top1_accuracy, 2),
        'top3_accuracy': round(hybrid_top3_accuracy, 2),
        'top5_accuracy': round(hybrid_top5_accuracy, 2),
        'recall': round(hybrid_recall, 2),
        'top1_correct': hybrid_top1_correct,
        'top3_correct': hybrid_top3_correct,
        'top5_correct': hybrid_top5_correct,
        'golden_found': hybrid_golden_found,
        'average_rank': round(hybrid_avg_rank, 2) if hybrid_avg_rank else None
    },
    'llm_evaluation_metrics': {
        'top1_accuracy': round(llm_top1_accuracy, 2),
        'top3_accuracy': round(llm_top3_accuracy, 2),
        'top5_accuracy': round(llm_top5_accuracy, 2),
        'recall': round(llm_recall, 2),
        'top1_correct': llm_top1_correct,
        'top3_correct': llm_top3_correct,
        'top5_correct': llm_top5_correct,
        'golden_found': llm_golden_found,
        'average_rank': round(llm_avg_rank, 2) if llm_avg_rank else None,
        'average_golden_priority_score': round(avg_llm_golden_priority, 2) if avg_llm_golden_priority else None,
        'average_golden_confidence_score': round(avg_llm_golden_confidence, 2) if avg_llm_golden_confidence else None,
        'average_top_priority_score': round(avg_llm_top_priority, 2) if avg_llm_top_priority else None,
        'average_top_confidence_score': round(avg_llm_top_confidence, 2) if avg_llm_top_confidence else None,
        'eligibility_assessment': {
            'eligible_count': eligible_count,
            'not_eligible_count': not_eligible_count,
            'need_more_info_count': need_info_count
        }
    },
    'improvement_analysis': {
        'top1_improvement': round(improvement_top1, 2),
        'top3_improvement': round(improvement_top3, 2),
        'top5_improvement': round(improvement_top5, 2),
        'average_rank_improvement': round(improvement_avg_rank, 2) if improvement_avg_rank else None
    },
    'detailed_results': results,
    'system_info': {
        'embedding_model': 'MedCPT (NCBI)',
        'matching_method': 'Hybrid (Embedding + BM25)',
        'llm_model': 'Google Gemini 2.0 Flash',
        'embedding_dimension': embedding_utils.dimension,
        'trials_in_index': len(trial_embeddings),
        'patients_processed': len(patient_embeddings)
    }
}

# Save report
report_file = Path("results") / f"accuracy_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
report_file.parent.mkdir(exist_ok=True)

with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print(f"✓ Report saved to: {report_file}")

# Print summary
print(f"\n{'='*100}")
print("EVALUATION SUMMARY")
print(f"{'='*100}")
print(f"\n📊 HYBRID MATCHING RESULTS (Before LLM):")
print(f"  Top-1 Accuracy: {hybrid_top1_accuracy:.2f}% ({hybrid_top1_correct}/{total_patients})")
print(f"  Top-3 Accuracy: {hybrid_top3_accuracy:.2f}% ({hybrid_top3_correct}/{total_patients})")
print(f"  Top-5 Accuracy: {hybrid_top5_accuracy:.2f}% ({hybrid_top5_correct}/{total_patients})")
print(f"  Average rank: {hybrid_avg_rank:.2f}" if hybrid_avg_rank else "  N/A")

print(f"\n🤖 LLM EVALUATION RESULTS (After LLM):")
print(f"  Top-1 Accuracy: {llm_top1_accuracy:.2f}% ({llm_top1_correct}/{total_patients})")
print(f"  Top-3 Accuracy: {llm_top3_accuracy:.2f}% ({llm_top3_correct}/{total_patients})")
print(f"  Top-5 Accuracy: {llm_top5_accuracy:.2f}% ({llm_top5_correct}/{total_patients})")
print(f"  Average rank: {llm_avg_rank:.2f}" if llm_avg_rank else "  N/A")

print(f"\n📈 IMPROVEMENT:")
print(f"  Top-1: {improvement_top1:+.2f}%")
print(f"  Top-3: {improvement_top3:+.2f}%")
print(f"  Top-5: {improvement_top5:+.2f}%")
if improvement_avg_rank:
    print(f"  Average rank: {improvement_avg_rank:.2f} positions better")

print(f"\n💾 Detailed report: {report_file}")
# Restore original config
config.LOCATION_ENABLED = original_location_enabled

print(f"\n{'='*100}")
print("EVALUATION COMPLETE")
print(f"{'='*100}")
print("\nNOTE: Location filtering was DISABLED for this evaluation")

