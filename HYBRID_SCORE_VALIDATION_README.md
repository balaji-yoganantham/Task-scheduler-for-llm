# Hybrid Score Validation Guide

This guide explains how to verify the accuracy of hybrid score calculations in the Clinical Trial Matching System.

## Overview

The hybrid score combines:
1. **Embedding Score** (semantic similarity from MedCPT embeddings)
2. **BM25 Score** (keyword-based similarity)

### Formula

```
normalized_bm25 = min(bm25_score / 10.0, 1.0)
hybrid_score = alpha * embedding_score + (1 - alpha) * normalized_bm25
```

Where:
- `alpha` = weight for embedding (default: 0.7 = 70%)
- `1 - alpha` = weight for BM25 (default: 0.3 = 30%)

## Validation Scripts

### 1. Quick Check Script (`quick_check_hybrid_scores.py`)

Fast validation for a single patient or trial match.

#### Usage

**Patient-to-Trial Matching:**
```bash
python quick_check_hybrid_scores.py patient <patient_id> [alpha] [top_n]
```

**Trial-to-Patient Matching:**
```bash
python quick_check_hybrid_scores.py trial <trial_id> [alpha] [top_n]
```

#### Examples

```bash
# Check patient 1 with default settings (alpha=0.7, top 5)
python quick_check_hybrid_scores.py patient 1

# Check patient 1 with custom alpha and top 10 results
python quick_check_hybrid_scores.py patient 1 0.7 10

# Check trial with default settings
python quick_check_hybrid_scores.py trial NCT04929223

# Check trial with custom alpha
python quick_check_hybrid_scores.py trial NCT04929223 0.8 10
```

#### Output

The script displays:
- Embedding score
- BM25 raw score
- BM25 normalized score
- Hybrid score (stored)
- Hybrid score (calculated)
- Validation status (✓ correct or ✗ with difference)

### 2. Comprehensive Validation Script (`test_hybrid_score_accuracy.py`)

Detailed validation with multiple test cases and statistics.

#### Usage

```bash
python test_hybrid_score_accuracy.py
```

#### What It Tests

1. **Formula Testing**: Tests the hybrid formula with various sample values
2. **Patient-to-Trial Validation**: Validates scores for patient matching
3. **Trial-to-Patient Validation**: Validates scores for trial matching
4. **Score Distribution Analysis**: Analyzes score distributions

#### Output

The script generates:
- Detailed score breakdowns
- Validation statistics (accuracy, average difference)
- JSON files with validation results:
  - `hybrid_score_validation_patient_<timestamp>.json`
  - `hybrid_score_validation_trial_<timestamp>.json`

## Understanding the Scores

### Embedding Score
- Range: Typically 0.0 to 1.0 (cosine similarity)
- Higher = more semantically similar
- Based on MedCPT embeddings (768 dimensions)

### BM25 Score (Raw)
- Range: Typically 0.0 to 10.0+ (can be higher)
- Higher = more keyword matches
- Based on BM25Okapi algorithm

### BM25 Score (Normalized)
- Formula: `min(bm25_raw / 10.0, 1.0)`
- Range: 0.0 to 1.0 (capped at 1.0)
- Normalizes BM25 to same scale as embedding

### Hybrid Score
- Formula: `alpha * embedding + (1-alpha) * normalized_bm25`
- Range: 0.0 to 1.0
- Weighted combination of both scores

## Validation Criteria

A score calculation is considered **correct** if:
```
|hybrid_stored - hybrid_calculated| < 0.0001
```

This accounts for floating-point precision differences.

## Common Issues to Check

### 1. BM25 Normalization
- Verify BM25 scores > 10.0 are capped at 1.0
- Check: `normalized_bm25 = min(bm25_raw / 10.0, 1.0)`

### 2. Alpha Weight
- Default alpha = 0.7 (70% embedding, 30% BM25)
- Verify weights sum to 1.0: `alpha + (1-alpha) = 1.0`

### 3. Missing Scores
- If embedding or BM25 score is missing, it defaults to 0.0
- This affects the hybrid score calculation

### 4. Score Ranges
- Embedding scores should be in range [0, 1]
- BM25 raw scores can be > 10 (then normalized)
- Hybrid scores should be in range [0, 1]

## Example Output

```
================================================================================
QUICK HYBRID SCORE VALIDATION - Patient 1
================================================================================
Patient: MRN PAT001
Alpha (embedding weight): 0.7
Formula: hybrid = 0.7 * embedding + 0.3 * normalized_bm25
================================================================================

Found 20 matching trials

Rank  Trial ID        Embedding    BM25 Raw     BM25 Norm    Hybrid       Calc         Status    
----------------------------------------------------------------------------------------------------
1     NCT04929223     0.854321     5.234567     0.523457     0.765432     0.765432     ✓         
2     NCT03947385     0.823456     4.567890     0.456789     0.723456     0.723456     ✓         
3     NCT01234567     0.812345     3.890123     0.389012     0.701234     0.701234     ✓         
```

## Troubleshooting

### If scores don't match:

1. **Check BM25 normalization**: Verify `min(bm25_raw / 10.0, 1.0)`
2. **Check alpha value**: Ensure correct alpha is used
3. **Check score sources**: Verify embedding and BM25 scores are correct
4. **Check formula**: Verify `alpha * embedding + (1-alpha) * normalized_bm25`

### If validation fails:

1. Check database connection
2. Verify embeddings are loaded
3. Check BM25 index is created
4. Verify patient/trial IDs exist

## Files Generated

- `hybrid_score_validation_patient_<timestamp>.json`: Patient validation results
- `hybrid_score_validation_trial_<timestamp>.json`: Trial validation results

## Next Steps

After validation, you can:
1. Adjust `alpha` value if needed (higher = more embedding weight)
2. Adjust BM25 normalization threshold if needed (currently 10.0)
3. Review score distributions to understand matching behavior
4. Fine-tune matching parameters based on validation results

