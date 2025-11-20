# Accuracy Test Results Analysis

## Executive Summary

The accuracy test successfully ran both matching flows, but results show **low precision (15%)** with **moderate recall (60-65%)**. This indicates the system is finding relevant matches but also returning many false positives.

## Key Findings

### Patient-to-Trial Matching
- **Precision: 15%** - Only 15% of predicted trials are correct
- **Recall: 65%** - System finds 65% of eligible trials
- **F1 Score: 24%** - Overall performance metric
- **13 out of 20 patients** had at least one correct match
- **All 20 patients** had some incorrect matches

### Trial-to-Patient Matching
- **Precision: 15%** - Only 15% of predicted patients are correct
- **Recall: 60%** - System finds 60% of eligible patients
- **F1 Score: 21%** - Overall performance metric
- **3 out of 5 trials** had at least one correct match
- **All 5 trials** had some incorrect matches

## Root Cause Analysis

### 1. **Small Dataset Problem**
- Only **5 trials** and **20 patients** in the test dataset
- The system is returning **all 5 trials** for most patients
- With such a small pool, similarity matching returns almost everything

### 2. **No Eligibility Filtering**
The current system only does **semantic similarity matching** (embeddings + BM25) but does NOT:
- Check eligibility criteria (age, gender, molecular markers, etc.)
- Apply exclusion criteria
- Filter based on trial requirements (e.g., MSI-H vs MSS, BRAF mutation status)

### 3. **No Score Thresholding**
- All matches are returned regardless of similarity score
- No minimum confidence threshold to filter low-quality matches
- The system returns top-K matches even if they're not very similar

### 4. **Example Issues**

**Patient 35294426:**
- Predicted: All 5 trials
- Ground Truth Eligible: 2 trials (NCT04835142, NCT06696768)
- Correct: 2, Incorrect: 3
- **Issue**: System predicted 3 trials that don't match eligibility criteria

**Trial NCT05672316:**
- Predicted: All 20 patients
- Ground Truth Eligible: 0 patients
- **Issue**: This trial requires "prior progression on chemotherapy" but all patients are treatment-naïve

## What's Working Well

✅ **Embedding Generation**: Successfully generating MedCPT embeddings  
✅ **Hybrid Search**: FAISS + BM25 combination is working  
✅ **Recall**: Finding 60-65% of eligible matches (good coverage)  
✅ **Infrastructure**: File-based persistence working correctly  

## Recommendations for Improvement

### 1. **Add Eligibility Criteria Filtering** (High Priority)
Implement rule-based filtering before/after similarity matching:
- Check molecular markers (RAS, BRAF, MSI-H/MSS)
- Verify age ranges
- Check performance status (ECOG)
- Validate prior treatment history
- Apply exclusion criteria

### 2. **Implement Score Thresholding** (High Priority)
- Set minimum similarity score threshold (e.g., 0.5)
- Only return matches above threshold
- This will improve precision significantly

### 3. **Add LLM-Based Eligibility Evaluation** (Medium Priority)
Use the existing LLM evaluation system to:
- Evaluate each match for eligibility
- Filter out ineligible matches
- Provide reasoning for matches

### 4. **Improve Text Representation** (Medium Priority)
- Include more detailed eligibility criteria in trial embeddings
- Add structured data (molecular markers, stage) to patient embeddings
- Use eligibility-focused text chunks

### 5. **Test with Larger Dataset** (Low Priority)
- Test with more trials (50+) and patients (100+)
- Current small dataset makes evaluation difficult
- Larger dataset will show true performance

## Expected Improvements

With eligibility filtering and score thresholding:
- **Precision**: 15% → **60-80%** (4-5x improvement)
- **Recall**: 65% → **50-60%** (slight decrease, but acceptable)
- **F1 Score**: 24% → **55-70%** (2-3x improvement)

## Next Steps

1. ✅ **Completed**: Basic similarity matching test
2. 🔄 **In Progress**: Add eligibility criteria filtering
3. ⏳ **Pending**: Implement score thresholding
4. ⏳ **Pending**: Integrate LLM evaluation for filtering
5. ⏳ **Pending**: Re-test with improvements

## Conclusion

The matching system is **functionally working** but needs **eligibility-based filtering** to improve precision. The current approach finds semantically similar matches but doesn't validate clinical eligibility, which is critical for clinical trial matching.

