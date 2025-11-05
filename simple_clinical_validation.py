"""
Simple Clinical Match Validation
Shows actual patient conditions vs trial conditions to validate matches
"""

import json
from services.trial_to_patient.trial_matcher import TrialMatcher
from services.shared.database_utils import DatabaseUtils


def validate_clinical_matches(trial_id: str, top_n: int = 5):
    """Validate clinical matches for a trial"""
    print("="*80)
    print("CLINICAL MATCH VALIDATION")
    print("="*80)
    print(f"Trial ID: {trial_id}")
    
    db_utils = DatabaseUtils()
    trial_matcher = TrialMatcher()
    
    # Get trial data
    trial_data = db_utils.get_trial_by_id(trial_id)
    if not trial_data:
        print(f"❌ Trial {trial_id} not found")
        return
    
    print(f"\nTrial: {trial_data.get('title', 'Unknown')}")
    trial_condition = trial_data.get('condition', 'Unknown')
    print(f"Condition: {trial_condition}")
    
    # Get matched patients
    matching_patients = trial_matcher.hybrid_search_patients_for_trial(trial_data, alpha=0.7)
    
    if not matching_patients:
        print("❌ No matching patients found")
        return
    
    print(f"\nFound {len(matching_patients)} matched patients")
    print(f"\n{'='*80}")
    print("MATCHED PATIENTS ANALYSIS")
    print(f"{'='*80}")
    
    for i, patient in enumerate(matching_patients[:top_n], 1):
        print(f"\n{i}. Patient Match #{i}")
        print(f"   Hybrid Score: {patient.get('hybrid_score', 0):.4f}")
        print(f"   Embedding Score: {patient.get('embedding_score', 0):.4f}")
        print(f"   BM25 Score: {patient.get('bm25_score', 0):.4f}")
        
        # Print patient data structure
        print(f"\n   Patient Data Structure:")
        for key, value in patient.items():
            if key not in ['combined_text', 'embedding_index']:
                if isinstance(value, str) and len(value) > 100:
                    print(f"     {key}: {value[:100]}...")
                else:
                    print(f"     {key}: {value}")
        
        # Try to get patient from database
        mrn = patient.get('mrn')
        patient_id = patient.get('patient_id')
        
        if not patient_id and mrn:
            patient_by_mrn = db_utils.get_patient_by_mrn(mrn)
            if patient_by_mrn:
                patient_id = patient_by_mrn.get('patient_id')
        
        if patient_id:
            patient_data = db_utils.get_patient_by_id(patient_id)
            if patient_data:
                print(f"\n   Patient Medical Information:")
                combined_text = patient_data.get('combined_text', '')
                
                # Extract key information
                if 'Chief Complaint:' in combined_text:
                    parts = combined_text.split('Chief Complaint:')
                    complaint = parts[1].split('\n\n')[0].strip() if len(parts) > 1 and '\n\n' in parts[1] else (parts[1].strip()[:200] if len(parts) > 1 else '')
                    print(f"     Chief Complaint: {complaint[:200]}")
                
                if 'Assessment:' in combined_text:
                    parts = combined_text.split('Assessment:')
                    assessment = parts[1].strip()[:300] if len(parts) > 1 else ''
                    print(f"     Assessment: {assessment[:300]}")
                
                # Get keywords
                keywords = db_utils.get_patient_keywords(patient_id)
                if keywords:
                    diagnosis = keywords.get('primary_diagnosis', '')
                    if diagnosis:
                        print(f"     Diagnosis: {diagnosis}")
                
                # Compare with trial condition
                print(f"\n   Condition Match Analysis:")
                print(f"     Trial Condition: {str(trial_condition)[:100]}")
                if keywords and keywords.get('primary_diagnosis'):
                    print(f"     Patient Diagnosis: {keywords.get('primary_diagnosis')}")
                    # Simple keyword matching
                    trial_cond_lower = str(trial_condition).lower()
                    patient_diag_lower = keywords.get('primary_diagnosis', '').lower()
                    
                    # Check for common medical terms
                    common_terms = ['cancer', 'carcinoma', 'tumor', 'neoplasm', 'leukemia', 
                                   'lymphoma', 'melanoma', 'sarcoma', 'breast', 'lung', 
                                   'prostate', 'colorectal', 'pancreatic']
                    
                    trial_has = [term for term in common_terms if term in trial_cond_lower]
                    patient_has = [term for term in common_terms if term in patient_diag_lower]
                    
                    common = set(trial_has) & set(patient_has)
                    
                    if common:
                        print(f"     ✓ Common medical terms: {list(common)}")
                        print(f"     ✓ MATCH - Found common medical terms")
                    else:
                        print(f"     ✗ No common medical terms found")
                        print(f"     ? UNLIKELY MATCH - Different conditions")
                else:
                    print(f"     ? Patient diagnosis not available")
        else:
            print(f"   ⚠️  Could not retrieve patient ID from match data")
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print("This validation shows:")
    print("1. The actual patient data returned by the matching system")
    print("2. Whether patient conditions match trial conditions")
    print("3. Whether the matches are clinically relevant")
    print("\nTo improve validation:")
    print("- Check if patient diagnosis matches trial condition")
    print("- Verify age/gender eligibility")
    print("- Review medical history compatibility")


if __name__ == "__main__":
    import sys
    
    trial_id = sys.argv[1] if len(sys.argv) > 1 else "NCT05334069"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    validate_clinical_matches(trial_id, top_n)

