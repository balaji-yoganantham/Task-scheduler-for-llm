"""
Validate Trial-to-Patient Matching Flow
Check if trials are correctly matched to patients based on medical conditions
"""

import json
from services.patient_to_trial.patient_matcher import PatientMatcher
from services.shared.database_utils import DatabaseUtils


def validate_trial_to_patient_matches(patient_id: int, top_n: int = 5):
    """Validate trial-to-patient matches for a specific patient"""
    print("="*80)
    print("TRIAL-TO-PATIENT MATCHING VALIDATION")
    print("="*80)
    print(f"Patient ID: {patient_id}")
    
    db_utils = DatabaseUtils()
    patient_matcher = PatientMatcher()
    
    # Get patient data
    patient_data = db_utils.get_patient_by_id(patient_id)
    if not patient_data:
        print(f"❌ Patient {patient_id} not found")
        return
    
    print(f"\nPatient: {patient_data.get('patient_name', 'Unknown')}")
    print(f"MRN: {patient_data.get('mrn', 'Unknown')}")
    print(f"Age: {patient_data.get('age', 'Unknown')}, Gender: {patient_data.get('gender', 'Unknown')}")
    
    # Get patient diagnosis
    keywords = db_utils.get_patient_keywords(patient_id)
    patient_diagnosis = ""
    if keywords:
        patient_diagnosis = keywords.get('primary_diagnosis', '')
    
    # Try to extract from combined_text
    combined_text = patient_data.get('combined_text', '')
    if not patient_diagnosis and combined_text:
        # Try Assessment section first
        if 'Assessment:' in combined_text:
            parts = combined_text.split('Assessment:')
            if len(parts) > 1:
                assessment = parts[1].strip()
                # Look for diagnosis or condition keywords
                if 'Diagnosis:' in assessment:
                    diagnosis_part = assessment.split('Diagnosis:')[1]
                    # Get first line or first sentence
                    diagnosis_part = diagnosis_part.split('\n')[0].split('.')[0].strip()
                    if diagnosis_part:
                        patient_diagnosis = diagnosis_part
                else:
                    # Get the full assessment line (often contains diagnosis)
                    assessment_line = assessment.split('\n')[0].strip()
                    if assessment_line and len(assessment_line) > 10:
                        # Check for medical condition keywords
                        medical_keywords = ['cancer', 'carcinoma', 'tumor', 'neoplasm', 'leukemia', 
                                           'lymphoma', 'melanoma', 'sarcoma', 'adenocarcinoma']
                        if any(keyword in assessment_line.lower() for keyword in medical_keywords):
                            patient_diagnosis = assessment_line
        
        # Try Chief Complaint
        if not patient_diagnosis and 'Chief Complaint:' in combined_text:
            parts = combined_text.split('Chief Complaint:')
            if len(parts) > 1:
                complaint = parts[1].split('\n\n')[0].strip() if '\n\n' in parts[1] else parts[1].strip()
                if complaint and len(complaint) > 10:
                    # Extract key terms
                    patient_diagnosis = complaint[:200]
    
    if patient_diagnosis:
        print(f"Diagnosis: {patient_diagnosis[:200]}")
    else:
        print(f"⚠️  Diagnosis not found - showing medical history preview")
        if combined_text:
            print(f"Medical History Preview: {combined_text[:300]}...")
    
    # Get matched trials
    print(f"\n{'='*80}")
    print("GETTING MATCHED TRIALS")
    print(f"{'='*80}")
    
    matching_trials = patient_matcher.hybrid_search_trials_for_patient(patient_data, alpha=0.7)
    
    if not matching_trials:
        print("❌ No matching trials found")
        return
    
    print(f"Found {len(matching_trials)} matched trials")
    print(f"\n{'='*80}")
    print("MATCHED TRIALS ANALYSIS")
    print(f"{'='*80}")
    
    validation_results = []
    
    for i, trial in enumerate(matching_trials[:top_n], 1):
        trial_id = trial.get('trial_id', 'Unknown')
        trial_title = trial.get('title', 'Unknown')
        trial_condition = trial.get('condition', 'Unknown')
        hybrid_score = trial.get('hybrid_score', 0.0)
        embedding_score = trial.get('embedding_score', 0.0)
        bm25_score = trial.get('bm25_score', 0.0)
        
        print(f"\n{i}. Trial Match #{i}")
        print(f"   Trial ID: {trial_id}")
        print(f"   Title: {trial_title[:80]}...")
        print(f"   Hybrid Score: {hybrid_score:.4f}")
        print(f"   Embedding Score: {embedding_score:.4f}")
        print(f"   BM25 Score: {bm25_score:.4f}")
        
        # Get full trial details
        trial_details = db_utils.get_trial_by_id(trial_id)
        
        if trial_details:
            full_condition = trial_details.get('condition', 'Unknown')
            print(f"\n   Trial Condition: {str(full_condition)[:200]}...")
            
            # Condition match analysis
            print(f"\n   Condition Match Analysis:")
            print(f"     Patient Diagnosis: {patient_diagnosis if patient_diagnosis else 'Not available'}")
            
            # Parse trial condition (could be string or list)
            trial_conditions = []
            if isinstance(full_condition, str):
                # Try to parse as JSON array
                try:
                    trial_conditions = json.loads(full_condition)
                except:
                    # If not JSON, treat as single condition
                    trial_conditions = [full_condition]
            elif isinstance(full_condition, list):
                trial_conditions = full_condition
            
            # Check for condition match
            match_found = False
            matching_conditions = []
            
            if patient_diagnosis:
                patient_diag_lower = patient_diagnosis.lower()
                
                for trial_cond in trial_conditions:
                    trial_cond_str = str(trial_cond).lower()
                    
                    # Check for exact matches or strong keyword matches
                    if patient_diag_lower in trial_cond_str or trial_cond_str in patient_diag_lower:
                        match_found = True
                        matching_conditions.append(trial_cond)
                    
                    # Check for related medical terms (e.g., ovarian neoplasm = ovarian cancer)
                    medical_synonyms = {
                        'ovarian': ['ovarian', 'ovary', 'fallopian', 'peritoneal'],
                        'colorectal': ['colorectal', 'colon', 'rectal'],
                        'breast': ['breast', 'mammary'],
                        'lung': ['lung', 'pulmonary', 'respiratory'],
                        'prostate': ['prostate', 'prostatic'],
                        'pancreatic': ['pancreatic', 'pancreas'],
                        'gastric': ['gastric', 'stomach', 'gastroesophageal'],
                        'hepatocellular': ['hepatocellular', 'liver', 'hepatic'],
                        'cancer': ['cancer', 'carcinoma', 'tumor', 'neoplasm', 'malignancy']
                    }
                    
                    # Check for synonym matches
                    synonym_matched = False
                    for key_term, synonyms in medical_synonyms.items():
                        if any(syn in patient_diag_lower for syn in synonyms):
                            if any(syn in trial_cond_str for syn in synonyms):
                                match_found = True
                                matching_conditions.append(f"{trial_cond} (matched on: {key_term})")
                                synonym_matched = True
                                break
                    
                    # If no synonym match, check for common medical terms
                    if not synonym_matched:
                        patient_terms = set(patient_diag_lower.split())
                        trial_terms = set(trial_cond_str.split())
                        
                        # Medical keywords
                        medical_keywords = ['cancer', 'carcinoma', 'tumor', 'neoplasm', 
                                          'leukemia', 'lymphoma', 'melanoma', 'sarcoma',
                                          'breast', 'lung', 'prostate', 'colorectal', 
                                          'pancreatic', 'gastric', 'esophageal', 'ovarian',
                                          'bladder', 'kidney', 'thyroid', 'metastatic',
                                          'stage', 'adenocarcinoma']
                        
                        patient_medical = {term for term in patient_terms if term in medical_keywords}
                        trial_medical = {term for term in trial_terms if term in medical_keywords}
                        
                        common_medical = patient_medical & trial_medical
                        
                        if len(common_medical) >= 2:
                            match_found = True
                            matching_conditions.append(f"{trial_cond} (matched on: {', '.join(common_medical)})")
            
            if match_found:
                print(f"     ✓ MATCH FOUND")
                print(f"     Matching conditions: {matching_conditions[:3]}")
                match_quality = "STRONG_MATCH"
            else:
                # Check for general medical term overlap
                if patient_diagnosis:
                    patient_lower = patient_diagnosis.lower()
                    trial_conditions_lower = ' '.join(str(tc).lower() for tc in trial_conditions)
                    
                    # Simple keyword check
                    patient_words = set(patient_lower.split())
                    trial_words = set(trial_conditions_lower.split())
                    common = patient_words & trial_words
                    
                    # Remove common stop words
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                                 'to', 'for', 'of', 'with', 'by', 'stage', 'i', 'ii', 'iii', 'iv'}
                    common = {w for w in common if w not in stop_words and len(w) > 3}
                    
                    if len(common) >= 1:
                        print(f"     ? POSSIBLE MATCH (common keywords: {list(common)[:5]})")
                        match_quality = "POSSIBLE_MATCH"
                    else:
                        print(f"     ✗ NO OBVIOUS MATCH")
                        match_quality = "UNLIKELY_MATCH"
                else:
                    print(f"     ? CANNOT VERIFY (patient diagnosis not available)")
                    match_quality = "UNKNOWN"
            
            validation_results.append({
                "rank": i,
                "trial_id": trial_id,
                "trial_title": trial_title,
                "trial_condition": str(trial_condition)[:200],
                "patient_diagnosis": patient_diagnosis,
                "hybrid_score": float(hybrid_score),
                "embedding_score": float(embedding_score),
                "bm25_score": float(bm25_score),
                "match_quality": match_quality,
                "matching_conditions": matching_conditions[:3] if 'matching_conditions' in locals() else []
            })
        else:
            print(f"   ⚠️  Could not retrieve full trial details")
            validation_results.append({
                "rank": i,
                "trial_id": trial_id,
                "trial_title": trial_title,
                "match_quality": "ERROR",
                "error": "Could not retrieve trial details"
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if validation_results:
        strong_matches = sum(1 for r in validation_results if r.get('match_quality') == 'STRONG_MATCH')
        possible_matches = sum(1 for r in validation_results if r.get('match_quality') == 'POSSIBLE_MATCH')
        unlikely_matches = sum(1 for r in validation_results if r.get('match_quality') == 'UNLIKELY_MATCH')
        unknown = sum(1 for r in validation_results if r.get('match_quality') == 'UNKNOWN')
        errors = sum(1 for r in validation_results if r.get('match_quality') == 'ERROR')
        
        print(f"Total trials validated: {len(validation_results)}")
        print(f"  ✓ Strong matches: {strong_matches} ({strong_matches/len(validation_results)*100:.1f}%)")
        print(f"  ? Possible matches: {possible_matches} ({possible_matches/len(validation_results)*100:.1f}%)")
        print(f"  ✗ Unlikely matches: {unlikely_matches} ({unlikely_matches/len(validation_results)*100:.1f}%)")
        if unknown > 0:
            print(f"  ? Unknown: {unknown} ({unknown/len(validation_results)*100:.1f}%)")
        if errors > 0:
            print(f"  ❌ Errors: {errors} ({errors/len(validation_results)*100:.1f}%)")
        
        avg_hybrid_score = sum(r.get('hybrid_score', 0) for r in validation_results) / len(validation_results)
        print(f"\nAverage Hybrid Score: {avg_hybrid_score:.4f}")
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"trial_to_patient_validation_patient{patient_id}_{timestamp}.json"
    
    results = {
        "patient_id": patient_id,
        "patient_mrn": patient_data.get('mrn', 'Unknown'),
        "patient_diagnosis": patient_diagnosis,
        "validation_timestamp": datetime.now().isoformat(),
        "total_trials_matched": len(matching_trials),
        "validated_count": len(validation_results),
        "summary": {
            "strong_matches": strong_matches if validation_results else 0,
            "possible_matches": possible_matches if validation_results else 0,
            "unlikely_matches": unlikely_matches if validation_results else 0,
            "avg_hybrid_score": float(avg_hybrid_score) if validation_results else 0.0
        },
        "validation_results": validation_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    patient_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Validating trial-to-patient matches for patient: {patient_id}")
    print(f"Top N: {top_n}\n")
    
    validate_trial_to_patient_matches(patient_id, top_n)

