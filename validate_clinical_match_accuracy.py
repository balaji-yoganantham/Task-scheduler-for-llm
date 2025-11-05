"""
Validate Clinical Match Accuracy
Check if matched patients are actually suitable for trials based on medical conditions
"""

import json
from typing import Dict, Any
from services.trial_to_patient.trial_matcher import TrialMatcher
from services.patient_to_trial.patient_matcher import PatientMatcher
from services.shared.database_utils import DatabaseUtils
from services.shared.llm_utils import LLMUtils


class ClinicalMatchValidator:
    """Validate clinical match accuracy"""
    
    def __init__(self):
        self.trial_matcher = TrialMatcher()
        self.patient_matcher = PatientMatcher()
        self.db_utils = DatabaseUtils()
        self.llm_utils = LLMUtils()
    
    def get_patient_medical_summary(self, patient_id: int) -> Dict[str, Any]:
        """Get patient medical summary for condition matching"""
        try:
            patient = self.db_utils.get_patient_by_id(patient_id)
            if not patient:
                return {}
            
            # Extract key medical information
            combined_text = patient.get('combined_text', '')
            
            # Try to extract chief complaint, diagnosis, assessment
            chief_complaint = ""
            diagnosis = ""
            assessment = ""
            
            if 'Chief Complaint:' in combined_text:
                parts = combined_text.split('Chief Complaint:')
                if len(parts) > 1:
                    chief_complaint = parts[1].split('\n\n')[0].strip() if '\n\n' in parts[1] else parts[1].strip()
            
            if 'Assessment:' in combined_text:
                parts = combined_text.split('Assessment:')
                if len(parts) > 1:
                    assessment = parts[1].strip()
            
            # Try to get primary diagnosis from keywords table
            keywords = self.db_utils.get_patient_keywords(patient_id)
            if keywords:
                diagnosis = keywords.get('primary_diagnosis', '')
            
            return {
                "patient_id": patient_id,
                "mrn": patient.get('mrn', ''),
                "age": patient.get('age', ''),
                "gender": patient.get('gender', ''),
                "chief_complaint": chief_complaint[:200] if chief_complaint else "Not available",
                "diagnosis": diagnosis if diagnosis else "Not available",
                "assessment": assessment[:300] if assessment else "Not available",
                "combined_text_preview": combined_text[:500] if combined_text else ""
            }
        except Exception as e:
            print(f"Error getting patient summary: {e}")
            return {}
    
    def validate_trial_patient_matches(self, trial_id: str, top_n: int = 10):
        """Validate if matched patients are actually suitable for the trial"""
        print("="*80)
        print("CLINICAL MATCH ACCURACY VALIDATION")
        print("="*80)
        print(f"Trial ID: {trial_id}")
        print(f"Top N matches to validate: {top_n}")
        
        # Get trial information
        trial_data = self.db_utils.get_trial_by_id(trial_id)
        if not trial_data:
            print(f"❌ Trial {trial_id} not found")
            return
        
        print(f"\n{'='*80}")
        print("TRIAL INFORMATION")
        print(f"{'='*80}")
        print(f"Title: {trial_data.get('title', 'Unknown')}")
        print(f"Condition: {trial_data.get('condition', 'Unknown')}")
        print(f"Phase: {trial_data.get('phase', 'Unknown')}")
        
        # Get trial description/preview
        trial_description = trial_data.get('detailed_description', '')
        if trial_description:
            print(f"Description: {trial_description[:300]}...")
        
        # Get matched patients
        print(f"\n{'='*80}")
        print("GETTING MATCHED PATIENTS")
        print(f"{'='*80}")
        
        matching_patients = self.trial_matcher.hybrid_search_patients_for_trial(
            trial_data, alpha=0.7
        )
        
        if not matching_patients:
            print("❌ No matching patients found")
            return
        
        print(f"Found {len(matching_patients)} matched patients")
        
        # Validate each match
        print(f"\n{'='*80}")
        print("CLINICAL MATCH VALIDATION")
        print(f"{'='*80}")
        print(f"Trial Condition: {trial_data.get('condition', 'Unknown')}")
        print(f"\nValidating top {min(top_n, len(matching_patients))} matches...\n")
        
        validation_results = []
        
        for i, patient_match in enumerate(matching_patients[:top_n], 1):
            # Debug: print patient_match structure
            print(f"\nDEBUG: patient_match keys: {patient_match.keys()}")
            
            patient_id = patient_match.get('patient_id')
            mrn = patient_match.get('mrn', 'Unknown')
            hybrid_score = patient_match.get('hybrid_score', 0.0)
            
            # If patient_id is None, try to get it from MRN
            if patient_id is None and mrn:
                patient_by_mrn = self.db_utils.get_patient_by_mrn(mrn)
                if patient_by_mrn:
                    patient_id = patient_by_mrn.get('patient_id')
            
            # If still None, try to extract from metadata keys
            if patient_id is None:
                # Check if there's a key that looks like patient_id
                for key in ['id', 'patient_id', 'patient']:
                    if key in patient_match:
                        patient_id = patient_match[key]
                        break
            
            print(f"{'='*80}")
            print(f"Match #{i}")
            print(f"{'='*80}")
            print(f"Patient ID: {patient_id}")
            print(f"MRN: {mrn}")
            print(f"Hybrid Score: {hybrid_score:.4f}")
            print(f"Patient Match Data: {json.dumps(patient_match, indent=2, default=str)[:500]}")
            
            # Get patient medical summary
            if patient_id:
                patient_summary = self.get_patient_medical_summary(patient_id)
            else:
                patient_summary = {}
            
            if not patient_summary or not patient_id:
                print("❌ Could not retrieve patient information")
                validation_results.append({
                    "rank": i,
                    "patient_id": patient_id,
                    "mrn": mrn,
                    "hybrid_score": float(hybrid_score),
                    "match_quality": "ERROR",
                    "match_reason": "Could not retrieve patient information"
                })
                continue
            
            print(f"\nPatient Information:")
            print(f"  Age: {patient_summary.get('age', 'Unknown')}")
            print(f"  Gender: {patient_summary.get('gender', 'Unknown')}")
            print(f"  Chief Complaint: {patient_summary.get('chief_complaint', 'Not available')}")
            print(f"  Diagnosis: {patient_summary.get('diagnosis', 'Not available')}")
            print(f"  Assessment: {patient_summary.get('assessment', 'Not available')[:200]}")
            
            # Compare conditions
            trial_condition = trial_data.get('condition', '').lower()
            patient_diagnosis = patient_summary.get('diagnosis', '').lower()
            patient_complaint = patient_summary.get('chief_complaint', '').lower()
            patient_assessment = patient_summary.get('assessment', '').lower()
            
            # Check for keyword matches
            condition_keywords = set(trial_condition.split())
            diagnosis_keywords = set(patient_diagnosis.split()) if patient_diagnosis else set()
            complaint_keywords = set(patient_complaint.split()) if patient_complaint else set()
            assessment_keywords = set(patient_assessment.split()) if patient_assessment else set()
            
            # Find common keywords
            common_diagnosis = condition_keywords.intersection(diagnosis_keywords)
            common_complaint = condition_keywords.intersection(complaint_keywords)
            common_assessment = condition_keywords.intersection(assessment_keywords)
            
            all_patient_keywords = diagnosis_keywords | complaint_keywords | assessment_keywords
            all_common = condition_keywords.intersection(all_patient_keywords)
            
            print(f"\nCondition Match Analysis:")
            print(f"  Trial Condition: {trial_condition}")
            print(f"  Patient Diagnosis: {patient_diagnosis}")
            print(f"  Common keywords (diagnosis): {len(common_diagnosis)} - {list(common_diagnosis)[:5]}")
            print(f"  Common keywords (complaint): {len(common_complaint)} - {list(common_complaint)[:5]}")
            print(f"  Common keywords (assessment): {len(common_assessment)} - {list(common_assessment)[:5]}")
            print(f"  Total common keywords: {len(all_common)}")
            
            # Determine match quality
            match_quality = "UNKNOWN"
            match_reason = ""
            
            if len(all_common) >= 2:
                match_quality = "LIKELY_MATCH"
                match_reason = f"Found {len(all_common)} common keywords between trial condition and patient medical information"
            elif len(all_common) == 1:
                match_quality = "POSSIBLE_MATCH"
                match_reason = f"Found only {len(all_common)} common keyword - may not be a strong match"
            else:
                match_quality = "UNLIKELY_MATCH"
                match_reason = "No common keywords found between trial condition and patient medical information"
            
            # Check for medical terms that might indicate match
            medical_terms = ['cancer', 'tumor', 'carcinoma', 'neoplasm', 'malignancy', 
                            'metastatic', 'advanced', 'stage', 'treatment', 'therapy']
            
            trial_has_medical = any(term in trial_condition for term in medical_terms)
            patient_has_medical = any(term in patient_diagnosis or term in patient_complaint 
                                     for term in medical_terms)
            
            if trial_has_medical and patient_has_medical:
                if match_quality == "UNLIKELY_MATCH":
                    match_quality = "POSSIBLE_MATCH"
                    match_reason += " (Both have medical terms, but different specific conditions)"
            
            print(f"\nMatch Quality: {match_quality}")
            print(f"Reason: {match_reason}")
            
            validation_results.append({
                "rank": i,
                "patient_id": patient_id,
                "mrn": mrn,
                "hybrid_score": float(hybrid_score),
                "trial_condition": trial_data.get('condition', 'Unknown'),
                "patient_diagnosis": patient_summary.get('diagnosis', 'Not available'),
                "patient_chief_complaint": patient_summary.get('chief_complaint', 'Not available'),
                "common_keywords_count": len(all_common),
                "common_keywords": list(all_common),
                "match_quality": match_quality,
                "match_reason": match_reason,
                "patient_age": patient_summary.get('age', ''),
                "patient_gender": patient_summary.get('gender', '')
            })
        
        # Summary
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        likely_matches = sum(1 for r in validation_results if r['match_quality'] == 'LIKELY_MATCH')
        possible_matches = sum(1 for r in validation_results if r['match_quality'] == 'POSSIBLE_MATCH')
        unlikely_matches = sum(1 for r in validation_results if r['match_quality'] == 'UNLIKELY_MATCH')
        errors = sum(1 for r in validation_results if r['match_quality'] == 'ERROR')
        
        print(f"Total matches validated: {len(validation_results)}")
        print(f"  ✓ Likely matches: {likely_matches} ({likely_matches/len(validation_results)*100:.1f}%)")
        print(f"  ? Possible matches: {possible_matches} ({possible_matches/len(validation_results)*100:.1f}%)")
        print(f"  ✗ Unlikely matches: {unlikely_matches} ({unlikely_matches/len(validation_results)*100:.1f}%)")
        if errors > 0:
            print(f"  ❌ Errors: {errors} ({errors/len(validation_results)*100:.1f}%)")
        
        avg_common_keywords = sum(r['common_keywords_count'] for r in validation_results) / len(validation_results)
        print(f"\nAverage common keywords per match: {avg_common_keywords:.2f}")
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"clinical_match_validation_{trial_id}_{timestamp}.json"
        
        results = {
            "trial_id": trial_id,
            "trial_title": trial_data.get('title', 'Unknown'),
            "trial_condition": trial_data.get('condition', 'Unknown'),
            "validation_timestamp": datetime.now().isoformat(),
            "total_matches": len(matching_patients),
            "validated_count": len(validation_results),
            "summary": {
                "likely_matches": likely_matches,
                "possible_matches": possible_matches,
                "unlikely_matches": unlikely_matches,
                "errors": errors,
                "avg_common_keywords": float(avg_common_keywords)
            },
            "validation_results": validation_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    """Main function"""
    import sys
    
    validator = ClinicalMatchValidator()
    
    # Get a trial ID from database
    db_utils = DatabaseUtils()
    trials = db_utils.get_detailed_trial_data()
    
    if not trials:
        print("❌ No trials found in database")
        return
    
    # Use first trial or specified trial ID
    if len(sys.argv) > 1:
        trial_id = sys.argv[1]
    else:
        trial_id = trials[0].get('trial_id')
    
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Validating matches for trial: {trial_id}")
    print(f"Top N: {top_n}\n")
    
    validator.validate_trial_patient_matches(trial_id, top_n)


if __name__ == "__main__":
    main()

