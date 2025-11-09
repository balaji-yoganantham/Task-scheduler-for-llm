"""
Rule-Based Pre-Filtering for Patient-Trial Matching
Filters out clearly ineligible matches before LLM evaluation
"""

import re
from typing import Dict, Any, Optional, Tuple, List


class RuleBasedFilter:
    """Rule-based pre-filtering for age, gender, and diagnosis"""
    
    def __init__(self):
        # Common age patterns in trial eligibility
        self.age_patterns = [
            r'age\s*(?:≥|>=|greater than or equal to|at least)\s*(\d+)',
            r'age\s*(?:≤|<=|less than or equal to|at most|up to)\s*(\d+)',
            r'age\s*(?:>|greater than|over)\s*(\d+)',
            r'age\s*(?:<|less than|under)\s*(\d+)',
            r'age\s*(\d+)\s*to\s*(\d+)',
            r'age\s*(\d+)\s*-\s*(\d+)',
            r'between\s*(\d+)\s*and\s*(\d+)\s*years',
            r'(\d+)\s*to\s*(\d+)\s*years?\s*old',
        ]
        
        # Gender patterns
        self.gender_patterns = [
            r'gender[:\s]+(male|female|men|women)',
            r'(male|female|men|women)\s*only',
            r'eligible\s+for\s+(male|female|men|women)',
        ]
        
        # Diagnosis patterns
        self.diagnosis_keywords = {
            'colorectal': ['colorectal', 'colon', 'rectal', 'crc', 'colorectal cancer'],
            'breast': ['breast cancer', 'breast carcinoma'],
            'lung': ['lung cancer', 'non-small cell lung', 'nsclc', 'small cell lung', 'sclc'],
            'prostate': ['prostate cancer', 'prostate carcinoma'],
        }
    
    def extract_age_range(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract age range from text"""
        text_lower = text.lower()
        
        # Try to find age range patterns
        for pattern in self.age_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) == 2:
                    try:
                        min_age = int(groups[0])
                        max_age = int(groups[1])
                        return (min_age, max_age)
                    except ValueError:
                        continue
                elif len(groups) == 1:
                    # Single age value - check context
                    age_val = int(groups[0])
                    if '≥' in match.group() or '>=' in match.group() or 'at least' in match.group():
                        return (age_val, 150)  # Upper bound
                    elif '≤' in match.group() or '<=' in match.group() or 'at most' in match.group():
                        return (0, age_val)  # Lower bound
        
        return None
    
    def extract_gender_requirement(self, text: str) -> Optional[str]:
        """Extract gender requirement from text"""
        text_lower = text.lower()
        
        for pattern in self.gender_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                gender = match.group(1).lower()
                if gender in ['male', 'men']:
                    return 'Male'
                elif gender in ['female', 'women']:
                    return 'Female'
        
        return None
    
    def check_diagnosis_match(self, trial_condition: str, patient_diagnosis: str) -> bool:
        """Check if patient diagnosis matches trial condition"""
        trial_lower = trial_condition.lower()
        patient_lower = patient_diagnosis.lower()
        
        # Extract key diagnosis terms
        for key, keywords in self.diagnosis_keywords.items():
            trial_has_key = any(kw in trial_lower for kw in keywords)
            patient_has_key = any(kw in patient_lower for kw in keywords)
            
            if trial_has_key and not patient_has_key:
                return False
            if patient_has_key and not trial_has_key:
                return False
        
        # Basic keyword matching
        trial_words = set(trial_lower.split())
        patient_words = set(patient_lower.split())
        
        # Check for common medical terms
        common_terms = {'cancer', 'carcinoma', 'tumor', 'neoplasm', 'metastatic', 'advanced', 'stage'}
        trial_medical = trial_words & common_terms
        patient_medical = patient_words & common_terms
        
        if trial_medical and not patient_medical:
            return False
        
        return True
    
    def filter_by_age(self, patient_age: int, trial_text: str) -> Tuple[bool, Optional[str]]:
        """
        Filter by age
        Returns: (is_eligible, reason)
        """
        age_range = self.extract_age_range(trial_text)
        
        if age_range:
            min_age, max_age = age_range
            if patient_age < min_age:
                return False, f"Patient age {patient_age} is below minimum age {min_age}"
            if patient_age > max_age:
                return False, f"Patient age {patient_age} is above maximum age {max_age}"
        
        return True, None
    
    def filter_by_gender(self, patient_gender: str, trial_text: str) -> Tuple[bool, Optional[str]]:
        """
        Filter by gender
        Returns: (is_eligible, reason)
        """
        gender_req = self.extract_gender_requirement(trial_text)
        
        if gender_req:
            # Normalize patient gender
            patient_gender_norm = patient_gender.strip().title()
            if patient_gender_norm not in [gender_req, gender_req + 's']:
                return False, f"Trial requires {gender_req}, patient is {patient_gender_norm}"
        
        return True, None
    
    def filter_by_diagnosis(self, patient_diagnosis: str, trial_condition: str) -> Tuple[bool, Optional[str]]:
        """
        Filter by diagnosis
        Returns: (is_eligible, reason)
        """
        if not self.check_diagnosis_match(trial_condition, patient_diagnosis):
            return False, f"Patient diagnosis '{patient_diagnosis}' does not match trial condition '{trial_condition}'"
        
        return True, None
    
    def apply_rule_based_filter(
        self, 
        patient_data: Dict[str, Any], 
        trial_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Apply all rule-based filters
        Returns: (is_eligible, reason, filter_results)
        """
        filter_results = {
            'age_check': {'passed': True, 'reason': None},
            'gender_check': {'passed': True, 'reason': None},
            'diagnosis_check': {'passed': True, 'reason': None},
        }
        
        # Get patient info
        patient_age = patient_data.get('age')
        patient_gender = patient_data.get('gender', '')
        patient_diagnosis = patient_data.get('combined_text', '')  # Will check in combined text
        
        # Get trial info
        trial_condition = trial_data.get('condition', '')
        trial_text = f"{trial_data.get('title', '')} {trial_data.get('condition', '')} {trial_data.get('description', '')}"
        
        # Check age
        if patient_age:
            age_eligible, age_reason = self.filter_by_age(patient_age, trial_text)
            filter_results['age_check'] = {
                'passed': age_eligible,
                'reason': age_reason
            }
            if not age_eligible:
                return False, age_reason, filter_results
        
        # Check gender
        if patient_gender:
            gender_eligible, gender_reason = self.filter_by_gender(patient_gender, trial_text)
            filter_results['gender_check'] = {
                'passed': gender_eligible,
                'reason': gender_reason
            }
            if not gender_eligible:
                return False, gender_reason, filter_results
        
        # Check diagnosis
        if trial_condition and patient_diagnosis:
            diagnosis_eligible, diagnosis_reason = self.filter_by_diagnosis(patient_diagnosis, trial_condition)
            filter_results['diagnosis_check'] = {
                'passed': diagnosis_eligible,
                'reason': diagnosis_reason
            }
            if not diagnosis_eligible:
                return False, diagnosis_reason, filter_results
        
        return True, None, filter_results

