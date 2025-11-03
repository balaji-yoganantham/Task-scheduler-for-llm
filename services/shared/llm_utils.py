"""
Shared LLM Utilities
Common LLM API calls and prompt templates for both patient-to-trial and trial-to-patient matching
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import google.generativeai as genai
from config import GEMINI_API_KEY

class LLMUtils:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def generate_keywords_prompt(self, patient_data: Dict[str, Any]) -> str:
        """Generate prompt for patient keyword extraction"""
        return f"""
        You are a medical research assistant specializing in clinical trial matching. 
        Analyze the following patient medical record and generate comprehensive keywords for clinical trial matching.

        Patient Data:
        {patient_data['combined_text']}

        Instructions:
        1. Extract ALL relevant medical conditions, symptoms, treatments, and characteristics
        2. Generate at least 30-40 keywords covering:
           - Primary diagnosis and staging
           - Metastatic sites
           - Molecular markers
           - Comorbidities
           - Medications
           - Allergies
           - Performance status
           - Family history
           - Demographics
        3. Use standardized medical terminology
        4. Include both specific and general terms for better matching

        Return ONLY a valid JSON object with this structure:
        {{
            "patient_id": {patient_data['patient_id']},
            "mrn": "{patient_data['mrn']}",
            "age": {patient_data['age']},
            "gender": "{patient_data['gender']}",
            "summary": "Brief 2-3 sentence summary of patient's condition",
            "primary_diagnosis": "Main diagnosis",
            "stage": "Cancer stage if applicable",
            "metastatic_sites": ["site1", "site2", ...],
            "molecular_markers": ["marker1", "marker2", ...],
            "comorbidities": ["condition1", "condition2", ...],
            "medications": ["med1", "med2", ...],
            "allergies": ["allergy1", "allergy2", ...],
            "performance_status": "ECOG status",
            "family_history": ["condition1", "condition2", ...],
            "keywords": ["keyword1", "keyword2", "keyword3", ...],
            "generated_at": "{datetime.now().isoformat()}"
        }}
        """

    def patient_trial_evaluation_prompt(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
        """Generate prompt for patient-trial eligibility evaluation"""
        return f"""
        You are an expert clinical trial coordinator specializing in patient-trial matching for oncology trials.
        
        Evaluate the eligibility of this patient for the given clinical trial.
        
        CLINICAL TRIAL INFORMATION:
        Title: {trial_info['title']}
        Condition: {trial_info['condition']}
        Phase: {trial_info['phase']}
        Status: {trial_info['status']}
        Investigator: {trial_info['investigator']}
        
        PATIENT INFORMATION:
        MRN: {patient_info['mrn']}
        Age: {patient_info['age']}
        Gender: {patient_info['gender']}
        Oncologist: {patient_info['oncologist']}
        Date of Visit: {patient_info['date_of_visit']}
        
        PATIENT MEDICAL RECORD:
        {patient_info['combined_text']}
        
        EVALUATION CRITERIA:
        1. Primary diagnosis match
        2. Disease stage compatibility
        3. Age eligibility
        4. Gender eligibility
        5. Performance status
        6. Prior treatments
        7. Comorbidities
        8. Laboratory values
        9. Exclusion criteria
        10. Overall eligibility assessment
        
        Provide a comprehensive evaluation with:
        - Eligibility status (ELIGIBLE, NOT_ELIGIBLE, NEED_MORE_INFO)
        - Confidence score (0-100)
        - Detailed reasoning
        - Specific inclusion/exclusion criteria met or not met
        - Recommendations for next steps
        
        Return ONLY a valid JSON object with this structure:
        {{
            "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
            "confidence_score": 85,
            "reasoning": "Detailed explanation of eligibility assessment",
            "inclusion_criteria_met": ["criteria1", "criteria2", ...],
            "exclusion_criteria_violated": ["criteria1", "criteria2", ...],
            "recommendations": "Specific recommendations for next steps",
            "priority_score": 75,
            "evaluation_timestamp": "{datetime.now().isoformat()}"
        }}
        """

    def trial_patient_evaluation_prompt(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
        """Generate prompt for trial-patient eligibility evaluation"""
        return f"""
        You are an expert clinical trial coordinator specializing in patient-trial matching for oncology trials.
        
        Evaluate the eligibility of this patient for the given clinical trial.
        
        CLINICAL TRIAL INFORMATION:
        Title: {trial_info['title']}
        Condition: {trial_info['condition']}
        Phase: {trial_info['phase']}
        Status: {trial_info['status']}
        Investigator: {trial_info['investigator']}
        
        PATIENT INFORMATION:
        MRN: {patient_info['mrn']}
        Age: {patient_info['age']}
        Gender: {patient_info['gender']}
        Oncologist: {patient_info['oncologist']}
        Date of Visit: {patient_info['date_of_visit']}
        
        PATIENT MEDICAL RECORD:
        {patient_info['combined_text']}
        
        EVALUATION CRITERIA:
        1. Primary diagnosis match
        2. Disease stage compatibility
        3. Age eligibility
        4. Gender eligibility
        5. Performance status
        6. Prior treatments
        7. Comorbidities
        8. Laboratory values
        9. Exclusion criteria
        10. Overall eligibility assessment
        
        Provide a comprehensive evaluation with:
        - Eligibility status (ELIGIBLE, NOT_ELIGIBLE, NEED_MORE_INFO)
        - Confidence score (0-100)
        - Detailed reasoning
        - Specific inclusion/exclusion criteria met or not met
        - Recommendations for next steps
        
        Return ONLY a valid JSON object with this structure:
        {{
            "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
            "confidence_score": 85,
            "reasoning": "Detailed explanation of eligibility assessment",
            "inclusion_criteria_met": ["criteria1", "criteria2", ...],
            "exclusion_criteria_violated": ["criteria1", "criteria2", ...],
            "recommendations": "Specific recommendations for next steps",
            "priority_score": 75,
            "evaluation_timestamp": "{datetime.now().isoformat()}"
        }}
        """

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Make LLM API call and parse response"""
        try:
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return {
                    "error": "Empty response from LLM",
                    "raw_response": ""
                }
            
            try:
                # Clean the response - remove code block markers (more thorough)
                response_text = response.text.strip()
                
                # Remove markdown code blocks (handle various formats)
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                
                # Remove trailing code block markers
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                elif response_text.rstrip().endswith("```"):
                    response_text = response_text.rstrip()[:-3]
                
                response_text = response_text.strip()
                
                # Try to extract JSON if there's extra text before/after
                # Look for the first '{' and last '}'
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    response_text = response_text[first_brace:last_brace+1]
                
                # Parse JSON
                result = json.loads(response_text)
                return {"response": response.text, **result}
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {str(e)}")
                print(f"JSON Error at line {e.lineno}, column {e.colno}")
                print(f"Error message: {e.msg}")
                print(f"Response text (first 1000 chars): {response.text[:1000]}")
                print(f"Response text length: {len(response.text)}")
                # Save full response to file for debugging
                try:
                    import os
                    debug_dir = Path("results/debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_file = debug_dir / f"llm_response_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write("=== PROMPT ===\n")
                        f.write(prompt)
                        f.write("\n\n=== RESPONSE ===\n")
                        f.write(response.text)
                        f.write("\n\n=== ERROR ===\n")
                        f.write(str(e))
                    print(f"💾 Full response saved to: {debug_file}")
                except Exception as save_error:
                    print(f"Could not save debug file: {save_error}")
                
                return {
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": response.text,
                    "error_details": {
                        "line": e.lineno,
                        "column": e.colno,
                        "message": e.msg
                    }
                }
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e)
            }

    def generate_keywords_for_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate keywords for a single patient"""
        prompt = self.generate_keywords_prompt(patient_data)
        result = self.call_llm(prompt)
        
        if "error" not in result:
            result["patient_id"] = patient_data['patient_id']
            result["mrn"] = patient_data['mrn']
        
        return result

    def generate_keywords_for_patient_batch(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate keywords for multiple patients in one batch with size limit"""
        try:
            print(f"Processing {len(patients)} patients in single batch...")
            
            # Handle empty list case
            if not patients:
                return {
                    "successful": {},
                    "failed": {},
                    "metadata": {
                        "total_patients": 0,
                        "generated_at": datetime.now().isoformat(),
                        "model": "gemini-2.0-flash-exp",
                        "batch_processing": "single_batch_all_patients",
                        "note": "No patients provided"
                    }
                }
            
            # Limit batch size to prevent JSON truncation
            MAX_BATCH_SIZE = 10
            if len(patients) > MAX_BATCH_SIZE:
                print(f"⚠️ Batch size {len(patients)} exceeds limit of {MAX_BATCH_SIZE}. Processing first {MAX_BATCH_SIZE} patients.")
                patients = patients[:MAX_BATCH_SIZE]
            
            # Create batch prompt
            batch_prompt = self.generate_batch_keywords_prompt(patients)
            
            # Call LLM with batch prompt
            result = self.call_llm(batch_prompt)
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Parse batch results
            return self.parse_batch_keywords_result(result, patients)
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return {"error": str(e)}

    def generate_batch_keywords_prompt(self, patients: List[Dict[str, Any]]) -> str:
        """Generate prompt for batch patient keyword extraction"""
        if not patients:
            return "No patients provided for keyword generation."
        
        patients_text = ""
        
        for i, patient in enumerate(patients, 1):
            patients_text += f"\n{'='*50}\nPATIENT {i}:\n{'='*50}\n"
            patients_text += f"Patient ID: {patient['patient_id']}\n"
            patients_text += f"MRN: {patient['mrn']}\n"
            patients_text += f"Age: {patient['age']}\n"
            patients_text += f"Gender: {patient['gender']}\n"
            patients_text += f"Combined Medical Record:\n{patient['combined_text']}\n"
        
        return f"""
        You are a medical research assistant specializing in clinical trial matching. 
        Analyze the following {len(patients)} patient medical records and generate comprehensive keywords for clinical trial matching.

        PATIENTS DATA:
        {patients_text}

        Instructions:
        1. Extract ALL relevant medical conditions, symptoms, treatments, and characteristics for EACH patient
        2. Generate at least 30-40 keywords per patient covering:
           - Primary diagnosis and staging
           - Metastatic sites
           - Molecular markers
           - Comorbidities
           - Medications
           - Allergies
           - Performance status
           - Family history
           - Demographics
        3. Use standardized medical terminology
        4. Include both specific and general terms for better matching

        Return ONLY a valid JSON object with this structure:
        {{
            "batch_results": [
                {{
                    "patient_id": {patients[0]['patient_id']},
                    "mrn": "{patients[0]['mrn']}",
                    "age": {patients[0]['age']},
                    "gender": "{patients[0]['gender']}",
                    "summary": "Brief 2-3 sentence summary of patient's condition",
                    "primary_diagnosis": "Main diagnosis",
                    "stage": "Cancer stage if applicable",
                    "metastatic_sites": ["site1", "site2", ...],
                    "molecular_markers": ["marker1", "marker2", ...],
                    "comorbidities": ["condition1", "condition2", ...],
                    "medications": ["med1", "med2", ...],
                    "allergies": ["allergy1", "allergy2", ...],
                    "performance_status": "ECOG status",
                    "family_history": ["condition1", "condition2", ...],
                    "keywords": ["keyword1", "keyword2", "keyword3", ...],
                    "generated_at": "{datetime.now().isoformat()}"
                }}{f''',
                {{
                    "patient_id": {patients[1]['patient_id']},
                    "mrn": "{patients[1]['mrn']}",
                    "age": {patients[1]['age']},
                    "gender": "{patients[1]['gender']}",
                    "summary": "Brief 2-3 sentence summary of patient's condition",
                    "primary_diagnosis": "Main diagnosis",
                    "stage": "Cancer stage if applicable",
                    "metastatic_sites": ["site1", "site2", ...],
                    "molecular_markers": ["marker1", "marker2", ...],
                    "comorbidities": ["condition1", "condition2", ...],
                    "medications": ["med1", "med2", ...],
                    "allergies": ["allergy1", "allergy2", ...],
                    "performance_status": "ECOG status",
                    "family_history": ["condition1", "condition2", ...],
                    "keywords": ["keyword1", "keyword2", "keyword3", ...],
                    "generated_at": "{datetime.now().isoformat()}"
                }}''' if len(patients) > 1 else ''}
                // ... continue for all {len(patients)} patients
            ],
            "batch_metadata": {{
                "total_patients": {len(patients)},
                "generated_at": "{datetime.now().isoformat()}",
                "model": "gemini-2.0-flash-exp",
                "batch_processing": true
            }}
        }}
        """

    def parse_batch_keywords_result(self, result: Dict[str, Any], patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse batch keywords result with improved error handling"""
        try:
            response = result.get("response", "")
            
            # Clean the response - remove code block markers
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Try to parse JSON
            try:
                parsed_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                print(f"Response length: {len(response)}")
                print(f"Response preview (first 1000 chars): {response[:1000]}")
                print(f"Response preview (last 1000 chars): {response[-1000:]}")
                
                # Try to fix common JSON issues
                response = self._fix_json_response(response)
                try:
                    parsed_data = json.loads(response)
                except json.JSONDecodeError as e2:
                    print(f"JSON fix failed: {str(e2)}")
                    return {"error": f"Failed to parse JSON response: {str(e)}"}
            
            batch_results = parsed_data.get("batch_results", [])
            
            print(f"Successfully parsed JSON with {len(batch_results)} batch results")
            
            successful = {}
            failed = {}
            
            for i, patient_result in enumerate(batch_results):
                if i < len(patients):
                    patient_id = patients[i]['patient_id']
                    
                    if patient_result and "error" not in patient_result:
                        successful[patient_id] = patient_result
                    else:
                        failed[patient_id] = {
                            "patient_id": patient_id,
                            "error": patient_result.get("error", "Unknown error")
                        }
            
            print(f"Processed {len(successful)} successful and {len(failed)} failed results")
            
            return {
                "successful": successful,
                "failed": failed,
                "metadata": {
                    "total_patients": len(patients),
                    "generated_at": datetime.now().isoformat(),
                    "model": "gemini-2.0-flash-exp",
                    "batch_processing": "single_batch_all_patients",
                    "batch_metadata": parsed_data.get("batch_metadata", {})
                }
            }
            
        except Exception as e:
            print(f"Error parsing batch results: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to parse batch results: {str(e)}"}

    def _fix_json_response(self, response: str) -> str:
        """Try to fix common JSON issues"""
        try:
            # Remove any trailing incomplete content
            if response.count('{') > response.count('}'):
                # Find the last complete object
                brace_count = 0
                last_complete_pos = -1
                for i, char in enumerate(response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_complete_pos = i
                
                if last_complete_pos > 0:
                    response = response[:last_complete_pos + 1]
            
            # Try to complete incomplete strings
            if response.count('"') % 2 != 0:
                response += '"'
            
            return response
        except Exception as e:
            print(f"Error fixing JSON: {e}")
            return response

    def evaluate_patient_trial_match(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate patient-trial match"""
        prompt = self.patient_trial_evaluation_prompt(trial_info, patient_info)
        result = self.call_llm(prompt)
        
        if "error" not in result:
            result["patient_info"] = {
                "patient_id": patient_info['patient_id'],
                "mrn": patient_info['mrn'],
                "age": patient_info['age'],
                "gender": patient_info['gender'],
                "oncologist": patient_info['oncologist']
            }
            result["trial_info"] = {
                "trial_id": trial_info['trial_id'],
                "title": trial_info['title'],
                "condition": trial_info['condition'],
                "phase": trial_info['phase']
            }
        
        return result

    def evaluate_trial_patient_match(self, trial_info: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trial-patient match"""
        prompt = self.trial_patient_evaluation_prompt(trial_info, patient_info)
        result = self.call_llm(prompt)
        
        if "error" not in result:
            result["patient_info"] = {
                "patient_id": patient_info['patient_id'],
                "mrn": patient_info['mrn'],
                "age": patient_info['age'],
                "gender": patient_info['gender'],
                "oncologist": patient_info['oncologist']
            }
            result["trial_info"] = {
                "trial_id": trial_info['trial_id'],
                "title": trial_info['title'],
                "condition": trial_info['condition'],
                "phase": trial_info['phase']
            }
        
        return result

    def evaluate_patient_trial_matches_batch(self, trials: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multiple trials for a patient in one batch"""
        try:
            prompt = self.batch_patient_trial_evaluation_prompt(trials, patient_info)
            result = self.call_llm(prompt)
            
            if "error" not in result:
                # Parse the batch results
                parsed_results = self.parse_batch_evaluation_result(result.get("response", ""), trials, patient_info)
                return parsed_results
            else:
                return result
                
        except Exception as e:
            return {
                "error": f"Batch evaluation failed: {str(e)}",
                "trials_count": len(trials),
                "patient_id": patient_info.get('patient_id', 'Unknown')
            }

    def batch_patient_trial_evaluation_prompt(self, trials: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> str:
        """Generate prompt for batch patient-trial evaluation"""
        
        # Patient information
        patient_text = f"""
        PATIENT INFORMATION:
        Patient ID: {patient_info['patient_id']}
        MRN: {patient_info['mrn']}
        Age: {patient_info['age']}
        Gender: {patient_info['gender']}
        Oncologist: {patient_info['oncologist']}
        Combined Medical Record: {patient_info['combined_text']}
        """
        
        # Trials information
        trials_text = ""
        for i, trial in enumerate(trials, 1):
            trials_text += f"""
        {'='*60}
        TRIAL {i}:
        {'='*60}
        Trial ID: {trial.get('trial_id', 'Unknown')}
        Title: {trial.get('title', 'Unknown')}
        Condition: {trial.get('condition', 'Unknown')}
        Phase: {trial.get('phase', 'Unknown')}
        Status: {trial.get('status', 'Unknown')}
        Investigator: {trial.get('investigator', 'Unknown')}
        Age Range: {trial.get('minimum_age', 'Not specified')} - {trial.get('maximum_age', 'Not specified')}
        Gender: {trial.get('sex', 'Not specified')}
        Hybrid Score: {trial.get('hybrid_score', 0):.3f}
        
        Brief Summary: {trial.get('brief_summary', 'Not available')}
        
        Detailed Description: {trial.get('detailed_description', 'Not available')}
        
        Inclusion Criteria: {trial.get('inclusion_criteria', 'Not available')}
        
        Exclusion Criteria: {trial.get('exclusion_criteria', 'Not available')}
        
        Eligibility Criteria: {trial.get('eligibility_criteria', 'Not available')}
        """
        
        return f"""
        You are a medical research assistant specializing in clinical trial matching. 
        Evaluate the following {len(trials)} clinical trials for the given patient and determine eligibility.

        {patient_text}

        CLINICAL TRIALS TO EVALUATE:
        {trials_text}

        Instructions:
        1. For EACH trial, evaluate the patient's eligibility based on:
           - Medical condition match
           - Age requirements
           - Gender requirements
           - Inclusion/exclusion criteria
           - Overall medical compatibility
        2. Provide detailed reasoning for each evaluation
        3. Consider the patient's medical history, current condition, and trial requirements
        4. Be thorough but concise in your analysis
        5. IMPORTANT: Return ONLY valid JSON - no markdown code blocks, no ```json markers, no extra text
        6. Ensure all strings use double quotes and escape special characters properly
        7. Make sure all brackets and braces are properly closed

        Return ONLY a valid JSON object (no markdown formatting) with this structure:
        {{
            "batch_evaluations": [
                {{
                    "trial_id": "{trials[0].get('trial_id', 'Unknown')}",
                    "trial_title": "{trials[0].get('title', 'Unknown')}",
                    "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
                    "confidence_score": 85,
                    "priority_score": 90,
                    "reasoning": "Detailed explanation of eligibility decision",
                    "key_criteria_met": ["criteria1", "criteria2", "criteria3"],
                    "key_criteria_missed": ["criteria1", "criteria2"],
                    "recommendations": "Specific recommendations for this trial"
                }},
                {{
                    "trial_id": "{trials[1].get('trial_id', 'Unknown') if len(trials) > 1 else 'N/A'}",
                    "trial_title": "{trials[1].get('title', 'Unknown') if len(trials) > 1 else 'N/A'}",
                    "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
                    "confidence_score": 75,
                    "priority_score": 80,
                    "reasoning": "Detailed explanation of eligibility decision",
                    "key_criteria_met": ["criteria1", "criteria2"],
                    "key_criteria_missed": ["criteria1", "criteria2", "criteria3"],
                    "recommendations": "Specific recommendations for this trial"
                }}
                // ... continue for all {len(trials)} trials
            ],
            "batch_summary": {{
                "total_trials_evaluated": {len(trials)},
                "eligible_count": 5,
                "not_eligible_count": 10,
                "need_more_info_count": 5,
                "average_confidence": 78.5,
                "top_recommendations": [
                    "Trial 1 - Best match due to...",
                    "Trial 2 - Good alternative because...",
                    "Trial 3 - Consider if..."
                ],
                "generated_at": "{datetime.now().isoformat()}"
            }}
        }}
        """

    def parse_batch_evaluation_result(self, response: str, trials: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse batch evaluation result from LLM response"""
        try:
            # Clean the response more thoroughly
            response = response.strip()
            
            # Remove markdown code blocks (handle various formats)
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            
            # Remove trailing code block markers
            if response.endswith("```"):
                response = response[:-3]
            elif response.rstrip().endswith("```"):
                response = response.rstrip()[:-3]
            
            response = response.strip()
            
            # Try to extract JSON if there's extra text before/after
            # Look for the first '{' and last '}'
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                response = response[first_brace:last_brace+1]
            
            # Parse JSON with better error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as json_error:
                # Try to fix common JSON issues
                # Fix unescaped quotes in strings (basic attempt)
                import re
                # Try to find and fix incomplete JSON (missing closing braces)
                open_braces = response.count('{')
                close_braces = response.count('}')
                if open_braces > close_braces:
                    # Add missing closing braces
                    response += '}' * (open_braces - close_braces)
                    try:
                        result = json.loads(response)
                    except json.JSONDecodeError:
                        # If still failing, try to extract just the batch_evaluations array
                        evaluations_match = re.search(r'"batch_evaluations"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                        if evaluations_match:
                            # Try to parse just the array
                            try:
                                # Reconstruct minimal JSON
                                array_content = evaluations_match.group(1)
                                # Try to parse as array
                                # This is a fallback - we'll build a minimal structure
                                print(f"⚠️ Attempting to extract JSON from malformed response...")
                                print(f"JSON Error at line {json_error.lineno}, column {json_error.colno}")
                                print(f"Error message: {json_error.msg}")
                                raise json_error
                            except:
                                raise json_error
                        else:
                            raise json_error
                else:
                    raise json_error
            
            # Validate structure
            if "batch_evaluations" not in result:
                return {"error": "Invalid response format: missing batch_evaluations"}
            
            evaluations = result["batch_evaluations"]
            
            # Handle count mismatch - LLM sometimes returns more or fewer evaluations
            if len(evaluations) != len(trials):
                # Log warning but continue processing
                if len(evaluations) > len(trials):
                    # Truncate to expected number if we got more
                    print(f"⚠️ LLM returned {len(evaluations)} evaluations but expected {len(trials)}. Truncating to {len(trials)}.")
                    evaluations = evaluations[:len(trials)]
                else:
                    # If we got fewer, we'll process what we have but log the issue
                    print(f"⚠️ LLM returned {len(evaluations)} evaluations but expected {len(trials)}. Processing {len(evaluations)} evaluations.")
            
            # Add trial and patient info to each evaluation
            for i, evaluation in enumerate(evaluations):
                if i < len(trials):
                    evaluation['trial_info'] = {
                        "trial_id": trials[i].get('trial_id', 'Unknown'),
                        "title": trials[i].get('title', 'Unknown'),
                        "condition": trials[i].get('condition', 'Unknown'),
                        "phase": trials[i].get('phase', 'Unknown'),
                        "status": trials[i].get('status', 'Unknown')
                    }
                    evaluation['patient_info'] = {
                        "patient_id": patient_info['patient_id'],
                        "mrn": patient_info['mrn'],
                        "age": patient_info['age'],
                        "gender": patient_info['gender'],
                        "oncologist": patient_info['oncologist']
                    }
                    evaluation['hybrid_score'] = trials[i].get('hybrid_score', 0)
            
            return {
                "success": True,
                "evaluations": evaluations,
                "batch_summary": result.get("batch_summary", {}),
                "total_evaluations": len(evaluations),
                "generated_at": datetime.now().isoformat()
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   JSON Error at line {e.lineno}, column {e.colno}")
            print(f"   Error message: {e.msg}")
            print(f"   Response length: {len(response)} characters")
            print(f"   First 500 chars: {response[:500]}")
            print(f"   Last 500 chars: {response[-500:]}")
            # Save debug file
            try:
                debug_dir = Path("results/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"batch_evaluation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("=== PARSED RESPONSE (cleaned) ===\n")
                    f.write(response)
                    f.write("\n\n=== ERROR DETAILS ===\n")
                    f.write(f"Line: {e.lineno}, Column: {e.colno}\n")
                    f.write(f"Message: {e.msg}\n")
                    f.write(f"Error: {str(e)}\n")
                print(f"💾 Debug file saved to: {debug_file}")
            except Exception as save_error:
                print(f"Could not save debug file: {save_error}")
            return {
                "error": error_msg,
                "error_details": {
                    "line": e.lineno,
                    "column": e.colno,
                    "message": e.msg
                },
                "raw_response_preview": response[:1000] if len(response) > 1000 else response
            }
        except Exception as e:
            import traceback
            error_msg = f"Failed to parse batch evaluation result: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {"error": error_msg}

    def evaluate_trial_patient_matches_batch(self, patients: List[Dict[str, Any]], trial_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multiple patients for a trial in one batch"""
        try:
            prompt = self.batch_trial_patient_evaluation_prompt(patients, trial_info)
            result = self.call_llm(prompt)
            
            if "error" not in result:
                # Parse the batch results
                parsed_results = self.parse_batch_trial_patient_evaluation_result(result.get("response", ""), patients, trial_info)
                return parsed_results
            else:
                return result
                
        except Exception as e:
            return {
                "error": f"Batch evaluation failed: {str(e)}",
                "patients_count": len(patients),
                "trial_id": trial_info.get('trial_id', 'Unknown')
            }

    def batch_trial_patient_evaluation_prompt(self, patients: List[Dict[str, Any]], trial_info: Dict[str, Any]) -> str:
        """Generate prompt for batch trial-patient evaluation"""
        
        # Trial information
        trial_text = f"""
        CLINICAL TRIAL INFORMATION:
        Trial ID: {trial_info['trial_id']}
        Title: {trial_info['title']}
        Condition: {trial_info['condition']}
        Phase: {trial_info['phase']}
        Status: {trial_info['status']}
        Investigator: {trial_info['investigator']}
        Age Range: {trial_info.get('minimum_age', 'Not specified')} - {trial_info.get('maximum_age', 'Not specified')}
        Gender: {trial_info.get('sex', 'Not specified')}
        
        Brief Summary: {trial_info.get('brief_summary', 'Not available')}
        
        Detailed Description: {trial_info.get('detailed_description', 'Not available')}
        
        Inclusion Criteria: {trial_info.get('inclusion_criteria', 'Not available')}
        
        Exclusion Criteria: {trial_info.get('exclusion_criteria', 'Not available')}
        
        Eligibility Criteria: {trial_info.get('eligibility_criteria', 'Not available')}
        """
        
        # Patients information
        patients_text = ""
        for i, patient in enumerate(patients, 1):
            patients_text += f"""
        {'='*60}
        PATIENT {i}:
        {'='*60}
        Patient ID: {patient.get('patient_id', 'Unknown')}
        MRN: {patient.get('mrn', 'Unknown')}
        Age: {patient.get('age', 'Unknown')}
        Gender: {patient.get('gender', 'Unknown')}
        Oncologist: {patient.get('oncologist', 'Unknown')}
        Hybrid Score: {patient.get('hybrid_score', 0):.3f}
        
        Combined Medical Record: {patient.get('combined_text', 'Not available')}
        """
        
        return f"""
        You are a medical research assistant specializing in clinical trial matching. 
        Evaluate the following {len(patients)} patients for the given clinical trial and determine eligibility.

        {trial_text}

        PATIENTS TO EVALUATE:
        {patients_text}

        Instructions:
        1. For EACH patient, evaluate their eligibility for the trial based on:
           - Medical condition match
           - Age requirements
           - Gender requirements
           - Inclusion/exclusion criteria
           - Overall medical compatibility
        2. Provide detailed reasoning for each evaluation
        3. Consider each patient's medical history, current condition, and trial requirements
        4. Be thorough but concise in your analysis

        Return ONLY a valid JSON object with this structure:
        {{
            "batch_evaluations": [
                {{
                    "patient_id": {patients[0].get('patient_id', 'Unknown')},
                    "mrn": "{patients[0].get('mrn', 'Unknown')}",
                    "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
                    "confidence_score": 85,
                    "priority_score": 90,
                    "reasoning": "Detailed explanation of eligibility decision",
                    "key_criteria_met": ["criteria1", "criteria2", "criteria3"],
                    "key_criteria_missed": ["criteria1", "criteria2"],
                    "recommendations": "Specific recommendations for this patient"
                }},
                {{
                    "patient_id": {patients[1].get('patient_id', 'Unknown') if len(patients) > 1 else 'N/A'},
                    "mrn": "{patients[1].get('mrn', 'Unknown') if len(patients) > 1 else 'N/A'}",
                    "eligibility_status": "ELIGIBLE|NOT_ELIGIBLE|NEED_MORE_INFO",
                    "confidence_score": 75,
                    "priority_score": 80,
                    "reasoning": "Detailed explanation of eligibility decision",
                    "key_criteria_met": ["criteria1", "criteria2"],
                    "key_criteria_missed": ["criteria1", "criteria2", "criteria3"],
                    "recommendations": "Specific recommendations for this patient"
                }}
                // ... continue for all {len(patients)} patients
            ],
            "batch_summary": {{
                "total_patients_evaluated": {len(patients)},
                "eligible_count": 5,
                "not_eligible_count": 10,
                "need_more_info_count": 5,
                "average_confidence": 78.5,
                "top_recommendations": [
                    "Patient 1 - Best match due to...",
                    "Patient 2 - Good alternative because...",
                    "Patient 3 - Consider if..."
                ],
                "generated_at": "{datetime.now().isoformat()}"
            }}
        }}
        """

    def parse_batch_trial_patient_evaluation_result(self, response: str, patients: List[Dict[str, Any]], trial_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse batch trial-patient evaluation result from LLM response"""
        try:
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            result = json.loads(response)
            
            # Validate structure
            if "batch_evaluations" not in result:
                return {"error": "Invalid response format: missing batch_evaluations"}
            
            evaluations = result["batch_evaluations"]
            
            # Handle count mismatch - LLM sometimes returns more or fewer evaluations
            if len(evaluations) != len(patients):
                # Log warning but continue processing
                if len(evaluations) > len(patients):
                    # Truncate to expected number if we got more
                    print(f"⚠️ LLM returned {len(evaluations)} evaluations but expected {len(patients)}. Truncating to {len(patients)}.")
                    evaluations = evaluations[:len(patients)]
                else:
                    # If we got fewer, we'll process what we have but log the issue
                    print(f"⚠️ LLM returned {len(evaluations)} evaluations but expected {len(patients)}. Processing {len(evaluations)} evaluations.")
            
            # Add patient and trial info to each evaluation
            for i, evaluation in enumerate(evaluations):
                if i < len(patients):
                    evaluation['patient_info'] = {
                        "patient_id": patients[i].get('patient_id', 'Unknown'),
                        "mrn": patients[i].get('mrn', 'Unknown'),
                        "age": patients[i].get('age', 'Unknown'),
                        "gender": patients[i].get('gender', 'Unknown'),
                        "oncologist": patients[i].get('oncologist', 'Unknown')
                    }
                    evaluation['trial_info'] = {
                        "trial_id": trial_info['trial_id'],
                        "title": trial_info['title'],
                        "condition": trial_info['condition'],
                        "phase": trial_info['phase'],
                        "status": trial_info['status']
                    }
                    evaluation['hybrid_score'] = patients[i].get('hybrid_score', 0)
            
            return {
                "success": True,
                "evaluations": evaluations,
                "batch_summary": result.get("batch_summary", {}),
                "total_evaluations": len(evaluations),
                "generated_at": datetime.now().isoformat()
            }
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to parse batch evaluation result: {str(e)}"}
