"""
Evaluation Results Database Utilities
Handles saving LLM evaluation results to PostgreSQL database
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

class EvaluationResultsDB:
    """Database utility class for saving evaluation results"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
    
    def save_patient_to_trial_evaluation(self, evaluation_data: Dict[str, Any]) -> Optional[int]:
        """
        Save patient-to-trial evaluation results to database
        Updates existing record if found, otherwise inserts new record
        
        Args:
            evaluation_data: Dictionary containing evaluation results
            
        Returns:
            int: ID of the inserted/updated record, or None if failed
        """
        try:
            # Extract data from evaluation results
            patient_info = evaluation_data.get('patient_info', {})
            llm_evaluation = evaluation_data.get('llm_evaluation', {})
            hybrid_matching = evaluation_data.get('hybrid_matching', {})
            summary = evaluation_data.get('summary', {})
            
            patient_id = patient_info.get('patient_id')
            
            # Prepare data for insertion/update
            insert_data = {
                'patient_id': patient_id,
                'patient_mrn': patient_info.get('mrn'),
                'evaluation_timestamp': datetime.now(),
                'patient_age': patient_info.get('age'),
                'patient_gender': patient_info.get('gender'),
                'patient_oncologist': patient_info.get('oncologist'),
                'total_trials_found': summary.get('total_trials_found', 0),
                'total_trials_evaluated': summary.get('trials_evaluated', 0),
                'evaluation_method': llm_evaluation.get('evaluation_method', 'batch'),
                'batch_summary': json.dumps(llm_evaluation.get('batch_summary', {})),
                'eligible_trials_count': summary.get('eligible_trials', 0),
                'not_eligible_trials_count': llm_evaluation.get('summary', {}).get('not_eligible_count', 0),
                'need_more_info_trials_count': llm_evaluation.get('summary', {}).get('need_more_info_count', 0),
                'average_confidence_score': summary.get('average_confidence', 0.0),
                'trial_evaluations': json.dumps(llm_evaluation.get('evaluations', []))
            }
            
            # Check if record exists for this patient_id (get most recent)
            with self.engine.connect() as connection:
                # Check for existing record
                check_query = text("""
                    SELECT id FROM insightsedge.patient_to_trial_evaluations
                    WHERE patient_id = :patient_id
                    ORDER BY evaluation_timestamp DESC, id DESC
                    LIMIT 1
                """)
                
                result = connection.execute(check_query, {'patient_id': patient_id})
                existing_record = result.fetchone()
                
                if existing_record:
                    # Update existing record
                    evaluation_id = existing_record[0]
                    update_query = text("""
                        UPDATE insightsedge.patient_to_trial_evaluations SET
                            patient_mrn = :patient_mrn,
                            evaluation_timestamp = :evaluation_timestamp,
                            patient_age = :patient_age,
                            patient_gender = :patient_gender,
                            patient_oncologist = :patient_oncologist,
                            total_trials_found = :total_trials_found,
                            total_trials_evaluated = :total_trials_evaluated,
                            evaluation_method = :evaluation_method,
                            batch_summary = :batch_summary,
                            eligible_trials_count = :eligible_trials_count,
                            not_eligible_trials_count = :not_eligible_trials_count,
                            need_more_info_trials_count = :need_more_info_trials_count,
                            average_confidence_score = :average_confidence_score,
                            trial_evaluations = :trial_evaluations,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                        RETURNING id
                    """)
                    
                    insert_data['id'] = evaluation_id
                    result = connection.execute(update_query, insert_data)
                    evaluation_id = result.fetchone()[0]
                    connection.commit()
                    
                    print(f"✅ Patient-to-Trial evaluation updated with ID: {evaluation_id}")
                    return evaluation_id
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO insightsedge.patient_to_trial_evaluations (
                            patient_id, patient_mrn, evaluation_timestamp, patient_age, 
                            patient_gender, patient_oncologist, total_trials_found, 
                            total_trials_evaluated, evaluation_method, batch_summary,
                            eligible_trials_count, not_eligible_trials_count, 
                            need_more_info_trials_count, average_confidence_score, 
                            trial_evaluations
                        ) VALUES (
                            :patient_id, :patient_mrn, :evaluation_timestamp, :patient_age,
                            :patient_gender, :patient_oncologist, :total_trials_found,
                            :total_trials_evaluated, :evaluation_method, :batch_summary,
                            :eligible_trials_count, :not_eligible_trials_count,
                            :need_more_info_trials_count, :average_confidence_score,
                            :trial_evaluations
                        ) RETURNING id
                    """)
                    
                    result = connection.execute(insert_query, insert_data)
                    evaluation_id = result.fetchone()[0]
                    connection.commit()
                    
                    print(f"✅ Patient-to-Trial evaluation saved with ID: {evaluation_id}")
                    return evaluation_id
                
        except SQLAlchemyError as e:
            print(f"❌ Database error saving patient-to-trial evaluation: {e}")
            return None
        except Exception as e:
            print(f"❌ Error saving patient-to-trial evaluation: {e}")
            return None
    
    def save_trial_to_patient_evaluation(self, evaluation_data: Dict[str, Any]) -> Optional[int]:
        """
        Save trial-to-patient evaluation results to database
        Updates existing record if found, otherwise inserts new record
        
        Args:
            evaluation_data: Dictionary containing evaluation results
            
        Returns:
            int: ID of the inserted/updated record, or None if failed
        """
        try:
            # Extract data from evaluation results
            trial_info = evaluation_data.get('trial_info', {})
            llm_evaluation = evaluation_data.get('llm_evaluation', {})
            hybrid_matching = evaluation_data.get('hybrid_matching', {})
            summary = evaluation_data.get('summary', {})
            
            trial_id = evaluation_data.get('trial_id') or trial_info.get('trial_id') or trial_info.get('nct_id', 'Unknown')
            
            # Prepare data for insertion/update
            insert_data = {
                'trial_id': trial_id,
                'trial_title': trial_info.get('title'),
                'evaluation_timestamp': datetime.now(),
                'trial_condition': trial_info.get('condition'),
                'trial_phase': trial_info.get('phase'),
                'trial_status': trial_info.get('status'),
                'trial_investigator': trial_info.get('investigator'),
                'total_patients_found': summary.get('total_patients_found', 0),
                'total_patients_evaluated': summary.get('patients_evaluated', 0),
                'evaluation_method': llm_evaluation.get('evaluation_method', 'batch'),
                'batch_summary': json.dumps(llm_evaluation.get('batch_summary', {})),
                'eligible_patients_count': summary.get('eligible_patients', 0),
                'not_eligible_patients_count': llm_evaluation.get('summary', {}).get('not_eligible_count', 0),
                'need_more_info_patients_count': llm_evaluation.get('summary', {}).get('need_more_info_count', 0),
                'average_confidence_score': summary.get('average_confidence', 0.0),
                'patient_evaluations': json.dumps(llm_evaluation.get('evaluations', []))
            }
            
            # Check if record exists for this trial_id (get most recent)
            with self.engine.connect() as connection:
                # Check for existing record
                check_query = text("""
                    SELECT id FROM insightsedge.trial_to_patient_evaluations
                    WHERE trial_id = :trial_id
                    ORDER BY evaluation_timestamp DESC, id DESC
                    LIMIT 1
                """)
                
                result = connection.execute(check_query, {'trial_id': trial_id})
                existing_record = result.fetchone()
                
                if existing_record:
                    # Update existing record
                    evaluation_id = existing_record[0]
                    update_query = text("""
                        UPDATE insightsedge.trial_to_patient_evaluations SET
                            trial_title = :trial_title,
                            evaluation_timestamp = :evaluation_timestamp,
                            trial_condition = :trial_condition,
                            trial_phase = :trial_phase,
                            trial_status = :trial_status,
                            trial_investigator = :trial_investigator,
                            total_patients_found = :total_patients_found,
                            total_patients_evaluated = :total_patients_evaluated,
                            evaluation_method = :evaluation_method,
                            batch_summary = :batch_summary,
                            eligible_patients_count = :eligible_patients_count,
                            not_eligible_patients_count = :not_eligible_patients_count,
                            need_more_info_patients_count = :need_more_info_patients_count,
                            average_confidence_score = :average_confidence_score,
                            patient_evaluations = :patient_evaluations,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                        RETURNING id
                    """)
                    
                    insert_data['id'] = evaluation_id
                    result = connection.execute(update_query, insert_data)
                    evaluation_id = result.fetchone()[0]
                    connection.commit()
                    
                    print(f"✅ Trial-to-Patient evaluation updated with ID: {evaluation_id}")
                    return evaluation_id
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO insightsedge.trial_to_patient_evaluations (
                            trial_id, trial_title, evaluation_timestamp, trial_condition,
                            trial_phase, trial_status, trial_investigator, total_patients_found,
                            total_patients_evaluated, evaluation_method, batch_summary,
                            eligible_patients_count, not_eligible_patients_count,
                            need_more_info_patients_count, average_confidence_score,
                            patient_evaluations
                        ) VALUES (
                            :trial_id, :trial_title, :evaluation_timestamp, :trial_condition,
                            :trial_phase, :trial_status, :trial_investigator, :total_patients_found,
                            :total_patients_evaluated, :evaluation_method, :batch_summary,
                            :eligible_patients_count, :not_eligible_patients_count,
                            :need_more_info_patients_count, :average_confidence_score,
                            :patient_evaluations
                        ) RETURNING id
                    """)
                    
                    result = connection.execute(insert_query, insert_data)
                    evaluation_id = result.fetchone()[0]
                    connection.commit()
                    
                    print(f"✅ Trial-to-Patient evaluation saved with ID: {evaluation_id}")
                    return evaluation_id
                
        except SQLAlchemyError as e:
            print(f"❌ Database error saving trial-to-patient evaluation: {e}")
            return None
        except Exception as e:
            print(f"❌ Error saving trial-to-patient evaluation: {e}")
            return None
    
    def get_patient_evaluation_history(self, patient_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a specific patient
        
        Args:
            patient_id: Patient ID to get history for
            limit: Maximum number of records to return
            
        Returns:
            List of evaluation records
        """
        try:
            with self.engine.connect() as connection:
                query = text("""
                    SELECT 
                        id, patient_id, patient_mrn, evaluation_timestamp,
                        total_trials_found, total_trials_evaluated, evaluation_method,
                        eligible_trials_count, not_eligible_trials_count, 
                        need_more_info_trials_count, average_confidence_score,
                        created_at
                    FROM insightsedge.patient_to_trial_evaluations
                    WHERE patient_id = :patient_id
                    ORDER BY evaluation_timestamp DESC
                    LIMIT :limit
                """)
                
                result = connection.execute(query, {'patient_id': patient_id, 'limit': limit})
                records = []
                
                for row in result:
                    records.append({
                        'id': row[0],
                        'patient_id': row[1],
                        'patient_mrn': row[2],
                        'evaluation_timestamp': row[3].isoformat() if row[3] else None,
                        'total_trials_found': row[4],
                        'total_trials_evaluated': row[5],
                        'evaluation_method': row[6],
                        'eligible_trials_count': row[7],
                        'not_eligible_trials_count': row[8],
                        'need_more_info_trials_count': row[9],
                        'average_confidence_score': float(row[10]) if row[10] else 0.0,
                        'created_at': row[11].isoformat() if row[11] else None
                    })
                
                return records
                
        except Exception as e:
            print(f"❌ Error getting patient evaluation history: {e}")
            return []
    
    def get_trial_evaluation_history(self, trial_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a specific trial
        
        Args:
            trial_id: Trial ID to get history for
            limit: Maximum number of records to return
            
        Returns:
            List of evaluation records
        """
        try:
            with self.engine.connect() as connection:
                query = text("""
                    SELECT 
                        id, trial_id, trial_title, evaluation_timestamp,
                        total_patients_found, total_patients_evaluated, evaluation_method,
                        eligible_patients_count, not_eligible_patients_count,
                        need_more_info_patients_count, average_confidence_score,
                        created_at
                    FROM insightsedge.trial_to_patient_evaluations
                    WHERE trial_id = :trial_id
                    ORDER BY evaluation_timestamp DESC
                    LIMIT :limit
                """)
                
                result = connection.execute(query, {'trial_id': trial_id, 'limit': limit})
                records = []
                
                for row in result:
                    records.append({
                        'id': row[0],
                        'trial_id': row[1],
                        'trial_title': row[2],
                        'evaluation_timestamp': row[3].isoformat() if row[3] else None,
                        'total_patients_found': row[4],
                        'total_patients_evaluated': row[5],
                        'evaluation_method': row[6],
                        'eligible_patients_count': row[7],
                        'not_eligible_patients_count': row[8],
                        'need_more_info_patients_count': row[9],
                        'average_confidence_score': float(row[10]) if row[10] else 0.0,
                        'created_at': row[11].isoformat() if row[11] else None
                    })
                
                return records
                
        except Exception as e:
            print(f"❌ Error getting trial evaluation history: {e}")
            return []
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get overall evaluation statistics
        
        Returns:
            Dictionary containing evaluation statistics
        """
        try:
            with self.engine.connect() as connection:
                # Patient-to-Trial statistics
                p2t_query = text("""
                    SELECT 
                        COUNT(*) as total_evaluations,
                        SUM(eligible_trials_count) as total_eligible_trials,
                        SUM(not_eligible_trials_count) as total_not_eligible_trials,
                        SUM(need_more_info_trials_count) as total_need_more_info_trials,
                        AVG(average_confidence_score) as avg_confidence_score,
                        COUNT(DISTINCT patient_id) as unique_patients_evaluated
                    FROM insightsedge.patient_to_trial_evaluations
                """)
                
                p2t_result = connection.execute(p2t_query).fetchone()
                
                # Trial-to-Patient statistics
                t2p_query = text("""
                    SELECT 
                        COUNT(*) as total_evaluations,
                        SUM(eligible_patients_count) as total_eligible_patients,
                        SUM(not_eligible_patients_count) as total_not_eligible_patients,
                        SUM(need_more_info_patients_count) as total_need_more_info_patients,
                        AVG(average_confidence_score) as avg_confidence_score,
                        COUNT(DISTINCT trial_id) as unique_trials_evaluated
                    FROM insightsedge.trial_to_patient_evaluations
                """)
                
                t2p_result = connection.execute(t2p_query).fetchone()
                
                return {
                    'patient_to_trial': {
                        'total_evaluations': p2t_result[0] or 0,
                        'total_eligible_trials': p2t_result[1] or 0,
                        'total_not_eligible_trials': p2t_result[2] or 0,
                        'total_need_more_info_trials': p2t_result[3] or 0,
                        'average_confidence_score': float(p2t_result[4]) if p2t_result[4] else 0.0,
                        'unique_patients_evaluated': p2t_result[5] or 0
                    },
                    'trial_to_patient': {
                        'total_evaluations': t2p_result[0] or 0,
                        'total_eligible_patients': t2p_result[1] or 0,
                        'total_not_eligible_patients': t2p_result[2] or 0,
                        'total_need_more_info_patients': t2p_result[3] or 0,
                        'average_confidence_score': float(t2p_result[4]) if t2p_result[4] else 0.0,
                        'unique_trials_evaluated': t2p_result[5] or 0
                    },
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error getting evaluation statistics: {e}")
            return {}

def main():
    """Test the evaluation results database utilities"""
    db = EvaluationResultsDB()
    
    # Test statistics
    print("📊 Evaluation Statistics:")
    stats = db.get_evaluation_statistics()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
