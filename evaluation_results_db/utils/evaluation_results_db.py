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
        self._table_created = False
    
    def _ensure_patient_trial_evaluations_table(self):
        """Ensure the patient_to_trial table exists, create if not"""
        if self._table_created:
            return True
        
        try:
            with self.engine.connect() as connection:
                # Check if table exists
                check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'insightsedge' 
                        AND table_name = 'patient_to_trial'
                    )
                """)
                result = connection.execute(check_query)
                table_exists = result.fetchone()[0]
                
                # Always ensure the update_updated_at_column function exists (needed for trigger)
                function_check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc p
                        JOIN pg_namespace n ON p.pronamespace = n.oid
                        WHERE n.nspname = 'insightsedge'
                        AND p.proname = 'update_updated_at_column'
                    )
                """)
                function_result = connection.execute(function_check_query)
                function_exists = function_result.fetchone()[0]
                
                if not function_exists:
                    print("📊 Creating update_updated_at_column function...")
                    create_function_query = text("""
                        CREATE OR REPLACE FUNCTION insightsedge.update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = NOW();
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql';
                    """)
                    connection.execute(create_function_query)
                    connection.commit()
                    print("✅ Function created successfully!")
                
                if table_exists:
                    # Table exists - check if columns need to be added
                    print("📊 Table exists, checking for missing columns...")
                    
                    # Check if mrn column exists
                    mrn_check = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'insightsedge' 
                            AND table_name = 'patient_to_trial'
                            AND column_name = 'mrn'
                        )
                    """)
                    mrn_result = connection.execute(mrn_check)
                    mrn_exists = mrn_result.fetchone()[0]
                    
                    # Check if reasoning column exists
                    reasoning_check = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'insightsedge' 
                            AND table_name = 'patient_to_trial'
                            AND column_name = 'reasoning'
                        )
                    """)
                    reasoning_result = connection.execute(reasoning_check)
                    reasoning_exists = reasoning_result.fetchone()[0]
                    
                    # Add missing columns
                    if not mrn_exists:
                        print("📊 Adding mrn column to existing table...")
                        alter_mrn = text("ALTER TABLE insightsedge.patient_to_trial ADD COLUMN IF NOT EXISTS mrn VARCHAR(50)")
                        connection.execute(alter_mrn)
                        # Create index for mrn
                        index_mrn = text("CREATE INDEX IF NOT EXISTS idx_p2t_mrn ON insightsedge.patient_to_trial(mrn)")
                        connection.execute(index_mrn)
                        print("✅ mrn column added successfully!")
                    
                    if not reasoning_exists:
                        print("📊 Adding reasoning column to existing table...")
                        alter_reasoning = text("ALTER TABLE insightsedge.patient_to_trial ADD COLUMN IF NOT EXISTS reasoning TEXT")
                        connection.execute(alter_reasoning)
                        print("✅ reasoning column added successfully!")
                    
                    if not mrn_exists or not reasoning_exists:
                        connection.commit()
                    
                    self._table_created = True
                else:
                    # Table doesn't exist - create it
                    print("📊 Creating patient_to_trial table...")
                    # Create the table
                    create_table_query = text("""
                        CREATE TABLE IF NOT EXISTS insightsedge.patient_to_trial (
                            id SERIAL PRIMARY KEY,
                            patient_id INTEGER NOT NULL,
                            mrn VARCHAR(50),
                            trial_id VARCHAR(50) NOT NULL,
                            eligibility_status VARCHAR(20) NOT NULL,
                            confidence_score DECIMAL(5,2) DEFAULT 0.00,
                            reasoning TEXT,
                            key_criteria_met JSONB,
                            key_criteria_missed JSONB,
                            recommendations TEXT,
                            isevaluated INTEGER DEFAULT 1,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW(),
                            CONSTRAINT chk_eligibility_status CHECK (eligibility_status IN ('ELIGIBLE', 'NOT_ELIGIBLE', 'NEED_MORE_INFO')),
                            CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 100),
                            CONSTRAINT chk_isevaluated CHECK (isevaluated IN (0, 1)),
                            CONSTRAINT uq_patient_trial UNIQUE (patient_id, trial_id)
                        )
                    """)
                    connection.execute(create_table_query)
                    
                    # Create indexes
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_p2t_patient_id ON insightsedge.patient_to_trial(patient_id)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_mrn ON insightsedge.patient_to_trial(mrn)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_trial_id ON insightsedge.patient_to_trial(trial_id)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_eligibility_status ON insightsedge.patient_to_trial(eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_confidence_score ON insightsedge.patient_to_trial(confidence_score DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_isevaluated ON insightsedge.patient_to_trial(isevaluated)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_patient_eligibility ON insightsedge.patient_to_trial(patient_id, eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_trial_eligibility ON insightsedge.patient_to_trial(trial_id, eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_p2t_created_at ON insightsedge.patient_to_trial(created_at DESC)"
                    ]
                    
                    for index_query in indexes:
                        connection.execute(text(index_query))
                    
                    # Create trigger for updated_at
                    trigger_query = text("""
                        DROP TRIGGER IF EXISTS update_p2t_updated_at ON insightsedge.patient_to_trial;
                        CREATE TRIGGER update_p2t_updated_at
                            BEFORE UPDATE ON insightsedge.patient_to_trial
                            FOR EACH ROW
                            EXECUTE FUNCTION insightsedge.update_updated_at_column();
                    """)
                    connection.execute(trigger_query)
                    
                    connection.commit()
                    print("✅ patient_to_trial table created successfully!")
                    self._table_created = True
                
                return True
        except Exception as e:
            print(f"⚠️ Error checking/creating patient_to_trial table: {e}")
            print("⚠️ You may need to run the SQL script manually: evaluation_results_db/sql/02_create_patient_trial_evaluations_table.sql")
            return False
    
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
    
    def save_patient_trial_evaluations(self, evaluation_data: Dict[str, Any]) -> int:
        """
        Save individual patient-trial evaluation results to normalized table
        Extracts individual trial evaluations and saves each as a separate row
        
        Args:
            evaluation_data: Dictionary containing evaluation results with llm_evaluation.evaluations array
            
        Returns:
            int: Number of records saved (0 if failed)
        """
        # Ensure table exists before trying to save
        if not self._ensure_patient_trial_evaluations_table():
            print("❌ Failed to ensure table exists, cannot save evaluations")
            return 0
        
        try:
            # Extract data from evaluation results
            patient_info = evaluation_data.get('patient_info', {})
            llm_evaluation = evaluation_data.get('llm_evaluation', {})
            
            patient_id = patient_info.get('patient_id')
            if not patient_id:
                print("❌ No patient_id found in evaluation data")
                return 0
            
            evaluations = llm_evaluation.get('evaluations', [])
            if not evaluations:
                print(f"⚠️ No individual evaluations found for patient {patient_id}")
                return 0
            
            saved_count = 0
            failed_count = 0
            
            with self.engine.connect() as connection:
                for eval_item in evaluations:
                    try:
                        # Extract trial_id - try multiple possible fields
                        trial_id = (
                            eval_item.get('trial_id') or 
                            eval_item.get('trial_info', {}).get('trial_id') or
                            eval_item.get('trial_info', {}).get('nct_id') or
                            None
                        )
                        
                        if not trial_id:
                            print(f"⚠️ Skipping evaluation: missing trial_id")
                            failed_count += 1
                            continue
                        
                        # Extract evaluation fields
                        eligibility_status = eval_item.get('eligibility_status', 'NEED_MORE_INFO')
                        confidence_score = eval_item.get('confidence_score', 0.0)
                        reasoning = eval_item.get('reasoning', '')
                        key_criteria_met = eval_item.get('key_criteria_met', [])
                        key_criteria_missed = eval_item.get('key_criteria_missed', [])
                        recommendations = eval_item.get('recommendations', '')
                        
                        # Extract MRN from patient_info
                        patient_info = eval_item.get('patient_info', {})
                        mrn = patient_info.get('mrn') or patient_info.get('patient_mrn')
                        
                        # Convert arrays to JSON strings for JSONB storage
                        # Handle empty arrays - store as NULL instead of empty JSON array
                        if key_criteria_met and len(key_criteria_met) > 0:
                            key_criteria_met_json = json.dumps(key_criteria_met)
                        else:
                            key_criteria_met_json = None
                        
                        if key_criteria_missed and len(key_criteria_missed) > 0:
                            key_criteria_missed_json = json.dumps(key_criteria_missed)
                        else:
                            key_criteria_missed_json = None
                        
                        # Prepare insert data - build query based on NULL values
                        insert_data = {
                            'patient_id': patient_id,
                            'mrn': mrn,
                            'trial_id': trial_id,
                            'eligibility_status': eligibility_status,
                            'confidence_score': float(confidence_score) if confidence_score else 0.0,
                            'reasoning': reasoning,
                            'recommendations': recommendations,
                            'isevaluated': 1
                        }
                        
                        # Build query based on which JSON values are NULL
                        # Use COALESCE to handle NULL values in SQL
                        if key_criteria_met_json is None and key_criteria_missed_json is None:
                            # Both are NULL
                            upsert_query = text("""
                                INSERT INTO insightsedge.patient_to_trial (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, NULL::jsonb, NULL::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (patient_id, trial_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        elif key_criteria_met_json is None:
                            # key_criteria_met is NULL
                            insert_data['key_criteria_missed'] = key_criteria_missed_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.patient_to_trial (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, NULL::jsonb, CAST(:key_criteria_missed AS TEXT)::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (patient_id, trial_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        elif key_criteria_missed_json is None:
                            # key_criteria_missed is NULL
                            insert_data['key_criteria_met'] = key_criteria_met_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.patient_to_trial (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, CAST(:key_criteria_met AS TEXT)::jsonb, NULL::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (patient_id, trial_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        else:
                            # Both have values
                            insert_data['key_criteria_met'] = key_criteria_met_json
                            insert_data['key_criteria_missed'] = key_criteria_missed_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.patient_to_trial (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, CAST(:key_criteria_met AS TEXT)::jsonb, CAST(:key_criteria_missed AS TEXT)::jsonb, 
                                    :recommendations, :isevaluated
                                )
                                ON CONFLICT (patient_id, trial_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        
                        result = connection.execute(upsert_query, insert_data)
                        record_id = result.fetchone()[0]
                        saved_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error saving individual evaluation for trial {trial_id}: {e}")
                        failed_count += 1
                        continue
                
                connection.commit()
                
                if saved_count > 0:
                    print(f"✅ Saved {saved_count} patient-trial evaluations to database")
                if failed_count > 0:
                    print(f"⚠️ Failed to save {failed_count} evaluations")
                
                return saved_count
                
        except SQLAlchemyError as e:
            print(f"❌ Database error saving patient-trial evaluations: {e}")
            return 0
        except Exception as e:
            print(f"❌ Error saving patient-trial evaluations: {e}")
            return 0
    
    def _ensure_trial_patient_evaluations_table(self) -> bool:
        """Ensure the trial_to_patient table exists, create if not"""
        if hasattr(self, '_t2p_table_created') and self._t2p_table_created:
            return True
        
        try:
            with self.engine.connect() as connection:
                # Check if table exists
                check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'insightsedge' 
                        AND table_name = 'trial_to_patient'
                    )
                """)
                result = connection.execute(check_query)
                table_exists = result.fetchone()[0]
                
                # Always ensure the update_updated_at_column function exists (needed for trigger)
                function_check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc p
                        JOIN pg_namespace n ON p.pronamespace = n.oid
                        WHERE n.nspname = 'insightsedge'
                        AND p.proname = 'update_updated_at_column'
                    )
                """)
                function_result = connection.execute(function_check_query)
                function_exists = function_result.fetchone()[0]
                
                if not function_exists:
                    print("📊 Creating update_updated_at_column function...")
                    create_function_query = text("""
                        CREATE OR REPLACE FUNCTION insightsedge.update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = NOW();
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql';
                    """)
                    connection.execute(create_function_query)
                    connection.commit()
                    print("✅ Function created successfully!")
                
                if table_exists:
                    # Table exists - check if columns need to be added
                    print("📊 Table exists, checking for missing columns...")
                    
                    # Check if mrn column exists
                    mrn_check = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'insightsedge' 
                            AND table_name = 'trial_to_patient'
                            AND column_name = 'mrn'
                        )
                    """)
                    mrn_result = connection.execute(mrn_check)
                    mrn_exists = mrn_result.fetchone()[0]
                    
                    # Check if reasoning column exists
                    reasoning_check = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'insightsedge' 
                            AND table_name = 'trial_to_patient'
                            AND column_name = 'reasoning'
                        )
                    """)
                    reasoning_result = connection.execute(reasoning_check)
                    reasoning_exists = reasoning_result.fetchone()[0]
                    
                    # Add missing columns
                    if not mrn_exists:
                        print("📊 Adding mrn column to existing table...")
                        alter_mrn = text("ALTER TABLE insightsedge.trial_to_patient ADD COLUMN IF NOT EXISTS mrn VARCHAR(50)")
                        connection.execute(alter_mrn)
                        # Create index for mrn
                        index_mrn = text("CREATE INDEX IF NOT EXISTS idx_t2p_mrn ON insightsedge.trial_to_patient(mrn)")
                        connection.execute(index_mrn)
                        print("✅ mrn column added successfully!")
                    
                    if not reasoning_exists:
                        print("📊 Adding reasoning column to existing table...")
                        alter_reasoning = text("ALTER TABLE insightsedge.trial_to_patient ADD COLUMN IF NOT EXISTS reasoning TEXT")
                        connection.execute(alter_reasoning)
                        print("✅ reasoning column added successfully!")
                    
                    if not mrn_exists or not reasoning_exists:
                        connection.commit()
                    
                    self._t2p_table_created = True
                else:
                    # Table doesn't exist - create it
                    print("📊 Creating trial_to_patient table...")
                    # Create the table
                    create_table_query = text("""
                        CREATE TABLE IF NOT EXISTS insightsedge.trial_to_patient (
                            id SERIAL PRIMARY KEY,
                            patient_id INTEGER NOT NULL,
                            mrn VARCHAR(50),
                            trial_id VARCHAR(50) NOT NULL,
                            eligibility_status VARCHAR(20) NOT NULL,
                            confidence_score DECIMAL(5,2) DEFAULT 0.00,
                            reasoning TEXT,
                            key_criteria_met JSONB,
                            key_criteria_missed JSONB,
                            recommendations TEXT,
                            isevaluated INTEGER DEFAULT 1,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW(),
                            CONSTRAINT chk_t2p_eligibility_status CHECK (eligibility_status IN ('ELIGIBLE', 'NOT_ELIGIBLE', 'NEED_MORE_INFO')),
                            CONSTRAINT chk_t2p_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 100),
                            CONSTRAINT chk_t2p_isevaluated CHECK (isevaluated IN (0, 1)),
                            CONSTRAINT uq_trial_patient UNIQUE (trial_id, patient_id)
                        )
                    """)
                    connection.execute(create_table_query)
                    
                    # Create indexes
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_t2p_patient_id ON insightsedge.trial_to_patient(patient_id)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_mrn ON insightsedge.trial_to_patient(mrn)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_trial_id ON insightsedge.trial_to_patient(trial_id)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_eligibility_status ON insightsedge.trial_to_patient(eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_confidence_score ON insightsedge.trial_to_patient(confidence_score DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_isevaluated ON insightsedge.trial_to_patient(isevaluated)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_patient_eligibility ON insightsedge.trial_to_patient(patient_id, eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_trial_eligibility ON insightsedge.trial_to_patient(trial_id, eligibility_status)",
                        "CREATE INDEX IF NOT EXISTS idx_t2p_created_at ON insightsedge.trial_to_patient(created_at DESC)"
                    ]
                    
                    for index_query in indexes:
                        connection.execute(text(index_query))
                    
                    # Create trigger for updated_at
                    trigger_query = text("""
                        DROP TRIGGER IF EXISTS update_t2p_updated_at ON insightsedge.trial_to_patient;
                        CREATE TRIGGER update_t2p_updated_at
                            BEFORE UPDATE ON insightsedge.trial_to_patient
                            FOR EACH ROW
                            EXECUTE FUNCTION insightsedge.update_updated_at_column();
                    """)
                    connection.execute(trigger_query)
                    
                    connection.commit()
                    print("✅ trial_to_patient table created successfully!")
                    self._t2p_table_created = True
                
                return True
        except Exception as e:
            print(f"⚠️ Error checking/creating trial_to_patient table: {e}")
            print("⚠️ You may need to run the SQL script manually: evaluation_results_db/sql/03_create_trial_patient_evaluations_table.sql")
            return False
    
    def save_trial_patient_evaluations(self, evaluation_data: Dict[str, Any]) -> int:
        """
        Save individual trial-patient evaluation results to normalized table
        Extracts individual patient evaluations and saves each as a separate row
        
        Args:
            evaluation_data: Dictionary containing evaluation results with llm_evaluation.evaluations array
            
        Returns:
            int: Number of records saved (0 if failed)
        """
        # Ensure table exists before trying to save
        if not self._ensure_trial_patient_evaluations_table():
            print("❌ Failed to ensure table exists, cannot save evaluations")
            return 0
        
        try:
            # Extract data from evaluation results
            trial_info = evaluation_data.get('trial_info', {})
            llm_evaluation = evaluation_data.get('llm_evaluation', {})
            
            trial_id = evaluation_data.get('trial_id') or trial_info.get('trial_id') or trial_info.get('nct_id')
            if not trial_id:
                print("❌ No trial_id found in evaluation data")
                return 0
            
            evaluations = llm_evaluation.get('evaluations', [])
            if not evaluations:
                print(f"⚠️ No individual evaluations found for trial {trial_id}")
                return 0
            
            saved_count = 0
            failed_count = 0
            
            with self.engine.connect() as connection:
                for eval_item in evaluations:
                    try:
                        # Extract patient_id - try multiple possible fields
                        patient_info = eval_item.get('patient_info', {})
                        patient_id = (
                            patient_info.get('patient_id') or
                            eval_item.get('patient_id') or
                            None
                        )
                        
                        if not patient_id:
                            print(f"⚠️ Skipping evaluation: missing patient_id")
                            failed_count += 1
                            continue
                        
                        # Extract evaluation fields
                        eligibility_status = eval_item.get('eligibility_status', 'NEED_MORE_INFO')
                        confidence_score = eval_item.get('confidence_score', 0.0)
                        reasoning = eval_item.get('reasoning', '')
                        key_criteria_met = eval_item.get('key_criteria_met', [])
                        key_criteria_missed = eval_item.get('key_criteria_missed', [])
                        recommendations = eval_item.get('recommendations', '')
                        
                        # Extract MRN from patient_info
                        mrn = patient_info.get('mrn') or patient_info.get('patient_mrn')
                        
                        # Convert arrays to JSON strings for JSONB storage
                        # Handle empty arrays - store as NULL instead of empty JSON array
                        if key_criteria_met and len(key_criteria_met) > 0:
                            key_criteria_met_json = json.dumps(key_criteria_met)
                        else:
                            key_criteria_met_json = None
                        
                        if key_criteria_missed and len(key_criteria_missed) > 0:
                            key_criteria_missed_json = json.dumps(key_criteria_missed)
                        else:
                            key_criteria_missed_json = None
                        
                        # Prepare insert data - build query based on NULL values
                        insert_data = {
                            'patient_id': patient_id,
                            'mrn': mrn,
                            'trial_id': trial_id,
                            'eligibility_status': eligibility_status,
                            'confidence_score': float(confidence_score) if confidence_score else 0.0,
                            'reasoning': reasoning,
                            'recommendations': recommendations,
                            'isevaluated': 1
                        }
                        
                        # Build query based on which JSON values are NULL
                        if key_criteria_met_json is None and key_criteria_missed_json is None:
                            # Both are NULL
                            upsert_query = text("""
                                INSERT INTO insightsedge.trial_to_patient (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, NULL::jsonb, NULL::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (trial_id, patient_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        elif key_criteria_met_json is None:
                            # Only key_criteria_met is NULL
                            insert_data['key_criteria_missed'] = key_criteria_missed_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.trial_to_patient (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, NULL::jsonb, CAST(:key_criteria_missed AS TEXT)::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (trial_id, patient_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        elif key_criteria_missed_json is None:
                            # Only key_criteria_missed is NULL
                            insert_data['key_criteria_met'] = key_criteria_met_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.trial_to_patient (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, CAST(:key_criteria_met AS TEXT)::jsonb, NULL::jsonb, :recommendations, :isevaluated
                                )
                                ON CONFLICT (trial_id, patient_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        else:
                            # Both are non-NULL
                            insert_data['key_criteria_met'] = key_criteria_met_json
                            insert_data['key_criteria_missed'] = key_criteria_missed_json
                            upsert_query = text("""
                                INSERT INTO insightsedge.trial_to_patient (
                                    patient_id, mrn, trial_id, eligibility_status, confidence_score,
                                    reasoning, key_criteria_met, key_criteria_missed, recommendations, isevaluated
                                ) VALUES (
                                    :patient_id, :mrn, :trial_id, :eligibility_status, :confidence_score,
                                    :reasoning, CAST(:key_criteria_met AS TEXT)::jsonb, CAST(:key_criteria_missed AS TEXT)::jsonb, 
                                    :recommendations, :isevaluated
                                )
                                ON CONFLICT (trial_id, patient_id) 
                                DO UPDATE SET
                                    mrn = EXCLUDED.mrn,
                                    eligibility_status = EXCLUDED.eligibility_status,
                                    confidence_score = EXCLUDED.confidence_score,
                                    reasoning = EXCLUDED.reasoning,
                                    key_criteria_met = EXCLUDED.key_criteria_met,
                                    key_criteria_missed = EXCLUDED.key_criteria_missed,
                                    recommendations = EXCLUDED.recommendations,
                                    isevaluated = EXCLUDED.isevaluated,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """)
                        
                        result = connection.execute(upsert_query, insert_data)
                        row_id = result.fetchone()
                        if row_id:
                            saved_count += 1
                        connection.commit()
                        
                    except Exception as e:
                        print(f"⚠️ Error saving individual evaluation for patient {patient_id}: {e}")
                        failed_count += 1
                        connection.rollback()
                        continue
            
            if saved_count > 0:
                print(f"✅ Saved {saved_count} individual trial-patient evaluations")
            if failed_count > 0:
                print(f"⚠️ Failed to save {failed_count} evaluations")
            
            return saved_count
            
        except Exception as e:
            print(f"❌ Error saving individual trial-patient evaluations: {e}")
            return 0
    
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
