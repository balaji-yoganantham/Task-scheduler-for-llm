-- =====================================================
-- Clinical Trial Matching System - Patient-Trial Evaluations Table
-- =====================================================
-- This script creates a normalized table to store individual
-- patient-trial evaluation results (one row per patient-trial pair)
-- =====================================================

-- =====================================================
-- Table: Patient-to-Trial Evaluations
-- =====================================================
-- Stores individual patient-trial evaluation results
-- One row per patient-trial combination
CREATE TABLE IF NOT EXISTS insightsedge.patient_to_trial (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Identifiers
    patient_id INTEGER NOT NULL,
    mrn VARCHAR(50),
    trial_id VARCHAR(50) NOT NULL,
    
    -- Evaluation Results
    eligibility_status VARCHAR(20) NOT NULL, -- ELIGIBLE, NOT_ELIGIBLE, NEED_MORE_INFO
    confidence_score DECIMAL(5,2) DEFAULT 0.00,
    reasoning TEXT,
    
    -- Criteria Analysis
    key_criteria_met JSONB, -- Array of criteria that were met
    key_criteria_missed JSONB, -- Array of criteria that were missed
    
    -- Recommendations
    recommendations TEXT,
    
    -- Evaluation Flag
    isevaluated INTEGER DEFAULT 1, -- 1 = evaluated, 0 = not evaluated
    
    -- System Fields
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_eligibility_status CHECK (eligibility_status IN ('ELIGIBLE', 'NOT_ELIGIBLE', 'NEED_MORE_INFO')),
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 100),
    CONSTRAINT chk_isevaluated CHECK (isevaluated IN (0, 1)),
    
    -- Unique constraint: one evaluation per patient-trial pair
    CONSTRAINT uq_patient_trial UNIQUE (patient_id, trial_id)
);

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Index on patient_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_p2t_patient_id ON insightsedge.patient_to_trial(patient_id);

-- Index on mrn for fast lookups
CREATE INDEX IF NOT EXISTS idx_p2t_mrn ON insightsedge.patient_to_trial(mrn);

-- Index on trial_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_p2t_trial_id ON insightsedge.patient_to_trial(trial_id);

-- Index on eligibility_status for filtering
CREATE INDEX IF NOT EXISTS idx_p2t_eligibility_status ON insightsedge.patient_to_trial(eligibility_status);

-- Index on confidence_score for sorting
CREATE INDEX IF NOT EXISTS idx_p2t_confidence_score ON insightsedge.patient_to_trial(confidence_score DESC);

-- Index on isevaluated for filtering evaluated records
CREATE INDEX IF NOT EXISTS idx_p2t_isevaluated ON insightsedge.patient_to_trial(isevaluated);

-- Composite index for common queries (patient_id + eligibility_status)
CREATE INDEX IF NOT EXISTS idx_p2t_patient_eligibility ON insightsedge.patient_to_trial(patient_id, eligibility_status);

-- Composite index for common queries (trial_id + eligibility_status)
CREATE INDEX IF NOT EXISTS idx_p2t_trial_eligibility ON insightsedge.patient_to_trial(trial_id, eligibility_status);

-- Index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_p2t_created_at ON insightsedge.patient_to_trial(created_at DESC);

-- =====================================================
-- Trigger for Auto-Update Timestamps
-- =====================================================

-- Trigger for patient_to_trial
DROP TRIGGER IF EXISTS update_p2t_updated_at ON insightsedge.patient_to_trial;
CREATE TRIGGER update_p2t_updated_at
    BEFORE UPDATE ON insightsedge.patient_to_trial
    FOR EACH ROW
    EXECUTE FUNCTION insightsedge.update_updated_at_column();

-- =====================================================
-- Comments for Documentation
-- =====================================================

COMMENT ON TABLE insightsedge.patient_to_trial IS 'Stores individual patient-trial evaluation results (one row per patient-trial pair)';

COMMENT ON COLUMN insightsedge.patient_to_trial.patient_id IS 'Patient identifier';
COMMENT ON COLUMN insightsedge.patient_to_trial.mrn IS 'Patient Medical Record Number (MRN)';
COMMENT ON COLUMN insightsedge.patient_to_trial.trial_id IS 'Clinical trial identifier (NCT ID)';
COMMENT ON COLUMN insightsedge.patient_to_trial.eligibility_status IS 'Eligibility status: ELIGIBLE, NOT_ELIGIBLE, or NEED_MORE_INFO';
COMMENT ON COLUMN insightsedge.patient_to_trial.confidence_score IS 'Confidence score (0-100) for the evaluation';
COMMENT ON COLUMN insightsedge.patient_to_trial.reasoning IS 'Detailed reasoning for the eligibility assessment';
COMMENT ON COLUMN insightsedge.patient_to_trial.key_criteria_met IS 'JSONB array of key criteria that were met';
COMMENT ON COLUMN insightsedge.patient_to_trial.key_criteria_missed IS 'JSONB array of key criteria that were missed';
COMMENT ON COLUMN insightsedge.patient_to_trial.recommendations IS 'Recommendations for next steps';
COMMENT ON COLUMN insightsedge.patient_to_trial.isevaluated IS 'Flag indicating if evaluation has been completed (1 = evaluated, 0 = not evaluated)';

-- =====================================================
-- Success Message
-- =====================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Patient-to-trial table created successfully!';
    RAISE NOTICE '📊 Table created: insightsedge.patient_to_trial';
    RAISE NOTICE '🔍 Indexes and triggers configured for optimal performance';
END $$;

