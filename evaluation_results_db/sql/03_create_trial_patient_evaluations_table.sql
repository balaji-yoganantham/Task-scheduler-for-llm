-- =====================================================
-- Clinical Trial Matching System - Trial-Patient Evaluations Table
-- =====================================================
-- This script creates a normalized table to store individual
-- trial-patient evaluation results (one row per trial-patient pair)
-- =====================================================

-- =====================================================
-- Table: Trial-to-Patient Evaluations
-- =====================================================
-- Stores individual trial-patient evaluation results
-- One row per trial-patient combination
CREATE TABLE IF NOT EXISTS insightsedge.trial_to_patient (
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
    CONSTRAINT chk_t2p_eligibility_status CHECK (eligibility_status IN ('ELIGIBLE', 'NOT_ELIGIBLE', 'NEED_MORE_INFO')),
    CONSTRAINT chk_t2p_confidence_score CHECK (confidence_score >= 0 AND confidence_score <= 100),
    CONSTRAINT chk_t2p_isevaluated CHECK (isevaluated IN (0, 1)),
    
    -- Unique constraint: one evaluation per trial-patient pair
    CONSTRAINT uq_trial_patient UNIQUE (trial_id, patient_id)
);

-- Unique constraint on (mrn, trial_id) for conflict resolution when mrn is provided
-- This allows checking by MRN and trial_id as requested
CREATE UNIQUE INDEX IF NOT EXISTS idx_t2p_mrn_trial_unique 
    ON insightsedge.trial_to_patient (mrn, trial_id) 
    WHERE mrn IS NOT NULL;

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Index on patient_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_t2p_patient_id ON insightsedge.trial_to_patient(patient_id);

-- Index on mrn for fast lookups
CREATE INDEX IF NOT EXISTS idx_t2p_mrn ON insightsedge.trial_to_patient(mrn);

-- Index on trial_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_t2p_trial_id ON insightsedge.trial_to_patient(trial_id);

-- Index on eligibility_status for filtering
CREATE INDEX IF NOT EXISTS idx_t2p_eligibility_status ON insightsedge.trial_to_patient(eligibility_status);

-- Index on confidence_score for sorting
CREATE INDEX IF NOT EXISTS idx_t2p_confidence_score ON insightsedge.trial_to_patient(confidence_score DESC);

-- Index on isevaluated for filtering
CREATE INDEX IF NOT EXISTS idx_t2p_isevaluated ON insightsedge.trial_to_patient(isevaluated);

-- Composite index for patient eligibility queries
CREATE INDEX IF NOT EXISTS idx_t2p_patient_eligibility ON insightsedge.trial_to_patient(patient_id, eligibility_status);

-- Composite index for trial eligibility queries
CREATE INDEX IF NOT EXISTS idx_t2p_trial_eligibility ON insightsedge.trial_to_patient(trial_id, eligibility_status);

-- Index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_t2p_created_at ON insightsedge.trial_to_patient(created_at DESC);

-- =====================================================
-- Trigger for Auto-Updating updated_at
-- =====================================================
-- Note: This assumes the update_updated_at_column function already exists
-- If not, it will be created by the Python code

DROP TRIGGER IF EXISTS update_t2p_updated_at ON insightsedge.trial_to_patient;
CREATE TRIGGER update_t2p_updated_at
    BEFORE UPDATE ON insightsedge.trial_to_patient
    FOR EACH ROW
    EXECUTE FUNCTION insightsedge.update_updated_at_column();

-- =====================================================
-- Comments for Documentation
-- =====================================================

COMMENT ON TABLE insightsedge.trial_to_patient IS 'Stores individual trial-patient evaluation results (one row per trial-patient pair)';

COMMENT ON COLUMN insightsedge.trial_to_patient.patient_id IS 'Patient identifier';
COMMENT ON COLUMN insightsedge.trial_to_patient.mrn IS 'Patient Medical Record Number (MRN)';
COMMENT ON COLUMN insightsedge.trial_to_patient.trial_id IS 'Clinical trial identifier (NCT ID)';
COMMENT ON COLUMN insightsedge.trial_to_patient.eligibility_status IS 'Eligibility status: ELIGIBLE, NOT_ELIGIBLE, or NEED_MORE_INFO';
COMMENT ON COLUMN insightsedge.trial_to_patient.confidence_score IS 'Confidence score (0-100) for the evaluation';
COMMENT ON COLUMN insightsedge.trial_to_patient.reasoning IS 'Detailed reasoning for the eligibility assessment';
COMMENT ON COLUMN insightsedge.trial_to_patient.key_criteria_met IS 'JSONB array of key criteria that were met';
COMMENT ON COLUMN insightsedge.trial_to_patient.key_criteria_missed IS 'JSONB array of key criteria that were missed';
COMMENT ON COLUMN insightsedge.trial_to_patient.recommendations IS 'Recommendations for next steps';
COMMENT ON COLUMN insightsedge.trial_to_patient.isevaluated IS 'Flag indicating if evaluation has been completed (1 = evaluated, 0 = not evaluated)';



