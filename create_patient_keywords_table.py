"""
Script to create patient_keywords table if it doesn't exist
"""

from sqlalchemy import create_engine, text, inspect
from config import DATABASE_URL

def create_patient_keywords_table():
    """Create patient_keywords table if it doesn't exist"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS insightsedge.patient_keywords (
        id SERIAL PRIMARY KEY,
        patient_id INTEGER NOT NULL UNIQUE,
        mrn VARCHAR(50),
        age INTEGER,
        gender VARCHAR(20),
        summary TEXT,
        primary_diagnosis TEXT,
        stage VARCHAR(50),
        metastatic_sites JSONB,
        molecular_markers JSONB,
        comorbidities JSONB,
        medications JSONB,
        allergies JSONB,
        performance_status VARCHAR(50),
        family_history JSONB,
        keywords JSONB,
        keywords_text TEXT,
        generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create index on patient_id for faster lookups
    CREATE INDEX IF NOT EXISTS idx_patient_keywords_patient_id 
    ON insightsedge.patient_keywords(patient_id);
    
    -- Create index on mrn for faster lookups
    CREATE INDEX IF NOT EXISTS idx_patient_keywords_mrn 
    ON insightsedge.patient_keywords(mrn);
    
    -- Create trigger function to update updated_at timestamp
    CREATE OR REPLACE FUNCTION insightsedge.update_patient_keywords_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    
    -- Create trigger to auto-update updated_at
    DROP TRIGGER IF EXISTS update_patient_keywords_updated_at ON insightsedge.patient_keywords;
    CREATE TRIGGER update_patient_keywords_updated_at
    BEFORE UPDATE ON insightsedge.patient_keywords
    FOR EACH ROW
    EXECUTE FUNCTION insightsedge.update_patient_keywords_updated_at();
    """
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Creating patient_keywords table if it doesn't exist...")
            print("-" * 80)
            
            # Check if table exists
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema='insightsedge')
            
            if 'patient_keywords' in tables:
                print("Table 'patient_keywords' already exists.")
                print("Checking columns...")
                
                # Get existing columns
                columns = inspector.get_columns('patient_keywords', schema='insightsedge')
                print(f"Found {len(columns)} columns in existing table.")
                for col in columns:
                    print(f"  - {col['name']}: {col['type']}")
            else:
                print("Table 'patient_keywords' does not exist. Creating it...")
                
                # Execute the table creation
                connection.execute(text(create_table_sql))
                connection.commit()
                
                print("SUCCESS: patient_keywords table created successfully!")
                print("\nTable structure:")
                print("  - id (SERIAL PRIMARY KEY)")
                print("  - patient_id (INTEGER NOT NULL UNIQUE)")
                print("  - mrn (VARCHAR(50))")
                print("  - age (INTEGER)")
                print("  - gender (VARCHAR(20))")
                print("  - summary (TEXT)")
                print("  - primary_diagnosis (TEXT)")
                print("  - stage (VARCHAR(50))")
                print("  - metastatic_sites (JSONB)")
                print("  - molecular_markers (JSONB)")
                print("  - comorbidities (JSONB)")
                print("  - medications (JSONB)")
                print("  - allergies (JSONB)")
                print("  - performance_status (VARCHAR(50))")
                print("  - family_history (JSONB)")
                print("  - keywords (JSONB)")
                print("  - keywords_text (TEXT)")
                print("  - generated_at (TIMESTAMPTZ)")
                print("  - created_at (TIMESTAMPTZ)")
                print("  - updated_at (TIMESTAMPTZ)")
                print("\nIndexes created:")
                print("  - idx_patient_keywords_patient_id")
                print("  - idx_patient_keywords_mrn")
                print("\nTrigger created:")
                print("  - update_patient_keywords_updated_at (auto-updates updated_at)")
            
    except Exception as e:
        print(f"ERROR: Error creating patient_keywords table: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_patient_keywords_table()

