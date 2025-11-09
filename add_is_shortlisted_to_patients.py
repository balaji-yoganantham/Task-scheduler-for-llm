"""
Script to add is_shortlisted column to patient_medical_history table
and set all existing patients to is_shortlisted = 0
"""

import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from config import DATABASE_URL
from loguru import logger

# Configure logger
logger.add(
    "results/add_is_shortlisted_column_{time:YYYYMMDD_HHmmss}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

def add_is_shortlisted_column():
    """
    Add is_shortlisted column to insightsedge.patient_medical_history table
    and set all existing rows to 0
    """
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        logger.info("Database connection established")
        
        with engine.connect() as connection:
            # Start a transaction
            trans = connection.begin()
            
            try:
                # Check if column already exists
                check_column_query = text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'insightsedge' 
                    AND table_name = 'patient_medical_history' 
                    AND column_name = 'is_shortlisted'
                """)
                
                result = connection.execute(check_column_query)
                column_exists = result.fetchone() is not None
                
                if column_exists:
                    logger.warning("Column 'is_shortlisted' already exists in patient_medical_history table")
                    logger.info("Updating all existing rows to is_shortlisted = 0")
                    
                    # Update all rows to 0
                    update_query = text("""
                        UPDATE insightsedge.patient_medical_history 
                        SET is_shortlisted = 0
                    """)
                    connection.execute(update_query)
                    
                    # Get count of updated rows
                    count_query = text("""
                        SELECT COUNT(*) as total_count 
                        FROM insightsedge.patient_medical_history
                    """)
                    count_result = connection.execute(count_query)
                    total_count = count_result.fetchone()[0]
                    
                    logger.info(f"Successfully updated {total_count} patients to is_shortlisted = 0")
                    
                else:
                    logger.info("Column 'is_shortlisted' does not exist. Creating it...")
                    
                    # Add the column with default value 0
                    alter_table_query = text("""
                        ALTER TABLE insightsedge.patient_medical_history 
                        ADD COLUMN is_shortlisted INTEGER DEFAULT 0 NOT NULL
                    """)
                    connection.execute(alter_table_query)
                    logger.info("Successfully added 'is_shortlisted' column to patient_medical_history table")
                    
                    # Get count of rows
                    count_query = text("""
                        SELECT COUNT(*) as total_count 
                        FROM insightsedge.patient_medical_history
                    """)
                    count_result = connection.execute(count_query)
                    total_count = count_result.fetchone()[0]
                    
                    logger.info(f"Total patients in table: {total_count}")
                    logger.info(f"All {total_count} patients have been set to is_shortlisted = 0")
                
                # Commit the transaction
                trans.commit()
                logger.info("Transaction committed successfully")
                
                # Verify the column and values
                verify_query = text("""
                    SELECT 
                        COUNT(*) as total_patients,
                        COUNT(CASE WHEN is_shortlisted = 0 THEN 1 END) as patients_with_zero,
                        COUNT(CASE WHEN is_shortlisted != 0 THEN 1 END) as patients_with_non_zero
                    FROM insightsedge.patient_medical_history
                """)
                verify_result = connection.execute(verify_query)
                verify_row = verify_result.fetchone()
                
                logger.info("Verification Results:")
                logger.info(f"  - Total patients: {verify_row[0]}")
                logger.info(f"  - Patients with is_shortlisted = 0: {verify_row[1]}")
                logger.info(f"  - Patients with is_shortlisted != 0: {verify_row[2]}")
                
                return True
                
            except SQLAlchemyError as e:
                # Rollback on error
                trans.rollback()
                logger.error(f"Database error occurred: {e}")
                raise
            except Exception as e:
                # Rollback on error
                trans.rollback()
                logger.error(f"Unexpected error occurred: {e}")
                raise
                
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to database: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()
            logger.info("Database connection closed")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting script to add is_shortlisted column to patient_medical_history")
    logger.info("=" * 60)
    
    success = add_is_shortlisted_column()
    
    if success:
        logger.info("=" * 60)
        logger.info("Script completed successfully!")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("Script failed! Check logs for details.")
        logger.error("=" * 60)
        sys.exit(1)

