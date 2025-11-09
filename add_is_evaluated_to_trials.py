"""
Script to add is_evaluated column to clinical_trial_details table
and set all existing trials to is_evaluated = 0
"""

import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from config import DATABASE_URL
from loguru import logger

# Configure logger
logger.add(
    "results/add_is_evaluated_column_{time:YYYYMMDD_HHmmss}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

def add_is_evaluated_column():
    """
    Add is_evaluated column to insightsedge.clinical_trial_details table
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
                    AND table_name = 'clinical_trial_details' 
                    AND column_name = 'is_evaluated'
                """)
                
                result = connection.execute(check_column_query)
                column_exists = result.fetchone() is not None
                
                if column_exists:
                    logger.warning("Column 'is_evaluated' already exists in clinical_trial_details table")
                    logger.info("Updating all existing rows to is_evaluated = 0")
                    
                    # Update all rows to 0
                    update_query = text("""
                        UPDATE insightsedge.clinical_trial_details 
                        SET is_evaluated = 0
                    """)
                    connection.execute(update_query)
                    
                    # Get count of updated rows
                    count_query = text("""
                        SELECT COUNT(*) as total_count 
                        FROM insightsedge.clinical_trial_details
                    """)
                    count_result = connection.execute(count_query)
                    total_count = count_result.fetchone()[0]
                    
                    logger.info(f"Successfully updated {total_count} trials to is_evaluated = 0")
                    
                else:
                    logger.info("Column 'is_evaluated' does not exist. Creating it...")
                    
                    # Add the column with default value 0
                    alter_table_query = text("""
                        ALTER TABLE insightsedge.clinical_trial_details 
                        ADD COLUMN is_evaluated INTEGER DEFAULT 0 NOT NULL
                    """)
                    connection.execute(alter_table_query)
                    logger.info("Successfully added 'is_evaluated' column to clinical_trial_details table")
                    
                    # Get count of rows
                    count_query = text("""
                        SELECT COUNT(*) as total_count 
                        FROM insightsedge.clinical_trial_details
                    """)
                    count_result = connection.execute(count_query)
                    total_count = count_result.fetchone()[0]
                    
                    logger.info(f"Total trials in table: {total_count}")
                    logger.info(f"All {total_count} trials have been set to is_evaluated = 0")
                
                # Commit the transaction
                trans.commit()
                logger.info("Transaction committed successfully")
                
                # Verify the column and values
                verify_query = text("""
                    SELECT 
                        COUNT(*) as total_trials,
                        COUNT(CASE WHEN is_evaluated = 0 THEN 1 END) as trials_with_zero,
                        COUNT(CASE WHEN is_evaluated != 0 THEN 1 END) as trials_with_non_zero
                    FROM insightsedge.clinical_trial_details
                """)
                verify_result = connection.execute(verify_query)
                verify_row = verify_result.fetchone()
                
                logger.info("Verification Results:")
                logger.info(f"  - Total trials: {verify_row[0]}")
                logger.info(f"  - Trials with is_evaluated = 0: {verify_row[1]}")
                logger.info(f"  - Trials with is_evaluated != 0: {verify_row[2]}")
                
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
    logger.info("Starting script to add is_evaluated column to clinical_trial_details")
    logger.info("=" * 60)
    
    success = add_is_evaluated_column()
    
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

