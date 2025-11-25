"""
Manual script to update the is_evaluated flag in clinical_trial_details table
"""
import sys
from sqlalchemy import create_engine, text
from config import DATABASE_URL
from typing import Optional, List, Dict, Any


class IsevaluationFlagUpdater:
    """Utility class to update is_evaluated flag in insightsedge.clinical_trial_details table"""
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = create_engine(
            DATABASE_URL,
            pool_size=2,
            max_overflow=3,
            pool_recycle=3600,
            pool_pre_ping=True,
            pool_timeout=30,
            echo=False
        )
    
    def get_connection(self):
        """Get database connection"""
        return self.engine.connect()
    
    def update_by_nct_id(self, nct_id: str, is_evaluated: bool) -> bool:
        """
        Update is_evaluated flag for a specific trial by NCT ID
        
        Args:
            nct_id: The NCT ID of the trial to update (e.g., "NCT03093116")
            is_evaluated: The new value for is_evaluated flag (True/False)
        
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                query = text("""
                    UPDATE insightsedge.clinical_trial_details 
                    SET is_evaluated = :is_evaluated 
                    WHERE nct_id = :nct_id
                """)
                result = connection.execute(query, {
                    "nct_id": nct_id,
                    "is_evaluated": 1 if is_evaluated else 0
                })
                connection.commit()
                
                if result.rowcount > 0:
                    print(f"✓ Successfully updated trial {nct_id}: is_evaluated = {is_evaluated}")
                    return True
                else:
                    print(f"✗ No trial found with NCT ID {nct_id}")
                    return False
        except Exception as e:
            print(f"✗ Error updating trial {nct_id}: {e}")
            return False
    
    def update_by_nct_ids(self, nct_ids: List[str], is_evaluated: bool) -> Dict[str, int]:
        """
        Update is_evaluated flag for multiple trials by NCT IDs
        
        Args:
            nct_ids: List of NCT IDs to update
            is_evaluated: The new value for is_evaluated flag (True/False)
        
        Returns:
            Dictionary with success and failure counts
        """
        success_count = 0
        failure_count = 0
        
        for nct_id in nct_ids:
            if self.update_by_nct_id(nct_id, is_evaluated):
                success_count += 1
            else:
                failure_count += 1
        
        return {"success": success_count, "failure": failure_count}
    
    def update_all(self, is_evaluated: bool, condition: Optional[str] = None) -> int:
        """
        Update is_evaluated flag for all trials (or matching a condition)
        
        Args:
            is_evaluated: The new value for is_evaluated flag (True/False)
            condition: Optional WHERE clause condition (without WHERE keyword)
                      Example: "overall_status = 'Recruiting'" or "is_evaluated = 0"
        
        Returns:
            Number of records updated
        """
        try:
            with self.get_connection() as connection:
                if condition:
                    query = text(f"""
                        UPDATE insightsedge.clinical_trial_details 
                        SET is_evaluated = :is_evaluated 
                        WHERE {condition}
                    """)
                else:
                    query = text("""
                        UPDATE insightsedge.clinical_trial_details 
                        SET is_evaluated = :is_evaluated
                    """)
                
                result = connection.execute(query, {"is_evaluated": 1 if is_evaluated else 0})
                connection.commit()
                
                updated_count = result.rowcount
                condition_text = f" (condition: {condition})" if condition else ""
                print(f"✓ Successfully updated {updated_count} record(s){condition_text}: is_evaluated = {is_evaluated}")
                return updated_count
        except Exception as e:
            print(f"✗ Error updating records: {e}")
            return 0
    
    def get_trial_info(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific trial
        
        Args:
            nct_id: The NCT ID of the trial to query
        
        Returns:
            Dictionary with trial information or None if not found
        """
        try:
            with self.get_connection() as connection:
                query = text("""
                    SELECT nct_id, study_title, is_evaluated, overall_status, created_at
                    FROM insightsedge.clinical_trial_details 
                    WHERE nct_id = :nct_id
                """)
                result = connection.execute(query, {"nct_id": nct_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "nct_id": row[0],
                        "study_title": row[1],
                        "is_evaluated": row[2],
                        "overall_status": row[3],
                        "created_at": row[4]
                    }
                else:
                    return None
        except Exception as e:
            print(f"✗ Error getting trial info: {e}")
            return None
    
    def count_records(self, condition: Optional[str] = None) -> int:
        """
        Count trials matching a condition
        
        Args:
            condition: Optional WHERE clause condition (without WHERE keyword)
        
        Returns:
            Number of trials matching the condition
        """
        try:
            with self.get_connection() as connection:
                if condition:
                    query = text(f"""
                        SELECT COUNT(*) 
                        FROM insightsedge.clinical_trial_details 
                        WHERE {condition}
                    """)
                else:
                    query = text("SELECT COUNT(*) FROM insightsedge.clinical_trial_details")
                
                result = connection.execute(query)
                count = result.scalar()
                return count
        except Exception as e:
            print(f"✗ Error counting records: {e}")
            return 0


def main():
    """Main function for command-line usage"""
    updater = IsevaluationFlagUpdater()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python update_isevaluation_flag.py <command> [arguments]")
        print("\nCommands:")
        print("  update-by-id <nct_id> <true|false>     - Update a single trial by NCT ID")
        print("  update-by-ids <nct_id1,nct_id2,...> <true|false> - Update multiple trials by NCT IDs")
        print("  update-all <true|false> [condition] - Update all trials (optionally with WHERE condition)")
        print("  info <nct_id>                          - Show information about a trial")
        print("  count [condition]                  - Count trials (optionally with WHERE condition)")
        print("\nExamples:")
        print("  python update_isevaluation_flag.py update-by-id NCT03093116 true")
        print("  python update_isevaluation_flag.py update-by-ids NCT03093116,NCT06298916 false")
        print("  python update_isevaluation_flag.py update-all false")
        print("  python update_isevaluation_flag.py update-all true \"overall_status = 'Recruiting'\"")
        print("  python update_isevaluation_flag.py info NCT03093116")
        print("  python update_isevaluation_flag.py count \"is_evaluated = 0\"")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "update-by-id":
        if len(sys.argv) < 4:
            print("Error: update-by-id requires <nct_id> and <true|false>")
            sys.exit(1)
        
        nct_id = sys.argv[2]
        is_evaluated = sys.argv[3].lower() == "true"
        
        # Show current info before update
        info = updater.get_trial_info(nct_id)
        if info:
            print(f"Current trial info: NCT ID={info['nct_id']}, Title={info['study_title'][:50]}..., is_evaluated={info['is_evaluated']}")
        
        updater.update_by_nct_id(nct_id, is_evaluated)
    
    elif command == "update-by-ids":
        if len(sys.argv) < 4:
            print("Error: update-by-ids requires <nct_id1,nct_id2,...> and <true|false>")
            sys.exit(1)
        
        nct_ids = [id.strip() for id in sys.argv[2].split(",")]
        is_evaluated = sys.argv[3].lower() == "true"
        
        print(f"Updating {len(nct_ids)} trial(s)...")
        result = updater.update_by_nct_ids(nct_ids, is_evaluated)
        print(f"\nSummary: {result['success']} successful, {result['failure']} failed")
    
    elif command == "update-all":
        if len(sys.argv) < 3:
            print("Error: update-all requires <true|false>")
            sys.exit(1)
        
        is_evaluated = sys.argv[2].lower() == "true"
        condition = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Show count before update
        count = updater.count_records(condition)
        print(f"Found {count} trial(s) to update")
        
        if count > 0:
            confirm = input(f"Are you sure you want to update {count} trial(s)? (yes/no): ").strip().lower()
            if confirm in ["yes", "y"]:
                updater.update_all(is_evaluated, condition)
            else:
                print("Update cancelled")
        else:
            print("No trials found to update")
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("Error: info requires <nct_id>")
            sys.exit(1)
        
        nct_id = sys.argv[2]
        info = updater.get_trial_info(nct_id)
        
        if info:
            print(f"\nTrial Information:")
            print(f"  NCT ID: {info['nct_id']}")
            print(f"  Study Title: {info['study_title']}")
            print(f"  is_evaluated: {info['is_evaluated']}")
            print(f"  Overall Status: {info['overall_status']}")
            print(f"  Created At: {info['created_at']}")
        else:
            print(f"Trial with NCT ID {nct_id} not found")
    
    elif command == "count":
        condition = sys.argv[2] if len(sys.argv) > 2 else None
        count = updater.count_records(condition)
        condition_text = f" (condition: {condition})" if condition else ""
        print(f"Total trials{condition_text}: {count}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
