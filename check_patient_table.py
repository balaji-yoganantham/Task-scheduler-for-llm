"""
Script to manually check patient_medical_history table structure and values
"""

from sqlalchemy import create_engine, text, inspect
from config import DATABASE_URL
import json
from datetime import datetime

def check_patient_table():
    """Check patient_medical_history table structure and sample data"""
    
    print("=" * 80)
    print("PATIENT_MEDICAL_HISTORY TABLE INSPECTION")
    print("=" * 80)
    print()
    
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            # 1. Get table structure (columns and data types)
            print("1. TABLE STRUCTURE (Columns and Data Types)")
            print("-" * 80)
            
            inspector = inspect(engine)
            columns = inspector.get_columns('patient_medical_history', schema='insightsedge')
            
            if not columns:
                print("ERROR: Table 'insightsedge.patient_medical_history' not found!")
                return
            
            print(f"\nFound {len(columns)} columns in patient_medical_history table:\n")
            
            column_info = []
            for col in columns:
                col_name = col['name']
                col_type = str(col['type'])
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                default = f" DEFAULT {col['default']}" if col.get('default') is not None else ""
                
                column_info.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': nullable,
                    'default': default
                })
                
                print(f"  {col_name:40} | {col_type:30} | {nullable}")
            
            print()
            print("=" * 80)
            print()
            
            # 2. Get total row count
            print("2. TABLE STATISTICS")
            print("-" * 80)
            
            count_query = text("SELECT COUNT(*) FROM insightsedge.patient_medical_history")
            result = connection.execute(count_query)
            total_rows = result.fetchone()[0]
            print(f"Total rows in table: {total_rows}")
            print()
            
            # 3. Get sample rows with all column values
            print("3. SAMPLE DATA (First 5 rows)")
            print("-" * 80)
            
            # Get all column names
            column_names = [col['name'] for col in columns]
            
            # Build SELECT query with all columns
            columns_str = ", ".join([f"pmh.{col}" for col in column_names])
            
            sample_query = text(f"""
                SELECT {columns_str}
                FROM insightsedge.patient_medical_history pmh
                ORDER BY pmh.id ASC
                LIMIT 5
            """)
            
            result = connection.execute(sample_query)
            rows = result.fetchall()
            
            if not rows:
                print("No data found in table!")
            else:
                for idx, row in enumerate(rows, 1):
                    print(f"\n--- ROW {idx} ---")
                    print()
                    
                    for i, col_name in enumerate(column_names):
                        value = row[i]
                        
                        # Format value for display
                        if value is None:
                            display_value = "NULL"
                        elif isinstance(value, (datetime,)):
                            display_value = value.isoformat()
                        elif isinstance(value, (str,)) and len(str(value)) > 100:
                            display_value = str(value)[:100] + "... (truncated)"
                        else:
                            display_value = str(value)
                        
                        print(f"  {col_name:40} = {display_value}")
            
            print()
            print("=" * 80)
            print()
            
            # 4. Get column statistics (non-null counts, unique values for key columns)
            print("4. COLUMN STATISTICS")
            print("-" * 80)
            
            # Check non-null counts for important columns
            important_columns = ['id', 'mrn', 'age', 'gender', 'chief_complaint', 
                                'history_of_present_illness', 'assessment', 'date_of_visit']
            
            for col_name in important_columns:
                if col_name in column_names:
                    stats_query = text(f"""
                        SELECT 
                            COUNT(*) as total,
                            COUNT({col_name}) as non_null,
                            COUNT(DISTINCT {col_name}) as unique_count
                        FROM insightsedge.patient_medical_history
                    """)
                    result = connection.execute(stats_query)
                    stats = result.fetchone()
                    
                    total, non_null, unique_count = stats
                    null_count = total - non_null
                    
                    print(f"\n  {col_name}:")
                    print(f"    Total rows: {total}")
                    print(f"    Non-null: {non_null} ({non_null/total*100:.1f}%)")
                    print(f"    Null: {null_count} ({null_count/total*100:.1f}%)")
                    print(f"    Unique values: {unique_count}")
            
            print()
            print("=" * 80)
            print()
            
            # 5. Sample values for text columns (to see what kind of data they contain)
            print("5. SAMPLE TEXT VALUES (to understand data content)")
            print("-" * 80)
            
            text_columns = ['chief_complaint', 'history_of_present_illness', 
                          'past_medical_history', 'assessment', 'imaging']
            
            for col_name in text_columns:
                if col_name in column_names:
                    sample_query = text(f"""
                        SELECT {col_name}
                        FROM insightsedge.patient_medical_history
                        WHERE {col_name} IS NOT NULL 
                          AND {col_name} != ''
                          AND LENGTH({col_name}) > 10
                        LIMIT 1
                    """)
                    result = connection.execute(sample_query)
                    row = result.fetchone()
                    
                    if row and row[0]:
                        value = row[0]
                        # Truncate long values
                        display_value = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                        print(f"\n  {col_name}:")
                        print(f"    Sample: {display_value}")
            
            print()
            print("=" * 80)
            print("INSPECTION COMPLETE")
            print("=" * 80)
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_patient_table()

