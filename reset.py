"""
Script to clear all data from database tables without affecting schema
"""
import sqlite3

DATABASE = 'wakeup_call.db'

def clear_all_data():
    """Clear all data from all tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    try:
        # Delete all data from tables (in order to respect foreign keys)
        print("Clearing all data from database...")
        
        cursor.execute('DELETE FROM auth_tokens')
        print("✓ Cleared auth_tokens")
        
        cursor.execute('DELETE FROM user_surveys')
        print("✓ Cleared user_surveys")
        
        cursor.execute('DELETE FROM users')
        print("✓ Cleared users")
        
        conn.commit()
        print("\n✅ All data cleared successfully!")
        print("Database schema remains intact.")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    response = input("⚠️  This will delete ALL data from the database. Continue? (yes/no): ")
    if response.lower() == 'yes':
        clear_all_data()
    else:
        print("Operation cancelled.")