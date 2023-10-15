import sqlite3

def clear_tables(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear (delete all rows) from the Runs, Metrics, and CheckpointPaths tables
        cursor.execute('DELETE FROM Runs;')
        cursor.execute('DELETE FROM Metrics;')
        cursor.execute('DELETE FROM CheckpointPaths;')
        
        cursor.execute('DROP TABLE IF EXISTS Metrics;')
        cursor.execute('DROP TABLE IF EXISTS CheckpointPaths;')
        
        # Commit the changes
        conn.commit()
        print("Tables `Runs`, `Metrics`, and `CheckpointPaths` have been cleared.")
        
    except sqlite3.Error as e:
        print("An error occurred:", e.args[0])
    
    finally:
        # Always close the connection
        conn.close()

# Specify your database path
db_path = '/scratch/vgn2004/metrics.sqlite3'

# Call the function to clear the tables
clear_tables(db_path)