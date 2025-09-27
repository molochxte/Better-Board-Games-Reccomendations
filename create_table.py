#!/usr/bin/env python3
"""
Script to create the board_games_embeddings table in your Neon database.
This script reads the SQL file and executes it against your database.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys

def create_table():
    """Create the board_games_embeddings table in the Neon database."""
    
    # Get database connection string from environment
    connection_string = os.getenv('NEON_CONNECTION_STRING')
    
    if not connection_string:
        print("❌ Error: NEON_CONNECTION_STRING environment variable not found!")
        print("Please set your database connection string in the .env file or environment.")
        return False
    
    try:
        # Read the SQL file
        with open('create_board_games_table.sql', 'r') as file:
            sql_script = file.read()
        
        print("🔗 Connecting to Neon database...")
        
        # Connect to the database
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("📝 Executing SQL script to create table...")
        
        # Execute the SQL script
        cursor.execute(sql_script)
        
        print("✅ Table 'board_games_embeddings' created successfully!")
        print("")
        print("📊 Table structure:")
        print("   - Primary key: id (SERIAL)")
        print("   - BGG ID: bgg_id (VARCHAR, UNIQUE)")
        print("   - Game info: name, year_published, min/max_players, etc.")
        print("   - Vector embedding: embedding (VECTOR(1536))")
        print("   - Timestamps: created_at, updated_at")
        print("   - Indexes: Optimized for vector similarity search")
        print("")
        print("🎯 Ready for your custom Neon database component to populate!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except FileNotFoundError:
        print("❌ Error: create_board_games_table.sql file not found!")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Creating board_games_embeddings table in Neon database...")
    print("")
    
    success = create_table()
    
    if success:
        print("")
        print("🎉 Setup complete! You can now use your custom Neon database component")
        print("   to populate this table with the 8000 board games from your CSV file.")
        sys.exit(0)
    else:
        print("")
        print("💥 Setup failed! Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
