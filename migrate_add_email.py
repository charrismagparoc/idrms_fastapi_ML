 
"""
migrate_add_email.py
====================
Run this ONCE to add the `email` column to the existing `residents` table.
Safe to run even if the column already exists (it checks first).

Usage:
    python migrate_add_email.py
"""

import sqlite3
import os

# Adjust this path if your db file has a different name/location
DB_PATH = os.path.join(os.path.dirname(__file__), "idrms.db")

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        print("If your db file has a different name, edit DB_PATH in this script.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if email column already exists
    cursor.execute("PRAGMA table_info(residents)")
    columns = [row[1] for row in cursor.fetchall()]

    if "email" in columns:
        print("Column 'email' already exists in residents table. Nothing to do.")
    else:
        cursor.execute("ALTER TABLE residents ADD COLUMN email TEXT DEFAULT ''")
        conn.commit()
        print("SUCCESS: Added 'email' column to residents table.")

    conn.close()

if __name__ == "__main__":
    migrate()