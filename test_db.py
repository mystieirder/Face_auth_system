from db import get_connection

try:
    conn = get_connection()
    print("Connected successfully!")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sys.tables")
    tables = cursor.fetchall()
    print("Tables found:", tables)
    conn.close()
except Exception as e:
    print("Connection failed:", e)