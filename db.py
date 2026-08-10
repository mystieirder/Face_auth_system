import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LOCHAN\\LOCHAN;"
        "DATABASE=multimodal_auth_db;"
        "Trusted_Connection=yes;"
    )
    return conn