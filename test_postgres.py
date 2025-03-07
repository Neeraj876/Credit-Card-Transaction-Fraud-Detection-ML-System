import psycopg2

try:
    conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="feast_db",
    user="postgres",
    password="neeraj"
)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)


