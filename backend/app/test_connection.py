import psycopg2

conn = psycopg2.connect(
    "postgresql://postgres:Mahima12@localhost:5433/VirtualTryOn"
)

print("Connected successfully!")