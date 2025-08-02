import pandas as pd
import mysql.connector

# Update this path to your CSV location
csv_path = ("C:/Users/Navy/Downloads/online_retail.csv")

# Load the CSV file
df = pd.read_csv(csv_path)
df = df.dropna(subset=['CustomerID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df.fillna('')

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Naman@0305',  # Replace with your MySQL password
    database='retaildb'
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS online_retail (
    InvoiceNo VARCHAR(30),
    StockCode VARCHAR(20),
    Description TEXT,
    Quantity INT,
    InvoiceDate DATETIME,
    UnitPrice FLOAT,
    CustomerID INT,
    Country VARCHAR(50)
)
""")

# Insert data into the MySQL table
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO online_retail (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(row['InvoiceNo']),
        str(row['StockCode']),
        str(row['Description']),
        int(row['Quantity']),
        row['InvoiceDate'].strftime('%Y-%m-%d %H:%M:%S'),
        float(row['UnitPrice']),
        int(row['CustomerID']),
        str(row['Country'])
    ))

conn.commit()
cursor.close()
conn.close()
print("Data imported successfully!")
