import pandas as pd
import mysql.connector
import numpy as np

# Load CSV file
df = pd.read_csv("data/SZ000001940.csv")

# Connect to MySQL
conn = mysql.connector.connect(host="localhost", user="root", password="Bastians11284s11284", database="weather_data_international")
cursor = conn.cursor()

# Create new table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tmp (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE,
        latitude FLOAT,
        longitude FLOAT,
        altitude FLOAT,
        name VARCHAR(255),
        tavg FLOAT
    )
""")

# Insert data into MySQL
for _, row in df.iterrows():
    date = row['DATE']
    latitude = row['LATITUDE']
    longitude = row['LONGITUDE']
    altitude = row['ELEVATION']
    name = row['NAME']
    tavg = row['TAVG']
    # print(tavg, type(tavg))
    if pd.isna(tavg):
        tmin = row['TMIN']
        tmax = row['TMAX']
        if pd.isna(tmin) or pd.isna(tmax):
            print("None in ", row)
            continue
        tavg = (tmin + tmax) / 2

    if pd.isna(tavg):
        print("None in ", row)
        continue

    tavg = tavg / 10
    
    cursor.execute("INSERT INTO tmp (date, latitude, longitude, altitude, name, tavg) VALUES (%s, %s, %s, %s, %s, %s)", (date, latitude, longitude, altitude, name, tavg))

# Commit and close
conn.commit()
cursor.close()
conn.close()