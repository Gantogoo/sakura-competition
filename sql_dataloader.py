import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os
import csv
from datetime import datetime, timedelta
import math

load_dotenv()

database_name = os.getenv("MYSQL_DATABASE_INT")

# Connect to MySQL
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=database_name
)
cursor = conn.cursor()

def calculate_first_bloom(full_bloom, bloom_doy):
    days_offset = 4 + (6 * math.exp(-((bloom_doy - 100) / 20) ** 2))
    return full_bloom - timedelta(days=int(days_offset))

def process_data(station_name = "Liestal_Weideli"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if station_name == "Liestal_Weideli":
        file_name = 'liestal.csv'
        target_station_bloom = "liestal_Weideli_bloom"
    elif station_name == "Washington":
        file_name = 'washingtondc.csv'
        target_station_bloom = "Washington_bloom"
    elif station_name == "New_York":
        file_name = 'nyc.csv'
        target_station_bloom = "New_York_bloom"
    elif station_name == "Vancouver":
        file_name = 'vancouver.csv'
        target_station_bloom = "Vancouver_bloom"
    else:
        print("Invalid station name")
        return

    # Load CSV file
    with open(os.path.join(script_dir, 'data', file_name), 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    # # Create new table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {target_station_bloom} (
            year INT PRIMARY KEY,
            latitude DOUBLE,
            longitude DOUBLE,
            altitude DOUBLE,
            bloom_date DATE,
            first_bloom_date DATE
        )
    """)

    for row in rows[1:]:
    # CREATE TABLE Washington_first_bloom AS SELECT year, DATE_SUB(bloom_date, INTERVAL (4 + (6 * EXP(-POW((bloom_doy - 100) / 20, 2)))) DAY) AS first_bloom_date FROM Washington_bloom;
        bloom_date = datetime.strptime(row[5], "%Y-%m-%d")
        first_bloom_date = calculate_first_bloom(bloom_date, int(row[6]))
        query = f"""INSERT INTO {target_station_bloom} (year, latitude, longitude, altitude, bloom_date, first_bloom_date) 
                  VALUES (%s, %s, %s, %s, %s, %s)
                  ON DUPLICATE KEY UPDATE latitude = VALUES(latitude), longitude = VALUES(longitude), altitude = VALUES(altitude), bloom_date = VALUES(bloom_date), first_bloom_date = VALUES(first_bloom_date);"""
        
        cursor.execute(query, (row[4], row[1], row[2], row[3], bloom_date, first_bloom_date))
    
    print("Data inserted successfully for ", station_name)

if __name__ == '__main__':
    name_list = ["Liestal_Weideli", "Washington", "New_York", "Vancouver"]
    for name in name_list:
        process_data(name)

# Commit and close
conn.commit()
cursor.close()
conn.close()

