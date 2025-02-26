import mysql.connector
import csv
from datetime import datetime, timedelta
import math

# Load environment variables or credentials
import os
from dotenv import load_dotenv
load_dotenv()

inputstring = input("Database_name (int or japan (Default)): ").lower() or 'japan'

if inputstring == "int":
    database_name = os.getenv("MYSQL_DATABASE_INT")
elif inputstring == "japan":
    database_name = os.getenv("MYSQL_DATABASE_JAPAN")
else:
    print("Invalid input")
    exit()

# Database connection
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=database_name
)
cursor = conn.cursor()

def create_table(city_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {city_name}_bloom (
        year INT PRIMARY KEY,
        city_name VARCHAR(255),
        longitude DOUBLE,
        latitude DOUBLE,
        altitude DOUBLE,
        bloom_date DATE,
        first_bloom_date DATE,
        bloom_doy INT
    );
    """
    cursor.execute(create_table_query)

def calculate_first_bloom(full_bloom, bloom_doy):
    days_offset = 4 + (6 * math.exp(-((bloom_doy - 100) / 20) ** 2))
    return full_bloom - timedelta(days=int(days_offset))

def load_bloom_data(file_path, file_path_first_bloom):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows_full = list(reader)
    with open(file_path_first_bloom, 'r') as csv_file2:
        reader2 = csv.reader(csv_file2)
        rows_first_bloom = list(reader2)

    locations = [row[0] for row in rows_full[1:]]
    locations = list(set(locations))
    locations.sort()

    for location in locations:
        # print("Location: ", location.split('/')[1])
        city_name = location.split('/')[1]

        city_name_in_first_bloom = False
        saved_row = []
        for row in rows_first_bloom[1:]:
            if city_name == row[0]:
                saved_row = row
                city_name_in_first_bloom = True
                break

        create_table(city_name)
        filtered_rows = [row for row in rows_full[1:] if row[0].split('/')[1] == city_name]

        for row in filtered_rows:
            latitude = row[1]
            longitude = row[2]
            altitude = row[3]
            year = row[4]
            full_bloom_date = row[5]
            bloom_doy = row[6]

            if city_name_in_first_bloom:
                first_bloom_date = next((date for date in saved_row[2:-2] if date.startswith(year)), None)

            if first_bloom_date == None:
                first_bloom_date = calculate_first_bloom(datetime.strptime(full_bloom_date, "%Y-%m-%d"), int(bloom_doy))
            
            insert_query = f"""
                INSERT INTO {city_name}_bloom (year, city_name, longitude, latitude, altitude, bloom_date, first_bloom_date, bloom_doy)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE city_name = VALUES(city_name), longitude = VALUES(longitude), latitude = VALUES(latitude), altitude = VALUES(altitude), 
                bloom_date = VALUES(bloom_date), first_bloom_date = VALUES(first_bloom_date), bloom_doy = VALUES(bloom_doy);
            """

            cursor.execute(insert_query, (year, city_name, longitude, latitude, altitude, full_bloom_date, first_bloom_date, bloom_doy))
        
        conn.commit()

def search_for_none(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    locations = [row[0] for row in rows[1:]]
    locations = list(set(locations))
    locations.sort()
    # print(locations)
    no_none = True

    for location in locations:
        city_name = location.split('/')[1]

        query = f"""
            SELECT * FROM {city_name}_bloom;
        """

        cursor.execute(query)

        new_rows = cursor.fetchall()

        has_none = any(None in tup for tup in new_rows)
        if has_none:
            no_none = False
            print("The city ", city_name, " has None values.")

    if no_none:    
        print("No None values found.")


def main():
    # Provide the path to your CSV file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # 
    # file_name_long_lat = 'worldcities.csv'
    # file_name_long_lat_extra = 'lat_long_extra_cities.csv'
    file_name = 'japan.csv'
    file_name_first_bloom = 'sakura_first_bloom_dates_new.csv'

    load_bloom_data(os.path.join(script_dir, 'data', file_name), os.path.join(script_dir, 'data', file_name_first_bloom))
    search_for_none(os.path.join(script_dir, 'data', file_name))

    # Load the data
    # load_bloom_data(os.path.join(script_dir, 'data', file_name), os.path.join(script_dir, 'data', file_name_long_lat), os.path.join(script_dir, 'data', file_name_long_lat_extra))

if __name__ == '__main__':
    main()

# Close the cursor and connection
cursor.close()
conn.close()
