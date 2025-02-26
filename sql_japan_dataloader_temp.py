import csv
import mysql.connector
from dotenv import load_dotenv
import os

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
            CREATE TABLE IF NOT EXISTS `{city_name}` (
                date DATE PRIMARY KEY,
                temperature FLOAT,
                city_name VARCHAR(255)
            );
        """
    cursor.execute(create_table_query)

def main():
    csv_file = 'data/Japanese_City_Temps_new.csv'
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = list(csv_reader)

    city_names = header[1:]  # Exclude the first column (Date)
    
    insert_query = f"""
        SHOW TABLES LIKE '%_bloom';
    """

    cursor.execute(insert_query)
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    indices = []
    for name in table_names:
        indices.append(city_names.index(name.split('_')[0]))
    
    for table in table_names:
        city_name_table = table.split('_')[0]
        if city_name_table not in city_names:
            print(city_name_table, " has no temperature data")
            continue
    
        create_table(city_name_table)
        
    print("Tables created successfully.")

    # Insert data into corresponding city tables
    for row in rows:
        date = row[0]
        for idx in indices:
            city = city_names[idx]
            temperature = row[idx + 1]

            if temperature == '':
                temperature = None
            
            # Insert into the table for that city
            insert_query = f"""
                INSERT INTO `{city}` (date, temperature, city_name)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE temperature = VALUES(temperature);
            """
            cursor.execute(insert_query, (date, temperature, city))
        
        print(f"Data inserted for {date}.")

# def fill_missing_dates(city_name, indices):
#     query = f"""
#         SELECT * FROM {city_name};
#     """

#     cursor.execute(query)
#     new_rows = cursor.fetchall()
#     for i in indices:
#         # print("The city ", city_name, " has missing temperature data at: ", new_rows[i])
#         insert_query = f"""
#             INSERT INTO `{city_name}` (date, temperature, city_name)
#             VALUES (%s, %s, %s)
#             ON DUPLICATE KEY UPDATE temperature = VALUES(temperature);
#         """

#         if new_rows[i-1][1] is not None :
#             cursor.execute(insert_query, (new_rows[i][0], new_rows[i-1][1], city_name))
#         elif i > 1 and new_rows[i-2][1] is not None:
#             cursor.execute(insert_query, (new_rows[i][0], new_rows[i-2][1], city_name))

def search_for_repairable_none(file_path):
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
            SELECT * FROM {city_name};
        """

        cursor.execute(query)

        new_rows = cursor.fetchall()

        has_none = any(None in tup for tup in new_rows)
        indices = []
        if has_none:
            none_counts = [sum(1 for item in tup if item is None) for tup in new_rows]
            for i, value in enumerate(none_counts):
                if value > 0:
                    indices.append(i)
            
            no_none = False

        cleaned_indices = [indices[i] for i in range(len(indices)) if i < 2 or indices[i] - indices[i-1] != 1 or indices[i-1] - indices[i-2] != 1]
        if 0 in cleaned_indices:
            cleaned_indices.remove(0)
        if 1 in cleaned_indices:
            cleaned_indices.remove(1)

        for idx in cleaned_indices:
            print("The city ", city_name, " has None values at: ", new_rows[idx])
            # print("The city ", city_name, " has None values at the positions: ", none_counts)

    if no_none:    
        print("No None values found.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = 'japan.csv'

    main()

    search_for_repairable_none(os.path.join(script_dir, 'data', file_name))

# Commit the changes and close connection
conn.commit()
cursor.close()
conn.close()

# print("Data successfully loaded into the database.")
