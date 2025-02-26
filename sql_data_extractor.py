import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

# Load database credentials from .env
load_dotenv()

inputstring = input("Database_name (int or japan (Default)): ").lower() or 'japan'

if inputstring == "int":
    database_name = os.getenv("MYSQL_DATABASE_INT")
elif inputstring == "japan":
    database_name = os.getenv("MYSQL_DATABASE_JAPAN")
else:
    print("Invalid input")
    exit()

# Connect to MySQL
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=database_name
)
cursor = conn.cursor()

def process_year_data(start_date = "1970-08-01", end_date= "1971-02-28", target_station = "Liestal_Weideli", target_station_bloom = None):
    if target_station_bloom is None:
        target_station_bloom = target_station + "_bloom"

    if inputstring == 'int':
        temp_name = "tavg"
    elif inputstring == 'japan':
        temp_name = "temperature"

    # Extract bloom data for the target year
    target_year = end_date[:4]  # Extract the year
    query_bloom = f"""
    SELECT * FROM {target_station_bloom} WHERE year = "{target_year}";
    """

    cursor.execute(query_bloom)
    bloom_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # print("Bloom Date: ", bloom_data['bloom_date'].values[0], " First Bloom Date: ", bloom_data['first_bloom_date'].values[0], " End Date: ", end_date)
    diff_full = (bloom_data['bloom_date'].values[0] - datetime.strptime(end_date, "%Y-%m-%d").date()).days - 1 # Subtract 1 to ignore the 28th of February
    diff_first = (bloom_data['first_bloom_date'].values[0] - datetime.strptime(end_date, "%Y-%m-%d").date()).days - 1 # Subtract 1 to ignore the 28th of February

    days_to_full = (bloom_data['bloom_date'].values[0] - datetime.strptime(end_date, "%Y-%m-%d").date()).days
    days_to_first = (bloom_data['first_bloom_date'].values[0] - datetime.strptime(start_date, "%Y-%m-%d").date()).days

    bloom_offset = (bloom_data['bloom_date'].values[0] - bloom_data['first_bloom_date'].values[0]).days

    print("Days to Full Bloom: ", days_to_full, " Days to First Bloom: ", days_to_first, " Bloom Offset: ", bloom_offset)
    return

    countdown_to_first = list(range(days_to_first, -bloom_offset - 1, -1))  # [days_to_first, ..., -bloom_offset]
    countdown_to_full = list(range(days_to_full, -1, -1))
    # print("Difference in days: ", diff_full, " First Bloom Date: ", diff_first)

    if diff_full == None:
        print("No bloom date")
        return
    elif diff_full < 0:
        print(f"Bloom date is in the past, there the data from {target_station} is not used!")
        return

    # Extract temperature data for the year before the target date
    query_temp = f"""
    SELECT {temp_name} FROM {target_station}
    WHERE date BETWEEN "{start_date}" AND "{end_date}"
    ORDER BY date;
    """

    cursor.execute(query_temp)
    temp_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # Prepare the input (X) - Temperature as an array and other features
    # Get temperature values for each day as an array of 365 values
    temperature_array = temp_data[temp_name].tolist()

    latitude = bloom_data['latitude'].iloc[0]  # Assuming consistent latitude
    longitude = bloom_data['longitude'].iloc[0]  # Assuming consistent longitude
    altitude = bloom_data['altitude'].iloc[0]  # Assuming consistent elevation

    # Create the input dataframe
    input_data = {
        "temperature_array": [temperature_array],
        "latitude": [latitude],
        "longitude": [longitude],
        "altitude": [altitude]
    }
    input_df = pd.DataFrame(input_data)

    output_data = {
        "bloom_days": [diff_full],
        "first_bloom_days": [diff_first]
    }
    output_df = pd.DataFrame(output_data)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if inputstring == 'int':
        input_file = os.path.join(script_dir,"data/sql/int/input", f"input_{target_station}_{start_date}_{end_date}.csv")
        output_file = os.path.join(script_dir,"data/sql/int/output", f"output_{target_station}_{start_date}_{end_date}.csv")
    elif inputstring == 'japan':
        input_file = os.path.join(script_dir,"data/sql/japan/input", f"input_{target_station}_{start_date}_{end_date}.csv")
        output_file = os.path.join(script_dir,"data/sql/japan/output", f"output_{target_station}_{start_date}_{end_date}.csv")
        
    input_df.to_csv(input_file, index=False)
    output_df.to_csv(output_file, index=False)

    # print(f"Input dataset saved as {input_file}")
    # print(f"Output dataset saved as {output_file}")

def international_data_loader():
    start_date = "08-01"
    end_date = "02-28"
    target_station_international = ["Liestal_Weideli", "New_York", "Vancouver", "Washington"]
    for station in target_station_international:
        station_bloom = station + "_bloom"

        query_years = f"SELECT year, bloom_date FROM {station_bloom};"
        query_years_check = f"SELECT DISTINCT YEAR(date) AS year FROM {station} ORDER BY year;"
        cursor.execute(query_years)
        years, bloom_dates = zip(*[[row[0], row[1]] for row in cursor.fetchall()])
        # print(years)

        cursor.execute(query_years_check)
        years_check = [row[0] for row in cursor.fetchall()]
        # print(years_check)
        # print("Years: ", years)
        # print("Bloom Dates: ", bloom_dates)

        # lowest_date = min(bloom_dates, key=lambda x: (x.month, x.day))

        # print("Lowest Date: ", lowest_date)

        for year in years:
            if year not in years_check and year - 1 not in years_check:
                continue
            process_year_data(str(year-1)+"-"+start_date, str(year)+"-"+end_date, station, station_bloom)   

def japan_data_loader():
    start_date = "08-01"
    end_date = "02-28"
    days_between_start_end = 212
    # target_station_japan = ["Abashiri", "Aikawa", "Akita", "Aomori", "Asahikawa", "Choshi", "Esashi", "Fukue", "Fukui", "Fukuoka", "Fukushima", "Gifu", "Hachijojima", "Hachinohe", "Hakodate", "Hamada", "Hamamatsu",
    #                         "Hikone", "Hiroo", "Hiroshima", "Iida", "Iwamizawa", "Kagoshima", "Kanazawa", "Kobe", "Kochi", "Kofu", "Kumagaya", "Kumamoto", "Kumejima", "Kushiro", "Kutchan", "Kyoto", "Maebashi", "Maizuru",
    #                         "Matsue", "Matsumoto", "Matsuyama", "Mito", "Miyakejima", "Miyako", "Miyakojima", "Miyazaki", "Monbetsu", "Morioka", "Muroran", "Nagano", "Nagasaki", "Nago", "Nagoya", "Naha", "Nara", "Naze",
    #                         "Nemuro", "Niigata", "Nobeoka", "Obihiro", "Oita", "Okayama", "Onahama", "Osaka", "Oshima", "Rumoi", "Saga", "Saigo", "Sakata", "Sapporo", "Sendai", "Shimonoseki", "Shinjo", "Shirakawa", "Shizuoka",
    #                         "Sumoto", "Takada", "Takamatsu", "Takayama", "Tanegashima", "Tateyama", "Tokushima", "Tokyo", "Tottori", "Toyama", "Toyooka", "Tsu", "Tsuruga", "Urakawa", "Utsunomiya", "Uwajima", "Wajima", 
    #                         "Wakayama", "Wakkanai", "Yakushima", "Yamagata", "Yokohama", "Yonago", "Yonaguni_Island", "Minami_Daito_Island", "Ishigaki_Island", "Iriomote_Island"]
    
    query_tables = "SHOW TABLES;"
    cursor.execute(query_tables)
    tables = [row[0] for row in cursor.fetchall()]
    target_station_japan = [table for table in tables if not table.endswith("_bloom")]

    for station in target_station_japan:
        station_bloom = station + "_bloom"

        query_years = f"SELECT year, bloom_date FROM {station_bloom};"
        cursor.execute(query_years)
        years, bloom_dates = zip(*[[row[0], row[1]] for row in cursor.fetchall()])
        # print("Station: ", station_bloom, " Years: ", years)

        query_years_check = f"SELECT DISTINCT YEAR(date) AS year FROM {station} ORDER BY year;"
        cursor.execute(query_years_check)
        years_check = [row[0] for row in cursor.fetchall()]

        # print("Years: ", years, " Check: ", years_check)
        for year in years:
            # if year not in years_check and year - 1 not in years_check:
            #     continue

            query_temp = f"""
                SELECT * FROM {station}
                WHERE date BETWEEN "{str(year-1)+"-"+start_date}" AND "{str(year)+"-"+end_date}"
                ORDER BY date;
            """

            cursor.execute(query_temp)

            temp_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            temperature_array = temp_data['temperature'].tolist()
            [temperature_array.remove(None) for i in range(temperature_array.count(None))]

            if len(temperature_array) < days_between_start_end:
                if len(temperature_array) != 0 and len(temperature_array) != 59 and len(temperature_array) != 154: # 0 ist das Jahr 1953, 1954 und 2021, 59 ist das Jahr 1955 und 154 ist das Jahr 2020
                    print("Only ", len(temperature_array), " days of data in ", year, " for ", station)
                continue

            process_year_data(str(year-1)+"-"+start_date, str(year)+"-"+end_date, station, station_bloom)  

def main():
    if inputstring == 'int':
        international_data_loader()
    elif inputstring == 'japan':
        japan_data_loader()


if __name__ == "__main__":
    main()

# Close connection
cursor.close()
conn.close()
