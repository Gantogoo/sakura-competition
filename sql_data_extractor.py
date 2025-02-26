import mysql.connector
import mysql.connector.errorcode as errorcode
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Optional

# Load database credentials from .env
load_dotenv(override=True)

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

    days_to_full = (bloom_data['bloom_date'].values[0] - datetime.strptime(start_date, "%Y-%m-%d").date()).days
    days_to_first = (bloom_data['first_bloom_date'].values[0] - datetime.strptime(start_date, "%Y-%m-%d").date()).days

    bloom_offset = (bloom_data['bloom_date'].values[0] - bloom_data['first_bloom_date'].values[0]).days

    # print("Days to Full Bloom: ", days_to_full, " Days to First Bloom: ", days_to_first, " Bloom Offset: ", bloom_offset)

    countdown_to_first = list(range(days_to_first, -bloom_offset - 1, -1))  # [days_to_first, ..., -bloom_offset]
    countdown_to_full = list(range(days_to_full, -1, -1))

    # print("Countdown to Full Bloom: ", countdown_to_full, " Countdown to First Bloom: ", countdown_to_first)

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
        "temperature_array": np.array(temperature_array),
        "latitude": [latitude] * len(temperature_array),
        "longitude": [longitude] * len(temperature_array),
        "altitude": [altitude] * len(temperature_array)
    }
    input_df = pd.DataFrame(input_data)

    # output_data = {
    #     "bloom_days": [diff_full],
    #     "first_bloom_days": [diff_first]
    # }
    output_data = {
        "bloom_day_countdown": np.array(countdown_to_full),
        "first_bloom_day_countdown": np.array(countdown_to_first)
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

    if not os.path.exists("tmp_full.txt"):
        with open("tmp_full.txt", 'w') as file:
            for number in countdown_to_full:
                file.write(str(number) + " ")
            file.write("\n")
    else:
        with open("tmp_full.txt", 'a') as file:
            for number in countdown_to_full:
                file.write(str(number) + " ")
            file.write("\n")

    if not os.path.exists("tmp_first.txt"):
        with open("tmp_first.txt", 'w') as file:
            for number in countdown_to_first:
                file.write(str(number) + " ")
            file.write("\n")
    else:
        with open("tmp_first.txt", 'a') as file:
            for number in countdown_to_first:
                file.write(str(number) + " ")
            file.write("\n")

    if not os.path.exists("tmp_temp.txt"):
        with open("tmp_temp.txt", 'w') as file:
            for number in temperature_array:
                file.write(str(number) + " ")
            file.write("\n")
    else:
        with open("tmp_temp.txt", 'a') as file:
            for number in temperature_array:
                file.write(str(number) + " ")
            file.write("\n")

    if not os.path.exists("tmp_lat.txt"):
        with open("tmp_lat.txt", 'w') as file:
            file.write(str(latitude) + "\n")
    else:
        with open("tmp_lat.txt", 'a') as file:
            file.write(str(latitude) + "\n")

    if not os.path.exists("tmp_lng.txt"):
        with open("tmp_lng.txt", 'w') as file:
            file.write(str(longitude) + "\n")
    else:
        with open("tmp_lng.txt", 'a') as file:
            file.write(str(longitude) + "\n")
    
    if not os.path.exists("tmp_alt.txt"):
        with open("tmp_alt.txt", 'w') as file:
            file.write(str(altitude) + "\n")
    else:
        with open("tmp_alt.txt", 'a') as file:
            file.write(str(altitude) + "\n")

    # print(f"Input dataset saved as {input_file}")
    # print(f"Output dataset saved as {output_file}")

def scale_data(script_dir):
    if inputstring == 'int':
        folder_path_input = os.path.join(script_dir, "data/sql/int/input")
        folder_path_output = os.path.join(script_dir, "data/sql/int/output")
    elif inputstring == 'japan':
        folder_path_input = os.path.join(script_dir, "data/sql/japan/input")
        folder_path_output = os.path.join(script_dir, "data/sql/japan/output")

    with open("tmp_full.txt", 'r') as file:
        countdown_to_full_lines = file.readlines()
    with open("tmp_first.txt", 'r') as file:
        countdown_to_first_lines = file.readlines()
    with open("tmp_temp.txt", 'r') as file:
        temp_lines = file.readlines()
    with open("tmp_lat.txt", 'r') as file:
        lat_lines = file.readlines()
    with open("tmp_lng.txt", 'r') as file:
        lng_lines = file.readlines()
    with open("tmp_alt.txt", 'r') as file:
        alt_lines = file.readlines()

    int_full_lines = [line.strip().split(" ") for line in countdown_to_full_lines]
    int_first_lines = [line.strip().split(" ") for line in countdown_to_first_lines]
    temp_lines = [line.strip().split(" ") for line in temp_lines]
    lat_lines = [line.strip().split(" ") for line in lat_lines]
    lng_lines = [line.strip().split(" ") for line in lng_lines]
    alt_lines = [line.strip().split(" ") for line in alt_lines]
    # print(temp_lines[0])

    for element in int_full_lines:
        element = [int(x) for x in element]
    for element in int_first_lines:
        element = [int(x) for x in element]
    for element in temp_lines:
        if len(element) <= 1:
            continue
        element = [float(x) for x in element]
    for element in lat_lines:
        element = [float(x) for x in element]
    for element in lng_lines:
        element = [float(x) for x in element]
    for element in alt_lines:
        element = [float(x) for x in element]

    all_values_first = np.concatenate([np.array(x).flatten() for x in int_full_lines if len(x) > 0])
    all_values_full = np.concatenate([np.array(x).flatten() for x in int_first_lines if len(x) > 0])
    all_values_temp = np.concatenate([np.array(x).flatten() for x in temp_lines if len(x) > 1])
    all_values_lat = np.concatenate([np.array(x).flatten() for x in lat_lines if len(x) > 0])
    all_values_lng = np.concatenate([np.array(x).flatten() for x in lng_lines if len(x) > 0])
    all_values_alt = np.concatenate([np.array(x).flatten() for x in alt_lines if len(x) > 0])

    scaler_kwargs: Optional[dict] = None
    scaler_kwargs = scaler_kwargs or {}
    scaler_kwargs['feature_range'] = (-1, 1)
    scaler_first = MinMaxScaler(**scaler_kwargs)
    scaler_full = MinMaxScaler(**scaler_kwargs)
    scaler_temp = MinMaxScaler(**scaler_kwargs)

    scaler_kwargs: Optional[dict] = {'quantile_range': (5, 95)}
    
    robust_lat = RobustScaler(**scaler_kwargs)
    robust_lng = RobustScaler(**scaler_kwargs)
    robust_alt = RobustScaler(**scaler_kwargs)

    # Fit scaler on all values
    scaler_first.fit(all_values_first.reshape(-1, 1))
    scaler_full.fit(all_values_full.reshape(-1, 1))
    scaler_temp.fit(all_values_temp.reshape(-1, 1))

    robust_lat.fit(all_values_lat.reshape(-1, 1))
    robust_lng.fit(all_values_lng.reshape(-1, 1))
    robust_alt.fit(all_values_alt.reshape(-1, 1))

        # Function to scale lists/arrays
    def scale_list_first(x):
        if isinstance(x, int): 
            x = [x]
        if len(x) == 0:
            return x
        return scaler_first.transform(np.array(x).reshape(-1, 1)).flatten().tolist()
    
    def scale_list_full(x):
        if isinstance(x, int):  
            x = [x]
        if len(x) == 0:
            return x
        return scaler_full.transform(np.array(x).reshape(-1, 1)).flatten().tolist()
    
    def scale_list_temp(x):
        if isinstance(x, float): 
            x = [x]
        if len(x) == 0:
            return x
        return scaler_temp.transform(np.array(x).reshape(-1, 1)).flatten().tolist()
    
    def scale_list_lat(x):
        if isinstance(x, float): 
            x = [x]
        if len(x) == 0:
            return x
        return robust_lat.transform(np.array(x).reshape(-1, 1)).flatten().tolist()
    
    def scale_list_lng(x):
        if isinstance(x, float): 
            x = [x]
        if len(x) == 0:
            return x
        return robust_lng.transform(np.array(x).reshape(-1, 1)).flatten().tolist()
    
    def scale_list_alt(x):
        if isinstance(x, float): 
            x = [x]
        if len(x) == 0:
            return x
        return robust_alt.transform(np.array(x).reshape(-1, 1)).flatten().tolist()

    # Get all files and folders in the directory
    files_and_folders = os.listdir(folder_path_output)

    # Filter out only files
    files = [file for file in files_and_folders if os.path.isfile(os.path.join(folder_path_output, file))]

    for file in files:
        # Read the file
        df = pd.read_csv(os.path.join(folder_path_output, file))
        # print(df["bloom_day_countdown"])
        # print(df["first_bloom_day_countdown"])

        # Apply scaling to each list in the column
        df['bloom_day_countdown'] = df['bloom_day_countdown'].apply(scale_list_full)
        df['first_bloom_day_countdown'] = df['first_bloom_day_countdown'].apply(scale_list_first)

        # Save the file
        df.to_csv(os.path.join(folder_path_output, file), index=False)

    files_and_folders = os.listdir(folder_path_input)
    files = [file for file in files_and_folders if os.path.isfile(os.path.join(folder_path_input, file))]

    for file in files:
        # Read the file
        df = pd.read_csv(os.path.join(folder_path_input, file))

        # Apply scaling to each list in the column
        df["temperature_array"] = df['temperature_array'].apply(scale_list_temp)
        df["latitude"] = df['latitude'].apply(scale_list_lat)
        df["longitude"] = df['longitude'].apply(scale_list_lng)
        df["altitude"] = df['altitude'].apply(scale_list_alt)

        # Save the file
        df.to_csv(os.path.join(folder_path_input, file), index=False)

    # Apply scaling to each list in the column
    # df[column_name] = df[column_name].apply(scale_list)

def international_data_loader():
    start_date = "08-01"
    end_date = "02-28"
    target_station_international = ["Washington", "Liestal_Weideli", "New_York", "Vancouver"]
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
            if year not in years_check or year - 1 not in years_check:
                continue
            process_year_data(str(year-1)+"-"+start_date, str(year)+"-"+end_date, station, station_bloom)  

    script_dir = os.path.dirname(os.path.realpath(__file__))
    scale_data(script_dir)
    os.remove(os.path.join(script_dir, "tmp_full.txt"))
    os.remove(os.path.join(script_dir, "tmp_first.txt"))
    os.remove(os.path.join(script_dir, "tmp_temp.txt"))
    os.remove(os.path.join(script_dir, "tmp_lat.txt"))
    os.remove(os.path.join(script_dir, "tmp_lng.txt"))
    os.remove(os.path.join(script_dir, "tmp_alt.txt"))

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
