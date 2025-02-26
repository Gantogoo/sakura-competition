import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_bloom_dates(parquet_file: str, output_csv: str, output_plot: str):
    # Load the predictions from the Parquet file
    df = pd.read_parquet(parquet_file)

    # Select relevant columns
    df_selected = df[['site_name', 'year', 'pred_first_bloom_date', 'pred_full_bloom_date']]

    # Convert bloom dates to datetime format
    df_selected['pred_first_bloom_date'] = pd.to_datetime(df_selected['pred_first_bloom_date'])
    df_selected['pred_full_bloom_date'] = pd.to_datetime(df_selected['pred_full_bloom_date'])

    # Save to CSV
    df_selected.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")

    # Plot bloom dates
    plt.figure(figsize=(10, 6))
    for site in df_selected['site_name'].unique():
        site_data = df_selected[df_selected['site_name'] == site]
        plt.scatter(site_data['site_name'], site_data['pred_first_bloom_date'], label=f"{site} - First Bloom", marker='o')
        plt.scatter(site_data['site_name'], site_data['pred_full_bloom_date'], label=f"{site} - Full Bloom", marker='x')

    plt.xticks(rotation=90)
    plt.ylabel("Bloom Date")
    plt.title("Predicted Sakura Bloom Dates by City")
    # plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(output_plot)
    print(f"Plot saved to: {output_plot}")
    plt.show()

# Define file paths
target_year = 2022
parquet_file = 'predictions/predictions_' + str(target_year) +'.parquet'
output_csv = 'predictions/bloom_dates_' + str(target_year) +'.csv'
output_plot = 'predictions/bloom_dates_' + str(target_year) +'.png'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
# Run the function
plot_bloom_dates(parquet_file, output_csv, output_plot)