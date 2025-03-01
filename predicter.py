import torch
import pandas as pd
import numpy as np
import os
from darts import TimeSeries
from transformer_prediction import SakuraTransformer  # Ensure this class is properly imported

def load_prediction_data(file_path: str) -> TimeSeries:
    """
    Load and preprocess the prediction data from CSV.
    """
    df = pd.read_csv(file_path)

    # Convert string arrays to float values
    df['temperature_array'] = df['temperature_array'].apply(lambda x: float(x.strip('[]')))
    df['latitude'] = df['latitude'].apply(lambda x: float(x.strip('[]')))
    df['longitude'] = df['longitude'].apply(lambda x: float(x.strip('[]')))
    df['altitude'] = df['altitude'].apply(lambda x: float(x.strip('[]')))

    features = pd.DataFrame({
        'date': pd.date_range(start="2024-08-01", periods=len(df), freq='D'),  # Assign date column
        'temperature': df['temperature_array'],
        'lat': df['latitude'],
        'lng': df['longitude'],
        'alt': df['altitude']
    })

    # Create a time series with the correct column name
    features_ts = TimeSeries.from_dataframe(features, time_col='date')

    return features_ts

def predict_for_2025(model_path: str, prediction_folders: list):
    """
    Load the trained model and use it to predict the bloom dates for 2025.
    """
    # Load trained model
    model = SakuraTransformer(load_pretrained_model=model_path)

    # Dummy training series (required before prediction)
    dates = pd.date_range(start="2024-08-01", periods=212, freq='D')

    dummy_series = TimeSeries.from_times_and_values(
        dates,
        np.zeros((212, 2))  # Assuming two target variables
    )

    dummy_features = TimeSeries.from_times_and_values(
        dates,
        np.zeros((212, 4))  # Assuming four feature variables
    )

    # Fit the model on dummy data to initialize it properly
    print("Fitting the model on dummy data to initialize...")
    model.model.fit(
        series=[dummy_series], 
        past_covariates=[dummy_features], 
        verbose=False
    )

    results = []

    for folder in prediction_folders:
        for file_name in os.listdir(folder):
            if file_name.endswith("_prediction.csv"):
                file_path = os.path.join(folder, file_name)
                print(f"Processing: {file_path}")

                # Extract location name
                location_name = file_name.split('_')[1]

                # Load prediction data
                features_ts = load_prediction_data(file_path)

                # Ensure features have the same length as dummy series
                features_ts = features_ts[:212]

                # Make prediction
                pred_targets = model.model.predict(
                    n=1,
                    series=dummy_series[:-1],  # Provide a valid series
                    past_covariates=features_ts[:-1]
                )

                pred_values = TimeSeries.values(pred_targets)[0]

                results.append({
                    'location': location_name,
                    'predicted_first_bloom': float(pred_values[0]),
                    'predicted_full_bloom': float(pred_values[1])
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("sakura_predictions_2025.csv", index=False)
    print("Predictions saved to sakura_predictions_2025.csv")

if __name__ == "__main__":
    model_path = "transformer_model_test_year_2024.pt"  # Adjust path as needed
    prediction_folders = [
        "data/sql/int/prediction",
        "data/sql/japan/prediction"
    ]
    predict_for_2025(model_path, prediction_folders)
