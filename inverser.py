import pickle
import pandas as pd
import numpy as np
import os

def inverse_scale_predictions(predictions_file: str, scalers_file: str, output_file: str):
    """
    Load the scaler, inverse transform the predictions, and save the results.
    """
    # Load scalers
    script_dir = os.path.dirname(os.path.realpath(__file__))
    scalers_path = os.path.join(script_dir, scalers_file)
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    scaler_first = scalers['countdown_to_first']
    scaler_full = scalers['countdown_to_full']

    # Load predictions
    predictions_path = os.path.join(script_dir, predictions_file)
    df = pd.read_csv(predictions_path)
    
    # Ensure columns are treated as numeric values
    df['predicted_first_bloom'] = pd.to_numeric(df['predicted_first_bloom'], errors='coerce')
    df['predicted_full_bloom'] = pd.to_numeric(df['predicted_full_bloom'], errors='coerce')
    
    # Drop rows with invalid values
    df = df.dropna()

    # Convert predictions to numpy arrays for inverse scaling
    first_bloom_scaled = df['predicted_first_bloom'].values.reshape(-1, 1)
    full_bloom_scaled = df['predicted_full_bloom'].values.reshape(-1, 1)

    # Inverse transform
    first_bloom_unscaled = scaler_first.inverse_transform(first_bloom_scaled).flatten()
    full_bloom_unscaled = scaler_full.inverse_transform(full_bloom_scaled).flatten()

    # Compute lower (full bloom - offset) and upper (full bloom + offset) based on location-specific offsets
    location_offsets = {
        'kyoto': 10,
        'washingtondc': 15,
        'newyorkcity': 16,
        'vancouver': 16,
        'liestal': 14
    }
    
    lower = [int(full_bloom_unscaled[i] - location_offsets.get(loc.lower(), 10)) for i, loc in enumerate(df['location'])]
    upper = [int(full_bloom_unscaled[i] + location_offsets.get(loc.lower(), 10)) for i, loc in enumerate(df['location'])]

    # Create output DataFrame
    output_df = pd.DataFrame({
        'location': df['location'],
        'prediction': full_bloom_unscaled.astype(int),
        'lower': lower,
        'upper': upper
    })

    # Save to new CSV file
    output_path = os.path.join(script_dir, output_file)
    output_df.to_csv(output_path, index=False, quoting=1)

    print(f"Inverse scaled predictions saved to {output_file}")

if __name__ == "__main__":
    predictions_file = "sakura_predictions_2025.csv"  # Adjust path as needed
    scalers_file = "new_scalers.pickle"
    output_file = "cherry_predictions.csv"

    inverse_scale_predictions(predictions_file, scalers_file, output_file)
