import torch
import numpy as np
import pandas as pd
import datetime
import os
import json
import sys
import warnings
from contextlib import contextmanager

# Suppress warnings (mostly from darts)
warnings.filterwarnings("ignore")

# --- Suppress Output Context Manager ---
@contextmanager
def suppress_output():
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        devnull.close()

# --- SakuraTransformer Class ---
class SakuraTransformer:
    def __init__(self, load_pretrained_model: str, training_end_year: int, test_year: int):
        from darts.models import TransformerModel
        from darts import TimeSeries

        self.training_end_year = training_end_year
        self.test_year = test_year
        self.df, self.scalers = self._load_sakura_data()

        self.model = TransformerModel.load(load_pretrained_model)
        self.test_indices = self.df.index[self.df['year'] == self.test_year].tolist()

    def _load_sakura_data(self):
        from sakura_data import load_sakura_data
        df, scalers = load_sakura_data()
        df['data_start_date'] = pd.to_datetime(df['data_start_date'])
        df.reset_index(drop=True, inplace=True)
        return df, scalers

    def _prepare_sequence(self, idx: int):
        from darts import TimeSeries
        row = self.df.iloc[idx]
        start_date = row['data_start_date']
        dates = pd.date_range(start=start_date, periods=len(row['temps_to_full']), freq='D')
        features = pd.DataFrame({
            'temperature': row['temps_to_full'],
            'lat': [row['lat']] * len(dates),
            'lng': [row['lng']] * len(dates)
        }, index=dates)
        targets = pd.DataFrame({
            'countdown_first': row['countdown_to_first'],
            'countdown_full': row['countdown_to_full']
        }, index=dates)
        features_ts = TimeSeries.from_dataframe(features)
        targets_ts = TimeSeries.from_dataframe(targets)
        return features_ts, targets_ts

    def test(self, test_cutoff_date: datetime.datetime = None):
        from darts import TimeSeries
        predictions = []
        for test_idx in self.test_indices:
            row = self.df.iloc[test_idx]
            features, targets = self._prepare_sequence(test_idx)
            start_year = row['data_start_date'].year
            test_start_date = datetime.datetime(start_year, 8, 1)
            cutoff_date = datetime.datetime(start_year + 1, 2, 18) if test_cutoff_date is None else test_cutoff_date
            cutoff_days = (cutoff_date - test_start_date).days + 1
            cutoff = min(cutoff_days, len(features))
            pred_features = features[:cutoff]
            true_targets = targets[:cutoff]
            pred_targets = self.model.predict(n=1, series=true_targets[:-1], past_covariates=pred_features[:-1])
            pred_values = TimeSeries.values(pred_targets)[0]

            # Ensure proper reshaping for inverse_transform
            unscaled_pred_first = self.scalers['countdown_to_first'].inverse_transform(
                np.array(pred_values[0]).reshape(-1, 1)
            )
            unscaled_pred_full = self.scalers['countdown_to_full'].inverse_transform(
                np.array(pred_values[1]).reshape(-1, 1)
            )

            predictions.append({
                'site_name': row['site_name'],
                'year': row['year'],
                'start_date': row['data_start_date'],
                'pred_first_bloom_date': (test_start_date + datetime.timedelta(days=cutoff) +
                                          datetime.timedelta(days=float(unscaled_pred_first))).strftime("%Y-%m-%d"),
                'pred_full_bloom_date': (test_start_date + datetime.timedelta(days=cutoff) +
                                         datetime.timedelta(days=float(unscaled_pred_full))).strftime("%Y-%m-%d")
            })

        if not predictions:
            print("Warning: No predictions were generated!")

        predictions_df = pd.DataFrame(predictions)

        print(f"Number of rows in predictions: {predictions_df.shape[0]}")
        print(f"Columns in predictions: {predictions_df.columns}")
        return predictions_df

# --- Prediction for 2025 ---
def predict_for_2025(pretrained_model_path: str, save_results_dir: str):
    test_year = 2020
    training_end_year = 2022
    cutoff_date = datetime.datetime(test_year, 2, 18)

    sakura_transformer = SakuraTransformer(
        load_pretrained_model=pretrained_model_path,
        training_end_year=training_end_year,
        test_year=test_year
    )

    with suppress_output():
        predictions_df = sakura_transformer.test(test_cutoff_date=cutoff_date)

    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    print(f"Number of rows in predictions: {predictions_df.shape[0]}")
    print(f"Columns in predictions: {predictions_df.columns}")
    predictions_path = os.path.join(save_results_dir, "prediction.parquet")
    predictions_df.to_parquet(predictions_path)

    print(f"Predictions saved to: {predictions_path}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Load a pretrained Sakura Transformer and predict 2025 blossom dates."
    )
    parser.add_argument('--pretrained_model_path', type=str,
                        default='src_test/transformer_d64_h4/sim_id_3/transformer_model_test_year_2024.pt',
                        help="Path to the pretrained transformer model (.pt file)")
    parser.add_argument('--save_results_dir', type=str, default='predictions/',
                        help="Directory where prediction results will be saved")
    args = parser.parse_args()
    predict_for_2025(args.pretrained_model_path, args.save_results_dir)
