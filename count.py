import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Constants ---
MODEL_PATH = 'rul_model.h5'
SCALER_PATH = 'rul_scaler.joblib'
FEATURES_PATH = 'feature_cols.json'
TEST_DATA_PATH = 'test_FD004.txt'

SEQUENCE_LENGTH = 50
ROLLING_WINDOW_SIZE = 5

# --- Helper Functions (Copied from our dashboard) ---

def load_assets():
    """Loads the model, scaler, and feature list from disk."""
    print("Loading predictive assets...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None, None

def load_test_data():
    """Loads the test data and assigns column names."""
    print("Loading test data...")
    try:
        operational_settings = ['setting_1', 'setting_2', 'setting_3']
        sensor_measurements = [f'sensor_{i}' for i in range(1, 22)]
        column_names = ['engine_id', 'cycle'] + operational_settings + sensor_measurements
        
        data = pd.read_csv(TEST_DATA_PATH, sep='\s+', header=None, names=column_names)
        data = data.dropna(axis=1, how='all')
        return data
    except FileNotFoundError:
        print(f"Test data file '{TEST_DATA_PATH}' not found.")
        return None

def add_rolling_features(df, sensor_cols, window_size):
    """Adds rolling features to the dataframe."""
    grouped = df.groupby('engine_id')
    
    non_constant_sensors = []
    for col in sensor_cols:
        if df[col].nunique() > 1:
            non_constant_sensors.append(col)

    for col in non_constant_sensors:
        df[f'{col}_roll_avg'] = grouped[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{col}_roll_std'] = grouped[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    all_rolling_cols = [f'{c}_roll_avg' for c in sensor_cols] + [f'{c}_roll_std' for c in sensor_cols]
    for col in all_rolling_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    return df

def get_health_status(rul):
    """Returns a status string based on the RUL."""
    if rul > 100:
        return "Healthy"
    elif rul > 50:
        return "Good"
    elif rul > 20:
        return "Warning"
    else:
        return "Critical"

# --- Main Script ---
def analyze_fleet():
    model, scaler, feature_cols = load_assets()
    test_data_raw = load_test_data()

    if not all([model, scaler, feature_cols, test_data_raw is not None]):
        print("Could not load all assets. Exiting.")
        return

    print(f"\nAnalyzing fleet of {test_data_raw['engine_id'].nunique()} engines...")
    
    fleet_status = []
    
    # Loop through every engine in the test set
    for engine_id in test_data_raw['engine_id'].unique():
        
        # 1. Isolate engine data
        engine_data = test_data_raw[test_data_raw['engine_id'] == engine_id].copy()
        
        # 2. Pre-process the data
        sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
        engine_data_with_rolling = add_rolling_features(engine_data.copy(), sensor_cols, ROLLING_WINDOW_SIZE)
        
        engine_data_for_prediction = engine_data_with_rolling.copy()
        for col in feature_cols:
            if col not in engine_data_for_prediction.columns:
                engine_data_for_prediction[col] = 0.0
        
        engine_data_scaled = scaler.transform(engine_data_for_prediction[feature_cols])
            
        # 3. Get the last sequence
        if len(engine_data_scaled) < SEQUENCE_LENGTH:
            padded_sequence = np.zeros((SEQUENCE_LENGTH, len(feature_cols)))
            padded_sequence[-len(engine_data_scaled):, :] = engine_data_scaled
            input_sequence = padded_sequence
        else:
            input_sequence = engine_data_scaled[-SEQUENCE_LENGTH:, :]
        
        input_sequence = np.expand_dims(input_sequence, axis=0)

        # 4. Make prediction
        predicted_rul = model.predict(input_sequence, verbose=0)[0][0]
        
        # 5. Get status and append to list
        status = get_health_status(predicted_rul)
        fleet_status.append(status)
        
        # Optional: uncomment to see live progress
        # print(f"Engine {engine_id}: RUL {predicted_rul:.2f} -> {status}")

    print("...Fleet analysis complete.")
    
    # 6. Count and print the results
    status_counts = pd.Series(fleet_status).value_counts()
    
    print("\n--- ✈️ Fleet Health Summary ---")
    print(status_counts.to_string())
    print("---------------------------------")
    print(f"Total Engines Analyzed: {len(fleet_status)}")

if __name__ == "__main__":
    analyze_fleet()
