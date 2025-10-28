import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

# --- Constants ---
MODEL_PATH = 'rul_model.h5'
SCALER_PATH = 'rul_scaler.joblib'
FEATURES_PATH = 'feature_cols.json'
TEST_DATA_PATH = 'test_FD004.txt'

SEQUENCE_LENGTH = 50
ROLLING_WINDOW_SIZE = 5

# --- UPDATED: Fewer, highly relevant sensors for plotting ---
KEY_SENSORS_TO_PLOT_TRENDS = ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_15'] 
# We will plot their rolling averages

# --- Caching ---
# Cache the loaded model, scaler, and feature list for performance
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and feature list from disk."""
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        st.error(
            f"Error loading assets: {e}. "
            "Please make sure 'rul_model.h5', 'rul_scaler.joblib', "
            "and 'feature_cols.json' are in the same directory."
        )
        return None, None, None

@st.cache_data
def load_test_data():
    """Loads the test data and assigns column names."""
    try:
        operational_settings = ['setting_1', 'setting_2', 'setting_3']
        sensor_measurements = [f'sensor_{i}' for i in range(1, 22)]
        column_names = ['engine_id', 'cycle'] + operational_settings + sensor_measurements
        
        data = pd.read_csv(TEST_DATA_PATH, sep='\s+', header=None, names=column_names)
        data = data.dropna(axis=1, how='all')
        return data
    except FileNotFoundError:
        st.error(f"Test data file '{TEST_DATA_PATH}' not found.")
        return None

# --- Helper Functions (Copied from Colab) ---
def add_rolling_features(df, sensor_cols, window_size):
    """Adds rolling features to the dataframe."""
    grouped = df.groupby('engine_id')
    
    # Identify non-constant sensor columns *within this engine*
    non_constant_sensors = []
    for col in sensor_cols:
        if df[col].nunique() > 1:
            non_constant_sensors.append(col)

    for col in non_constant_sensors:
        df[f'{col}_roll_avg'] = grouped[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{col}_roll_std'] = grouped[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Fill NaNs created by rolling
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Handle columns that were constant (std=0) and thus didn't get rolling features
    # This ensures all feature columns exist
    all_rolling_cols = [f'{c}_roll_avg' for c in sensor_cols] + [f'{c}_roll_std' for c in sensor_cols]
    for col in all_rolling_cols:
        if col not in df.columns:
            df[col] = 0.0 # If it was constant, rolling features are 0
            
    return df

# --- UI Helper Functions ---
def get_health_status(rul):
    """Returns a status, color, and emoji based on the RUL."""
    if rul > 100:
        return "Healthy", "success", "✅"
    elif rul > 50:
        return "Good", "success", "👍"
    elif rul > 20:
        return "Warning", "warning", "⚠️"
    else:
        return "Critical", "error", "🚨"

# --- Main Application ---
st.title("✈️ Turbofan Engine Predictive Maintenance")
st.markdown("This dashboard simulates RUL (Remaining Useful Life) prediction for engines from the NASA `test_FD004` dataset.")

# Load all assets
model, scaler, feature_cols = load_assets()
test_data_raw = load_test_data()

if model and scaler and feature_cols and test_data_raw is not None:
    
    # --- Sidebar for Engine Selection ---
    st.sidebar.header("Engine Selection")
    engine_ids = test_data_raw['engine_id'].unique()
    selected_engine_id = st.sidebar.selectbox(
        "Select an Engine ID to analyze:",
        engine_ids
    )

    # --- Prediction Logic ---
    if st.sidebar.button("Predict RUL", type="primary"):
        with st.spinner(f"Analyzing Engine {selected_engine_id}..."):
            
            # 1. Isolate selected engine data
            engine_data = test_data_raw[test_data_raw['engine_id'] == selected_engine_id].copy()
            
            # --- IMPORTANT: We need the rolling features *before* scaling for the plot ---
            sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
            engine_data_with_rolling = add_rolling_features(engine_data.copy(), sensor_cols, ROLLING_WINDOW_SIZE)
            
            # 2. Pre-process the data (EXACTLY as in Colab)
            # Ensure all feature columns are present, even if not in this engine's data (e.g., constant)
            engine_data_for_prediction = engine_data_with_rolling.copy() # Use the one with rolling features
            for col in feature_cols:
                if col not in engine_data_for_prediction.columns:
                    engine_data_for_prediction[col] = 0.0 # Fill missing (constant) rolling features
            
            try:
                engine_data_scaled = scaler.transform(engine_data_for_prediction[feature_cols])
            except Exception as e:
                st.error(f"Error during scaling. This might be a feature mismatch. Details: {e}")
                st.stop()
                
            # 3. Get the last sequence (the most recent data)
            
            if len(engine_data_scaled) < SEQUENCE_LENGTH:
                padded_sequence = np.zeros((SEQUENCE_LENGTH, len(feature_cols)))
                padded_sequence[-len(engine_data_scaled):, :] = engine_data_scaled
                input_sequence = padded_sequence
            else:
                input_sequence = engine_data_scaled[-SEQUENCE_LENGTH:, :]
            
            input_sequence = np.expand_dims(input_sequence, axis=0)

            # 4. Make prediction
            predicted_rul = model.predict(input_sequence)[0][0]
            
            # --- Display results ---
            status, color, emoji = get_health_status(predicted_rul)
            
            st.subheader(f"Prediction for Engine {selected_engine_id}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Remaining Useful Life (RUL)",
                    value=f"{predicted_rul:.2f} cycles"
                )
            with col2:
                st.metric(
                    label="Engine Status",
                    value=f"{status} {emoji}"
                )
            
            if color == "error":
                st.error(f"**{status.upper()}:** This engine is predicted to require maintenance within {int(predicted_rul) + 1} cycles.", icon=emoji)
            elif color == "warning":
                st.warning(f"**{status.upper()}:** This engine is predicted to require maintenance within {int(predicted_rul) + 1} cycles.", icon=emoji)
            else:
                st.success(f"**{status.upper()}:** No immediate maintenance required.", icon=emoji)
            
            st.balloons()
            
            # --- UPDATED: Interactive Sensor Trend Plot ---
            st.subheader("Key Sensor Degradation Trends (Engine Lifetime)")
            st.markdown("This chart shows the **smoothed rolling average** of critical sensor values, highlighting degradation over time.")
            
            fig = go.Figure()
            for sensor in KEY_SENSORS_TO_PLOT_TRENDS:
                # Plot the rolling average, not the raw sensor
                trend_col_name = f'{sensor}_roll_avg'
                if trend_col_name in engine_data_with_rolling.columns:
                    fig.add_trace(go.Scatter(
                        x=engine_data_with_rolling['cycle'], 
                        y=engine_data_with_rolling[trend_col_name],
                        mode='lines',
                        name=f'{sensor} (Rolling Avg)'
                    ))
            
            fig.update_layout(
                title=f'Smoothed Sensor Trends for Engine {selected_engine_id}',
                xaxis_title='Cycle (Flight)',
                yaxis_title='Smoothed Sensor Value',
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- NEW: Sensor Descriptions ---
            with st.expander("What do these sensors mean?"):
                st.markdown("""
                These sensors are critical indicators of engine health:
                - **sensor_2**: Total pressure at HPC outlet (psig) - *Measures the efficiency of the high-pressure compressor.*
                - **sensor_7**: Ratio of fuel flow to Ps30 (pps/psia) - *Indicates efficiency of fuel combustion and power generation.*
                - **sensor_11**: Total pressure at LPT outlet (psig) - *Measures the efficiency of the low-pressure turbine.*
                - **sensor_15**: Total pressure at bypass-duct (psig) - *Measures the pressure in the bypass duct, indicating overall fan performance.*
                
                A clear upward or downward trend in these smoothed values is a strong sign of degradation.
                """)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app loads the trained Keras model, applies the *exact* "
        "same preprocessing (rolling features + scaling) to the selected "
        "engine's test data, and predicts the RUL."
    )
    
else:
    st.error("Could not load all required assets. Dashboard cannot start.")

