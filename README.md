✈️ Turbofan Engine Predictive Maintenance Dashboard

This repository contains a Streamlit web application that predicts the Remaining Useful Life (RUL) of turbofan engines using a deep learning model. The project is based on the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD004 dataset.

The deployed application allows a user to select any of the 100 engines from the test dataset and get a live RUL prediction, health status, and a visualization of key sensor degradation trends.

Key Features

RUL Prediction: Uses a trained Stacked LSTM (Long Short-Term Memory) model to predict the remaining useful life in cycles.

Health Status: Classifies the engine's health as "Healthy", "Good", "Warning", or "Critical" based on the RUL.

Interactive Visualization: Plots the smoothed rolling averages of 4 critical sensors (sensor_2, sensor_7, sensor_11, sensor_15) to visualize degradation trends over the engine's life.

Feature Engineering: The model's 50% error reduction (from a baseline of ~42 RMSE to 20.97 RMSE) was achieved by robust feature engineering, including 5-cycle rolling averages and standard deviations.

Tech Stack

Model: TensorFlow (Keras)

Data Processing: Pandas, NumPy, Scikit-learn

Dashboard: Streamlit

Plotting: Plotly

How to Run Locally

Clone or download this repository.

Install dependencies (preferably in a virtual environment):

pip install -r requirements.txt


Run the Streamlit app from your terminal:

streamlit run dashboard.py
