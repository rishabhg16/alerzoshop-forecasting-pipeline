# Demand Forecasting Time-Series Pipeline for Alerzoshop

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 1. Load and Preprocess Data
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
    df = df.sort_index()
    df = df.fillna(method='ffill')  # Forward fill for missing values
    return df

# 2. Create Sequences for Time-Series Data
def create_sequences(data, window_size):
    sequences, targets = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)

# 3. Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Train Model
def train_model(model, X_train, y_train, epochs=20):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
    return history

# 5. Save Model and Scaler
def save_model_and_scaler(model, scaler, output_path='model'):
    os.makedirs(output_path, exist_ok=True)
    model.save(os.path.join(output_path, 'lstm_model'))
    np.save(os.path.join(output_path, 'scaler.npy'), scaler.scale_)

# 6. Pipeline Entry Point
def run_pipeline(csv_path, window_size=30):
    df = load_and_preprocess(csv_path)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['sales'].values.reshape(-1, 1))

    X, y = create_sequences(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model((window_size, 1))
    history = train_model(model, X, y)

    save_model_and_scaler(model, scaler)
    print("Pipeline completed. Model and scaler saved.")

    return history

if __name__ == '__main__':
    run_pipeline('sales_data.csv')
