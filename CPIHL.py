"""
Physics-Informed Neural Network Hybrid Model for Battery Voltage Prediction
Author: Sahar Qaadan
Date: 2025-04-01

This script implements a hybrid PINN model combined with machine learning (Random Forest) to predict battery voltage. It includes configurable parameters, physics-informed loss, and baseline model comparisons.
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_data(config):
    all_data = []
    for batch in range(config['batch_range'][0], config['batch_range'][1] + 1):
        batch_path = os.path.join(config['data_directory'], f"{config['batch_prefix']}{batch}")
        if os.path.exists(batch_path):
            files = [f for f in os.listdir(batch_path) if f.startswith(config['file_prefix']) and f.endswith('.csv')]
            for file in files[:config['max_files_per_batch']]:
                df = pd.read_csv(os.path.join(batch_path, file))
                all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    data.dropna(inplace=True)
    return data


def split_data(df, config):
    X = df[config['feature_cols']].values
    y = df[config['target_col']].values
    split_idx = int(len(X) * (1 - config['validation_split']))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_pinn(input_dim, config):
    model = Sequential()
    for units in config['pinn_layers']:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    return model


def physics_loss(y_true, y_pred, temp, age, lambda_term):
    physics = (y_pred - (0.5 * temp - 0.002 * age)) ** 2
    return lambda_term * np.mean(physics)


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} R2 Score: {r2_score(y_test, y_pred):.3f}")
        print(f"{name} RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}\n")


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='PINN Prediction')
    plt.xlabel("Samples")
    plt.ylabel("Voltage")
    plt.title("PINN Prediction vs Actual Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_pipeline(config):
    data = load_data(config)
    X_train, X_test, y_train, y_test = split_data(data, config)

    if config.get("model_comparison", False):
        print("\nBaseline Model Comparison:")
        evaluate_models(X_train, X_test, y_train, y_test)

    pinn = build_pinn(X_train.shape[1], config)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    pinn.fit(X_train, y_train, 
             validation_data=(X_test, y_test),
             epochs=config['epochs'],
             batch_size=config['batch_size'],
             callbacks=[early_stop],
             verbose=1)

    y_pred = pinn.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"\nPINN R2 Score: {r2:.3f}")
    print(f"PINN RMSE: {rmse:.3f}")

    temp_col = config['physics_temp_col']
    age_col = config['physics_age_col']
    phys_loss = physics_loss(y_test, y_pred, data[temp_col].values[:len(y_test)],
                             data[age_col].values[:len(y_test)], config['physics_lambda'])
    print(f"Physics-Informed Loss Component: {phys_loss:.5f}")

    if config.get("visualize", True):
        plot_predictions(y_test, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN hybrid model")
    parser.add_argument("--config", type=str, default="config/pinn_config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)
