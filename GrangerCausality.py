#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Granger Causality and Transfer Entropy Analysis for Battery Data
Author: Sahar Qaadan
Date: 2025-04-01

This script loads battery measurement data, calculates resistance, and performs both Granger causality and transfer entropy analyses between temperature, voltage, and resistance. Parameters are externalized in a YAML configuration file.
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.stattools import grangercausalitytests


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_data(config):
    sample_csvs = []
    for batch_num in range(config['batch_range'][0], config['batch_range'][1] + 1):
        batch_path = os.path.join(config['data_directory'], f"{config['batch_prefix']}{batch_num}")
        if os.path.exists(batch_path):
            csv_files = [os.path.join(batch_path, f) for f in os.listdir(batch_path) 
                         if f.endswith('.csv') and f.startswith(config['file_prefix'])]
            sample_csvs.extend(csv_files[:config['max_files_per_batch']])
        if len(sample_csvs) >= config['max_total_files']:
            break

    data = pd.concat([pd.read_csv(f) for f in sample_csvs], ignore_index=True)
    return data


def preprocess_data(data, config):
    data.fillna(method='ffill', inplace=True)
    data['Resistance'] = np.where(data[config['voltage_col']] != 0,
                                  data[config['voltage_col']] / data[config['current_col']], np.nan)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['Resistance'], inplace=True)
    data['Smoothed_Resistance'] = data['Resistance'].rolling(window=config['smoothing_window'], min_periods=1).mean()
    return data


def discretize_data(data, n_bins):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return discretizer.fit_transform(data.reshape(-1, 1)).flatten()


def calculate_transfer_entropy(source, target, lag, n_bins):
    source_discrete = discretize_data(source, n_bins)
    target_discrete = discretize_data(target, n_bins)
    source_lagged = source_discrete[:-lag]
    target_future = target_discrete[lag:]
    p_xy = np.histogram2d(source_lagged, target_future, bins=n_bins)[0] / len(source_lagged)
    p_y = np.histogram(target_future, bins=n_bins)[0] / len(target_future)
    p_x = np.histogram(source_lagged, bins=n_bins)[0] / len(source_lagged)
    p_y_given_x = p_xy / (p_x + 1e-10)
    te = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0:
                te += p_xy[i, j] * np.log((p_y_given_x[i, j] + 1e-10) / (p_y[j] + 1e-10))
    return te


def run_analysis(config):
    data = load_data(config)
    data = preprocess_data(data, config)

    segment = data.iloc[config['segment'][0]:config['segment'][1]]
    target = segment[config['target_col']].dropna().values

    for predictor in config['predictor_cols']:
        predictor_values = segment[predictor].dropna().values
        print(f"\nGranger Causality Test: {predictor} -> {config['target_col']}")
        try:
            grangercausalitytests(segment[[predictor, config['target_col']]].dropna(), maxlag=config['granger_max_lag'])
        except Exception as e:
            print(f"Failed Granger causality for {predictor}: {e}")

        try:
            te = calculate_transfer_entropy(predictor_values, target, config['te_lag'], config['te_bins'])
            print(f"Transfer Entropy ({predictor} -> {config['target_col']}): {te:.5f}")
        except Exception as e:
            print(f"Failed transfer entropy for {predictor}: {e}")

    if config['visualize']:
        plt.figure(figsize=(10, 6))
        for col in config['predictor_cols'] + [config['target_col']]:
            plt.plot(segment[col], label=col)
        plt.xlabel("Time (Index)", fontsize=14)
        plt.ylabel("Normalized Value", fontsize=14)
        plt.title("Causality Relationships", fontsize=16)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Granger Causality and Transfer Entropy Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_analysis(config)