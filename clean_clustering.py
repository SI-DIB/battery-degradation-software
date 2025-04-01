#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 02:55:26 2025

@author: saharqaadan
"""

"""
Clustering and Feature Engineering for Lithium-Ion Battery Degradation Analysis
Refactored for CLI-only configuration (no YAML required).

Example usage:

python clean_clustering.py \
  --excel_path ./data/Batch_List.xlsx \
  --output_path ./outputs/clustered_data.csv \
  --selected_batches 19 20 21 22 \
  --n_clusters 3

Clustering and Feature Engineering for Lithium-Ion Battery Degradation Analysis
Refactored for CLI-only configuration (no YAML required).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_soh(data):
    data['SOH_Lower'] = data['SOH (supplier)'].str.extract(r'(\d+)<SOH<\d+').astype(float)
    data['SOH_Upper'] = data['SOH (supplier)'].str.extract(r'\d+<SOH<(\d+)').astype(float)
    data['SOH'] = (data['SOH_Lower'] + data['SOH_Upper']) / 2
    return data


def encode_visible_quality(data):
    data['Visible Quality Encoded'] = data['Visible Quality'].astype('category').cat.codes
    return data


def cluster_data(data, features, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features].fillna(data[features].mean()))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    return data, kmeans


def visualize_clusters(data, features):
    sns.scatterplot(x=features[1], y=features[0], hue='Cluster', data=data)
    plt.title('Clustering of Battery Cells')
    plt.xlabel(features[1])
    plt.ylabel(features[0])
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_pipeline(args):
    df = pd.read_excel(args.excel_path)
    df = df[df['Batch'].isin(args.selected_batches)].copy()
    df = df[['Batch', 'Measured Voltage', 'SOH (supplier)', 'Visible Quality']]

    df = preprocess_soh(df)
    df = encode_visible_quality(df)

    features = ["SOH", "Measured Voltage", "Visible Quality Encoded"]
    df, kmeans = cluster_data(df, features, args.n_clusters)

    cluster_summary = df.groupby('Cluster')[features].mean()
    print("\nCluster Summary:")
    print(cluster_summary)

    visualize_clusters(df, features)

    df.to_csv(args.output_path, index=False)
    print(f"Clustered data saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run battery clustering analysis without a config file")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to Excel file with battery metadata")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV path for clustered results")
    parser.add_argument("--selected_batches", type=int, nargs='+', required=True, help="List of selected batch numbers")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters to form")

    args = parser.parse_args()
    run_pipeline(args)
