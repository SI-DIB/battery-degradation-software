# battery-degradation-software
# Granger Causality and Transfer Entropy Analysis for Battery Degradation

This repository provides a reproducible Python implementation to assess the causal relationships between temperature, voltage, and resistance in Lithium-ion battery datasets using **Granger Causality** and **Transfer Entropy** techniques.

## 🔍 Features

- Generalized for multiple battery datasets (supports batch processing).
- Fully configurable via an external YAML file.
- Performs:
  - Granger Causality Tests
  - Transfer Entropy Calculations
  - Resistance Smoothing and Calculation
  - Visualization of variable relationships
- Modular and ready for reproducibility certification (e.g., Zenodo, Code Ocean).

---

## 🧾 Requirements

Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt

## ⚙️ Configuration

Set up `config.yaml` like this:

```yaml
data_directory: "./data/HPPCs"
batch_prefix: "Batch"
file_prefix: "Cleaned_"
batch_range: [19, 50]
max_files_per_batch: 1
max_total_files: 10

current_col: "I"
voltage_col: "U"
target_col: "Smoothed_Resistance"
predictor_cols: ["Temp_Cell", "U"]

smoothing_window: 50
segment: [1000, 2000]

granger_max_lag: 6
te_lag: 1
te_bins: 10

visualize: true

## Run the script as follows:
python granger_causality.py --config config.yaml
