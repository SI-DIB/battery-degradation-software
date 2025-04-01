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



## 🧾 Run the script as follows:
python granger_causality.py --config config.yaml

---
## 🧠 CPIHL – Causality-informed PINN Hybrid Learning

This module implements a hybrid deep learning framework for predicting battery voltage using a **Physics-Informed Neural Network (PINN)** enhanced by traditional machine learning regressors.

It is based on the methodology presented in the accepted IEEE paper:

**"Causality-informed PINN Hybrid Learning for Lithium-Ion Battery Voltage Prediction"**  
[IEEE Xplore Link](https://ieeexplore.ieee.org/document/10945873)

**Filename**: `cpihl.py`  
**Configuration file**: `config/cpihl_config.yaml`
---

### ⚙️ Configuration Parameters (`cpihl_config.yaml`)

```yaml
data_directory: "./data/HPPCs"
batch_prefix: "Batch"
file_prefix: "Cleaned_"
batch_range: [19, 50]
max_files_per_batch: 1

feature_cols:
  - ClimaTemp
  - I
  - Itarget
  - P
  - Q
  - Qneg
  - Qpos
  - Temp_Cell
target_col: U

pinn_layers: [128, 128, 128]
learning_rate: 0.001
epochs: 50
batch_size: 32
validation_split: 0.2

physics_lambda: 0.01
physics_temp_col: Temp_Cell
physics_age_col: Q

model_comparison: true
visualize: true

