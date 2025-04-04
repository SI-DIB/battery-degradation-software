# 🔋 battery-degradation-software

**A modular suite for causality analysis, hybrid PINN modeling, advanced feature engineering, and unsupervised clustering to support battery health analytics.**

This repository provides Python implementations aligned with two published works:

- 📘 IEEE SMARTTECH: [Causality-informed PINN Hybrid Learning (CPIHL)](https://ieeexplore.ieee.org/document/10945873)
- 🧾 Data in Brief (submitted): Feature engineering on Samsung INR21700-50E dataset

---

## 📌 Modules Overview

### 1. 🧠 CPIHL — Causality-informed PINN Hybrid Learning
A hybrid deep learning framework integrating **Ohm's law** with **neural networks** to predict voltage under battery aging. Physics-informed regularization improves generalization and scientific interpretability.

- File: `cpihl.py`
- Config: `config/cpihl_config.yaml`

### 2. 🔍 Granger Causality + Transfer Entropy Analysis
Identify statistically significant relations between **voltage**, **resistance**, and **temperature** using Granger causality and entropy metrics.

- File: `granger_causality.py`
- Config: `config/granger_causality_config.yaml`

### 3. ⚙️ Feature Extraction Tool
Automatically derive 20+ engineered features such as:

- Capacity Fade Rate
- Internal Resistance
- Thermal Runaway Risk
- Energy Throughput
- Voltage Rolling Average

- File: `extract_features.py`

### 4. 📊 Clustering and Health Metadata
Segment battery batches using K-Means or GMM with options to visualize intra- and inter-batch similarities.

- File: `clean_clustering.py`
- CLI Parameters (no config file required)

---

## 🧬 Feature Highlights

- Modular structure: Each module works standalone
- Physics + Machine Learning integration
- YAML or CLI based configuration
- End-to-end reproducibility
- Real-time feasible: 0.002s inference time (MacBook M3 Pro)
- Engineered feature dataset aligned with Data in Brief submission

---

## 📁 Repository Structure

battery-degradation-software/ │ ├── cpihl.py # Physics-Informed Hybrid Learning ├── granger_causality.py # Granger & Transfer Entropy ├── extract_features.py # Advanced feature extraction ├── clean_clustering.py # Metadata clustering │ ├── config/ │ ├── cpihl_config.yaml │ └── granger_causality_config.yaml │ ├── data/ # Your input HPPC datasets ├── outputs/ # Cluster results, plots, etc. ├── requirements.txt └── README.md


---

## ⚙️ Installation & Requirements

```bash
pip install -r requirements.txt

python granger_causality.py --config config/granger_causality_config.yaml

python cpihl.py --config config/cpihl_config.yaml

python extract_features.py --dir ./data/HPPCs --samples 10

python clean_clustering.py \
  --excel_path ./data/Batch_List.xlsx \
  --output_path ./outputs/clustered_data.csv \
  --selected_batches 19 20 21 22 \
  --n_clusters 3


📚 Publications
CPIHL for Li-ion Voltage Prediction
IEEE ACCESS 2024 — DOI:10.1109/SMARTTECH58661.2024.10945873

Samsung INR21700 Dataset Feature Analysis
Data in Brief, Elsevier — Zenodo DOI

👩‍💻 Author & Contact
For questions or collaboration inquiries:

Prof. Dr. Sahar Qaadan
German Jordanian University
📧 sahar.qadan@gju.edu.jo
