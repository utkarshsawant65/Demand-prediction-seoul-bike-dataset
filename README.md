# Data-Driven Approaches to Optimizing Urban Bike Sharing Systems

**Master Thesis** | SRH University Heidelberg | M.Sc. Applied Data Science and Analytics

**Author:** Utkarsh Sawant (Matriculation no: 11038703)

**Supervisors:** Prof. Dr. Binh Vu, Prof. Dr. Swati Chandna

**Date:** 10 March 2026

---

## Abstract

Bike-sharing programmes have become an important part of urban transport infrastructure, and there is a need to forecast short-term demand to prevent stations running out of bicycles or parking spaces. This thesis assesses various deep learning architectures, including standalone and hybrid models, for hourly bike rental demand forecasting in Seoul. A complete feature engineering pipeline is constructed and seven models are trained using historical bike rental, weather, and time-related information from the UCI Seoul Bike Sharing dataset. All models are evaluated under a strict one-step-ahead temporal protocol to ensure realistic performance estimates.

---

## Research Questions

1. Can deep learning models accurately capture the nonlinear temporal patterns of Seoul bike-sharing demand, offering accurate short-term forecasts?
2. How does the combination of weather, temporal, and historical demand features contribute to prediction performance?
3. Do hybrid architectures combining multiple model paradigms outperform individual baselines?
4. What is the best architectural configuration for bike demand forecasting and what trade-offs exist among prediction accuracy, model complexity, and computational efficiency?

---

## Results

All seven models evaluated under one-step-ahead temporal protocol on an unseen test set (Sep 19 - Nov 30, 2018).

| Rank | Model | Parameters | Train R2 | Test R2 | Test RMSE | Test MAE |
|------|-------|-----------|----------|---------|-----------|----------|
| 1 | Multi-Scale TCN+LSTM | 92,849 | 97.30% | 88.83% | 204.24 | 141.27 |
| 2 | LSTM-XGBoost | 222,272+XGB | 97.50% | 86.67% | 223.09 | 151.89 |
| 3 | TCN-GRU-Attention | 294,177 | 95.60% | 85.58% | 231.98 | 151.91 |
| 4 | TCN-LSTM | 484,641 | 89.90% | 84.75% | 238.60 | 158.25 |
| 5 | TCN-CBAM-LSTM | 330,382 | 94.37% | 84.37% | 241.59 | 173.92 |
| 6 | TCN | - | 96.92% | 81.92% | 260.47 | 198.89 |
| 7 | LSTM | - | 95.66% | 75.76% | 300.58 | 218.80 |

The Multi-Scale TCN+LSTM achieves the highest accuracy with the fewest parameters among all hybrid models, demonstrating that architecture design matters more than model size.

---

## Dataset

**Source:** [UCI Machine Learning Repository - Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

- 8,760 hourly observations (Dec 2017 - Nov 2018)
- 30 engineered features derived from weather, temporal, and demand history variables
- Temporal split: 80% training (Dec 2017 - Sep 18, 2018) / 20% testing (Sep 19 - Nov 30, 2018)
- Sliding window of 24 hours for sequence input

---

## Repository Structure

```
.
├── feature_engineering.py          # Shared feature engineering pipeline
├── data/
│   ├── raw/                        # Original Seoul bike dataset
│   └── feature_data/               # Engineered train/test CSVs (30 features)
├── lstm/                           # LSTM baseline (Rank 7)
│   └── train_lstm_enhanced.py
├── tcn/                            # TCN baseline (Rank 6)
│   └── train_tcn_enhanced.py
├── hybrid/                         # TCN-LSTM hybrid (Rank 4)
│   └── train_hybrid.py
├── tcn_gru_attention/              # TCN-GRU-Attention (Rank 3)
│   └── train_tcn_gru_attention.py
├── tcn_cbam_lstm/                  # TCN-CBAM-LSTM (Rank 5)
│   └── train_tcn_cbam_lstm.py
├── lstm_xgboost/                   # LSTM-XGBoost ensemble (Rank 2)
│   └── train_lstm_xgboost.py
├── multi_scale_tcn/                # Multi-Scale TCN+LSTM (Rank 1)
│   └── train_multi_scale_tcn_lstm.py
├── r/                              # Cubist model (R baseline for comparison)
├── reports/                        # EDA figures and comparison outputs
└── requirements.txt
```

---

## Methodology

This thesis follows the CRISP-DM methodology and uses a strict **one-step-ahead evaluation protocol**: at each test timestep, the model predicts hour t+1 using only data available up to hour t. This avoids the inflated accuracy that same-time prediction produces and reflects real-world deployment conditions.

**Feature engineering pipeline** processes raw data into 30 features across six domains:
- Demand history (lag features, rolling statistics)
- Temperature (raw, squared, interaction terms)
- Weather conditions (precipitation, visibility, comfort index)
- Cyclical time encoding (hour, day-of-week, month via sine/cosine)
- Categorical indicators (season, weekend, holiday)
- Rush hour flags

**Data leakage prevention:** all scalers are fitted exclusively on the training partition; an automated safety check verifies that target-correlated columns are excluded from model inputs.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.9, TensorFlow/Keras 2.2 |
| Gradient Boosting | XGBoost 3.1 |
| Data Processing | pandas 2.3, NumPy 2.3, scikit-learn 1.7 |
| Hardware | Intel Core i7, 16 GB RAM (CPU-only) |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature engineering
python feature_engineering.py

# Train any model (example: Multi-Scale TCN+LSTM)
cd multi_scale_tcn
python train_multi_scale_tcn_lstm.py
```

Each training script is self-contained: it loads the preprocessed data, trains the model, evaluates on the test set, and saves metrics/weights to its local `results/` and `models/` directories.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
