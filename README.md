# Seoul Bike Demand Prediction


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

**Source:** [UCI Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

- 8,760 hourly observations (Dec 2017 - Nov 2018)
- 30 engineered features (weather, temporal, demand history)
- Temporal split: 80% train / 20% test
- Sliding window: 24 hours

## Results

| Rank | Model | Parameters | Test R2 | Test RMSE | Test MAE |
|------|-------|-----------|---------|-----------|----------|
| 1 | Multi-Scale TCN+LSTM | 92,849 | 88.83% | 204.24 | 141.27 |
| 2 | LSTM-XGBoost | 222,272+XGB | 86.67% | 223.09 | 151.89 |
| 3 | TCN-GRU-Attention | 294,177 | 85.58% | 231.98 | 151.91 |
| 4 | TCN-LSTM | 484,641 | 84.75% | 238.60 | 158.25 |
| 5 | TCN-CBAM-LSTM | 330,382 | 84.37% | 241.59 | 173.92 |
| 6 | TCN | - | 81.92% | 260.47 | 198.89 |
| 7 | LSTM | - | 75.76% | 300.58 | 218.80 |

## Project Structure

```
.
├── feature_engineering.py          # Feature engineering pipeline (30 features)
├── data/
│   ├── raw/                        # Original Seoul bike dataset
│   └── feature_data/               # Processed train/test CSVs
├── lstm/                           # LSTM baseline
├── tcn/                            # TCN baseline
├── hybrid/                         # TCN-LSTM hybrid
├── tcn_gru_attention/              # TCN-GRU-Attention
├── tcn_cbam_lstm/                  # TCN-CBAM-LSTM
├── lstm_xgboost/                   # LSTM-XGBoost ensemble
├── multi_scale_tcn/                # Multi-Scale TCN+LSTM (best model)
├── r/                              # Cubist model (R baseline)
├── reports/                        # EDA figures
└── requirements.txt
```

## Features

The feature engineering pipeline (`feature_engineering.py`) generates 30 features across six domains:

- **Demand history** — lag features (1h, 24h, 168h), rolling mean/std/max
- **Temperature** — raw, squared, hour interaction
- **Weather** — precipitation, visibility, comfort index
- **Cyclical time** — hour, day-of-week, month (sine/cosine encoded)
- **Categorical** — season, weekend, holiday
- **Rush hour** — morning and evening peak flags

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.9, TensorFlow/Keras 2.2 |
| Gradient Boosting | XGBoost 3.1 |
| Data Processing | pandas 2.3, NumPy 2.3, scikit-learn 1.7 |

## How to Run

```bash
pip install -r requirements.txt

# Generate features
python feature_engineering.py

# Train a model (example)
cd multi_scale_tcn
python train_multi_scale_tcn_lstm.py
```

Each training script loads the preprocessed data, trains the model, evaluates on the test set, and saves metrics and weights locally.

## License

MIT - see [LICENSE](LICENSE)
