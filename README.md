<p align="center">
  <h1 align="center">Seoul Bike Demand Prediction</h1>
  <p align="center">
    Deep learning pipeline for hourly bike rental demand forecasting using hybrid neural architectures
    <br />
    <a href="#models">View Models</a>
    &middot;
    <a href="#results">Results</a>
    &middot;
    <a href="#quick-start">Quick Start</a>
  </p>
</p>

## Overview


> All hybrid architectures outperform both baselines. The Multi-Scale TCN+LSTM achieves the highest accuracy with the smallest parameter count among hybrids.

## Models

### Baselines
| Model | Description | Script |
|-------|-------------|--------|
| **LSTM** | 2-layer stacked LSTM with batch normalization and dropout | `lstm/train_lstm_enhanced.py` |
| **TCN** | 3-block dilated causal convolution network with residual connections | `tcn/train_tcn_enhanced.py` |

### Hybrid Architectures
| Model | Description | Script |
|-------|-------------|--------|
| **TCN-LSTM** | Sequential pipeline: 5-block TCN feature extraction followed by 2-layer LSTM refinement | `hybrid/train_hybrid.py` |
| **TCN-GRU-Attention** | TCN + GRU with multi-head self-attention for selective temporal weighting | `tcn_gru_attention/train_tcn_gru_attention.py` |
| **TCN-CBAM-LSTM** | TCN + Convolutional Block Attention Module (channel + spatial attention) + LSTM | `tcn_cbam_lstm/train_tcn_cbam_lstm.py` |
| **LSTM-XGBoost** | Two-stage ensemble: LSTM feature extractor feeding XGBoost regressor | `lstm_xgboost/train_lstm_xgboost.py` |
| **Multi-Scale TCN+LSTM** | Three parallel TCN branches (kernel sizes 2, 3, 5) + LSTM with mixup augmentation | `multi_scale_tcn/train_multi_scale_tcn_lstm.py` |

## Architecture

```
Raw Data (8,760 hourly records)
    |
    v
Feature Engineering Pipeline ──> 30 features across 6 domains
    |
    v
Temporal Split (80/20 chronological)
    |
    v
Sliding Window (24h lookback) ──> Input tensor: (batch, 24, 30)
    |
    v
┌──────────────────────────────────────────────────────────┐
│  7 Model Architectures (trained independently)           │
│  ├── LSTM baseline                                       │
│  ├── TCN baseline                                        │
│  ├── TCN-LSTM                                            │
│  ├── TCN-GRU-Attention                                   │
│  ├── TCN-CBAM-LSTM                                       │
│  ├── LSTM-XGBoost                                        │
│  └── Multi-Scale TCN+LSTM                                │
└──────────────────────────────────────────────────────────┘
    |
    v
One-Step-Ahead Evaluation ──> R2, RMSE, MAE
```

## Feature Engineering

The pipeline (`feature_engineering.py`) transforms 14 raw columns into **30 optimized features**:

| Domain | Features | Examples |
|--------|:--------:|---------|
| Demand History | 6 | `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`, `rolling_3h_mean`, `rolling_24h_std`, `rolling_24h_max` |
| Temperature | 3 | `temperature`, `temp_squared`, `temp_x_hour` |
| Weather | 4 | `humidity`, `visibility`, `solar_radiation`, `has_precipitation` |
| Cyclical Time | 6 | `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos` |
| Categorical | 7 | `is_weekend`, `is_holiday`, `is_functioning`, `season_*` |
| Rush Hour | 4 | `is_rush_hour`, `is_evening_rush`, `is_comfortable_weather`, `bad_weather` |

Scalers are fitted **exclusively on training data** to prevent information leakage. An automated safety check verifies no target-correlated columns are present in model inputs.

## Dataset

**Source:** [UCI Machine Learning Repository: Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

| Property | Value |
|----------|-------|
| Records | 8,760 hourly observations |
| Period | Dec 1, 2017 to Nov 30, 2018 |
| Train set | 6,840 samples (Dec 2017 to Sep 18, 2018) |
| Test set | 1,728 samples (Sep 19 to Nov 30, 2018) |
| Input shape | `(24, 30)` : 24-hour window, 30 features |
| Target | Hourly rented bike count (next hour) |

## Project Structure

```
.
├── feature_engineering.py              # Shared feature engineering pipeline
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/                            # Original UCI dataset
│   └── feature_data/                   # Processed train.csv / test.csv
│
├── lstm/                               # LSTM baseline
│   ├── train_lstm_enhanced.py          #   Main training script
│   └── train_lstm_basic.py             #   Initial experiment
│
├── tcn/                                # TCN baseline
│   ├── train_tcn_enhanced.py           #   Main training script
│   └── train_tcn_basic.py              #   Initial experiment
│
├── hybrid/                             # TCN-LSTM hybrid
│   ├── train_hybrid.py                 #   Main training script
│   ├── train_hybrid_ensemble.py        #   Ensemble experiment
│   └── train_hybrid_final.py           #   Regularization experiment
│
├── tcn_gru_attention/                  # TCN-GRU-Attention
│   └── train_tcn_gru_attention.py
│
├── tcn_cbam_lstm/                      # TCN-CBAM-LSTM
│   └── train_tcn_cbam_lstm.py
│
├── lstm_xgboost/                       # LSTM-XGBoost ensemble
│   └── train_lstm_xgboost.py
│
├── multi_scale_tcn/                    # Multi-Scale TCN+LSTM (best)
│   ├── train_multi_scale_tcn_lstm.py   #   Main training script
│   ├── train_multi_scale_tcn.py        #   Ablation: TCN-only variant
│   ├── train_multi_scale_tcn_v2.py     #   Generalization experiment
│   └── train_multi_scale_tcn_regularized.py
│
├── r/                                  # R-based Cubist baseline
│   ├── cubist_model.r
│   └── cubist_model_temporal.r
│
└── reports/                            # EDA visualizations
    ├── figures/
    └── results/
```

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Deep Learning | PyTorch | 2.9 |
| Deep Learning | TensorFlow / Keras | 2.2 |
| Gradient Boosting | XGBoost | 3.1 |
| Data Processing | pandas | 2.3 |
| Numerical Computing | NumPy | 2.3 |
| ML Utilities | scikit-learn | 1.7 |
| Language | Python | 3.12 |
| Hardware | Intel Core i7, 16 GB RAM | CPU-only |

## Quick Start

### Prerequisites

```bash
python >= 3.12
pip install -r requirements.txt
```

### 1. Generate Features

```bash
python feature_engineering.py
```

This reads the raw dataset from `data/raw/`, applies all transformations, and writes `train.csv` and `test.csv` to `data/feature_data/`.

### 2. Train a Model

Each model directory contains self-contained training scripts. Example:

```bash
# Train the best model (Multi-Scale TCN+LSTM)
cd multi_scale_tcn
python train_multi_scale_tcn_lstm.py

# Train the LSTM baseline
cd lstm
python train_lstm_enhanced.py

# Train any other model
cd tcn_gru_attention
python train_tcn_gru_attention.py
```

### 3. Outputs

Each script automatically:
- Trains the model with early stopping
- Evaluates on the held-out test set
- Saves model weights to `models/`
- Saves evaluation metrics (R2, RMSE, MAE) to `results/`
- Saves training history to `results/`

## Future Work

- **Interactive Streamlit Dashboard**: Building a multi-page web dashboard for exploring model predictions, comparing architectures, and visualizing feature importance interactively. The dashboard will load saved predictions and metrics (no live inference) and allow users to filter by time range, model, and weather conditions.
- **Multi-step Forecasting**: Extending the pipeline to predict demand multiple hours ahead instead of single-step
- **Additional Datasets**: Validating the architectures on bike-sharing systems from other cities

## License

MIT. See [LICENSE](LICENSE) for details.
