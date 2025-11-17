# LSTM Model for Seoul Bike Prediction

This folder contains the LSTM model training script for predicting bike rental counts.

## Structure

```
lstm/
├── train_lstm.py          # Main training script
├── models/                # Saved models and scalers (created after training)
│   ├── lstm_model.keras
│   ├── best_lstm_model.keras
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
└── results/               # Training results and metrics (created after training)
    ├── lstm_metrics.json
    ├── lstm_metrics_summary.csv
    └── training_history.csv
```

## Requirements

The script uses the following Python packages:
- numpy
- pandas
- tensorflow
- scikit-learn
- joblib

## Usage

Run the training script from the project root directory:

```bash
python lstm/train_lstm.py
```

## Data

The script automatically loads data from:
- Training data: `data/model_data/train.csv`
- Testing data: `data/model_data/test.csv`

**Important**: Test data is never used for training, only for final evaluation.

## Model Architecture

- Input: Sequences of 24 hours (24 timesteps)
- LSTM Layer 1: 64 units with dropout (0.2)
- LSTM Layer 2: 32 units with dropout (0.2)
- Dense Layer: 32 units
- Output: Single value (bike count prediction)

## Outputs

After training, the following files are created:

### Models
- `models/lstm_model.keras`: Final trained model
- `models/best_lstm_model.keras`: Best model based on validation loss
- `models/feature_scaler.pkl`: Scaler for input features
- `models/target_scaler.pkl`: Scaler for target variable

### Results
- `results/lstm_metrics.json`: Complete metrics and metadata
- `results/lstm_metrics_summary.csv`: Summary table of train/test metrics
- `results/training_history.csv`: Loss and MAE per epoch

## Metrics

The model is evaluated using:
- R² (R-squared)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- CV (Coefficient of Variation)
- MAPE (Mean Absolute Percentage Error)
