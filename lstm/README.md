# LSTM Enhanced Model for Seoul Bike Prediction

This folder contains the LSTM Enhanced model training script for predicting bike rental counts.

## Structure

```
lstm/
├── train_lstm_enhanced.py     # Main training script
├── models/                    # Saved models and scalers (created after training)
│   ├── lstm_enhanced_model.keras
│   ├── best_lstm_enhanced_model.keras
│   ├── feature_scaler_enhanced.pkl
│   └── target_scaler_enhanced.pkl
└── results/                   # Training results and metrics (created after training)
    ├── lstm_enhanced_metrics.json
    ├── lstm_enhanced_metrics_summary.csv
    └── training_history_enhanced.csv
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
python lstm/train_lstm_enhanced.py
```

## Data

The script automatically loads data from:
- Training data: `data/feature_data/train.csv`
- Testing data: `data/feature_data/test.csv`

**Important**: Test data is never used for training, only for final evaluation.

## Model Architecture

**LSTM Enhanced:**
```
Input (24h × 30 features)
  ↓
LSTM(128) + BatchNorm + Dropout(0.3)
  ↓
LSTM(64) + BatchNorm + Dropout(0.3)
  ↓
LSTM(32) + BatchNorm + Dropout(0.3)
  ↓
Dense(16) → Dense(1)
```

**Parameters:** ~110,000

**Characteristics:**
- Deeper architecture with 3 LSTM layers
- Batch normalization for training stability
- Higher capacity model
- Regularization with dropout (0.3)

## Outputs

After training, the following files are created:

### Models
- `models/lstm_enhanced_model.keras`: Final trained model
- `models/best_lstm_enhanced_model.keras`: Best model based on validation loss
- `models/feature_scaler_enhanced.pkl`: Scaler for input features
- `models/target_scaler_enhanced.pkl`: Scaler for target variable

### Results
- `results/lstm_enhanced_metrics.json`: Complete metrics and metadata
- `results/lstm_enhanced_metrics_summary.csv`: Summary table of train/test metrics
- `results/training_history_enhanced.csv`: Loss and MAE per epoch

## Metrics

The model is evaluated using:
- R² (R-squared)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- CV (Coefficient of Variation)
- MAPE (Mean Absolute Percentage Error)

## Performance

- Training R²: 0.9156 (91.56%)
- Testing R²: 0.6357 (63.57%)
- Test RMSE: 368.80 bikes
