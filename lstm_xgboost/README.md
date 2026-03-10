# Hybrid LSTM-XGBoost Model for Seoul Bike Demand Forecasting

## Overview

This hybrid model combines the temporal pattern recognition of LSTM with the powerful gradient boosting of XGBoost to create a two-stage prediction system.

## Architecture

### Model Flow

```
Input Sequence (24 hours × 64 features)
        |
        ├─────────────────────────────────────┐
        |                                     |
   LSTM Branch                          Original Features
(Temporal Patterns)                    (Last timestep)
        |                                     |
  2 × LSTM(128)                          64 features
        |                                     |
   Dropout(0.3)                               |
        |                                     |
   Dense(128→64)                              |
        |                                     |
  LSTM Features (64D)                         |
        |                                     |
        └─────────────────┬───────────────────┘
                          |
                  Concatenate (128D)
                  (64 LSTM + 64 Original)
                          |
                    XGBoost Model
                  (Gradient Boosting)
                          |
                  Prediction (1 value)
```

### Two-Stage Training Process

#### Stage 1: LSTM Feature Extraction
- **Purpose:** Extract meaningful temporal features from sequences
- **Training:** LSTM is trained with a temporary regression head to learn features
- **Output:** 64-dimensional feature vector per sample
- **Architecture:**
  - 2 stacked LSTM layers (128 hidden units each)
  - Dense layer (128 → 64) for feature compression
  - Dropout (0.3) for regularization

#### Stage 2: XGBoost Prediction
- **Purpose:** Final prediction using combined features
- **Input:** Concatenated LSTM features (64D) + Original point features (64D) = 128D
- **Training:** Gradient boosting on combined feature set
- **Advantages:**
  - Handles complex feature interactions
  - Robust to outliers
  - Built-in feature importance

## Why LSTM + XGBoost?

### Complementary Strengths

1. **LSTM Component**
   - Excels at capturing temporal dependencies
   - Learns sequential patterns over 24-hour window
   - Extracts high-level temporal representations

2. **XGBoost Component**
   - Excels at handling complex feature interactions
   - Robust to noisy data and outliers
   - Fast inference time
   - Provides feature importance metrics

3. **Hybrid Synergy**
   - LSTM provides temporal context
   - XGBoost leverages both temporal and point features
   - Better than either model alone

### Comparison with Other Hybrids

| Aspect | LSTM-XGBoost | LSTM-GRU | LSTM-TCN |
|--------|-------------|----------|----------|
| **Temporal Learning** | LSTM only | LSTM + GRU | LSTM + TCN |
| **Final Prediction** | XGBoost (tree-based) | Dense layers | Dense layers |
| **Feature Interactions** | Excellent | Good | Good |
| **Training Time** | Moderate | Fast | Slow |
| **Interpretability** | High (feature importance) | Low | Low |
| **Robustness** | High | Moderate | Moderate |

## Model Specifications

| Component | Specification |
|-----------|--------------|
| **Input Features** | 64 engineered features |
| **Sequence Length** | 24 hours (1-step ahead prediction) |
| **LSTM Hidden Units** | 128 |
| **LSTM Layers** | 2 |
| **LSTM Output Features** | 64 |
| **LSTM Dropout** | 0.3 |
| **Combined Features** | 128 (64 LSTM + 64 original) |
| **XGBoost Max Depth** | 6 |
| **XGBoost Learning Rate** | 0.1 |
| **XGBoost Estimators** | 500 (with early stopping) |
| **XGBoost Regularization** | L1=0.1, L2=1.0 |

## Expected Performance

Based on the hybrid architecture combining deep learning and gradient boosting:

- **Expected R²:** 0.84 - 0.88
- **Expected RMSE:** 220 - 260 bikes
- **Expected MAE:** 150 - 180 bikes

Should achieve competitive performance with benefits of interpretability and robustness.

## Usage

### Training the Model

```bash
cd lstm_xgboost
python train_lstm_xgboost.py
```

### Training Process

1. **LSTM Training (Stage 1)**
   - Trains LSTM feature extractor with temporary regression head
   - Uses early stopping (patience=15)
   - Saves best model based on validation loss

2. **Feature Extraction**
   - Uses trained LSTM to extract 64D features from all samples
   - Combines with original 64D point features → 128D total

3. **XGBoost Training (Stage 2)**
   - Trains gradient boosting on 128D combined features
   - Uses early stopping (50 rounds)
   - Automatically tunes tree depth and learning

### Data Requirements

- **Training Data:** `data/feature_data/train.csv`
- **Testing Data:** `data/feature_data/test.csv`
- **Features:** 64 optimized engineered features (temporal, lag, rolling, interactions)

### Output Files

#### Models
- `models/lstm_extractor.pth` - Trained LSTM feature extractor
- `models/best_lstm_extractor.pth` - Best LSTM (lowest validation loss)
- `models/xgboost_model.json` - Trained XGBoost model
- `models/architecture.json` - Model architecture details
- `models/feature_scaler.pkl` - Feature normalization scaler
- `models/target_scaler.pkl` - Target normalization scaler

#### Results
- `results/lstm_xgboost_metrics.json` - Detailed metrics and architecture
- `results/lstm_xgboost_metrics_summary.csv` - Performance summary
- `results/lstm_training_history.csv` - LSTM training history

## Advantages of LSTM-XGBoost

### 1. Interpretability
- XGBoost provides feature importance scores
- Can identify which temporal vs. point features matter most
- Helps understand model decisions

### 2. Robustness
- XGBoost handles outliers well
- Less sensitive to distribution shifts
- Regularization prevents overfitting

### 3. Flexibility
- Easy to tune XGBoost parameters independently
- Can replace LSTM with other temporal models
- Can add custom features to XGBoost stage

### 4. Efficiency
- XGBoost inference is fast (tree traversal)
- Good balance of accuracy and speed
- Suitable for production deployment

## When to Use LSTM-XGBoost

**Use LSTM-XGBoost when:**
- You need model interpretability (feature importance)
- You want robustness to outliers and noise
- You need fast inference time
- You want to understand feature contributions
- You need a balanced model (accuracy + speed)

**Use Other Hybrids when:**
- LSTM-GRU: Need pure deep learning, faster training
- LSTM-TCN: Need maximum accuracy, multi-scale patterns

## Feature Importance Analysis

After training, you can extract feature importance from XGBoost:

```python
# Get feature importance
importance = hybrid_model.xgb_model.feature_importances_

# First 64 features are LSTM temporal features
# Last 64 features are original point features
lstm_importance = importance[:64].sum()
point_importance = importance[64:].sum()

print(f"LSTM features contribution: {lstm_importance}")
print(f"Point features contribution: {point_importance}")
```

## References

- Original LSTM: Hochreiter & Schmidhuber (1997)
- XGBoost: Chen & Guestrin (2016)
- Seoul Bike Sharing Dataset: UCI Machine Learning Repository

## Implementation Details

### LSTM Feature Extractor
- Uses bidirectional information flow through stacked layers
- Dropout prevents overfitting on temporal patterns
- Dense compression layer creates richer features

### XGBoost Configuration
- Tree-based ensemble for handling non-linear interactions
- L1/L2 regularization for generalization
- Early stopping prevents overfitting
- Handles missing values automatically

### Combined Architecture Benefits
- LSTM: Temporal sequence modeling
- XGBoost: Complex feature interaction modeling
- Together: Best of both worlds
