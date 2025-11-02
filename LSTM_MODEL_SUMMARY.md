# LSTM Model for Seoul Bike Sharing Demand Prediction

## Overview

This document describes the Long Short-Term Memory (LSTM) neural network implementation for predicting bike sharing demand, trained with both random and temporal splits for comparison with CUBIST.

---

## Model Architecture

### LSTM Configuration

**Type**: Sequence-to-sequence forecasting using stacked LSTM layers

**Architecture**:
```
Input Layer (14 features)
    ↓
LSTM Layer 1 (128 hidden units)
    ↓
Dropout (20%)
    ↓
LSTM Layer 2 (128 hidden units)
    ↓
Dropout (20%)
    ↓
Fully Connected Layer 1 (128 → 64 units)
    ↓
ReLU Activation + Dropout (20%)
    ↓
Fully Connected Layer 2 (64 → 32 units)
    ↓
ReLU Activation + Dropout (20%)
    ↓
Output Layer (32 → 1 unit)
```

**Total Parameters**: ~216,193 trainable parameters

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sequence Length** | 24 hours | Lookback window (past 24 hours) |
| **Hidden Size** | 128 | LSTM hidden units per layer |
| **Number of Layers** | 2 | Stacked LSTM layers |
| **Dropout Rate** | 0.2 | Regularization (20%) |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Batch Size** | 64 | Training batch size |
| **Max Epochs** | 100 | Maximum training epochs |
| **Early Stopping Patience** | 15 | Stop if no improvement for 15 epochs |

---

## Data Preparation

### Features (14 total)

**Numeric Features (9)**:
- `Hour`: Hour of day (0-23)
- `Temp`: Temperature (°C)
- `Hum`: Humidity (%)
- `Wind`: Wind speed (m/s)
- `Visb`: Visibility (10m)
- `Dew`: Dew point temperature (°C)
- `Solar`: Solar radiation (MJ/m²)
- `Rain`: Rainfall (mm)
- `Snow`: Snowfall (cm)

**Categorical Features (5)** - Label encoded:
- `Season`: Winter, Spring, Summer, Autumn
- `Holiday`: Holiday, Workday
- `Fday`: Func (functioning), NoFunc (non-functioning)
- `WeekStatus`: Weekday, Weekend
- `DayName`: Monday-Sunday

### Preprocessing Steps

1. **Date Sorting**: Sort by Date and Hour to maintain temporal order
2. **Label Encoding**: Convert categorical features to integers (0, 1, 2, ...)
3. **Feature Scaling**: StandardScaler normalization (mean=0, std=1) for both X and y
4. **Sequence Creation**: Create rolling windows of 24-hour sequences
   - Input: Past 24 hours of data
   - Output: Next hour's bike count
5. **Train/Test Split**:
   - **Random**: 75/25 stratified split (paper replication)
   - **Temporal**: First 9 months train, last 3 months test (real-world)

### Sequence Example

```
Input Sequence (24 timesteps × 14 features):
[
  [Hour=0, Temp=-5.2, Hum=37, ..., Season=0, ...],  # t-24
  [Hour=1, Temp=-5.5, Hum=38, ..., Season=0, ...],  # t-23
  ...
  [Hour=23, Temp=1.9, Hum=43, ..., Season=0, ...]   # t-1
]
    ↓
Output: Bike Count at hour t
```

---

## Training Process

### Loss Function
**Mean Squared Error (MSE)**: Minimizes squared difference between predictions and actual values

### Optimizer
**Adam** with learning rate 0.001
- Adaptive learning rate
- Momentum-based optimization
- Default beta parameters (β₁=0.9, β₂=0.999)

### Learning Rate Scheduling
**ReduceLROnPlateau**:
- Monitors validation loss
- Reduces LR by factor of 0.5 if no improvement for 5 epochs
- Helps model fine-tune in later epochs

### Early Stopping
- Monitors validation loss
- Stops training if no improvement for 15 consecutive epochs
- Saves best model based on lowest validation loss

### Training Flow

1. **Forward Pass**: Input sequence → LSTM → FC layers → Prediction
2. **Loss Calculation**: MSE between prediction and actual
3. **Backward Pass**: Compute gradients via backpropagation through time (BPTT)
4. **Parameter Update**: Adam optimizer updates weights
5. **Validation**: Evaluate on test set (no gradient updates)
6. **LR Scheduling**: Adjust learning rate if validation loss plateaus
7. **Early Stopping Check**: Stop if patience counter exceeds threshold

---

## Evaluation Metrics

Same metrics as CUBIST for fair comparison:

### R² Score (Coefficient of Determination)
- **Formula**: R² = 1 - (SS_res / SS_tot)
- **Range**: -∞ to 1 (1 is perfect)
- **Interpretation**: Proportion of variance explained by the model

### RMSE (Root Mean Squared Error)
- **Formula**: RMSE = √(Σ(y_true - y_pred)² / n)
- **Unit**: Bikes
- **Interpretation**: Average prediction error (penalizes large errors more)

### MAE (Mean Absolute Error)
- **Formula**: MAE = Σ|y_true - y_pred| / n
- **Unit**: Bikes
- **Interpretation**: Average absolute prediction error

### CV (Coefficient of Variation)
- **Formula**: CV = (RMSE / mean(y_true)) × 100
- **Unit**: Percentage
- **Interpretation**: Normalized error relative to mean

---

## Comparison with CUBIST

### Model Characteristics

| Aspect | CUBIST | LSTM |
|--------|--------|------|
| **Type** | Rule-based + Linear | Deep Neural Network |
| **Training** | Grid search hyperparameters | Gradient descent |
| **Features** | All at once | Sequences (past 24 hours) |
| **Interpretability** | High (rules + coefficients) | Low (black box) |
| **Training Time** | ~2 minutes (R) | ~5-10 minutes (GPU) |
| **Parameters** | Varies (41 committees, 3 neighbors) | 216,193 parameters |
| **Memory Usage** | Low | Moderate-High |
| **Temporal Modeling** | Implicit (via features) | Explicit (sequences) |

### Expected Performance Patterns

**Random Split**:
- LSTM may perform similarly or slightly better than CUBIST
- LSTM can learn complex temporal patterns from scattered samples
- Both benefit from seeing data distributed across all months

**Temporal Split**:
- LSTM might struggle more than CUBIST if unseen season is very different
- However, LSTM's sequence modeling could help generalize better
- Depends on whether past 24-hour patterns transfer across seasons

---

## Files Structure

```
seoul-bike-thesis/
├── src/
│   └── lstm_model.py                 # LSTM implementation
├── train_lstm.py                     # Training script
├── models/
│   ├── lstm_random.pth               # Random split model
│   ├── lstm_temporal.pth             # Temporal split model
│   └── lstm_best.pth                 # Temporary best model during training
├── results/
│   └── lstm_results.json             # Performance metrics
└── reports/
    └── figures/
        └── lstm/                      # LSTM-specific figures
```

---

## Expected Results Structure

```json
{
  "random": {
    "train": {
      "Set": "Training",
      "R2": 0.XXXX,
      "RMSE": XX.XX,
      "MAE": XX.XX,
      "CV": XX.XX
    },
    "test": {
      "Set": "Testing",
      "R2": 0.XXXX,
      "RMSE": XX.XX,
      "MAE": XX.XX,
      "CV": XX.XX
    },
    "config": {...},
    "split_info": {...}
  },
  "temporal": {
    "train": {...},
    "test": {...},
    "config": {...},
    "split_info": {...}
  }
}
```

---

## Advantages of LSTM

1. **Temporal Dependencies**: Explicitly models sequential patterns
2. **Long-Term Memory**: Can remember patterns over 24+ hours
3. **Non-Linear Modeling**: Captures complex relationships
4. **Feature Interactions**: Automatically learns feature combinations
5. **No Manual Rules**: Data-driven approach

---

## Limitations of LSTM

1. **Interpretability**: Black box - hard to explain predictions
2. **Training Time**: Slower than CUBIST
3. **Hyperparameter Tuning**: Requires careful architecture design
4. **Data Hungry**: Needs sufficient training data
5. **Overfitting Risk**: Can memorize training data if not regularized

---

## Training Commands

### Train both models
```bash
python train_lstm.py
```

### Expected Output
```
================================================================================
LSTM MODEL TRAINING FOR SEOUL BIKE DATA
================================================================================

################################################################################
# TRAINING LSTM WITH RANDOM SPLIT
################################################################################
Using device: cpu (or cuda if GPU available)

[OK] Loading data...
[OK] Loaded 8760 samples
[OK] Creating RANDOM split (75/25)
[OK] Training samples: 6570
[OK] Testing samples: 2190
[OK] Created 6546 training sequences
[OK] Created 2166 testing sequences

[OK] Model architecture:
    - Total parameters: 216,193

[OK] Starting training for 100 epochs (patience=15)
Epoch [  1/100] - Train Loss: X.XXXXXX, Val Loss: X.XXXXXX
Epoch [  5/100] - Train Loss: X.XXXXXX, Val Loss: X.XXXXXX
...

[OK] Training completed
[OK] Best validation loss: X.XXXXXX

============================================================
TRAINING SET RESULTS
============================================================
R²:    X.XXXX (XX.XX%)
RMSE:  XXX.XX bikes
MAE:   XXX.XX bikes
CV:    XX.XX%

============================================================
TESTING SET RESULTS
============================================================
R²:    X.XXXX (XX.XX%)
RMSE:  XXX.XX bikes
MAE:   XXX.XX bikes
CV:    XX.XX%

################################################################################
# TRAINING LSTM WITH TEMPORAL SPLIT
################################################################################
[Similar output for temporal split...]

================================================================================
LSTM MODEL COMPARISON
================================================================================
Random Split (Paper Replication)     | Temporal Split (Real-World)
--------------------------------------------------------------------------------
Training R²: X.XXXX (XX.XX%)         | Training R²: X.XXXX (XX.XX%)
Testing R²:  X.XXXX (XX.XX%)         | Testing R²:  X.XXXX (XX.XX%)
...
```

---

## Reproducibility

**Random Seed**: 42 (for train/test split)
**PyTorch Determinism**: Not enforced (for performance)
- Results may vary slightly between runs due to random initialization
- General trends should be consistent

**To improve reproducibility**:
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Next Steps

1. **Training**: Wait for both models to train (~5-10 minutes)
2. **Results Analysis**: Compare LSTM vs CUBIST performance
3. **Visualization**: Create comparison charts
4. **Discussion**: Analyze why each model performs as it does
5. **Thesis Integration**: Document findings

---

## Technical Notes

### Sequence Length Choice (24 hours)
- **Rationale**: Daily patterns (morning commute, evening, night)
- **Trade-off**: Longer sequences = more context but fewer samples
- **Alternative**: Could try 12, 48, or 168 (1 week) hours

### LSTM vs GRU
- Used LSTM instead of GRU for established track record
- GRU is simpler (fewer parameters) but LSTM often performs better for time series
- Could experiment with GRU in future work

### GPU Acceleration
- Training is CPU-compatible
- GPU significantly speeds up training (5-10x faster)
- Model automatically uses GPU if available

---

**Model Status**: Training in progress...
**Expected completion**: ~5-10 minutes
**Output file**: `results/lstm_results.json`
