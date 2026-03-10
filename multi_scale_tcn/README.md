# Multi-Scale TCN for Seoul Bike Demand Forecasting

## Overview

Multi-Scale TCN uses **parallel TCN branches with different kernel sizes** to capture temporal patterns at multiple time scales simultaneously. This architecture is particularly effective for bike demand forecasting where patterns exist at multiple temporal granularities:

- **Hourly patterns**: Rush hour peaks, lunch breaks
- **Daily patterns**: Weekday vs weekend differences
- **Weekly patterns**: Day-of-week variations

## Architecture

```
Input Sequence (batch, 24, features)
         │
         ├──────────────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ TCN Branch  │   │ TCN Branch  │   │ TCN Branch  │   │ TCN Branch  │
   │ Kernel = 2  │   │ Kernel = 3  │   │ Kernel = 5  │   │ Kernel = 7  │
   │ (Fine)      │   │ (Short)     │   │ (Medium)    │   │ (Long)      │
   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
         │                  │                  │                  │
         └──────────────────┴──────────────────┴──────────────────┘
                                    │
                           ┌────────▼────────┐
                           │ Scale Attention │
                           │   (Learnable    │
                           │    Weights)     │
                           └────────┬────────┘
                                    │
                           ┌────────▼────────┐
                           │  FC Layers      │
                           │  128 → 64 → 32  │
                           │     → 1         │
                           └────────┬────────┘
                                    │
                                    ▼
                              Prediction
```

## Key Features

### 1. Multi-Scale Temporal Convolutions
Each branch uses a different kernel size to capture patterns at different time scales:

| Branch | Kernel Size | Receptive Field | Captures |
|--------|-------------|-----------------|----------|
| 1 | 2 | 13 timesteps | Fine-grained hourly changes |
| 2 | 3 | 25 timesteps | Short-term (2-3 hour) patterns |
| 3 | 5 | 49 timesteps | Medium-term (half-day) trends |
| 4 | 7 | 73 timesteps | Longer daily patterns |

### 2. Scale Attention Mechanism
A learnable attention mechanism weights the contribution of each scale:
- Automatically learns which time scales are most predictive
- Adapts to the data characteristics
- Provides interpretability through attention weights

### 3. Dilated Causal Convolutions
Each branch maintains causality (no future information leakage) while achieving large receptive fields through exponentially increasing dilation rates.

## Usage

### Training
```bash
cd c:\Git\seoul-bike-thesis
python multi_scale_tcn/train_multi_scale_tcn.py
```

### Configuration
Key hyperparameters can be modified in the `main()` function:

```python
SEQUENCE_LENGTH = 24      # Input sequence length (hours)
NUM_CHANNELS = [64, 64, 32]  # Channels per TCN level
KERNEL_SIZES = [2, 3, 5, 7]  # Different time scales
DROPOUT_RATE = 0.25
USE_ATTENTION = True      # Enable scale attention
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

## Output Files

After training, the following files are generated:

### Models
- `models/multi_scale_tcn_model.pth` - Final trained model
- `models/best_multi_scale_tcn_model.pth` - Best model (lowest validation loss)
- `models/feature_scaler.pkl` - Feature StandardScaler
- `models/target_scaler.pkl` - Target StandardScaler

### Results
- `results/multi_scale_tcn_metrics.json` - Complete metrics and configuration
- `results/multi_scale_tcn_metrics_summary.csv` - Summary metrics table
- `results/training_history.csv` - Epoch-by-epoch training history

### Visualizations
- `results/multi_scale_tcn_predictions.png` - Prediction vs actual plots
- `results/scale_attention_weights.png` - Attention weight distribution

## Model Architecture Details

### TCNBranch
Each branch is a standard TCN with:
- Multiple TemporalBlocks with exponentially increasing dilation
- Residual connections for gradient flow
- ReLU activation and dropout regularization

### ScaleAttention
Attention mechanism:
```
Concat(branch_outputs) → Linear(total, total/2) → ReLU → Linear(total/2, num_scales) → Softmax
```

### Output Head
```
Weighted_features → Linear(32, 128) → ReLU → Dropout
                 → Linear(128, 64) → ReLU → Dropout
                 → Linear(64, 32) → ReLU
                 → Linear(32, 1)
```

## Comparison with Standard TCN

| Aspect | Standard TCN | Multi-Scale TCN |
|--------|--------------|-----------------|
| Kernel Size | Single (3) | Multiple (2, 3, 5, 7) |
| Time Scales | One | Multiple simultaneous |
| Parameters | ~45K | ~120K |
| Interpretability | Limited | Scale attention weights |
| Flexibility | Fixed receptive field | Adaptive scale weighting |

## References

- [TCN Paper](https://arxiv.org/abs/1803.01271) - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- Multi-scale architectures in time series forecasting
