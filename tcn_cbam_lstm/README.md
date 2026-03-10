# TCN-CBAM-LSTM: Novel Hybrid Model for Bike Demand Forecasting

## Overview

This directory contains the implementation of a **novel TCN-CBAM-LSTM** architecture for Seoul bike demand forecasting. The key innovation is the integration of **CBAM (Convolutional Block Attention Module)** into the TCN-LSTM hybrid model.

## Novelty & Scientific Contribution

### What makes this model different?

Standard TCN-LSTM models process temporal data without explicitly learning which features or time steps are most important. Our model introduces **CBAM attention** which provides:

1. **Channel Attention**: Dynamically learns which feature channels (e.g., temperature, humidity, hour encodings) are most important at each layer
2. **Spatial Attention**: Learns which temporal positions in the sequence are most relevant for prediction

### Why is this different from standard temporal attention?

| Attention Type | What it Does | Used In |
|----------------|--------------|---------|
| **Temporal Attention** | Weights across time steps only | Standard attention mechanisms |
| **CBAM (Ours)** | Weights across BOTH channels AND spatial positions | This novel model |

### Research Backing

- CBAM was originally proposed by Woo et al. (2018) for computer vision
- Recent 2024-2025 research shows CBAM applied to time series can reduce:
  - MSE by 22.04%
  - MAE by 14.62%
  - MAPE by 6.18%

## Architecture

```
Input (batch, 24, 30)
    |
    v
+-------------------+
|   TCN Block 1     |
|   (Conv + Dilate) |
+-------------------+
    |
    v
+-------------------+
|      CBAM 1       |  <-- Channel Attention + Spatial Attention
+-------------------+
    |
    v
+-------------------+
|   TCN Block 2     |
+-------------------+
    |
    v
+-------------------+
|      CBAM 2       |
+-------------------+
    |
    ... (repeat for all TCN blocks)
    |
    v
+-------------------+
|       LSTM        |
|   (2 layers)      |
+-------------------+
    |
    v
+-------------------+
|   Dense Layers    |
|   (128 -> 64 -> 1)|
+-------------------+
    |
    v
Output (predicted demand)
```

## CBAM Module Explained

### Channel Attention
```
Input Feature Map
    |
    +---> Global Avg Pool ---> MLP ---+
    |                                  |
    +---> Global Max Pool ---> MLP ---+
                                       |
                                       v
                                   Add + Sigmoid
                                       |
                                       v
                              Channel Attention Weights
```

### Spatial Attention
```
Input Feature Map
    |
    +---> Channel Avg ---> Concat ---> Conv1D ---> Sigmoid
    |                        ^
    +---> Channel Max -------+
                                       |
                                       v
                              Spatial Attention Weights
```

## Files

```
tcn_cbam_lstm/
├── train_tcn_cbam_lstm.py    # Main training script
├── models/
│   ├── best_tcn_cbam_lstm_model.pth  # Best model checkpoint
│   ├── tcn_cbam_lstm_model.pth       # Final model
│   ├── feature_scaler.pkl            # Feature scaler
│   └── target_scaler.pkl             # Target scaler
├── results/
│   ├── tcn_cbam_lstm_metrics.json    # Full metrics
│   ├── tcn_cbam_lstm_metrics_summary.csv
│   └── training_history.csv
└── README.md
```

## Usage

### Training

```bash
cd tcn_cbam_lstm
python train_tcn_cbam_lstm.py
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 24 | Hours of input |
| Batch Size | 32 | Training batch size |
| Learning Rate | 0.001 | Initial LR |
| TCN Channels | [128, 128, 64, 64, 32] | TCN layer sizes |
| LSTM Hidden | 128 | LSTM hidden size |
| LSTM Layers | 2 | Number of LSTM layers |
| Dropout | 0.3 | Dropout rate |
| **CBAM Reduction** | 16 | Channel attention reduction ratio |

## Expected Results

### Comparison with Baseline

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| TCN-LSTM (Baseline) | 88.37% | 208.36 | 138.77 |
| **TCN-CBAM-LSTM (Ours)** | TBD | TBD | TBD |

## Thesis Contribution Statement

> "We propose a novel TCN-CBAM-LSTM architecture that integrates Convolutional Block Attention Module (CBAM) into the temporal convolutional network. Unlike standard temporal attention mechanisms that only weight across time steps, CBAM applies dual attention: (1) Channel Attention that dynamically learns feature importance, and (2) Spatial Attention that identifies critical temporal positions. This enables the model to focus on the most relevant weather features and time periods for bike demand prediction."

## References

1. Woo, S., Park, J., Lee, J.Y., & Kweon, I.S. (2018). CBAM: Convolutional Block Attention Module. ECCV.
2. TCN-CBAM for Chaotic Time Series Prediction (2021). Chaos, Solitons & Fractals.
3. MSACN-LSTM with CBAM (2025). International Journal of Machine Learning and Cybernetics.

## Author

Utkarsh - Seoul Bike Demand Forecasting Thesis
