# Hybrid LSTM-TCN Model - Summary

## 🎯 Overview

A novel hybrid architecture combining **LSTM** and **TCN** (Temporal Convolutional Network) to leverage the strengths of both approaches for Seoul bike demand forecasting.

---

## 🏗️ Architecture

```
Input (24h × 64 features)
         |
    ┌────┴────┐
    │         │
  TCN      LSTM
 Branch   Branch
    │         │
[32 dim] [128 dim]
    │         │
    └────┬────┘
         |
   Concatenate
    [160 dim]
         |
    Fusion Layers
   [128→64→1]
         |
   Prediction
```

### Components:

1. **TCN Branch**
   - Layers: [128, 128, 64, 64, 32]
   - Dilations: 1, 2, 4, 8, 16
   - Receptive field: 63 timesteps
   - Output: 32-dimensional features

2. **LSTM Branch**
   - 2 stacked LSTM layers
   - Hidden size: 128
   - Output: 128-dimensional features

3. **Fusion Module**
   - Dense layers: 160 → 128 → 64 → 1
   - Dropout: 0.3
   - Final prediction

---

## 📊 Model Specifications

| Specification | Value |
|--------------|-------|
| **Total Parameters** | ~522,000 |
| **Input Features** | 64 (enhanced features) |
| **Sequence Length** | 24 hours |
| **Output** | 1 hour ahead prediction |
| **Dropout** | 0.3 |
| **Optimizer** | Adam (lr=0.001) |
| **Batch Size** | 32 |
| **Early Stopping** | Patience = 15 |

---

## 💡 Why Hybrid?

### TCN Strengths:
✅ Large receptive field (63 timesteps)
✅ Parallel processing (fast training)
✅ Stable gradients
✅ Multi-scale pattern capture

### LSTM Strengths:
✅ Sequential dependency modeling
✅ Gated memory mechanisms
✅ Long-term pattern retention
✅ Proven effectiveness

### Hybrid Benefits:
🎯 Complementary learning
🎯 Ensemble-like robustness
🎯 Better generalization
🎯 Enhanced feature utilization

---

## 📈 Expected Performance

Based on individual model results:
- **TCN Enhanced**: R² = 0.6645, RMSE = 353.89
- **LSTM Enhanced**: R² = 0.6357, RMSE = 368.80

**Expected Hybrid Performance**:
- **Target R²**: 0.68 - 0.72 (68-72%)
- **Target RMSE**: 340 - 360 bikes
- **Target MAE**: 240 - 270 bikes
- **Improvement**: +2-5% over best individual model

---

## 🔧 Training Details

### Data Split:
- **Training**: 5,453 sequences (2017-12-08 to 2018-09-18)
- **Validation**: 1,363 sequences (20% of training)
- **Testing**: 1,728 sequences (2018-09-19 to 2018-11-30)

### Features (64 total):
- Temporal: 20 features
- Lag: 8 features
- Rolling: 10 features
- Interactions: 13 features
- Weather change: 3 features
- Categorical: 5 features
- Original weather: 8 features

### Training Strategy:
- Loss: MSE (Mean Squared Error)
- Learning rate scheduler: ReduceLROnPlateau
- Regularization: Dropout (0.3) + Early stopping
- Validation: 1-step ahead prediction

---

## 📁 Project Structure

```
hybrid/
├── train_hybrid.py                 # Main training script
├── README.md                       # Usage guide
├── HYBRID_MODEL_ARCHITECTURE.md    # Detailed architecture docs
├── models/
│   ├── hybrid_model.pth           # Final trained model
│   ├── best_hybrid_model.pth      # Best validation model
│   ├── feature_scaler.pkl         # Feature normalization
│   └── target_scaler.pkl          # Target normalization
└── results/
    ├── hybrid_metrics.json         # Detailed metrics
    ├── hybrid_metrics_summary.csv  # Performance summary
    └── training_history.csv        # Training progress
```

---

## 🚀 Usage

### Training:
```bash
cd hybrid
python train_hybrid.py
```

### Loading Model:
```python
import torch

# Load model
model = HybridLSTMTCN(num_features=64)
model.load_state_dict(torch.load('models/best_hybrid_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

---

## ⚖️ Model Comparison

| Model | Parameters | R² (Expected) | RMSE (Expected) | Speed | Complexity |
|-------|-----------|---------------|-----------------|-------|------------|
| LSTM Basic | ~47K | 0.6537 | 359.56 | Fast | Low |
| TCN Enhanced | ~180K | 0.6645 | 353.89 | Very Fast | Medium |
| LSTM Enhanced | ~110K | 0.6357 | 368.80 | Medium | High |
| **Hybrid** | **~522K** | **0.68-0.72** | **340-360** | **Medium** | **High** |

---

## 🎓 Key Innovations

1. **Parallel Feature Extraction**
   - TCN and LSTM process same input independently
   - Captures different aspects of temporal patterns

2. **Intelligent Fusion**
   - Learned combination of branch outputs
   - Non-linear feature interactions

3. **Balanced Architecture**
   - TCN for multi-scale patterns
   - LSTM for sequential dependencies
   - Fusion for optimal combination

4. **Production-Ready**
   - Early stopping prevents overfitting
   - Dropout for regularization
   - Scalable to larger datasets

---

## 📊 Training Output

The model will show:
```
Epoch [1/100] - Train Loss: 0.xxxx, Val Loss: 0.xxxx
Epoch [5/100] - Train Loss: 0.xxxx, Val Loss: 0.xxxx
...
Early stopping triggered at epoch XX
Training completed. Best validation loss: 0.xxxx

TRAINING SET EVALUATION
  R²:    0.xxxx (xx.xx%)
  RMSE:  xxx.xx
  MAE:   xxx.xx

TESTING SET EVALUATION
  R²:    0.xxxx (xx.xx%)
  RMSE:  xxx.xx
  MAE:   xxx.xx
```

---

## 🔮 Future Enhancements

1. **Attention Mechanism**
   - Add attention layer to fusion module
   - Learn which branch to focus on

2. **Multi-Task Learning**
   - Predict demand + uncertainty
   - Joint optimization

3. **Adaptive Weighting**
   - Dynamic branch importance
   - Context-dependent fusion

4. **Feature Importance**
   - Analyze which features each branch uses
   - Feature selection optimization

---

## 📝 Citation

```
Hybrid LSTM-TCN Model for Seoul Bike Demand Prediction
Combining Temporal Convolutional Networks and Long Short-Term Memory
Enhanced with Comprehensive Feature Engineering
2025
```

---

## ✅ Advantages Over Individual Models

| Advantage | Description |
|-----------|-------------|
| **Better Accuracy** | +2-5% R² improvement expected |
| **Robust Predictions** | Ensemble-like behavior |
| **Feature Utilization** | Both branches leverage different feature types |
| **Generalization** | Reduced overfitting through diversity |
| **Flexibility** | Can weight branches for different scenarios |

---

## ⚠️ Considerations

| Aspect | Note |
|--------|------|
| **Training Time** | Longer than individual models (~20-40 min CPU) |
| **Memory** | Requires ~1.5GB RAM |
| **Complexity** | More parameters = more data needed |
| **Interpretability** | Harder to explain than simple models |

---

## 🎯 When to Use Hybrid Model

### ✅ Use When:
- Maximum accuracy is required
- Sufficient computational resources available
- Production forecasting system
- Complex temporal patterns in data

### ❌ Don't Use When:
- Need real-time inference (<1ms)
- Limited computational resources
- Simple patterns (use LSTM Basic instead)
- Need high interpretability

---

## 📚 Documentation

- **[README.md](hybrid/README.md)** - Usage guide
- **[HYBRID_MODEL_ARCHITECTURE.md](hybrid/HYBRID_MODEL_ARCHITECTURE.md)** - Detailed architecture
- **[train_hybrid.py](hybrid/train_hybrid.py)** - Implementation code

---

**Status**: ✅ Training in progress
**Framework**: PyTorch 2.0
**Python**: 3.12+
**Created**: 2025-11-18
