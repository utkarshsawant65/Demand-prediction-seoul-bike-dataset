# TCN Model for Seoul Bike Demand Prediction

This folder contains the Temporal Convolutional Network (TCN) implementation for predicting Seoul bike rental demand.

## 🏆 Performance Summary

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| **TCN (Enhanced)** | **84.29%** | **242.20** | **171.69** |

## What is TCN?

Temporal Convolutional Network (TCN) is a powerful deep learning architecture specifically designed for sequential data and time series forecasting. Unlike RNNs/LSTMs, TCN uses:

- **Dilated Causal Convolutions**: Enables the model to have very large receptive fields with few layers
- **Residual Connections**: Helps with gradient flow and allows training of deeper networks
- **Parallel Processing**: Unlike sequential RNN/LSTM, TCN can process all time steps in parallel, making it faster to train

## Model Architecture

The Enhanced TCN model consists of:
- **5 Temporal Blocks**: Multiple layers of dilated causal convolutions with exponentially increasing dilation rates [1, 2, 4, 8, 16]
- **Channel Configuration**: [128, 128, 64, 64, 32]
- **Receptive Field**: 125 timesteps (covers full sequence + history)
- **Residual Connections**: Skip connections that help training deeper networks
- **Dropout Layers**: 0.3 dropout rate for regularization
- **Final Dense Layers**: Multi-layer output head for better mapping

## Files

- `train_tcn_enhanced.py`: Enhanced TCN training script
- `MODEL_COMPARISON.md`: Detailed analysis of TCN vs LSTM performance
- `models/`: Directory containing saved models and scalers
  - `tcn_enhanced_model.pth`: Trained Enhanced TCN model weights
  - `best_tcn_enhanced_model.pth`: Best model during training (lowest validation loss)
  - `feature_scaler_enhanced.pkl`: Scaler for input features
  - `target_scaler_enhanced.pkl`: Scaler for target variable
- `results/`: Directory containing training results and metrics
  - `tcn_enhanced_metrics.json`: Detailed metrics and configuration
  - `training_history_enhanced.csv`: Training and validation loss per epoch
  - `tcn_enhanced_metrics_summary.csv`: Summary of metrics on train and test sets

## Usage

### Training the Model

Run the training script from the repository root:

```bash
cd tcn
python train_tcn_enhanced.py
```

Or from the repository root:

```bash
python tcn/train_tcn_enhanced.py
```

### Model Configuration

The default hyperparameters in `train_tcn_enhanced.py`:

```python
SEQUENCE_LENGTH = 24      # Use 24 hours of history
NUM_CHANNELS = [128, 128, 64, 64, 32]  # TCN channel sizes per level
KERNEL_SIZE = 3           # Convolution kernel size
DROPOUT_RATE = 0.3        # Dropout rate for regularization
EPOCHS = 150              # Maximum training epochs
BATCH_SIZE = 32           # Batch size for training
LEARNING_RATE = 0.001     # Learning rate for Adam optimizer
WEIGHT_DECAY = 1e-5       # L2 regularization
VALIDATION_SPLIT = 0.2    # 20% of training data for validation
```

## Data Requirements

The model expects the following data structure:

- Training data: `data/feature_data/train.csv`
- Test data: `data/feature_data/test.csv`

Required columns:
- `Date`: Date and time information
- `Rented Bike Count`: Target variable (number of bikes rented)
- `Hour`: Hour of the day (0-23)
- `Temperature(°C)`: Temperature in Celsius
- `Humidity(%)`: Humidity percentage
- `Wind speed (m/s)`: Wind speed
- `Visibility (10m)`: Visibility
- `Dew point temperature(°C)`: Dew point temperature
- `Solar Radiation (MJ/m2)`: Solar radiation
- `Rainfall(mm)`: Rainfall amount
- `Snowfall (cm)`: Snowfall amount
- `Seasons`: Season (Winter/Spring/Summer/Autumn)
- `Holiday`: Whether it's a holiday (Holiday/No Holiday)
- `Functioning Day`: Whether bikes are available (Yes/No)

## Model Features

1. **Data Preprocessing**:
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features
   - Sequence creation for temporal patterns

2. **Training Features**:
   - Early stopping with patience of 20 epochs
   - Model checkpointing (saves best model based on validation loss)
   - Train/validation split for model evaluation
   - MSE loss function with Adam optimizer
   - Learning rate scheduling (ReduceLROnPlateau)
   - Gradient clipping for stable training
   - Weight decay for regularization

3. **Evaluation Metrics**:
   - R² (Coefficient of Determination)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - CV (Coefficient of Variation)
   - MAPE (Mean Absolute Percentage Error)

## Advantages of TCN

1. **Long-term Dependencies**: Dilated convolutions enable large receptive fields
2. **Parallel Processing**: Faster training compared to RNNs
3. **Stable Gradients**: Residual connections help avoid vanishing gradients
4. **Flexibility**: Easy to control receptive field size by adjusting network depth and dilation factors
5. **Lower Memory Footprint**: No need to maintain hidden states like in RNNs

## Requirements

The model requires PyTorch and standard data science libraries. Make sure you have:

```
torch>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib
tqdm
```

## Output

After training, the model will generate:

1. **Saved Models**: Best model checkpoint and final model
2. **Scalers**: Feature and target scalers for inference
3. **Metrics**: Comprehensive evaluation metrics in JSON and CSV formats
4. **Training History**: Loss curves for monitoring training progress

## Notes

- The model uses GPU if available, otherwise falls back to CPU
- Training typically takes longer than traditional ML models but less than LSTM
- The model automatically implements early stopping to prevent overfitting
- All random seeds are set for reproducibility
