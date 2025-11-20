# TCN Model for Seoul Bike Demand Prediction

This folder contains the Temporal Convolutional Network (TCN) implementation for predicting Seoul bike rental demand.

## 🏆 Performance Summary

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| TCN (Baseline) | 54.71% | 411.19 | 309.90 |
| **TCN (Enhanced)** | **66.45%** | **353.89** | **264.32** |
| LSTM | 65.37% | 359.56 | 250.12 |

**The Enhanced TCN outperforms both the baseline TCN and LSTM!** See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for detailed analysis.

## What is TCN?

Temporal Convolutional Network (TCN) is a powerful deep learning architecture specifically designed for sequential data and time series forecasting. Unlike RNNs/LSTMs, TCN uses:

- **Dilated Causal Convolutions**: Enables the model to have very large receptive fields with few layers
- **Residual Connections**: Helps with gradient flow and allows training of deeper networks
- **Parallel Processing**: Unlike sequential RNN/LSTM, TCN can process all time steps in parallel, making it faster to train

## Model Architecture

The TCN model consists of:
- **Temporal Blocks**: Multiple layers of dilated causal convolutions with exponentially increasing dilation rates
- **Residual Connections**: Skip connections that help training deeper networks
- **Dropout Layers**: For regularization
- **Final Dense Layer**: Maps TCN output to bike rental count prediction

## Files

- `train_tcn.py`: Baseline TCN training script
- `train_tcn_enhanced.py`: **Enhanced TCN training script (RECOMMENDED)**
- `MODEL_COMPARISON.md`: Detailed analysis of TCN vs LSTM performance
- `models/`: Directory containing saved models and scalers
  - `tcn_model.pth`: Trained TCN model weights
  - `best_tcn_model.pth`: Best model during training (lowest validation loss)
  - `feature_scaler.pkl`: Scaler for input features
  - `target_scaler.pkl`: Scaler for target variable
- `results/`: Directory containing training results and metrics
  - `tcn_metrics.json`: Detailed metrics and configuration
  - `training_history.csv`: Training and validation loss per epoch
  - `tcn_metrics_summary.csv`: Summary of metrics on train and test sets

## Usage

### Training the Model

Run the training script from the repository root:

```bash
cd tcn
python train_tcn.py
```

Or from the repository root:

```bash
python tcn/train_tcn.py
```

### Model Configuration

The default hyperparameters in `train_tcn.py`:

```python
SEQUENCE_LENGTH = 24      # Use 24 hours of history
NUM_CHANNELS = [64, 64, 32]  # TCN channel sizes per level
KERNEL_SIZE = 3           # Convolution kernel size
DROPOUT_RATE = 0.2        # Dropout rate for regularization
EPOCHS = 100              # Maximum training epochs
BATCH_SIZE = 32           # Batch size for training
LEARNING_RATE = 0.001     # Learning rate for Adam optimizer
VALIDATION_SPLIT = 0.2    # 20% of training data for validation
```

## Data Requirements

The model expects the following data structure:

- Training data: `data/model_data/train.csv`
- Test data: `data/model_data/test.csv`

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
   - Early stopping with patience of 15 epochs
   - Model checkpointing (saves best model based on validation loss)
   - Train/validation split for model evaluation
   - MSE loss function with Adam optimizer

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
