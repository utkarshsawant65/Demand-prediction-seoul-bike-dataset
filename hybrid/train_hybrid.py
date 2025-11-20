"""
HYBRID LSTM-TCN Model Training for Seoul Bike Data
Combines LSTM and TCN architectures for enhanced temporal pattern learning

Architecture:
    - TCN branch: Captures multi-scale temporal patterns with dilated convolutions
    - LSTM branch: Captures sequential dependencies
    - Feature fusion: Concatenate outputs from both branches
    - Dense layers: Final prediction from combined features

This hybrid approach leverages:
    - TCN's strength in capturing long-term dependencies with efficient receptive field
    - LSTM's strength in sequential pattern modeling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# TCN COMPONENTS
# ============================================================================

class Chomp1d(nn.Module):
    """Removes extra padding from convolution to ensure causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal Block with dilated causal convolutions"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranch(nn.Module):
    """TCN branch for the hybrid model"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size,
                                       padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.output_size = num_channels[-1]

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take the last time step
        y = y[:, :, -1]
        return y


# ============================================================================
# LSTM COMPONENTS
# ============================================================================

class LSTMBranch(nn.Module):
    """LSTM branch for the hybrid model"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(LSTMBranch, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        return self.dropout(last_output)


# ============================================================================
# HYBRID MODEL
# ============================================================================

class HybridLSTMTCN(nn.Module):
    """
    Hybrid LSTM-TCN Model

    Architecture:
        1. Parallel processing through TCN and LSTM branches
        2. Feature concatenation
        3. Dense layers for final prediction
    """
    def __init__(self, num_features,
                 tcn_channels=[128, 128, 64, 64, 32],
                 lstm_hidden=128,
                 lstm_layers=2,
                 fusion_hidden=128,
                 dropout=0.3):
        super(HybridLSTMTCN, self).__init__()

        # TCN branch
        self.tcn = TCNBranch(
            num_inputs=num_features,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )

        # LSTM branch
        self.lstm = LSTMBranch(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # Calculate combined feature size
        combined_size = self.tcn.output_size + self.lstm.output_size

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Process through both branches
        tcn_out = self.tcn(x)      # (batch, tcn_output_size)
        lstm_out = self.lstm(x)    # (batch, lstm_output_size)

        # Concatenate features from both branches
        combined = torch.cat([tcn_out, lstm_out], dim=1)

        # Final prediction
        output = self.fusion(combined)
        return output


# ============================================================================
# DATASET
# ============================================================================

class BikeDataset(Dataset):
    """Dataset class for bike rental data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_data(train_path='../data/feature_data/train.csv', test_path='../data/feature_data/test.csv'):
    """Load train and test data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess data for Hybrid model training"""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    # Target column - feature_data has both 'Rented Bike Count' and 'target'
    target_col = 'target'

    # Columns to exclude from features
    exclude_cols = [target_col, 'Rented Bike Count']

    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    print(f"Features used: {len(feature_cols)}")
    print(f"First 10 features: {feature_cols[:10]}")
    print(f"Last 10 features: {feature_cols[-10:]}")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"\nTrain shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Test shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler, target_scaler, feature_cols)


def create_sequences(X, y, sequence_length=24):
    """Create sequences for model input"""
    print(f"\nCreating sequences with length {sequence_length}...")

    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

    return X_seq, y_seq


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train the hybrid model"""
    print("\n" + "="*80)
    print("TRAINING HYBRID LSTM-TCN MODEL")
    print("="*80)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Early stopping patience: {patience}")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_hybrid_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load('models/best_hybrid_model.pth'))
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, target_scaler, device, set_name='Test'):
    """Evaluate model and calculate metrics"""
    print("\n" + "="*80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("="*80)

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform to original scale
    y_pred = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100

    # Handle MAPE carefully (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    print(f"\nMetrics:")
    print(f"  R²:    {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  CV:    {cv:.2f}%")
    print(f"  MAPE:  {mape:.2f}%")

    metrics = {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'CV': float(cv),
        'MAPE': float(mape)
    }

    return metrics, y_true, y_pred


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(train_metrics, test_metrics, history, feature_cols, model_info):
    """Save training results and metrics"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save metrics
    results = {
        'model_type': 'Hybrid LSTM-TCN',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'model_architecture': model_info,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('results/hybrid_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: results/hybrid_metrics.json")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })

    history_df.to_csv('results/training_history.csv', index=False)
    print("Saved: results/training_history.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })

    metrics_df.to_csv('results/hybrid_metrics_summary.csv', index=False)
    print("Saved: results/hybrid_metrics_summary.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# HYBRID LSTM-TCN MODEL TRAINING #")
    print("#"*80 + "\n")

    # Hyperparameters
    SEQUENCE_LENGTH = 24
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Model architecture
    TCN_CHANNELS = [128, 128, 64, 64, 32]
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    FUSION_HIDDEN = 128
    DROPOUT = 0.3

    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    # Split training data into train/validation
    val_size = int(len(X_train_seq) * VALIDATION_SPLIT)
    X_train_final = X_train_seq[:-val_size]
    y_train_final = y_train_seq[:-val_size]
    X_val = X_train_seq[-val_size:]
    y_val = y_train_seq[-val_size:]

    print(f"\nFinal data split:")
    print(f"  Train: {len(X_train_final)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test_seq)} samples")

    # Create datasets and dataloaders
    train_dataset = BikeDataset(X_train_final, y_train_final)
    val_dataset = BikeDataset(X_val, y_val)
    test_dataset = BikeDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n" + "="*80)
    print("BUILDING HYBRID LSTM-TCN MODEL")
    print("="*80)

    num_features = X_train_seq.shape[2]

    model = HybridLSTMTCN(
        num_features=num_features,
        tcn_channels=TCN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        fusion_hidden=FUSION_HIDDEN,
        dropout=DROPOUT
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"  TCN Channels: {TCN_CHANNELS}")
    print(f"  LSTM Hidden: {LSTM_HIDDEN}")
    print(f"  LSTM Layers: {LSTM_LAYERS}")
    print(f"  Fusion Hidden: {FUSION_HIDDEN}")
    print(f"  Dropout: {DROPOUT}")
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    model_info = {
        'tcn_channels': TCN_CHANNELS,
        'lstm_hidden': LSTM_HIDDEN,
        'lstm_layers': LSTM_LAYERS,
        'fusion_hidden': FUSION_HIDDEN,
        'dropout': DROPOUT,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=device
    )

    # Evaluate on training set
    train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        model, train_loader_eval, target_scaler, device, 'Training'
    )

    # Evaluate on test set
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        model, test_loader, target_scaler, device, 'Testing'
    )

    # Save scalers
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    print("\nSaved scalers:")
    print("  models/feature_scaler.pkl")
    print("  models/target_scaler.pkl")

    # Save final model
    torch.save(model.state_dict(), 'models/hybrid_model.pth')
    print("\nSaved final model:")
    print("  models/hybrid_model.pth")

    # Save results
    save_results(train_metrics, test_metrics, history, feature_cols, model_info)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Training R²: {train_metrics['R2']:.4f}")
    print(f"  Testing R²: {test_metrics['R2']:.4f}")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")
    print("\n")


if __name__ == "__main__":
    main()
