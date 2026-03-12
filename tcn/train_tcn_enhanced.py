"""
Enhanced TCN (Temporal Convolutional Network) Model Training for Seoul Bike Data
Improved version with better hyperparameters for higher performance
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


class Chomp1d(nn.Module):
    """
    Removes the extra padding from the convolution output
    to ensure causality (no future information leak)
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """
    Temporal Block with dilated causal convolutions
    """
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


class EnhancedTCN(nn.Module):
    """
    Enhanced Temporal Convolutional Network with deeper architecture
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(EnhancedTCN, self).__init__()
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

        # Enhanced output layers
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take the last time step
        y = y[:, :, -1]

        # Enhanced output layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)

        return y


class BikeDataset(Dataset):
    """Dataset class for bike rental data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(train_path='data/feature_data/train.csv', test_path='data/feature_data/test.csv'):
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
    """Preprocess data for TCN training"""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    # Make copies to avoid modifying original data
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    # Target column - try both possible names
    if 'target' in train_processed.columns:
        target_col = 'target'
    else:
        target_col = 'Rented Bike Count'

    exclude_cols = [target_col, 'Rented Bike Count', 'target']

    # Separate features and target
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

    X_train = train_processed[feature_cols].values
    y_train = train_processed[target_col].values

    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

    print(f"Features used: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature names: {feature_cols}")

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
    """Create sequences for TCN input"""
    print(f"\nCreating sequences with length {sequence_length}...")

    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

    return X_seq, y_seq


def build_tcn_model(num_features, num_channels=[128, 128, 64, 64, 32], kernel_size=3, dropout=0.3):
    """Build enhanced TCN model architecture"""
    print("\n" + "="*80)
    print("BUILDING ENHANCED TCN MODEL")
    print("="*80)

    model = EnhancedTCN(
        num_inputs=num_features,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )

    # Calculate receptive field
    receptive_field = 1
    for i in range(len(num_channels)):
        receptive_field += 2 * (kernel_size - 1) * (2 ** i)

    print(f"\nModel Configuration:")
    print(f"  Input features: {num_features}")
    print(f"  Channel sizes: {num_channels}")
    print(f"  Number of levels: {len(num_channels)}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Dropout rate: {dropout}")
    print(f"  Receptive field: {receptive_field} timesteps")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001):
    """Train TCN model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    # Create output directories
    os.makedirs('tcn/models', exist_ok=True)
    os.makedirs('tcn/results', exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-2)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: 1e-5")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': []
    }

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'tcn/models/best_tcn_enhanced_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load('tcn/models/best_tcn_enhanced_model.pth'))

    return model, history


def evaluate_model(model, X, y, target_scaler, device, set_name='Test'):
    """Evaluate model and calculate metrics"""
    print("\n" + "="*80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("="*80)

    model.eval()

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)

    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    # Inverse transform to original scale
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100

    # Calculate MAPE, avoiding division by zero
    mask = y_true > 10
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


def save_results(train_metrics, test_metrics, history, feature_cols, model_config):
    """Save training results and metrics"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save metrics
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'model_config': model_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('tcn/results/tcn_enhanced_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: tcn/results/tcn_enhanced_metrics.json")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })

    history_df.to_csv('tcn/results/training_history_enhanced.csv', index=False)
    print("Saved: tcn/results/training_history_enhanced.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })

    metrics_df.to_csv('tcn/results/tcn_enhanced_metrics_summary.csv', index=False)
    print("Saved: tcn/results/tcn_enhanced_metrics_summary.csv")


def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# SEOUL BIKE ENHANCED TCN MODEL TRAINING #")
    print("#"*80 + "\n")

    SEQUENCE_LENGTH = 8
    NUM_CHANNELS = [32, 16, 16]
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.5
    EPOCHS = 150
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

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

    # Create data loaders
    train_dataset = BikeDataset(X_train_final, y_train_final)
    val_dataset = BikeDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    num_features = X_train_seq.shape[2]
    model = build_tcn_model(num_features, NUM_CHANNELS, KERNEL_SIZE, DROPOUT_RATE)

    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, learning_rate=LEARNING_RATE
    )

    # Evaluate on training set
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        model, X_train_final, y_train_final, target_scaler, device, 'Training'
    )

    # Evaluate on test set
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        model, X_test_seq, y_test_seq, target_scaler, device, 'Testing'
    )

    # Save scalers
    os.makedirs('tcn/models', exist_ok=True)
    joblib.dump(scaler, 'tcn/models/feature_scaler_enhanced.pkl')
    joblib.dump(target_scaler, 'tcn/models/target_scaler_enhanced.pkl')
    print("\nSaved scalers:")
    print("  tcn/models/feature_scaler_enhanced.pkl")
    print("  tcn/models/target_scaler_enhanced.pkl")

    # Save final model
    torch.save(model.state_dict(), 'tcn/models/tcn_enhanced_model.pth')
    print("\nSaved final model:")
    print("  tcn/models/tcn_enhanced_model.pth")

    # Save model configuration
    model_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_channels': NUM_CHANNELS,
        'kernel_size': KERNEL_SIZE,
        'dropout_rate': DROPOUT_RATE,
        'num_features': num_features,
        'receptive_field': 1 + sum(2 * (KERNEL_SIZE - 1) * (2 ** i) for i in range(len(NUM_CHANNELS)))
    }

    # Save results
    save_results(train_metrics, test_metrics, history, feature_cols, model_config)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Training R²: {train_metrics['R2']:.4f}")
    print(f"  Testing R²: {test_metrics['R2']:.4f}")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")

    # Compare with baseline
    print("\n" + "="*80)
    print("IMPROVEMENTS OVER BASELINE TCN:")
    print("="*80)
    baseline_test_r2 = 0.5471
    improvement = ((test_metrics['R2'] - baseline_test_r2) / baseline_test_r2) * 100
    print(f"  Baseline Test R²: {baseline_test_r2:.4f}")
    print(f"  Enhanced Test R²: {test_metrics['R2']:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print("\n")


if __name__ == "__main__":
    main()
