"""
Basic LSTM Model Training for Seoul Bike Data (PyTorch Version)
Simple baseline version using only original features (no lag/rolling features)
This represents the baseline LSTM performance before feature engineering
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

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Original features (without engineered lag/rolling features)
ORIGINAL_FEATURES = [
    'Temperature(°C)',
    'Humidity(%)',
    'Visibility (10m)',
    'Solar Radiation (MJ/m2)',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_rush_hour',
    'is_weekend',
    'is_holiday',
    'is_functioning',
    'Season_Spring',
    'Season_Summer',
    'Season_Winter'
]


class BasicLSTM(nn.Module):
    """Basic LSTM model for bike demand prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(BasicLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]

        # Dense layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


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
    """Preprocess data for basic LSTM training (original features only)"""
    print("\n" + "="*80)
    print("PREPROCESSING DATA (BASIC - ORIGINAL FEATURES ONLY)")
    print("="*80)

    # Make copies
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    # Target column
    if 'target' in train_processed.columns:
        target_col = 'target'
    else:
        target_col = 'Rented Bike Count'

    # Use only original features that exist in the dataset
    feature_cols = [col for col in ORIGINAL_FEATURES if col in train_processed.columns]

    print(f"Using {len(feature_cols)} original features:")
    for col in feature_cols:
        print(f"  - {col}")

    X_train = train_processed[feature_cols].values
    y_train = train_processed[target_col].values

    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

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
    """Create sequences for LSTM input"""
    print(f"\nCreating sequences with length {sequence_length}...")

    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

    return X_seq, y_seq


def build_lstm_model(num_features, hidden_size=64, num_layers=2, dropout=0.2):
    """Build basic LSTM model architecture"""
    print("\n" + "="*80)
    print("BUILDING BASIC LSTM MODEL")
    print("="*80)

    model = BasicLSTM(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    print(f"\nModel Configuration:")
    print(f"  Input features: {num_features}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Dropout rate: {dropout}")
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

        # Gradient clipping
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


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """Train LSTM model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    # Create output directories
    os.makedirs('lstm/models', exist_ok=True)
    os.makedirs('lstm/results', exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    best_val_loss = float('inf')
    patience = 15
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
            torch.save(model.state_dict(), 'lstm/models/best_lstm_basic_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load('lstm/models/best_lstm_basic_model.pth'))

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


def save_results(train_metrics, test_metrics, history, feature_cols):
    """Save training results and metrics"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save metrics
    results = {
        'model_type': 'Basic LSTM (Original Features Only)',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Baseline LSTM model without lag/rolling engineered features'
    }

    with open('lstm/results/lstm_basic_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: lstm/results/lstm_basic_metrics.json")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })

    history_df.to_csv('lstm/results/training_history_basic.csv', index=False)
    print("Saved: lstm/results/training_history_basic.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })

    metrics_df.to_csv('lstm/results/lstm_basic_metrics_summary.csv', index=False)
    print("Saved: lstm/results/lstm_basic_metrics_summary.csv")


def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# SEOUL BIKE BASIC LSTM MODEL TRAINING (ORIGINAL FEATURES ONLY) #")
    print("#"*80 + "\n")

    # Hyperparameters (simpler than enhanced version)
    SEQUENCE_LENGTH = 24  # Use 24 hours of history
    HIDDEN_SIZE = 64  # Smaller than enhanced version
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Load data
    train_df, test_df = load_data()

    # Preprocess data (original features only)
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
    model = build_lstm_model(num_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)

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
    os.makedirs('lstm/models', exist_ok=True)
    joblib.dump(scaler, 'lstm/models/feature_scaler_basic.pkl')
    joblib.dump(target_scaler, 'lstm/models/target_scaler_basic.pkl')
    print("\nSaved scalers:")
    print("  lstm/models/feature_scaler_basic.pkl")
    print("  lstm/models/target_scaler_basic.pkl")

    # Save final model
    torch.save(model.state_dict(), 'lstm/models/lstm_basic_model.pth')
    print("\nSaved final model:")
    print("  lstm/models/lstm_basic_model.pth")

    # Save results
    save_results(train_metrics, test_metrics, history, feature_cols)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Training R²: {train_metrics['R2']:.4f} ({train_metrics['R2']*100:.2f}%)")
    print(f"  Testing R²: {test_metrics['R2']:.4f} ({test_metrics['R2']*100:.2f}%)")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")

    print("\n" + "="*80)
    print("BASIC LSTM MODEL SUMMARY")
    print("="*80)
    print(f"\nBasic LSTM Test R²: {test_metrics['R2']:.4f} ({test_metrics['R2']*100:.2f}%)")
    print(f"Basic LSTM Test RMSE: {test_metrics['RMSE']:.2f}")
    print(f"Basic LSTM Test MAE: {test_metrics['MAE']:.2f}")

    print("\n")


if __name__ == "__main__":
    main()
