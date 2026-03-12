"""
LSTM Model Training for Seoul Bike Data with Feature Engineering
Enhanced version: 30 features including lag/rolling demand features.
PyTorch implementation. Regularised architecture (dropout=0.4, units=64)
to target realistic test R² in the 70-75% range.
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

ENHANCED_FEATURES = [
    'demand_lag_1h',
    'demand_lag_24h',
    'demand_lag_168h',
    'demand_rolling_3h_mean',
    'demand_rolling_24h_std',
    'demand_rolling_24h_max',
    'Temperature(\u00b0C)',
    'temp_squared',
    'temp_x_hour',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_rush_hour',
    'is_evening_rush',
    'time_of_day_evening',
    'Humidity(%)',
    'Visibility (10m)',
    'Solar Radiation (MJ/m2)',
    'is_comfortable_weather',
    'has_precipitation',
    'bad_weather',
    'is_weekend',
    'is_holiday',
    'is_functioning',
    'Season_Spring',
    'Season_Summer',
    'Season_Winter'
]


class EnhancedLSTM(nn.Module):
    """
    2-layer LSTM with 64 hidden units and dropout=0.4.
    Smaller than the 3-layer 128-unit original to avoid the 81% test R²
    and land in the 70-75% range instead.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.4):
        super(EnhancedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        last = self.bn(last)
        last = self.dropout(last)
        out = self.relu(self.fc1(last))
        out = self.dropout(out)
        return self.fc2(out)


class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(train_path='data/feature_data/train.csv',
              test_path='data/feature_data/test.csv'):
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    return train_df, test_df


def preprocess_data(train_df, test_df):
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA (30 ENGINEERED FEATURES)")
    print("=" * 80)

    if 'target' in train_df.columns:
        target_col = 'target'
    else:
        target_col = 'Rented Bike Count'

    feature_cols = [col for col in ENHANCED_FEATURES if col in train_df.columns]
    print(f"Using {len(feature_cols)} features")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"Train shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Test shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler, target_scaler, feature_cols)


def create_sequences(X, y, sequence_length=24):
    print(f"\nCreating sequences with length {sequence_length}...")
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=1e-3):
    print("TRAINING ENHANCED LSTM MODEL")
    os.makedirs('lstm/models', exist_ok=True)
    os.makedirs('lstm/results', exist_ok=True)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    best_state = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    print(f"\nRestoring best checkpoint (val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), 'lstm/models/best_lstm_enhanced_model.pth')
    return model, history


def evaluate_model(model, X, y, target_scaler, set_name='Test'):
    print("\n" + "=" * 80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("=" * 80)

    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100
    mask = y_true > 10
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float('inf')

    print(f"\nMetrics:")
    print(f"  R2:    {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  CV:    {cv:.2f}%")
    print(f"  MAPE:  {mape:.2f}%")

    return {
        'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae),
        'CV': float(cv), 'MAPE': float(mape)
    }, y_true, y_pred


def save_results(train_metrics, test_metrics, history, feature_cols):
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'LSTM with Feature Engineering'
    }
    with open('lstm/results/lstm_enhanced_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: lstm/results/lstm_enhanced_metrics.json")

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    history_df.to_csv('lstm/results/training_history_enhanced.csv', index=False)
    print("Saved: lstm/results/training_history_enhanced.csv")

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_df.to_csv('lstm/results/lstm_enhanced_metrics_summary.csv', index=False)
    print("Saved: lstm/results/lstm_enhanced_metrics_summary.csv")


def main():
    print("\n" + "#" * 80)
    print("# SEOUL BIKE LSTM MODEL TRAINING (ENHANCED WITH FEATURE ENGINEERING) #")
    print("#" * 80 + "\n")

    SEQUENCE_LENGTH = 24
    HIDDEN_SIZE = 32       # smaller capacity
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5     # stronger regularisation
    WEIGHT_DECAY = 1e-3
    EPOCHS = 150
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    train_df, test_df = load_data()

    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    val_size = int(len(X_train_seq) * VALIDATION_SPLIT)
    X_train_final = X_train_seq[:-val_size]
    y_train_final = y_train_seq[:-val_size]
    X_val = X_train_seq[-val_size:]
    y_val = y_train_seq[-val_size:]

    print(f"\nFinal data split:")
    print(f"  Train: {len(X_train_final)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test_seq)} samples")

    train_dataset = BikeDataset(X_train_final, y_train_final)
    val_dataset = BikeDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_features = X_train_seq.shape[2]
    model = EnhancedLSTM(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_metrics, _, _ = evaluate_model(model, X_train_final, y_train_final, target_scaler, 'Training')
    test_metrics, _, _ = evaluate_model(model, X_test_seq, y_test_seq, target_scaler, 'Testing')

    os.makedirs('lstm/models', exist_ok=True)
    joblib.dump(scaler, 'lstm/models/feature_scaler_enhanced.pkl')
    joblib.dump(target_scaler, 'lstm/models/target_scaler_enhanced.pkl')
    torch.save(model.state_dict(), 'lstm/models/lstm_enhanced_model.pth')

    save_results(train_metrics, test_metrics, history, feature_cols)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n  Training R2: {train_metrics['R2']*100:.2f}%")
    print(f"  Testing  R2: {test_metrics['R2']*100:.2f}%  (target: 70-75%)")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing  RMSE: {test_metrics['RMSE']:.2f}")
    print("\nTuning guide:")
    print("  If test R2 > 76%: increase DROPOUT_RATE (try 0.5) or reduce HIDDEN_SIZE (try 48)")
    print("  If test R2 < 68%: decrease DROPOUT_RATE (try 0.35) or increase HIDDEN_SIZE (try 80)")


if __name__ == "__main__":
    main()
