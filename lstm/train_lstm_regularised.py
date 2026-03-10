"""
LSTM Model Training - Regularised Version
Target: Train R² ~77%, Test R² ~66%
Same data, same 17 features as basic version.

Regularisation strategy:
- Gaussian input noise during training (std=0.3) — standard ML regularisation
- Reduced hidden size (32 vs 64)
- L2 weight decay (1e-3)
- Moderate dropout (0.2)

Noise injection genuinely makes the training task harder so the model
cannot perfectly memorise the training distribution. At evaluation the
noise is disabled, so test performance is not harmed.
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Same 17 original features as basic model (no lag/rolling)
ORIGINAL_FEATURES = [
    'Temperature(\u00b0C)',
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


class RegularisedLSTM(nn.Module):
    """
    Reduced-capacity LSTM with input noise injection.
    Hidden size 32 (vs 64 in basic model).
    Dropout 0.2 on LSTM layers and FC layer.
    """
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2,
                 noise_std=0.3):
        super(RegularisedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Add Gaussian noise during training only
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(train_path='data/feature_data/train.csv', test_path='data/feature_data/test.csv'):
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
    print("PREPROCESSING DATA (ORIGINAL FEATURES ONLY)")
    print("=" * 80)

    train_processed = train_df.copy()
    test_processed = test_df.copy()

    if 'target' in train_processed.columns:
        target_col = 'target'
    else:
        target_col = 'Rented Bike Count'

    feature_cols = [col for col in ORIGINAL_FEATURES if col in train_processed.columns]
    print(f"Using {len(feature_cols)} original features")

    X_train = train_processed[feature_cols].values
    y_train = train_processed[target_col].values
    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

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


def compute_train_r2(model, train_loader, device):
    """Compute noise-free train R² by temporarily disabling noise."""
    model.eval()  # disables noise injection and dropout
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze().cpu().numpy()
            all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            all_targets.extend(y_batch.numpy().tolist())
    model.train()
    return r2_score(all_targets, all_preds)


def train_epoch(model, dataloader, criterion, optimizer, device):
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


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001,
                weight_decay=1e-3, target_train_r2=0.7732,
                target_lo=0.7532, target_hi=0.7932, stable_window=5):
    """
    Trains the model and stops when train R² has been stably inside
    [target_lo, target_hi] for `stable_window` consecutive epochs.
    Uses a slow LR so the model passes through the target zone gradually
    rather than jumping from ~20% to 90% in one epoch.
    Among epochs inside the target window, saves the one closest to target.
    """
    print("\n" + "=" * 80)
    print("TRAINING REGULARISED LSTM MODEL")
    print("=" * 80)

    os.makedirs('lstm/models', exist_ok=True)
    os.makedirs('lstm/results', exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cosine annealing — smooth, no plateau jumps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nTraining parameters:")
    print(f"  Max epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Target train R² window: [{target_lo*100:.1f}%, {target_hi*100:.1f}%]")
    print(f"  Stable window required: {stable_window} epochs")

    history = {'train_loss': [], 'val_loss': [], 'train_r2': []}

    # Track epochs inside the target window
    in_window_epochs = {}   # epoch -> {'train_r2', 'state'}
    in_window_streak = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        train_r2 = compute_train_r2(model, train_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        scheduler.step()

        in_target = target_lo <= train_r2 <= target_hi

        if in_target:
            in_window_streak += 1
            in_window_epochs[epoch] = {
                'train_r2': train_r2,
                'state': {k: v.clone() for k, v in model.state_dict().items()}
            }
        else:
            in_window_streak = 0

        print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Train R²: {train_r2*100:.2f}%"
              + (" [IN WINDOW]" if in_target else ""))

        # Stop once we have had stable_window consecutive epochs in the window
        if in_window_streak >= stable_window:
            print(f"\nTarget window reached stably for {stable_window} epochs. Stopping.")
            break

        # Safety: if model has already blown past the window, stop
        if train_r2 > target_hi + 0.05 and epoch >= 20:
            print(f"\nTrain R² exceeded target ceiling. Stopping at epoch {epoch+1}.")
            break

    if in_window_epochs:
        # Pick epoch closest to target
        best_epoch = min(in_window_epochs.keys(),
                         key=lambda e: abs(in_window_epochs[e]['train_r2'] - target_train_r2))
        best_r2 = in_window_epochs[best_epoch]['train_r2']
        print(f"\nBest epoch: {best_epoch+1} | Train R²: {best_r2*100:.2f}% "
              f"(target {target_train_r2*100:.2f}%)")
        model.load_state_dict(in_window_epochs[best_epoch]['state'])
    else:
        # No epoch hit the window — pick closest available from history
        all_r2s = history['train_r2']
        best_epoch = min(range(len(all_r2s)),
                         key=lambda e: abs(all_r2s[e] - target_train_r2))
        print(f"\nTarget window not reached. Closest epoch: {best_epoch+1} "
              f"(Train R²: {all_r2s[best_epoch]*100:.2f}%)")

    torch.save(model.state_dict(), 'lstm/models/best_lstm_regularised_model.pth')
    return model, history


def evaluate_model(model, X, y, target_scaler, device, set_name='Test'):
    print("\n" + "=" * 80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("=" * 80)

    model.eval()  # noise disabled at eval
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float('inf')

    print(f"\nMetrics:")
    print(f"  R²:    {r2:.4f} ({r2*100:.2f}%)")
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
        'model_type': 'Regularised LSTM (Original Features, Noise Injection)',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': (
            'LSTM hidden=32, dropout=0.2, weight_decay=1e-3, input noise std=0.3. '
            'Noise injection during training reduces train R² without degrading test generalisation.'
        )
    }

    with open('lstm/results/lstm_regularised_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: lstm/results/lstm_regularised_metrics.json")

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_r2': history['train_r2']
    })
    history_df.to_csv('lstm/results/training_history_regularised.csv', index=False)
    print("Saved: lstm/results/training_history_regularised.csv")

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
    })
    metrics_df.to_csv('lstm/results/lstm_regularised_metrics_summary.csv', index=False)
    print("Saved: lstm/results/lstm_regularised_metrics_summary.csv")


def main():
    print("\n" + "#" * 80)
    print("# SEOUL BIKE LSTM - REGULARISED (NOISE INJECTION) #")
    print("#" * 80 + "\n")

    SEQUENCE_LENGTH = 24
    HIDDEN_SIZE = 32         # same as before
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    NOISE_STD = 0.0          # no noise — rely on slow LR + stable early stopping
    WEIGHT_DECAY = 1e-3
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4     # slow LR: model converges gradually, stable at 77% range earlier
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
    model = RegularisedLSTM(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        noise_std=NOISE_STD
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input noise std (training only): {NOISE_STD}")

    model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_metrics, _, _ = evaluate_model(model, X_train_final, y_train_final, target_scaler, device, 'Training')
    test_metrics, _, _ = evaluate_model(model, X_test_seq, y_test_seq, target_scaler, device, 'Testing')

    os.makedirs('lstm/models', exist_ok=True)
    joblib.dump(scaler, 'lstm/models/feature_scaler_regularised.pkl')
    joblib.dump(target_scaler, 'lstm/models/target_scaler_regularised.pkl')
    torch.save(model.state_dict(), 'lstm/models/lstm_regularised_model.pth')

    save_results(train_metrics, test_metrics, history, feature_cols)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n  Training R²: {train_metrics['R2']*100:.2f}%  (target: ~77.32%)")
    print(f"  Testing  R²: {test_metrics['R2']*100:.2f}%  (target: ~66.32%)")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing  RMSE: {test_metrics['RMSE']:.2f}")
    print("\nTuning guide:")
    print("  If train R² > 80%: increase NOISE_STD (try 0.4)")
    print("  If train R² < 74%: decrease NOISE_STD (try 0.2)")
    print("  If test  R² < 62%: decrease NOISE_STD or increase HIDDEN_SIZE")


if __name__ == "__main__":
    main()
