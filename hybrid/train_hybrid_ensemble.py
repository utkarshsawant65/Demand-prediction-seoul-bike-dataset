"""
ENSEMBLE HYBRID TCN-LSTM Model - Target 88.37% R²

Strategy:
1. Train multiple models with different random seeds
2. Use mixup augmentation for better generalization
3. Ensemble predictions for better accuracy
4. Test-time augmentation (TTA)
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
import math
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================================
# TCN COMPONENTS
# ============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, dropout):
        super().__init__()
        padding = (k - 1) * dilation

        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2
        )

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_in, channels, k=3, dropout=0.3):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = n_in if i == 0 else channels[i-1]
            layers.append(TemporalBlock(in_ch, ch, k, 2**i, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# HYBRID MODEL
# ============================================================================

class HybridTCNLSTM(nn.Module):
    def __init__(self, n_features, tcn_ch=[128, 128, 64, 64, 32],
                 lstm_h=128, lstm_l=2, fusion_h=128, dropout=0.3):
        super().__init__()

        self.tcn = TCN(n_features, tcn_ch, k=3, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=tcn_ch[-1],
            hidden_size=lstm_h,
            num_layers=lstm_l,
            batch_first=True,
            dropout=dropout if lstm_l > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fusion = nn.Sequential(
            nn.Linear(lstm_h, fusion_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_h, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p)

    def forward(self, x):
        x_tcn = x.transpose(1, 2)
        tcn_out = self.tcn(x_tcn)
        tcn_out = tcn_out.transpose(1, 2)

        lstm_out, _ = self.lstm(tcn_out)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)

        return self.fusion(last_out)


# ============================================================================
# DATASET WITH MIXUP
# ============================================================================

class BikeDataset(Dataset):
    def __init__(self, X, y, augment=False, mixup_alpha=0.2):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.augment and np.random.random() < 0.5:
            # Mixup with random sample
            mix_idx = np.random.randint(0, len(self.X))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * self.X[mix_idx]
            y = lam * y + (1 - lam) * self.y[mix_idx]

        return x, y


# ============================================================================
# DATA
# ============================================================================

def load_and_preprocess():
    train_df = pd.read_csv('../data/feature_data/train.csv')
    test_df = pd.read_csv('../data/feature_data/test.csv')
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    target = 'target'
    exclude = [target, 'Rented Bike Count']
    features = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_s = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"Features: {len(features)}")
    return X_train_s, y_train_s, X_test_s, y_test_s, scaler, target_scaler, features


def create_sequences(X, y, seq_len=24):
    X_seq = [X[i:i+seq_len] for i in range(len(X) - seq_len)]
    y_seq = [y[i+seq_len] for i in range(len(X) - seq_len)]
    return np.array(X_seq), np.array(y_seq)


# ============================================================================
# TRAINING
# ============================================================================

def train_single_model(seed, X_tr, y_tr, X_val, y_val, epochs=100, lr=0.001, device='cpu'):
    print(f"\n--- Training model with seed {seed} ---")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader = DataLoader(
        BikeDataset(X_tr, y_tr, augment=True, mixup_alpha=0.2),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(BikeDataset(X_val, y_val), batch_size=32)

    model = HybridTCNLSTM(
        n_features=X_tr.shape[2],
        tcn_ch=[128, 128, 64, 64, 32],
        lstm_h=128,
        lstm_l=2,
        fusion_h=128,
        dropout=0.3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val = float('inf')
    best_model = None
    patience = 20
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    model.load_state_dict(best_model)
    print(f"  Best val loss: {best_val:.4f}")
    return model


def ensemble_predict(models, X, device):
    """Ensemble prediction with TTA (Test-Time Augmentation)"""
    all_preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            # Regular prediction
            X_tensor = torch.FloatTensor(X).to(device)
            pred = model(X_tensor).squeeze().cpu().numpy()
            all_preds.append(pred)

            # TTA: Add small noise and predict again (more runs)
            for _ in range(4):
                X_noisy = X + np.random.normal(0, 0.01, X.shape)
                X_tensor = torch.FloatTensor(X_noisy).to(device)
                pred = model(X_tensor).squeeze().cpu().numpy()
                all_preds.append(pred)

    # Average all predictions
    return np.mean(all_preds, axis=0)


def evaluate(preds, actuals, target_scaler, name='Test'):
    print(f"\n{'='*60}")
    print(f"{name.upper()} EVALUATION")
    print("="*60)

    y_pred = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"R²:   {r2:.4f} ({r2*100:.2f}%)")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"CV:   {cv:.2f}%")

    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'CV': float(cv), 'MAPE': float(mape)}, y_true, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "#"*60)
    print("# ENSEMBLE HYBRID TCN-LSTM - TARGET: 88.37% R²")
    print("#"*60)

    SEQ_LEN = 24
    EPOCHS = 100
    LR = 0.001
    VAL_SPLIT = 0.2
    N_MODELS = 7  # Number of models in ensemble (increased)
    SEEDS = [42, 123, 456, 789, 1024, 2048, 3072]

    # Data
    X_train, y_train, X_test, y_test, scaler, target_scaler, features = load_and_preprocess()

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

    val_size = int(len(X_train_seq) * VAL_SPLIT)
    X_tr, y_tr = X_train_seq[:-val_size], y_train_seq[:-val_size]
    X_val, y_val = X_train_seq[-val_size:], y_train_seq[-val_size:]

    print(f"\nSplit: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test_seq)}")
    print(f"Training {N_MODELS} models for ensemble...")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Train ensemble
    models = []
    for i, seed in enumerate(SEEDS[:N_MODELS]):
        model = train_single_model(seed, X_tr, y_tr, X_val, y_val,
                                   epochs=EPOCHS, lr=LR, device=device)
        models.append(model)
        torch.save(model.state_dict(), f'models/ensemble_model_{i}.pth')

    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print("="*60)

    # Ensemble predictions
    train_preds = ensemble_predict(models, X_tr, device)
    test_preds = ensemble_predict(models, X_test_seq, device)

    train_metrics, _, _ = evaluate(train_preds, y_tr, target_scaler, 'Training')
    test_metrics, y_true, y_pred = evaluate(test_preds, y_test_seq, target_scaler, 'Testing')

    # Save
    joblib.dump(scaler, 'models/scaler_ensemble.pkl')
    joblib.dump(target_scaler, 'models/target_scaler_ensemble.pkl')

    pred_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    pred_df.to_csv('results/predictions_ensemble.csv', index=False)

    results = {
        'model': f'Ensemble Hybrid TCN-LSTM ({N_MODELS} models + TTA)',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'ensemble_size': N_MODELS,
        'seeds': SEEDS[:N_MODELS],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('results/ensemble_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train R²: {train_metrics['R2']*100:.2f}%")
    print(f"Test R²:  {test_metrics['R2']*100:.2f}%")
    print(f"Overfitting gap: {(train_metrics['R2'] - test_metrics['R2'])*100:.2f}%")

    target = 0.8837
    if test_metrics['R2'] >= target:
        print(f"\nTARGET ACHIEVED! {test_metrics['R2']*100:.2f}% >= 88.37%")
    else:
        gap = (target - test_metrics['R2']) * 100
        print(f"\nGap: {gap:.2f}% (Current: {test_metrics['R2']*100:.2f}%, Target: 88.37%)")


if __name__ == "__main__":
    main()
