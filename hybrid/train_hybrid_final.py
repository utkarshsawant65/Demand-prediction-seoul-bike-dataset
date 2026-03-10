"""
HYBRID TCN-LSTM Model - FINAL VERSION for 88.37% R²

Strategy to reduce overfitting gap:
    1. Stronger regularization (higher dropout, weight decay)
    2. Fewer parameters (simpler architecture)
    3. More training data by reducing validation split
    4. SWA (Stochastic Weight Averaging) for better generalization
    5. Label smoothing equivalent via regression smoothing
    6. Multi-head attention for better feature capture
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime
import math
import copy

# Seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

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
        self.bn1 = nn.BatchNorm1d(n_out)  # Add BatchNorm
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x):
        out = torch.relu(self.bn1(self.chomp1(self.conv1(x))))
        out = self.drop1(out)
        out = torch.relu(self.bn2(self.chomp2(self.conv2(out))))
        out = self.drop2(out)

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
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for LSTM output"""
    def __init__(self, hidden, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** 0.5

        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, hidden)
        B, S, H = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, H)
        return self.proj(out)


# ============================================================================
# HYBRID MODEL - SIMPLIFIED
# ============================================================================

class HybridTCNLSTMFinal(nn.Module):
    """
    Final Hybrid TCN-LSTM with:
    - Simpler TCN (fewer channels)
    - Multi-head attention
    - Strong regularization
    """
    def __init__(self, n_features,
                 tcn_ch=[64, 64, 32],  # Smaller TCN
                 lstm_h=96,
                 lstm_l=2,
                 dropout=0.4):  # Higher dropout
        super().__init__()

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features)

        # TCN
        self.tcn = TCN(n_features, tcn_ch, k=3, dropout=dropout)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=tcn_ch[-1],
            hidden_size=lstm_h,
            num_layers=lstm_l,
            batch_first=True,
            dropout=dropout if lstm_l > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(lstm_h)

        # Multi-head attention
        self.attention = MultiHeadSelfAttention(lstm_h, heads=4, dropout=dropout)

        # Simple output
        self.output = nn.Sequential(
            nn.Linear(lstm_h, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p, gain=0.5)

    def forward(self, x):
        # Input: (B, seq, features)
        x = self.input_norm(x)

        # TCN
        x = x.transpose(1, 2)  # (B, features, seq)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (B, seq, ch)

        # LSTM
        x, _ = self.lstm(x)
        x = self.lstm_norm(x)

        # Attention
        x = x + self.attention(x)  # Residual

        # Pool and output
        x = x[:, -1, :]  # Last timestep
        return self.output(x)


# ============================================================================
# DATASET
# ============================================================================

class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
# TRAINING WITH SWA
# ============================================================================

def train_with_swa(model, train_loader, val_loader, epochs=120, lr=0.001, swa_start=60, device='cpu'):
    print("\n" + "="*60)
    print("TRAINING WITH SWA")
    print("="*60)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)

    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    # Regular scheduler for pre-SWA epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-5)

    best_val = float('inf')
    best_model = None
    history = {'train': [], 'val': []}

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print(f"Epochs: {epochs}, SWA starts at: {swa_start}")

    for epoch in range(epochs):
        # Train
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

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # Update scheduler
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model.state_dict())

    # Update BatchNorm for SWA model
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # Evaluate both models and pick best
    model.load_state_dict(best_model)
    model.eval()
    swa_model.eval()

    # Test both on validation
    def get_val_loss(m):
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = m(X).squeeze()
                total += criterion(pred, y).item()
        return total / len(val_loader)

    base_val = get_val_loss(model)
    swa_val = get_val_loss(swa_model)

    print(f"\nBase model val: {base_val:.4f}, SWA model val: {swa_val:.4f}")

    if swa_val < base_val:
        print("Using SWA model (better generalization)")
        # Copy SWA weights to main model
        model.load_state_dict(swa_model.module.state_dict())
    else:
        print("Using base model")

    torch.save(model.state_dict(), 'models/best_hybrid_final.pth')
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, loader, target_scaler, device, name='Test'):
    print(f"\n{'='*60}")
    print(f"{name.upper()} EVALUATION")
    print("="*60)

    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X).squeeze()
            preds.extend(out.cpu().numpy())
            actuals.extend(y.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

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
    print("# HYBRID TCN-LSTM FINAL - TARGET: 88.37% R²")
    print("#"*60)

    # Config
    SEQ_LEN = 24
    BATCH_SIZE = 32
    EPOCHS = 120
    LR = 0.0015
    VAL_SPLIT = 0.15  # Less validation = more training data
    SWA_START = 60

    TCN_CH = [64, 64, 32]
    LSTM_H = 96
    LSTM_L = 2
    DROPOUT = 0.4

    # Data
    X_train, y_train, X_test, y_test, scaler, target_scaler, features = load_and_preprocess()

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

    # Split
    val_size = int(len(X_train_seq) * VAL_SPLIT)
    X_tr, y_tr = X_train_seq[:-val_size], y_train_seq[:-val_size]
    X_val, y_val = X_train_seq[-val_size:], y_train_seq[-val_size:]

    print(f"\nSplit: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test_seq)}")

    # Loaders
    train_loader = DataLoader(BikeDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BikeDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(BikeDataset(X_test_seq, y_test_seq), batch_size=BATCH_SIZE)

    # Model
    model = HybridTCNLSTMFinal(
        n_features=X_train_seq.shape[2],
        tcn_ch=TCN_CH,
        lstm_h=LSTM_H,
        lstm_l=LSTM_L,
        dropout=DROPOUT
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Train with SWA
    model, history = train_with_swa(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, swa_start=SWA_START, device=device
    )

    # Evaluate
    train_loader_eval = DataLoader(BikeDataset(X_tr, y_tr), batch_size=BATCH_SIZE)
    train_metrics, _, _ = evaluate(model, train_loader_eval, target_scaler, device, 'Training')
    test_metrics, y_true, y_pred = evaluate(model, test_loader, target_scaler, device, 'Testing')

    # Save
    joblib.dump(scaler, 'models/scaler_final.pkl')
    joblib.dump(target_scaler, 'models/target_scaler_final.pkl')
    torch.save(model.state_dict(), 'models/hybrid_final.pth')

    pred_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    pred_df.to_csv('results/predictions_final.csv', index=False)

    results = {
        'model': 'Hybrid TCN-LSTM Final (SWA)',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'architecture': {
            'tcn_channels': TCN_CH,
            'lstm_hidden': LSTM_H,
            'lstm_layers': LSTM_L,
            'dropout': DROPOUT,
            'swa_start': SWA_START,
            'parameters': params
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('results/hybrid_final_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv('results/training_history_final.csv', index=False)

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
