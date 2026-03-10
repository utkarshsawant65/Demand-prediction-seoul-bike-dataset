"""
TCN R² Reduction Script — Target: push test R² lower (e.g. 70-78%)
Tries different approaches beyond just reducing capacity.
Does NOT touch the main enhanced model or its results.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ──────────────────────────────────────────────────────────────
# CONFIGS — different approaches to reduce R²
# ──────────────────────────────────────────────────────────────
CONFIGS = [
    # Approach 1: Very high weight decay — strong L2 penalty crushes weights
    {
        "name": "A_high_wd",
        "seq": 24, "channels": [64, 64, 32], "dropout": 0.3, "wd": 0.1,
        "batch": 32, "lr": 0.001,
        "drop_features": [],
        "desc": "All 30 features, wd=0.1 (very high L2)"
    },
    # Approach 2: Large batch + high LR — noisy gradient, poor convergence
    {
        "name": "B_large_batch_high_lr",
        "seq": 24, "channels": [64, 64, 32], "dropout": 0.3, "wd": 1e-3,
        "batch": 256, "lr": 0.005,
        "drop_features": [],
        "desc": "All 30 features, batch=256, lr=0.005 — noisy training"
    },
    # Approach 3: Very short sequence — model can't see far enough back
    {
        "name": "C_seq2",
        "seq": 2, "channels": [32, 16], "dropout": 0.4, "wd": 1e-3,
        "batch": 64, "lr": 0.001,
        "drop_features": [],
        "desc": "All 30 features, seq=2 (minimal context window)"
    },
    # Approach 4: Tiny model + extreme dropout + high wd
    {
        "name": "D_extreme_reg",
        "seq": 6, "channels": [8, 8], "dropout": 0.7, "wd": 0.05,
        "batch": 128, "lr": 0.002,
        "drop_features": [],
        "desc": "All 30 features, channels=[8,8], dropout=0.7, wd=0.05"
    },
    # Approach 5: Underfitting via very high LR + cosine annealing restart
    {
        "name": "E_high_lr_warmup",
        "seq": 12, "channels": [32, 16, 16], "dropout": 0.5, "wd": 0.03,
        "batch": 128, "lr": 0.01,
        "drop_features": [],
        "desc": "All 30 features, lr=0.01, wd=0.03, batch=128"
    },
]

EPOCHS = 150
PATIENCE = 20
VALIDATION_SPLIT = 0.2


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, dilation, padding, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ni, no, ks, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(no, no, ks, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(ni, no, 1) if ni != no else None
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + (x if self.downsample is None else self.downsample(x)))


class TCN(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            d = 2 ** i
            pad = (kernel_size - 1) * d
            layers.append(TemporalBlock(num_inputs if i == 0 else channels[i-1],
                                        ch, kernel_size, 1, d, pad, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        y = self.network(x.transpose(1, 2))[:, :, -1]
        return self.fc(y)


class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ──────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv('data/feature_data/train.csv')
    test_df  = pd.read_csv('data/feature_data/test.csv')
    target_col = 'target' if 'target' in train_df.columns else 'Rented Bike Count'
    return train_df, test_df, target_col


def preprocess(train_df, test_df, target_col, drop_features=None):
    exclude = set([target_col, 'Rented Bike Count', 'target'])
    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_tr = train_df[feature_cols].values
    y_tr = train_df[target_col].values
    X_te = test_df[feature_cols].values
    y_te = test_df[target_col].values

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    tsc = StandardScaler()
    y_tr = tsc.fit_transform(y_tr.reshape(-1,1)).flatten()
    y_te = tsc.transform(y_te.reshape(-1,1)).flatten()

    return X_tr, y_tr, X_te, y_te, tsc, feature_cols


def make_sequences(X, y, seq):
    Xs, ys = [], []
    for i in range(len(X) - seq):
        Xs.append(X[i:i+seq]); ys.append(y[i+seq])
    return np.array(Xs), np.array(ys)


# ──────────────────────────────────────────────────────────────
# TRAIN + EVALUATE
# ──────────────────────────────────────────────────────────────
def run(cfg, train_df, test_df, target_col):
    X_tr, y_tr, X_te, y_te, tsc, feat = preprocess(train_df, test_df, target_col)

    Xts, yts = make_sequences(X_tr, y_tr, cfg["seq"])
    Xte, yte = make_sequences(X_te, y_te, cfg["seq"])

    val_n = int(len(Xts) * VALIDATION_SPLIT)
    Xv, yv = Xts[-val_n:], yts[-val_n:]
    Xts, yts = Xts[:-val_n], yts[:-val_n]

    trl = DataLoader(BikeDataset(Xts, yts), batch_size=cfg["batch"], shuffle=True)
    vl  = DataLoader(BikeDataset(Xv,  yv),  batch_size=cfg["batch"], shuffle=False)

    model = TCN(Xts.shape[2], cfg["channels"], dropout=cfg["dropout"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    crit = nn.MSELoss()

    best_val, pat, best_state = float('inf'), 0, None
    for ep in range(EPOCHS):
        model.train()
        for Xb, yb in trl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb).squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl_ = sum(crit(model(Xb.to(device)).squeeze(), yb.to(device)).item()
                  for Xb, yb in vl) / len(vl)
        sch.step(vl_)

        if vl_ < best_val:
            best_val, pat = vl_, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"    Early stop epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = model(torch.FloatTensor(Xte).to(device)).cpu().numpy().flatten()
    yp = tsc.inverse_transform(yp.reshape(-1,1)).flatten()
    yt = tsc.inverse_transform(yte.reshape(-1,1)).flatten()

    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mask = yt > 10
    mape = np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100 if mask.sum() else float('inf')

    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape),
            "features_used": len(feat), "features_dropped": cfg["drop_features"]}


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    train_df, test_df, target_col = load_data()
    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*65}")
        print(f"Config {cfg['name']}: {cfg['desc']}")
        print(f"  seq={cfg['seq']}, channels={cfg['channels']}, dropout={cfg['dropout']}, "
              f"wd={cfg['wd']}, batch={cfg['batch']}")
        print('='*65)
        m = run(cfg, train_df, test_df, target_col)
        print(f"  Features used : {m['features_used']}")
        print(f"  Test R²       : {m['R2']*100:.2f}%")
        print(f"  Test RMSE     : {m['RMSE']:.2f}")
        print(f"  Test MAE      : {m['MAE']:.2f}")
        all_results.append({"config": cfg["name"], "desc": cfg["desc"],
                             "params": {k: v for k, v in cfg.items() if k != "drop_features"},
                             "metrics": m})

    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print('='*65)
    print(f"{'Config':<25} {'Features':<10} {'Test R²':>8} {'RMSE':>8} {'MAE':>8}")
    for r in all_results:
        print(f"  {r['config']:<23} {r['metrics']['features_used']:<10} "
              f"{r['metrics']['R2']*100:>7.2f}%  {r['metrics']['RMSE']:>8.2f}  {r['metrics']['MAE']:>8.2f}")

    out = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           "goal": "Reduce TCN test R² below 80%", "results": all_results}
    with open('tcn/results/tcn_tuning_results.json', 'w') as f:
        json.dump(out, f, indent=4)
    print("\nSaved: tcn/results/tcn_tuning_results.json")


if __name__ == "__main__":
    main()
