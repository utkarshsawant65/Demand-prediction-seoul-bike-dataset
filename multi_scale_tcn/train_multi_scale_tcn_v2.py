"""
Multi-Scale TCN V2 - Optimized for Better Generalization
Key improvements:
- Even smaller model capacity
- Spatial Dropout (drops entire channels)
- Label smoothing
- Mixup data augmentation
- Cosine annealing with warm restarts
- Early fusion instead of late fusion
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

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SpatialDropout1d(nn.Module):
    """Drops entire channels instead of individual elements"""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x shape: (batch, channels, seq_len)
        mask = torch.bernoulli(torch.ones(x.size(0), x.size(1), 1, device=x.device) * (1 - self.p))
        return x * mask / (1 - self.p)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class LightTemporalBlock(nn.Module):
    """Lightweight temporal block with spatial dropout"""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Single conv layer instead of two (lighter)
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                             padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.spatial_dropout = SpatialDropout1d(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.spatial_dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiScaleTCNv2(nn.Module):
    """
    Multi-Scale TCN V2 with early fusion and aggressive regularization
    """
    def __init__(self, num_inputs, num_channels=[24, 16],
                 kernel_sizes=[2, 3, 5], dropout=0.35):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # Shared input projection (reduces parameters)
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_inputs, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SpatialDropout1d(dropout * 0.5)
        )

        # Lighter parallel branches
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = []
            in_ch = 32
            for i, out_ch in enumerate(num_channels):
                dilation = 2 ** i
                layers.append(LightTemporalBlock(in_ch, out_ch, k, dilation, dropout))
                in_ch = out_ch
            self.branches.append(nn.Sequential(*layers))

        # Simple averaging instead of attention (reduces overfitting)
        self.use_learned_weights = True
        if self.use_learned_weights:
            # Learnable but constrained scale weights
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

        # Very simple output head
        total_channels = num_channels[-1]  # After averaging
        self.output_head = nn.Sequential(
            nn.Linear(total_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

        self.last_attention_weights = None

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)

        # Shared projection
        x = self.input_proj(x)

        # Process through branches
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            out = out[:, :, -1]  # Last timestep
            branch_outputs.append(out)

        # Weighted average of branches
        if self.use_learned_weights:
            weights = torch.softmax(self.scale_weights, dim=0)
            self.last_attention_weights = weights.detach().unsqueeze(0).expand(x.size(0), -1)
            combined = sum(w * out for w, out in zip(weights, branch_outputs))
        else:
            combined = torch.stack(branch_outputs, dim=0).mean(dim=0)

        return self.output_head(combined)

    def get_receptive_fields(self):
        receptive_fields = {}
        num_levels = len(self.branches[0])
        for k in self.kernel_sizes:
            rf = 1
            for i in range(num_levels):
                rf += (k - 1) * (2 ** i)
            receptive_fields[f'kernel_{k}'] = rf
        return receptive_fields


class BikeDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.02, mixup_alpha=0.0):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.augment:
            # Small noise
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std

        return x, y


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for regression"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def load_data():
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    train_df = pd.read_csv('data/feature_data/train.csv')
    test_df = pd.read_csv('data/feature_data/test.csv')
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df


def preprocess_data(train_df, test_df):
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    target_col = 'target' if 'target' in train_df.columns else 'Rented Bike Count'
    exclude_cols = [target_col, 'Rented Bike Count', 'target']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    print(f"Features: {len(feature_cols)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler, target_scaler, feature_cols)


def create_sequences(X, y, seq_len=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def evaluate_model(model, X, y, target_scaler, device, set_name='Test'):
    print(f"\n{'='*80}\n{set_name.upper()} SET EVALUATION\n{'='*80}")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"  R²:   {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape)}, y_true, y_pred


def save_results(train_metrics, test_metrics, history, feature_cols, model_config, scale_weights=None):
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_gap': train_metrics['R2'] - test_metrics['R2'],
        'feature_count': len(feature_cols),
        'model_config': model_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if scale_weights is not None:
        results['scale_weights'] = {f'kernel_{k}': float(w)
                                    for k, w in zip(model_config['kernel_sizes'], scale_weights)}

    with open('multi_scale_tcn/results/multi_scale_tcn_v2_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(history).to_csv('multi_scale_tcn/results/training_history_v2.csv', index=False)

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_df.to_csv('multi_scale_tcn/results/multi_scale_tcn_v2_summary.csv', index=False)

    print("Results saved to multi_scale_tcn/results/")


def visualize_results(y_true, y_pred, history, scale_weights, kernel_sizes, save_path='multi_scale_tcn/results'):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Predictions
    n = min(200, len(y_true))
    axes[0, 0].plot(y_true[:n], label='Actual', alpha=0.7)
    axes[0, 0].plot(y_pred[:n], label='Predicted', alpha=0.7)
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Loss history
    axes[1, 0].plot(history['train_loss'], label='Train')
    axes[1, 0].plot(history['val_loss'], label='Val')
    axes[1, 0].set_title('Training History')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Scale weights
    if scale_weights is not None:
        axes[1, 1].bar([f'K={k}' for k in kernel_sizes], scale_weights)
        axes[1, 1].set_title('Scale Weights')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_path}/multi_scale_tcn_v2_results.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path}/multi_scale_tcn_v2_results.png")


def main():
    print("\n" + "#"*80)
    print("# MULTI-SCALE TCN V2 - OPTIMIZED FOR GENERALIZATION #")
    print("#"*80 + "\n")

    # Hyperparameters - more aggressive regularization
    SEQUENCE_LENGTH = 24
    NUM_CHANNELS = [24, 16]  # Even smaller
    KERNEL_SIZES = [2, 3, 5]  # Fewer scales
    DROPOUT = 0.35
    EPOCHS = 200
    BATCH_SIZE = 128  # Larger batch
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-3  # Strong L2
    MIXUP_ALPHA = 0.1  # Mixup augmentation
    NOISE_STD = 0.03

    print("Configuration:")
    print(f"  Channels: {NUM_CHANNELS}")
    print(f"  Kernels: {KERNEL_SIZES}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Mixup alpha: {MIXUP_ALPHA}")
    print(f"  Noise std: {NOISE_STD}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Load and preprocess
    train_df, test_df = load_data()
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    # Create sequences
    print(f"\nCreating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    # Split
    val_size = int(len(X_train_seq) * 0.2)
    X_train_final, y_train_final = X_train_seq[:-val_size], y_train_seq[:-val_size]
    X_val, y_val = X_train_seq[-val_size:], y_train_seq[-val_size:]

    print(f"Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test_seq)}")

    # Data loaders
    train_dataset = BikeDataset(X_train_final, y_train_final, augment=True, noise_std=NOISE_STD)
    val_dataset = BikeDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    num_features = X_train_seq.shape[2]
    model = MultiScaleTCNv2(
        num_inputs=num_features,
        num_channels=NUM_CHANNELS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive fields: {model.get_receptive_fields()}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    os.makedirs('multi_scale_tcn/models', exist_ok=True)
    os.makedirs('multi_scale_tcn/results', exist_ok=True)

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Apply mixup
            if MIXUP_ALPHA > 0:
                X_batch, y_batch = mixup_data(X_batch, y_batch, MIXUP_ALPHA)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'multi_scale_tcn/models/best_multi_scale_tcn_v2.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('multi_scale_tcn/models/best_multi_scale_tcn_v2.pth'))

    # Evaluate
    train_metrics, _, _ = evaluate_model(model, X_train_final, y_train_final, target_scaler, device, 'Training')
    test_metrics, y_test_true, y_test_pred = evaluate_model(model, X_test_seq, y_test_seq, target_scaler, device, 'Testing')

    # Get scale weights
    scale_weights = None
    if hasattr(model, 'scale_weights'):
        scale_weights = torch.softmax(model.scale_weights, dim=0).detach().cpu().numpy()
        print("\nScale weights:")
        for k, w in zip(KERNEL_SIZES, scale_weights):
            print(f"  Kernel {k}: {w:.4f} ({w*100:.2f}%)")

    # Save
    joblib.dump(scaler, 'multi_scale_tcn/models/feature_scaler_v2.pkl')
    joblib.dump(target_scaler, 'multi_scale_tcn/models/target_scaler_v2.pkl')
    torch.save(model.state_dict(), 'multi_scale_tcn/models/multi_scale_tcn_v2.pth')

    model_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_channels': NUM_CHANNELS,
        'kernel_sizes': KERNEL_SIZES,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'mixup_alpha': MIXUP_ALPHA,
        'num_features': num_features,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }

    save_results(train_metrics, test_metrics, history, feature_cols, model_config, scale_weights)
    visualize_results(y_test_true, y_test_pred, history, scale_weights, KERNEL_SIZES)

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    gap = train_metrics['R2'] - test_metrics['R2']
    print(f"\n  Training R²:  {train_metrics['R2']*100:.2f}%")
    print(f"  Testing R²:   {test_metrics['R2']*100:.2f}%")
    print(f"  Gap:          {gap*100:.2f}%")
    print(f"\n  Test RMSE:    {test_metrics['RMSE']:.2f}")
    print(f"  Test MAE:     {test_metrics['MAE']:.2f}")

    if gap < 0.05:
        print("\n  Excellent generalization (gap < 5%)")
    elif gap < 0.08:
        print("\n  Good generalization (gap < 8%)")
    elif gap < 0.10:
        print("\n  Acceptable generalization (gap < 10%)")
    else:
        print("\n  Consider more regularization")


if __name__ == "__main__":
    main()
