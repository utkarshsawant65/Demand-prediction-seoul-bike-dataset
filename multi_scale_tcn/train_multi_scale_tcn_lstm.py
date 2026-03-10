"""
Multi-Scale TCN + LSTM Hybrid Model
Architecture: Multi-Scale TCN (feature extraction) -> LSTM (temporal modeling) -> Output

Key Innovation:
- Multi-Scale TCN captures patterns at different time scales (kernels 2, 3, 5)
- LSTM processes the TCN features to capture long-term dependencies
- Combines the best of both worlds: local pattern extraction + sequential memory
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


class MultiScaleTCNLSTM(nn.Module):
    """Multi-Scale TCN + LSTM Hybrid Model
    Architecture:
    1. Input projection (reduce dimensionality)
    2. Parallel Multi-Scale TCN branches (kernels 2, 3, 5)
    3. Concatenate TCN outputs (preserving temporal dimension)
    4. LSTM layers to process TCN features sequentially
    5. Output head for final prediction
    Key difference from pure TCN:
    - TCN extracts local patterns at multiple scales
    - LSTM captures long-range temporal dependencies from TCN features"""
    def __init__(self, num_inputs, tcn_channels=[32, 24], lstm_hidden=64, lstm_layers=2,
                 kernel_sizes=[2, 3, 5], dropout=0.3):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        # Shared input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_inputs, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SpatialDropout1d(dropout * 0.5))
        # Parallel TCN branches (each captures different scale patterns)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = []
            in_ch = 32
            for i, out_ch in enumerate(tcn_channels):
                dilation = 2 ** i
                layers.append(LightTemporalBlock(in_ch, out_ch, k, dilation, dropout))
                in_ch = out_ch
            self.branches.append(nn.Sequential(*layers))
        # TC output channels (concatenated from all branches)
        tcn_output_channels = tcn_channels[-1] * self.num_scales
        # Layer norm before LSTM (helps with training stability)
        self.pre_lstm_norm = nn.LayerNorm(tcn_output_channels)
        # LSTM to process TCN features temporally
        self.lstm = nn.LSTM(
            input_size=tcn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False  )
        # Layer norm after LSTM
        self.post_lstm_norm = nn.LayerNorm(lstm_hidden)
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1))
        # Store attention weights for analysis
        self.last_scale_contributions = None
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        # Shared projection
        x = self.input_proj(x)  # (batch, 32, seq)
        # Process through parallel TCN branches
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # (batch, tcn_channels[-1], seq)
            branch_outputs.append(out)

        # Concatenate branch outputs along channel dimension
        # (batch, tcn_channels[-1] * num_scales, seq)
        tcn_out = torch.cat(branch_outputs, dim=1)

        # Store scale contributions for analysis
        with torch.no_grad():
            contributions = [out.abs().mean().item() for out in branch_outputs]
            total = sum(contributions)
            self.last_scale_contributions = [c/total for c in contributions]

        # Transpose for LSTM: (batch, seq, channels)
        tcn_out = tcn_out.transpose(1, 2)

        # Apply layer norm before LSTM
        tcn_out = self.pre_lstm_norm(tcn_out)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(tcn_out)
        # lstm_out: (batch, seq, lstm_hidden)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # Apply layer norm after LSTM
        last_output = self.post_lstm_norm(last_output)

        # Final prediction
        output = self.output_head(last_output)

        return output

    def get_receptive_fields(self):
        """Calculate receptive field for each TCN branch"""
        receptive_fields = {}
        num_levels = len(self.branches[0])
        for k in self.kernel_sizes:
            rf = 1
            for i in range(num_levels):
                rf += (k - 1) * (2 ** i)
            receptive_fields[f'kernel_{k}'] = rf
        return receptive_fields


class BikeDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.02):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment and self.noise_std > 0:
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
    cv = (rmse / np.mean(y_true)) * 100

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"  R²:   {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  CV:   {cv:.2f}%")
    print(f"  MAPE: {mape:.2f}%")

    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'CV': float(cv),
        'MAPE': float(mape)
    }, y_true, y_pred


def save_results(train_metrics, test_metrics, history, feature_cols, model_config, scale_contributions=None):
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results = {
        'model_name': 'Multi-Scale TCN + LSTM',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_gap': train_metrics['R2'] - test_metrics['R2'],
        'feature_count': len(feature_cols),
        'model_config': model_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if scale_contributions is not None:
        results['scale_contributions'] = {
            f'kernel_{k}': float(c)
            for k, c in zip(model_config['kernel_sizes'], scale_contributions)
        }

    os.makedirs('multi_scale_tcn/results', exist_ok=True)

    with open('multi_scale_tcn/results/multi_scale_tcn_lstm_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: multi_scale_tcn/results/multi_scale_tcn_lstm_metrics.json")

    pd.DataFrame(history).to_csv('multi_scale_tcn/results/training_history_tcn_lstm.csv', index=False)
    print("Saved: multi_scale_tcn/results/training_history_tcn_lstm.csv")

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_df.to_csv('multi_scale_tcn/results/multi_scale_tcn_lstm_summary.csv', index=False)
    print("Saved: multi_scale_tcn/results/multi_scale_tcn_lstm_summary.csv")


def visualize_results(y_true, y_pred, history, scale_contributions, kernel_sizes, save_path='multi_scale_tcn/results'):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Scale TCN + LSTM Results', fontsize=14, fontweight='bold')

    # Predictions vs Actual
    n = min(200, len(y_true))
    axes[0, 0].plot(y_true[:n], label='Actual', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(y_pred[:n], label='Predicted', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('Predictions vs Actual (First 200 samples)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Bike Rentals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual Bike Rentals')
    axes[0, 1].set_ylabel('Predicted Bike Rentals')
    axes[0, 1].set_title(f'Scatter Plot (R² = {r2_score(y_true, y_pred):.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training history
    axes[1, 0].plot(history['train_loss'], label='Train Loss', linewidth=1.5)
    axes[1, 0].plot(history['val_loss'], label='Val Loss', linewidth=1.5)
    axes[1, 0].set_title('Training History')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Scale contributions
    if scale_contributions is not None:
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = axes[1, 1].bar([f'Kernel {k}' for k in kernel_sizes], scale_contributions, color=colors)
        for bar, contrib in zip(bars, scale_contributions):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{contrib:.2%}', ha='center', va='bottom', fontsize=10)
        axes[1, 1].set_title('TCN Scale Contributions')
        axes[1, 1].set_ylabel('Contribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim(0, max(scale_contributions) * 1.2)

    plt.tight_layout()
    plt.savefig(f'{save_path}/multi_scale_tcn_lstm_results.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path}/multi_scale_tcn_lstm_results.png")


def main():
    print("\n" + "#"*80)
    print("# MULTI-SCALE TCN + LSTM HYBRID MODEL #")
    print("#"*80 + "\n")

    # Hyperparameters
    SEQUENCE_LENGTH = 24
    TCN_CHANNELS = [32, 24]       # TCN channel configuration
    LSTM_HIDDEN = 64              # LSTM hidden size
    LSTM_LAYERS = 2               # Number of LSTM layers
    KERNEL_SIZES = [2, 3, 5]      # Multi-scale kernels
    DROPOUT = 0.3
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3
    MIXUP_ALPHA = 0.1
    NOISE_STD = 0.02

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  TCN channels: {TCN_CHANNELS}")
    print(f"  LSTM hidden: {LSTM_HIDDEN}")
    print(f"  LSTM layers: {LSTM_LAYERS}")
    print(f"  Kernel sizes: {KERNEL_SIZES}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Mixup alpha: {MIXUP_ALPHA}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Load and preprocess data
    train_df, test_df = load_data()
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    # Create sequences
    print(f"\nCreating sequences with length {SEQUENCE_LENGTH}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    # Train/validation split
    val_size = int(len(X_train_seq) * 0.2)
    X_train_final, y_train_final = X_train_seq[:-val_size], y_train_seq[:-val_size]
    X_val, y_val = X_train_seq[-val_size:], y_train_seq[-val_size:]

    print(f"\nData split:")
    print(f"  Train: {len(X_train_final)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test_seq)} samples")

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
    model = MultiScaleTCNLSTM(
        num_inputs=num_features,
        tcn_channels=TCN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Architecture: Multi-Scale TCN -> LSTM -> Output")
    print(f"  Input features: {num_features}")
    print(f"  TCN channels: {TCN_CHANNELS}")
    print(f"  LSTM: {LSTM_LAYERS} layers x {LSTM_HIDDEN} hidden")
    print(f"  Kernel sizes: {KERNEL_SIZES}")
    print(f"  Receptive fields: {model.get_receptive_fields()}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    os.makedirs('multi_scale_tcn/models', exist_ok=True)
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'multi_scale_tcn/models/best_multi_scale_tcn_lstm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('multi_scale_tcn/models/best_multi_scale_tcn_lstm.pth'))

    # Evaluate
    train_metrics, _, _ = evaluate_model(model, X_train_final, y_train_final, target_scaler, device, 'Training')
    test_metrics, y_test_true, y_test_pred = evaluate_model(model, X_test_seq, y_test_seq, target_scaler, device, 'Testing')

    # Get scale contributions
    scale_contributions = model.last_scale_contributions
    if scale_contributions:
        print("\nTCN Scale Contributions:")
        for k, c in zip(KERNEL_SIZES, scale_contributions):
            print(f"  Kernel {k}: {c:.2%}")

    # Save scalers and model
    joblib.dump(scaler, 'multi_scale_tcn/models/feature_scaler_tcn_lstm.pkl')
    joblib.dump(target_scaler, 'multi_scale_tcn/models/target_scaler_tcn_lstm.pkl')
    torch.save(model.state_dict(), 'multi_scale_tcn/models/multi_scale_tcn_lstm.pth')
    print("\nSaved model and scalers")

    model_config = {
        'architecture': 'Multi-Scale TCN + LSTM',
        'sequence_length': SEQUENCE_LENGTH,
        'tcn_channels': TCN_CHANNELS,
        'lstm_hidden': LSTM_HIDDEN,
        'lstm_layers': LSTM_LAYERS,
        'kernel_sizes': KERNEL_SIZES,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'mixup_alpha': MIXUP_ALPHA,
        'num_features': num_features,
        'total_parameters': total_params,
        'receptive_fields': model.get_receptive_fields()
    }

    save_results(train_metrics, test_metrics, history, feature_cols, model_config, scale_contributions)
    visualize_results(y_test_true, y_test_pred, history, scale_contributions, KERNEL_SIZES)

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS - MULTI-SCALE TCN + LSTM")
    print("="*80)

    gap = train_metrics['R2'] - test_metrics['R2']

    print(f"\n  Training R²:  {train_metrics['R2']*100:.2f}%")
    print(f"  Testing R²:   {test_metrics['R2']*100:.2f}%")
    print(f"  Gap:          {gap*100:.2f}%")
    print(f"\n  Test RMSE:    {test_metrics['RMSE']:.2f}")
    print(f"  Test MAE:     {test_metrics['MAE']:.2f}")
    print(f"  Test MAPE:    {test_metrics['MAPE']:.2f}%")

    print("\n" + "-"*80)
    print("ARCHITECTURE BENEFITS:")
    print("-"*80)
    print("  1. Multi-Scale TCN: Captures local patterns at different time scales")
    print("     - Kernel 2: Fine-grained hourly patterns")
    print("     - Kernel 3: Short-term (2-3 hour) patterns")
    print("     - Kernel 5: Medium-term (half-day) patterns")
    print("  2. LSTM: Captures long-range temporal dependencies")
    print("     - Processes TCN features sequentially")
    print("     - Maintains memory of past patterns")
    print("  3. Hybrid: Best of both worlds")
    print("     - TCN for efficient local feature extraction")
    print("     - LSTM for sequential modeling and long-term memory")

    if gap < 0.05:
        print("\n  Excellent generalization (gap < 5%)")
    elif gap < 0.08:
        print("\n  Good generalization (gap < 8%)")
    elif gap < 0.10:
        print("\n  Acceptable generalization (gap < 10%)")
    else:
        print("\n  Consider more regularization (gap >= 10%)")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
