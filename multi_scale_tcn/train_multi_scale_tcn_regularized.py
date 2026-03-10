"""
Multi-Scale TCN with Enhanced Regularization for Seoul Bike Data
Addresses overfitting with stronger regularization techniques:
- Higher dropout rates
- Stronger weight decay
- Batch normalization
- Reduced model capacity
- Data augmentation (noise injection)
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
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Chomp1d(nn.Module):
    """Removes extra padding for causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlockRegularized(nn.Module):
    """
    Temporal Block with enhanced regularization:
    - Batch normalization after convolutions
    - Higher dropout
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.4):
        super(TemporalBlockRegularized, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranchRegularized(nn.Module):
    """TCN branch with batch normalization"""
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.4):
        super(TCNBranchRegularized, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlockRegularized(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)
        self.output_channels = num_channels[-1]

    def forward(self, x):
        return self.network(x)


class ScaleAttentionRegularized(nn.Module):
    """Scale attention with dropout"""
    def __init__(self, num_scales, channels_per_scale, dropout=0.3):
        super(ScaleAttentionRegularized, self).__init__()
        total_channels = num_scales * channels_per_scale

        self.attention = nn.Sequential(
            nn.Linear(total_channels, total_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_channels // 4, num_scales),
            nn.Softmax(dim=1)
        )

        self.num_scales = num_scales
        self.channels_per_scale = channels_per_scale

    def forward(self, scale_outputs):
        concat = torch.cat(scale_outputs, dim=1).squeeze(-1)
        attention_weights = self.attention(concat)

        weighted_outputs = []
        for i, scale_out in enumerate(scale_outputs):
            scale_out = scale_out.squeeze(-1)
            weight = attention_weights[:, i:i+1]
            weighted_outputs.append(scale_out * weight)

        weighted_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)
        return weighted_output, attention_weights


class MultiScaleTCNRegularized(nn.Module):
    """
    Multi-Scale TCN with enhanced regularization to prevent overfitting:
    - Reduced channel sizes
    - Higher dropout throughout
    - Batch normalization
    - Simplified output head
    """
    def __init__(self, num_inputs, num_channels=[32, 32, 16],
                 kernel_sizes=[2, 3, 5, 7], dropout=0.4, use_attention=True):
        super(MultiScaleTCNRegularized, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.use_attention = use_attention

        # Parallel TCN branches with reduced capacity
        self.branches = nn.ModuleList([
            TCNBranchRegularized(num_inputs, num_channels, k, dropout)
            for k in kernel_sizes
        ])

        self.channels_per_scale = num_channels[-1]
        total_channels = self.num_scales * self.channels_per_scale

        # Scale attention with dropout
        if use_attention:
            self.scale_attention = ScaleAttentionRegularized(
                self.num_scales, self.channels_per_scale, dropout=0.3
            )
            fc_input_channels = self.channels_per_scale
        else:
            fc_input_channels = total_channels

        # Simplified output head with strong regularization
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Slightly less dropout near output
            nn.Linear(16, 1)
        )

        self.last_attention_weights = None

    def forward(self, x):
        x = x.transpose(1, 2)

        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            out = out[:, :, -1:]
            branch_outputs.append(out)

        if self.use_attention:
            combined, attention_weights = self.scale_attention(branch_outputs)
            self.last_attention_weights = attention_weights.detach()
        else:
            combined = torch.cat([out.squeeze(-1) for out in branch_outputs], dim=1)

        output = self.fc_layers(combined)
        return output

    def get_receptive_fields(self):
        receptive_fields = {}
        for k in self.kernel_sizes:
            rf = 1
            for i in range(len(self.branches[0].network)):
                rf += 2 * (k - 1) * (2 ** i)
            receptive_fields[f'kernel_{k}'] = rf
        return receptive_fields


class BikeDatasetWithAugmentation(Dataset):
    """Dataset with optional noise augmentation for regularization"""
    def __init__(self, X, y, noise_std=0.0, training=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.noise_std = noise_std
        self.training = training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Add Gaussian noise during training for regularization
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, y


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
    """Preprocess data"""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    train_processed = train_df.copy()
    test_processed = test_df.copy()

    if 'target' in train_processed.columns:
        target_col = 'target'
    else:
        target_col = 'Rented Bike Count'

    exclude_cols = [target_col, 'Rented Bike Count', 'target']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

    X_train = train_processed[feature_cols].values
    y_train = train_processed[target_col].values
    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

    print(f"Features used: {len(feature_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"Train shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Test shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            scaler, target_scaler, feature_cols)


def create_sequences(X, y, sequence_length=24):
    """Create sequences for TCN input"""
    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return np.array(X_seq), np.array(y_seq)


def train_epoch(model, dataloader, criterion, optimizer, device, l2_lambda=0.01):
    """Train for one epoch with L2 regularization"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()

        # MSE loss
        mse_loss = criterion(y_pred, y_batch)

        # Additional L2 regularization on predictions (smoothness)
        loss = mse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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


def evaluate_model(model, X, y, target_scaler, device, set_name='Test'):
    """Evaluate model and calculate metrics"""
    print("\n" + "="*80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("="*80)

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

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float('inf')

    print(f"\nMetrics:")
    print(f"  R²:    {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  CV:    {cv:.2f}%")
    print(f"  MAPE:  {mape:.2f}%")

    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae),
            'CV': float(cv), 'MAPE': float(mape)}, y_true, y_pred


def save_results(train_metrics, test_metrics, history, feature_cols, model_config, attention_weights=None):
    """Save training results"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'model_config': model_config,
        'overfitting_gap': train_metrics['R2'] - test_metrics['R2'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if attention_weights is not None:
        results['scale_attention_weights'] = {
            f'kernel_{k}': float(w)
            for k, w in zip(model_config['kernel_sizes'], attention_weights)
        }

    with open('multi_scale_tcn/results/multi_scale_tcn_regularized_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: multi_scale_tcn/results/multi_scale_tcn_regularized_metrics.json")

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    history_df.to_csv('multi_scale_tcn/results/training_history_regularized.csv', index=False)
    print("Saved: multi_scale_tcn/results/training_history_regularized.csv")

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_df.to_csv('multi_scale_tcn/results/multi_scale_tcn_regularized_summary.csv', index=False)
    print("Saved: multi_scale_tcn/results/multi_scale_tcn_regularized_summary.csv")


def visualize_results(y_true, y_pred, history, model, kernel_sizes, save_path='multi_scale_tcn/results'):
    """Create visualizations"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Predictions vs Actual
    n_samples = min(200, len(y_true))
    axes[0, 0].plot(range(n_samples), y_true[:n_samples], label='Actual', alpha=0.7)
    axes[0, 0].plot(range(n_samples), y_pred[:n_samples], label='Predicted', alpha=0.7)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Bike Rentals')
    axes[0, 0].set_title('Predictions vs Actual (First 200 samples)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Prediction Scatter Plot')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training history
    axes[1, 0].plot(history['train_loss'], label='Train Loss')
    axes[1, 0].plot(history['val_loss'], label='Val Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training History')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Attention weights
    if model.last_attention_weights is not None:
        avg_attention = model.last_attention_weights.mean(dim=0).cpu().numpy()
        bars = axes[1, 1].bar([f'K={k}' for k in kernel_sizes], avg_attention,
                              color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
        for bar, weight in zip(bars, avg_attention):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{weight:.2f}', ha='center', va='bottom')
        axes[1, 1].set_xlabel('Kernel Size')
        axes[1, 1].set_ylabel('Attention Weight')
        axes[1, 1].set_title('Scale Attention Weights')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_path}/multi_scale_tcn_regularized_results.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path}/multi_scale_tcn_regularized_results.png")


def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# MULTI-SCALE TCN WITH ENHANCED REGULARIZATION #")
    print("#"*80 + "\n")

    # Regularized Hyperparameters
    SEQUENCE_LENGTH = 24
    NUM_CHANNELS = [32, 32, 16]  # Reduced capacity (was [64, 64, 32])
    KERNEL_SIZES = [2, 3, 5, 7]
    DROPOUT_RATE = 0.4  # Increased (was 0.25)
    USE_ATTENTION = True
    EPOCHS = 150
    BATCH_SIZE = 64  # Larger batch for more stable gradients
    LEARNING_RATE = 0.0005  # Lower learning rate
    WEIGHT_DECAY = 1e-3  # Stronger weight decay (was 1e-5)
    VALIDATION_SPLIT = 0.2
    NOISE_STD = 0.05  # Noise augmentation

    print("Regularization strategies:")
    print(f"  - Reduced model capacity: {NUM_CHANNELS}")
    print(f"  - Higher dropout: {DROPOUT_RATE}")
    print(f"  - Stronger weight decay: {WEIGHT_DECAY}")
    print(f"  - Batch normalization: Enabled")
    print(f"  - Noise augmentation: std={NOISE_STD}")
    print(f"  - Lower learning rate: {LEARNING_RATE}")
    print(f"  - Larger batch size: {BATCH_SIZE}")

    # Load and preprocess data
    train_df, test_df = load_data()
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    # Create sequences
    print(f"\nCreating sequences with length {SEQUENCE_LENGTH}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)
    print(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

    # Split training data
    val_size = int(len(X_train_seq) * VALIDATION_SPLIT)
    X_train_final = X_train_seq[:-val_size]
    y_train_final = y_train_seq[:-val_size]
    X_val = X_train_seq[-val_size:]
    y_val = y_train_seq[-val_size:]

    print(f"\nData split:")
    print(f"  Train: {len(X_train_final)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test_seq)} samples")

    # Create data loaders with noise augmentation
    train_dataset = BikeDatasetWithAugmentation(X_train_final, y_train_final,
                                                 noise_std=NOISE_STD, training=True)
    val_dataset = BikeDatasetWithAugmentation(X_val, y_val, noise_std=0, training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n" + "="*80)
    print("BUILDING REGULARIZED MULTI-SCALE TCN MODEL")
    print("="*80)

    num_features = X_train_seq.shape[2]
    model = MultiScaleTCNRegularized(
        num_inputs=num_features,
        num_channels=NUM_CHANNELS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT_RATE,
        use_attention=USE_ATTENTION
    )

    receptive_fields = model.get_receptive_fields()
    print(f"\nModel Configuration:")
    print(f"  Input features: {num_features}")
    print(f"  Channel sizes: {NUM_CHANNELS}")
    print(f"  Kernel sizes: {KERNEL_SIZES}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Receptive fields: {receptive_fields}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    os.makedirs('multi_scale_tcn/models', exist_ok=True)
    os.makedirs('multi_scale_tcn/results', exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\nTraining parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Early stopping patience: {patience}")

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate current overfitting gap
        gap = train_loss - val_loss if val_loss > train_loss else 0

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'multi_scale_tcn/models/best_multi_scale_tcn_regularized.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load('multi_scale_tcn/models/best_multi_scale_tcn_regularized.pth'))

    # Evaluate
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        model, X_train_final, y_train_final, target_scaler, device, 'Training'
    )
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        model, X_test_seq, y_test_seq, target_scaler, device, 'Testing'
    )

    # Analyze attention
    model.eval()
    with torch.no_grad():
        _ = model(torch.FloatTensor(X_test_seq[:100]).to(device))

    attention_weights = None
    if model.last_attention_weights is not None:
        attention_weights = model.last_attention_weights.mean(dim=0).cpu().numpy()
        print("\n" + "="*80)
        print("SCALE ATTENTION ANALYSIS")
        print("="*80)
        for k, w in zip(KERNEL_SIZES, attention_weights):
            print(f"  Kernel size {k}: {w:.4f} ({w*100:.2f}%)")

    # Save everything
    joblib.dump(scaler, 'multi_scale_tcn/models/feature_scaler_regularized.pkl')
    joblib.dump(target_scaler, 'multi_scale_tcn/models/target_scaler_regularized.pkl')
    torch.save(model.state_dict(), 'multi_scale_tcn/models/multi_scale_tcn_regularized.pth')

    model_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_channels': NUM_CHANNELS,
        'kernel_sizes': KERNEL_SIZES,
        'dropout_rate': DROPOUT_RATE,
        'weight_decay': WEIGHT_DECAY,
        'noise_std': NOISE_STD,
        'use_attention': USE_ATTENTION,
        'num_features': num_features,
        'receptive_fields': receptive_fields
    }

    save_results(train_metrics, test_metrics, history, feature_cols, model_config, attention_weights)
    visualize_results(y_test_true, y_test_pred, history, model, KERNEL_SIZES)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - OVERFITTING ANALYSIS")
    print("="*80)

    overfitting_gap = train_metrics['R2'] - test_metrics['R2']
    print(f"\n  Training R²:  {train_metrics['R2']:.4f} ({train_metrics['R2']*100:.2f}%)")
    print(f"  Testing R²:   {test_metrics['R2']:.4f} ({test_metrics['R2']*100:.2f}%)")
    print(f"  Gap:          {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

    if overfitting_gap < 0.10:
        print("\n  Status: Good generalization (gap < 10%)")
    elif overfitting_gap < 0.15:
        print("\n  Status: Moderate overfitting (gap 10-15%)")
    else:
        print("\n  Status: Significant overfitting (gap > 15%)")

    print(f"\n  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE:  {test_metrics['RMSE']:.2f}")


if __name__ == "__main__":
    main()
