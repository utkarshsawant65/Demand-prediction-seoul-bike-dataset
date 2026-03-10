"""
Multi-Scale TCN (Temporal Convolutional Network) Model Training for Seoul Bike Data
Uses parallel TCN branches with different kernel sizes to capture patterns at multiple time scales

Architecture:
- Multiple parallel TCN branches with different kernel sizes (e.g., 2, 3, 5, 7)
- Each branch captures temporal patterns at different scales
- Features from all branches are concatenated and fused
- Attention mechanism to weight the importance of each scale
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


class TCNBranch(nn.Module):
    """
    Single TCN branch with a specific kernel size
    Captures patterns at a specific time scale
    """
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(TCNBranch, self).__init__()
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
        self.output_channels = num_channels[-1]

    def forward(self, x):
        return self.network(x)


class ScaleAttention(nn.Module):
    """
    Attention mechanism to weight the importance of different scales
    Learns which time scales are most relevant for prediction
    """
    def __init__(self, num_scales, channels_per_scale):
        super(ScaleAttention, self).__init__()
        total_channels = num_scales * channels_per_scale

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(total_channels, total_channels // 2),
            nn.ReLU(),
            nn.Linear(total_channels // 2, num_scales),
            nn.Softmax(dim=1)
        )

        self.num_scales = num_scales
        self.channels_per_scale = channels_per_scale

    def forward(self, scale_outputs):
        """
        Args:
            scale_outputs: list of tensors, each (batch, channels, 1)
        Returns:
            weighted_output: (batch, total_channels)
            attention_weights: (batch, num_scales)
        """
        # Concatenate all scale outputs
        concat = torch.cat(scale_outputs, dim=1)  # (batch, total_channels, 1)
        concat = concat.squeeze(-1)  # (batch, total_channels)

        # Compute attention weights
        attention_weights = self.attention(concat)  # (batch, num_scales)

        # Apply attention weights to each scale
        weighted_outputs = []
        for i, scale_out in enumerate(scale_outputs):
            scale_out = scale_out.squeeze(-1)  # (batch, channels)
            weight = attention_weights[:, i:i+1]  # (batch, 1)
            weighted_outputs.append(scale_out * weight)

        # Sum weighted outputs
        weighted_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)  # (batch, channels)

        return weighted_output, attention_weights


class MultiScaleTCN(nn.Module):
    """
    Multi-Scale Temporal Convolutional Network

    Uses parallel TCN branches with different kernel sizes to capture
    patterns at multiple time scales simultaneously.

    Architecture:
    - Input → Multiple parallel TCN branches (kernel sizes: 2, 3, 5, 7)
    - Each branch has the same channel configuration but different receptive fields
    - Outputs from all branches are concatenated
    - Scale attention weights the contribution of each branch
    - Final fully connected layers for prediction
    """
    def __init__(self, num_inputs, num_channels=[64, 64, 32],
                 kernel_sizes=[2, 3, 5, 7], dropout=0.2, use_attention=True):
        super(MultiScaleTCN, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.use_attention = use_attention

        # Create parallel TCN branches with different kernel sizes
        self.branches = nn.ModuleList([
            TCNBranch(num_inputs, num_channels, k, dropout)
            for k in kernel_sizes
        ])

        # Output channels from each branch
        self.channels_per_scale = num_channels[-1]
        total_channels = self.num_scales * self.channels_per_scale

        # Scale attention mechanism
        if use_attention:
            self.scale_attention = ScaleAttention(self.num_scales, self.channels_per_scale)
            fc_input_channels = self.channels_per_scale  # After attention weighting
        else:
            fc_input_channels = total_channels  # Simple concatenation

        # Fully connected layers for final prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Store attention weights for analysis
        self.last_attention_weights = None

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Process through all branches in parallel
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            # Take the last time step from each branch
            out = out[:, :, -1:]  # (batch, channels, 1)
            branch_outputs.append(out)

        # Combine branch outputs
        if self.use_attention:
            # Use attention to weight scales
            combined, attention_weights = self.scale_attention(branch_outputs)
            self.last_attention_weights = attention_weights.detach()
        else:
            # Simple concatenation
            combined = torch.cat([out.squeeze(-1) for out in branch_outputs], dim=1)

        # Final prediction
        output = self.fc_layers(combined)

        return output

    def get_receptive_fields(self):
        """Calculate receptive field for each branch"""
        receptive_fields = {}
        for k in self.kernel_sizes:
            rf = 1
            for i in range(len(self.branches[0].network)):
                rf += 2 * (k - 1) * (2 ** i)
            receptive_fields[f'kernel_{k}'] = rf
        return receptive_fields


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
    """Preprocess data for Multi-Scale TCN training"""
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

    # Columns to exclude from features
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


def build_multi_scale_tcn(num_features, num_channels=[64, 64, 32],
                          kernel_sizes=[2, 3, 5, 7], dropout=0.25, use_attention=True):
    """Build Multi-Scale TCN model architecture"""
    print("\n" + "="*80)
    print("BUILDING MULTI-SCALE TCN MODEL")
    print("="*80)

    model = MultiScaleTCN(
        num_inputs=num_features,
        num_channels=num_channels,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        use_attention=use_attention
    )

    # Calculate receptive fields for each branch
    receptive_fields = model.get_receptive_fields()

    print(f"\nModel Configuration:")
    print(f"  Input features: {num_features}")
    print(f"  Channel sizes per branch: {num_channels}")
    print(f"  Number of scales (branches): {len(kernel_sizes)}")
    print(f"  Kernel sizes: {kernel_sizes}")
    print(f"  Dropout rate: {dropout}")
    print(f"  Use scale attention: {use_attention}")
    print(f"\n  Receptive fields:")
    for name, rf in receptive_fields.items():
        print(f"    {name}: {rf} timesteps")
    print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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


def train_model(model, train_loader, val_loader, epochs=150, learning_rate=0.001,
                model_save_path='multi_scale_tcn/models'):
    """Train Multi-Scale TCN model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    # Create output directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs('multi_scale_tcn/results', exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{model_save_path}/best_multi_scale_tcn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load(f'{model_save_path}/best_multi_scale_tcn_model.pth'))

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


def analyze_scale_attention(model, X_sample, device):
    """Analyze which scales the model focuses on"""
    print("\n" + "="*80)
    print("SCALE ATTENTION ANALYSIS")
    print("="*80)

    model.eval()
    X_tensor = torch.FloatTensor(X_sample).to(device)

    with torch.no_grad():
        _ = model(X_tensor)

    if model.last_attention_weights is not None:
        avg_attention = model.last_attention_weights.mean(dim=0).cpu().numpy()
        print("\nAverage attention weights per scale:")
        for i, (kernel, weight) in enumerate(zip(model.kernel_sizes, avg_attention)):
            print(f"  Kernel size {kernel}: {weight:.4f} ({weight*100:.2f}%)")
        return avg_attention
    return None


def save_results(train_metrics, test_metrics, history, feature_cols, model_config,
                 attention_weights=None):
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

    if attention_weights is not None:
        results['scale_attention_weights'] = {
            f'kernel_{k}': float(w)
            for k, w in zip(model_config['kernel_sizes'], attention_weights)
        }

    with open('multi_scale_tcn/results/multi_scale_tcn_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: multi_scale_tcn/results/multi_scale_tcn_metrics.json")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })

    history_df.to_csv('multi_scale_tcn/results/training_history.csv', index=False)
    print("Saved: multi_scale_tcn/results/training_history.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })

    metrics_df.to_csv('multi_scale_tcn/results/multi_scale_tcn_metrics_summary.csv', index=False)
    print("Saved: multi_scale_tcn/results/multi_scale_tcn_metrics_summary.csv")


def visualize_predictions(y_true, y_pred, save_path='multi_scale_tcn/results'):
    """Create visualization of predictions vs actual values"""
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time series comparison (first 200 samples)
    ax1 = axes[0, 0]
    n_samples = min(200, len(y_true))
    ax1.plot(range(n_samples), y_true[:n_samples], label='Actual', alpha=0.7)
    ax1.plot(range(n_samples), y_pred[:n_samples], label='Predicted', alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Bike Rentals')
    ax1.set_title('Predictions vs Actual (First 200 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual Bike Rentals')
    ax2.set_ylabel('Predicted Bike Rentals')
    ax2.set_title('Prediction Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = y_pred - y_true
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('Residual (Predicted - Actual)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals over time
    ax4 = axes[1, 1]
    ax4.plot(residuals, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residuals Over Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/multi_scale_tcn_predictions.png', dpi=150)
    plt.close()

    print(f"Saved: {save_path}/multi_scale_tcn_predictions.png")


def visualize_attention_weights(model, kernel_sizes, save_path='multi_scale_tcn/results'):
    """Visualize scale attention weights"""
    import matplotlib.pyplot as plt

    if model.last_attention_weights is None:
        return

    avg_attention = model.last_attention_weights.mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar([f'Kernel {k}' for k in kernel_sizes], avg_attention,
                   color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])

    # Add value labels on bars
    for bar, weight in zip(bars, avg_attention):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=12)

    ax.set_xlabel('TCN Branch (Kernel Size)', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Multi-Scale TCN: Scale Attention Weights', fontsize=14)
    ax.set_ylim(0, max(avg_attention) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_path}/scale_attention_weights.png', dpi=150)
    plt.close()

    print(f"Saved: {save_path}/scale_attention_weights.png")


def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# SEOUL BIKE MULTI-SCALE TCN MODEL TRAINING #")
    print("#"*80 + "\n")

    # Hyperparameters
    SEQUENCE_LENGTH = 24  # Use 24 hours of history
    NUM_CHANNELS = [64, 64, 32]  # Channel configuration for each branch
    KERNEL_SIZES = [2, 3, 5, 7]  # Different scales for capturing patterns
    DROPOUT_RATE = 0.25
    USE_ATTENTION = True  # Use scale attention mechanism
    EPOCHS = 150
    BATCH_SIZE = 32
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
    model = build_multi_scale_tcn(
        num_features, NUM_CHANNELS, KERNEL_SIZES, DROPOUT_RATE, USE_ATTENTION
    )

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

    # Analyze scale attention
    attention_weights = None
    if USE_ATTENTION:
        attention_weights = analyze_scale_attention(model, X_test_seq[:100], device)

    # Save scalers
    os.makedirs('multi_scale_tcn/models', exist_ok=True)
    joblib.dump(scaler, 'multi_scale_tcn/models/feature_scaler.pkl')
    joblib.dump(target_scaler, 'multi_scale_tcn/models/target_scaler.pkl')
    print("\nSaved scalers:")
    print("  multi_scale_tcn/models/feature_scaler.pkl")
    print("  multi_scale_tcn/models/target_scaler.pkl")

    # Save final model
    torch.save(model.state_dict(), 'multi_scale_tcn/models/multi_scale_tcn_model.pth')
    print("\nSaved final model:")
    print("  multi_scale_tcn/models/multi_scale_tcn_model.pth")

    # Save model configuration
    model_config = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_channels': NUM_CHANNELS,
        'kernel_sizes': KERNEL_SIZES,
        'dropout_rate': DROPOUT_RATE,
        'use_attention': USE_ATTENTION,
        'num_features': num_features,
        'receptive_fields': model.get_receptive_fields()
    }

    # Save results
    save_results(train_metrics, test_metrics, history, feature_cols, model_config,
                 attention_weights)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_predictions(y_test_true, y_test_pred)
    if USE_ATTENTION:
        visualize_attention_weights(model, KERNEL_SIZES)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Training R²: {train_metrics['R2']:.4f}")
    print(f"  Testing R²: {test_metrics['R2']:.4f}")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")

    print("\n" + "="*80)
    print("MULTI-SCALE TCN ARCHITECTURE BENEFITS:")
    print("="*80)
    print("  1. Captures patterns at multiple time scales simultaneously")
    print("  2. Kernel size 2: Fine-grained hourly patterns")
    print("  3. Kernel size 3: Short-term dependencies (2-3 hours)")
    print("  4. Kernel size 5: Medium-term patterns (half-day)")
    print("  5. Kernel size 7: Longer-term daily patterns")
    print("  6. Attention mechanism learns optimal scale weighting")
    print("\n")


if __name__ == "__main__":
    main()
