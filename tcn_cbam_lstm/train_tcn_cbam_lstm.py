"""
NOVEL TCN-CBAM-LSTM Model Training for Seoul Bike Data
=======================================================

This model introduces CBAM (Convolutional Block Attention Module) to enhance
the standard TCN-LSTM architecture. CBAM applies both Channel Attention and
Spatial Attention to help the model focus on important features and time steps.

NOVELTY:
    - Unlike standard temporal attention (which attends across time steps),
      CBAM applies:
      1. Channel Attention: Learns which feature channels are important
      2. Spatial Attention: Learns which temporal positions are important
    - This dual attention mechanism is novel for bike demand forecasting

Architecture:
    Input -> TCN Block 1 -> CBAM -> TCN Block 2 -> CBAM -> ... -> LSTM -> Dense -> Output

References:
    - CBAM: Convolutional Block Attention Module (Woo et al., 2018)
    - TCN-CBAM for time series (2024 research showing 22% MSE reduction)
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


# ============================================================================
# CBAM MODULE (NOVEL COMPONENT) - IMPROVED VERSION
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module - Learns feature importance

    Uses both average pooling and max pooling to capture different
    statistics of the channel information.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='sigmoid')
    def forward(self, x):
        batch_size, channels, _ = x.size()
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(2)
        return x * attention
class SpatialAttention(nn.Module):
    """Spatial Attention Module - Learns temporal position importance"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='sigmoid')
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention
class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)
    Combines Channel Attention and Spatial Attention with residual connection
    for stable training."""
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel_size=7, use_residual=True):
        super(CBAM, self).__init__()
        self.use_residual = use_residual
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        identity = x
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        if self.use_residual:
            out = identity + self.gamma * out
        return out


# ============================================================================
# TCN COMPONENTS
# ============================================================================

class Chomp1d(nn.Module):
    """Removes extra padding to ensure causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlockWithCBAM(nn.Module):
    """
    Temporal Block with optional CBAM Integration
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, use_cbam=True, cbam_reduction=8):
        super(TemporalBlockWithCBAM, self).__init__()

        self.use_cbam = use_cbam

        # First convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # CBAM module (Novel component)
        if use_cbam:
            self.cbam = CBAM(n_outputs, reduction_ratio=cbam_reduction, use_residual=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        out = out + res

        # Apply CBAM attention after residual
        if self.use_cbam:
            out = self.cbam(out)

        return out


class TCNCBAMBranch(nn.Module):
    """TCN Branch with CBAM Enhancement"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3,
                 use_cbam=True, cbam_reduction=8):
        super(TCNCBAMBranch, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            # Only apply CBAM to later layers for stable training
            apply_cbam = use_cbam and (i >= num_levels // 2)

            layers.append(
                TemporalBlockWithCBAM(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding,
                    dropout=dropout, use_cbam=apply_cbam,
                    cbam_reduction=cbam_reduction
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_size = num_channels[-1]

    def forward(self, x):
        return self.network(x)


# ============================================================================
# HYBRID TCN-CBAM-LSTM MODEL
# ============================================================================

class HybridTCNCBAMLSTM(nn.Module):
    """
    Novel Hybrid TCN-CBAM-LSTM Model

    Combines:
        1. TCN with CBAM: Extracts multi-scale temporal features with attention
        2. LSTM: Captures sequential dependencies
        3. Dense layers: Final prediction
    """
    def __init__(self, num_features,
                 tcn_channels=[64, 64, 64, 32, 32],
                 lstm_hidden=128,
                 lstm_layers=2,
                 fusion_hidden=128,
                 dropout=0.2,
                 use_cbam=True,
                 cbam_reduction=8):
        super(HybridTCNCBAMLSTM, self).__init__()
        self.use_cbam = use_cbam
        # TCN branch with CBAM
        self.tcn = TCNCBAMBranch(
            num_inputs=num_features,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction)
        # LSTM processes TCN output
        self.lstm = nn.LSTM(
            input_size=self.tcn.output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0)
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        # Final prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1))

        # Initialize fusion layers
        self._init_fusion_weights()

    def _init_fusion_weights(self):
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # Step 1: TCN with CBAM
        x_tcn = x.transpose(1, 2)  # (batch, features, seq_len)
        tcn_out = self.tcn(x_tcn)  # (batch, tcn_channels[-1], seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, tcn_channels[-1])

        # Step 2: LSTM
        lstm_out, _ = self.lstm(tcn_out)
        last_output = lstm_out[:, -1, :]

        # Layer norm and dropout
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)

        # Step 3: Final prediction
        output = self.fusion(last_output)
        return output


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
# DATA PROCESSING
# ============================================================================

def load_data(train_path='../data/feature_data/train.csv',
              test_path='../data/feature_data/test.csv'):
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df


def preprocess_data(train_df, test_df):
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    target_col = 'target'
    exclude_cols = [target_col, 'Rented Bike Count']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

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
    print(f"Creating sequences with length {sequence_length}...")

    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

    return X_seq, y_seq


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Early stopping patience: {patience}")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_tcn_cbam_lstm_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('models/best_tcn_cbam_lstm_model.pth'))
    print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")

    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, target_scaler, device, set_name='Test'):
    print("\n" + "="*80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("="*80)

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform
    y_pred = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float('inf')

    print(f"\nMetrics:")
    print(f"  R2:    {r2:.4f} ({r2*100:.2f}%)")
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


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(train_metrics, test_metrics, history, feature_cols, model_info):
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results = {
        'model_type': 'TCN-CBAM-LSTM (Novel Architecture)',
        'novelty': 'CBAM attention module integrated into TCN for channel and spatial attention',
        'architecture_flow': 'Input -> [TCN Block + CBAM] x N -> LSTM -> Dense -> Output',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'model_architecture': model_info,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('results/tcn_cbam_lstm_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved: results/tcn_cbam_lstm_metrics.json")

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    history_df.to_csv('results/training_history.csv', index=False)
    print("Saved: results/training_history.csv")

    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_df.to_csv('results/tcn_cbam_lstm_metrics_summary.csv', index=False)
    print("Saved: results/tcn_cbam_lstm_metrics_summary.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "#"*80)
    print("#" + " "*20 + "TCN-CBAM-LSTM MODEL TRAINING" + " "*20 + "#")
    print("#" + " "*15 + "(NOVEL ARCHITECTURE WITH CBAM ATTENTION)" + " "*14 + "#")
    print("#"*80 + "\n")

    # Hyperparameters - tuned for stability
    SEQUENCE_LENGTH = 24
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 0.002
    VALIDATION_SPLIT = 0.2

    # Model architecture - optimized
    TCN_CHANNELS = [64, 64, 64, 32, 32]
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    FUSION_HIDDEN = 128
    DROPOUT = 0.2

    # CBAM parameters
    USE_CBAM = True
    CBAM_REDUCTION = 8

    # Load data
    train_df, test_df = load_data()

    # Preprocess
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     scaler, target_scaler, feature_cols) = preprocess_data(train_df, test_df)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    # Split
    val_size = int(len(X_train_seq) * VALIDATION_SPLIT)
    X_train_final = X_train_seq[:-val_size]
    y_train_final = y_train_seq[:-val_size]
    X_val = X_train_seq[-val_size:]
    y_val = y_train_seq[-val_size:]

    print(f"\nData split: Train={len(X_train_final)}, Val={len(X_val)}, Test={len(X_test_seq)}")

    # Dataloaders
    train_dataset = BikeDataset(X_train_final, y_train_final)
    val_dataset = BikeDataset(X_val, y_val)
    test_dataset = BikeDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n" + "="*80)
    print("BUILDING TCN-CBAM-LSTM MODEL")
    print("="*80)

    num_features = X_train_seq.shape[2]

    model = HybridTCNCBAMLSTM(
        num_features=num_features,
        tcn_channels=TCN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        fusion_hidden=FUSION_HIDDEN,
        dropout=DROPOUT,
        use_cbam=USE_CBAM,
        cbam_reduction=CBAM_REDUCTION
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    cbam_params = sum(p.numel() for n, p in model.named_parameters() if 'cbam' in n)

    print(f"\nArchitecture: TCN{TCN_CHANNELS} -> LSTM({LSTM_HIDDEN}x{LSTM_LAYERS}) -> Dense")
    print(f"CBAM: Enabled={USE_CBAM}, Reduction={CBAM_REDUCTION}")
    print(f"Parameters: {total_params:,} total, {cbam_params:,} CBAM ({100*cbam_params/total_params:.1f}%)")

    model_info = {
        'tcn_channels': TCN_CHANNELS,
        'lstm_hidden': LSTM_HIDDEN,
        'lstm_layers': LSTM_LAYERS,
        'fusion_hidden': FUSION_HIDDEN,
        'dropout': DROPOUT,
        'use_cbam': USE_CBAM,
        'cbam_reduction': CBAM_REDUCTION,
        'total_params': total_params,
        'cbam_params': cbam_params
    }

    # Train
    model, history = train_model(model, train_loader, val_loader,
                                  epochs=EPOCHS, lr=LEARNING_RATE, device=device)

    # Evaluate
    train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_metrics, _, _ = evaluate_model(model, train_loader_eval, target_scaler, device, 'Training')
    test_metrics, _, _ = evaluate_model(model, test_loader, target_scaler, device, 'Testing')

    # Save
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    joblib.dump(target_scaler, 'models/target_scaler.pkl')
    torch.save(model.state_dict(), 'models/tcn_cbam_lstm_model.pth')
    save_results(train_metrics, test_metrics, history, feature_cols, model_info)

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nTCN-CBAM-LSTM (Novel):")
    print(f"  Training R2:  {train_metrics['R2']*100:.2f}%")
    print(f"  Testing R2:   {test_metrics['R2']*100:.2f}%")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")
    print(f"  Testing MAE:  {test_metrics['MAE']:.2f}")

    print(f"\nBaseline TCN-LSTM:")
    print(f"  Testing R2:   88.37%")
    print(f"  Testing RMSE: 208.36")

    improvement = test_metrics['R2'] * 100 - 88.37
    print(f"\nImprovement: {improvement:+.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
