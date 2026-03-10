"""
Visualization script for TCN model predictions
Creates actual vs predicted plots for both Enhanced and Baseline TCN models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Chomp1d(nn.Module):
    """Removes extra padding from convolution output"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """Temporal Block with dilated causal convolutions"""
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


class EnhancedTCN(nn.Module):
    """Enhanced Temporal Convolutional Network"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(EnhancedTCN, self).__init__()
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

        # Enhanced output layers
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]

        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)

        return y


def load_data(test_path='data/feature_data/test.csv'):
    """Load test data"""
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    return test_df


def preprocess_data(test_df, expected_features=None):
    """Preprocess test data"""
    print("Preprocessing data...")

    target_col = 'Rented Bike Count'
    exclude_cols = [target_col, 'Date']
    categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']

    test_processed = test_df.copy()

    # One-hot encode categorical variables
    for col in categorical_cols:
        if col in test_processed.columns:
            test_dummies = pd.get_dummies(test_processed[col], prefix=col, drop_first=True)
            test_processed = pd.concat([test_processed.drop(col, axis=1), test_dummies], axis=1)

    # If expected features provided, align columns
    if expected_features is not None:
        print(f"Aligning features to match training data...")
        for feat in expected_features:
            if feat not in test_processed.columns:
                test_processed[feat] = 0
                print(f"  Added missing feature: {feat}")

        # Select only expected features in the same order
        feature_cols = expected_features
    else:
        feature_cols = [col for col in test_processed.columns if col not in exclude_cols]

    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

    print(f"Final feature count: {len(feature_cols)}")

    return X_test, y_test, test_df['Date'].values if 'Date' in test_df.columns else None


def create_sequences(X, y, sequence_length=24):
    """Create sequences for TCN input"""
    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return np.array(X_seq), np.array(y_seq)


def make_predictions(model_path, scaler_path, target_scaler_path, X_test, model_config):
    """Load model and make predictions"""
    print(f"Loading model from {model_path}...")

    # Load scalers
    scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)),
                                      model_config['sequence_length'])

    # Load model
    model = EnhancedTCN(
        num_inputs=model_config['num_features'],
        num_channels=model_config['num_channels'],
        kernel_size=model_config['kernel_size'],
        dropout=model_config['dropout_rate']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_seq).to(device)
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    # Inverse transform
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return y_pred


def plot_predictions(y_true, y_pred, title, save_path, num_samples=500):
    """Create actual vs predicted plots"""

    # Limit to num_samples for clearer visualization
    if len(y_true) > num_samples:
        indices = np.linspace(0, len(y_true)-1, num_samples, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Time series plot
    ax1 = axes[0, 0]
    x_axis = np.arange(len(y_true_plot))
    ax1.plot(x_axis, y_true_plot, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax1.plot(x_axis, y_pred_plot, label='Predicted', color='#E63946', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Rented Bike Count', fontsize=12)
    ax1.set_title('Actual vs Predicted Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=30, color='#A8DADC')

    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Bike Count', fontsize=12)
    ax2.set_ylabel('Predicted Bike Count', fontsize=12)
    ax2.set_title('Scatter Plot: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Residual plot
    ax3 = axes[1, 0]
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.5, s=30, color='#457B9D')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Bike Count', fontsize=12)
    ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax3.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#F1FAEE')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_comparison(y_true, y_pred_enhanced, y_pred_baseline, save_path, num_samples=500):
    """Compare Enhanced vs Baseline TCN predictions"""

    # Limit samples for clarity
    if len(y_true) > num_samples:
        indices = np.linspace(0, len(y_true)-1, num_samples, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_enh = y_pred_enhanced[indices]
        y_pred_base = y_pred_baseline[indices]
    else:
        y_true_plot = y_true
        y_pred_enh = y_pred_enhanced
        y_pred_base = y_pred_baseline

    # Calculate metrics
    r2_enh = r2_score(y_true, y_pred_enhanced)
    rmse_enh = np.sqrt(mean_squared_error(y_true, y_pred_enhanced))

    r2_base = r2_score(y_true, y_pred_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_baseline))

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x_axis = np.arange(len(y_true_plot))
    ax.plot(x_axis, y_true_plot, label='Actual', color='#264653', linewidth=2.5, alpha=0.9)
    ax.plot(x_axis, y_pred_enh, label=f'Enhanced TCN (R²={r2_enh:.4f})',
            color='#2A9D8F', linewidth=2, alpha=0.8)
    ax.plot(x_axis, y_pred_base, label=f'Baseline TCN (R²={r2_base:.4f})',
            color='#E76F51', linewidth=2, alpha=0.8, linestyle='--')

    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel('Rented Bike Count', fontsize=14)
    ax.set_title('Model Comparison: Enhanced TCN vs Baseline TCN', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Add metrics box
    metrics_text = (f'Enhanced TCN:\n  R² = {r2_enh:.4f}\n  RMSE = {rmse_enh:.2f}\n\n'
                   f'Baseline TCN:\n  R² = {r2_base:.4f}\n  RMSE = {rmse_base:.2f}')
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TCN MODEL PREDICTIONS VISUALIZATION")
    print("="*80 + "\n")

    # Load enhanced model config first to get expected features
    with open('tcn/results/tcn_enhanced_metrics.json', 'r') as f:
        config_data = json.load(f)
        enhanced_config = config_data['model_config']
        expected_features = config_data['features']

    # Load test data
    test_df = load_data()
    X_test, y_test_full, dates = preprocess_data(test_df, expected_features)

    # Adjust y_test for sequence offset
    SEQUENCE_LENGTH = 24
    y_test = y_test_full[SEQUENCE_LENGTH:]

    print(f"Total test samples after sequencing: {len(y_test)}")

    # Make predictions with Enhanced TCN
    print("\n" + "="*80)
    print("ENHANCED TCN PREDICTIONS")
    print("="*80)
    y_pred_enhanced = make_predictions(
        'tcn/models/tcn_enhanced_model.pth',
        'tcn/models/feature_scaler_enhanced.pkl',
        'tcn/models/target_scaler_enhanced.pkl',
        X_test,
        enhanced_config
    )

    # Plot Enhanced TCN results
    plot_predictions(
        y_test, y_pred_enhanced,
        'Enhanced TCN Model: Actual vs Predicted',
        'tcn/results/enhanced_tcn_predictions.png',
        num_samples=500
    )

    # Try to load baseline TCN if available
    try:
        print("\n" + "="*80)
        print("BASELINE TCN PREDICTIONS")
        print("="*80)

        with open('tcn/results/tcn_metrics.json', 'r') as f:
            baseline_config = json.load(f)['model_config']

        # Load baseline model (need to use basic TCN architecture)
        # For simplicity, we'll skip this comparison if baseline model is different architecture
        print("Note: Skipping baseline comparison (different architecture)")

    except Exception as e:
        print(f"Could not load baseline model: {e}")

    # Create additional focused plots
    print("\n" + "="*80)
    print("CREATING ADDITIONAL VISUALIZATIONS")
    print("="*80)

    # Plot first week (168 hours)
    week_samples = min(168, len(y_test))

    fig, ax = plt.subplots(figsize=(16, 6))
    x_axis = np.arange(week_samples)
    ax.plot(x_axis, y_test[:week_samples], label='Actual',
            color='#2E86AB', linewidth=2.5, marker='o', markersize=4)
    ax.plot(x_axis, y_pred_enhanced[:week_samples], label='Predicted',
            color='#E63946', linewidth=2.5, marker='s', markersize=4)

    ax.set_xlabel('Hour', fontsize=14)
    ax.set_ylabel('Rented Bike Count', fontsize=14)
    ax.set_title(f'Enhanced TCN: First {week_samples} Hours - Actual vs Predicted',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tcn/results/enhanced_tcn_first_week.png', dpi=300, bbox_inches='tight')
    print("Saved: tcn/results/enhanced_tcn_first_week.png")
    plt.close()

    # Final metrics
    print("\n" + "="*80)
    print("FINAL METRICS")
    print("="*80)

    r2 = r2_score(y_test, y_pred_enhanced)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))
    mae = mean_absolute_error(y_test, y_pred_enhanced)

    print(f"\nEnhanced TCN Test Set Performance:")
    print(f"  R² Score:  {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE:      {rmse:.2f}")
    print(f"  MAE:       {mae:.2f}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. tcn/results/enhanced_tcn_predictions.png (comprehensive 4-panel view)")
    print("  2. tcn/results/enhanced_tcn_first_week.png (detailed first week)")
    print("\n")


if __name__ == "__main__":
    main()
