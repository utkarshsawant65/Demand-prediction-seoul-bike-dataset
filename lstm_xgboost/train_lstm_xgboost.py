"""
Hybrid LSTM-XGBoost Model for Seoul Bike Demand Forecasting

This script implements a hybrid model that combines:
- LSTM: Extracts temporal features and patterns from sequential data
- XGBoost: Performs final prediction using LSTM features + original features

Architecture:
1. LSTM Branch: Processes sequences to extract temporal representations
2. Feature Fusion: Combines LSTM outputs with original engineered features
3. XGBoost: Gradient boosting on combined feature set for final prediction

This hybrid approach leverages:
- LSTM's strength in capturing temporal dependencies
- XGBoost's strength in handling complex feature interactions
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
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




class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for temporal patterns.

    This model processes sequential data and outputs a feature representation
    that will be used as input to XGBoost along with original features.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # Additional dense layer to create richer feature representation
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.output_size = 64

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Extracted features of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        last_hidden = self.dropout(last_hidden)

        # Extract features
        features = self.feature_layer(last_hidden)  # (batch_size, 64)

        return features


# ================================================================================
# HYBRID LSTM-XGBOOST MODEL
# ================================================================================

class HybridLSTMXGBoost:
    """
    Hybrid model combining LSTM feature extraction with XGBoost prediction.

    Training Pipeline:
    1. Train LSTM to extract temporal features
    2. Use trained LSTM to generate features for all samples
    3. Concatenate LSTM features with original features
    4. Train XGBoost on combined feature set
    """
    def __init__(
        self,
        num_features,
        lstm_hidden=128,
        lstm_layers=2,
        lstm_dropout=0.3,
        xgb_params=None
    ):
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        # LSTM feature extractor
        self.lstm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # XGBoost model
        self.xgb_model = None

        # Default XGBoost parameters
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 50,
                'eval_metric': 'rmse'
            }
        else:
            self.xgb_params = xgb_params

    def build_lstm(self):
        """Build LSTM feature extractor."""
        self.lstm_model = LSTMFeatureExtractor(
            input_size=self.num_features,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout
        ).to(self.device)

    def train_lstm(self, train_loader, val_loader, epochs=50, lr=0.001, patience=15):
        """
        Train LSTM feature extractor.

        We train LSTM with a simple regression head to learn meaningful features,
        then discard the head and use only the feature extractor.
        """
        # Add temporary regression head for training
        regression_head = nn.Linear(self.lstm_model.output_size, 1).to(self.device)
        # Combine for training
        all_params = list(self.lstm_model.parameters()) + list(regression_head.parameters())
        optimizer = optim.Adam(all_params, lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        print(f"\nLSTM Training parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Patience: {patience}")

        for epoch in range(epochs):
            # Training
            self.lstm_model.train()
            regression_head.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                features = self.lstm_model(X_batch)
                outputs = regression_head(features).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            self.lstm_model.eval()
            regression_head.eval()
            val_losses = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    features = self.lstm_model(X_batch)
                    outputs = regression_head(features).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.lstm_model.state_dict(), 'models/best_lstm_extractor.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        self.lstm_model.load_state_dict(torch.load('models/best_lstm_extractor.pth'))
        print(f"\nLSTM training completed. Best validation loss: {best_val_loss:.4f}")

        return history

    def extract_lstm_features(self, X_seq):
        """Extract features from sequences using trained LSTM."""
        self.lstm_model.eval()

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            features = self.lstm_model(X_tensor).cpu().numpy()

        return features

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost on combined features."""
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)

        print(f"\nXGBoost Training parameters:")
        print(f"  Combined features: {X_train.shape[1]}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        print(f"  Max depth: {self.xgb_params['max_depth']}")
        print(f"  Learning rate: {self.xgb_params['learning_rate']}")
        print(f"  N estimators: {self.xgb_params['n_estimators']}")

        # Extract early stopping rounds
        early_stopping_rounds = self.xgb_params.pop('early_stopping_rounds', 50)
        eval_metric = self.xgb_params.pop('eval_metric', 'rmse')

        # Add early_stopping_rounds and callbacks to params
        self.xgb_params['early_stopping_rounds'] = early_stopping_rounds

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        print(f"\nXGBoost training completed.")
        print(f"Best iteration: {self.xgb_model.best_iteration}")
        print(f"Best score: {self.xgb_model.best_score:.4f}")

        return self.xgb_model

    def predict(self, X_seq, X_original):
        """
        Make predictions using the hybrid model.

        Args:
            X_seq: Sequential data for LSTM (batch_size, seq_len, features)
            X_original: Original features corresponding to prediction timestep

        Returns:
            Predictions
        """
        # Extract LSTM features
        lstm_features = self.extract_lstm_features(X_seq)

        # Combine with original features
        X_combined = np.concatenate([lstm_features, X_original], axis=1)

        # XGBoost prediction
        predictions = self.xgb_model.predict(X_combined)

        return predictions

    def save(self, model_dir='models'):
        """Save both LSTM and XGBoost models."""
        os.makedirs(model_dir, exist_ok=True)

        # Save LSTM
        torch.save(self.lstm_model.state_dict(), f'{model_dir}/lstm_extractor.pth')

        # Save XGBoost
        self.xgb_model.save_model(f'{model_dir}/xgboost_model.json')

        # Save architecture info
        architecture = {
            'num_features': self.num_features,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'lstm_dropout': self.lstm_dropout,
            'lstm_output_size': self.lstm_model.output_size,
            'xgb_params': self.xgb_params
        }

        with open(f'{model_dir}/architecture.json', 'w') as f:
            json.dump(architecture, f, indent=4)

    def load(self, model_dir='models'):
        """Load both LSTM and XGBoost models."""
        # Load architecture
        with open(f'{model_dir}/architecture.json', 'r') as f:
            architecture = json.load(f)

        # Build and load LSTM
        self.build_lstm()
        self.lstm_model.load_state_dict(torch.load(f'{model_dir}/lstm_extractor.pth'))

        # Load XGBoost
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(f'{model_dir}/xgboost_model.json')


# ================================================================================
# DATA LOADING AND PREPROCESSING
# ================================================================================

def load_data(train_path='../data/feature_data/train.csv',
              test_path='../data/feature_data/test.csv'):
    """Load training and testing data."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess data for hybrid model training."""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    # Target column
    target_col = 'target'
    exclude_cols = [target_col, 'Rented Bike Count']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    print(f"Features used: {len(feature_cols)}")
    print(f"First 10 features: {feature_cols[:10]}")
    print(f"Last 10 features: {feature_cols[-10:]}")

    # Separate features and target
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Scale features
    print("\nScaling features...")
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"\nTrain shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Test shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            feature_scaler, target_scaler, feature_cols)


def create_sequences(X, y, sequence_length=24):
    """
    Create sequences for LSTM and corresponding original features for XGBoost.

    Returns:
        X_seq: Sequences for LSTM (n_samples, seq_len, features)
        X_point: Point features for XGBoost (n_samples, features)
        y_seq: Targets (n_samples,)
    """
    X_seq, X_point, y_seq = [], [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        X_point.append(X[i+sequence_length-1])  # Last timestep features
        y_seq.append(y[i+sequence_length])

    return np.array(X_seq), np.array(X_point), np.array(y_seq)


# ================================================================================
# EVALUATION
# ================================================================================

def evaluate_model(model, X_seq, X_point, y, target_scaler, dataset_name='Test'):
    """Evaluate the hybrid model."""
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} SET EVALUATION")
    print("="*80)

    # Predictions (scaled)
    y_pred_scaled = model.predict(X_seq, X_point)

    # Inverse transform
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    metrics = {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'CV': float(cv),
        'MAPE': float(mape)
    }

    print(f"\nMetrics:")
    print(f"  R²:    {r2:.4f} ({r2*100:.2f}%)")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  CV:    {cv:.2f}%")
    print(f"  MAPE:  {mape:.2f}%")

    return metrics, y_true, y_pred


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function."""

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("#"*80)
    print("# HYBRID LSTM-XGBOOST MODEL TRAINING #")
    print("#"*80)

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    train_df, test_df = load_data()

    # ============================================================================
    # PREPROCESS DATA
    # ============================================================================

    X_train, y_train, X_test, y_test, feature_scaler, target_scaler, feature_cols = \
        preprocess_data(train_df, test_df)

    # ============================================================================
    # CREATE SEQUENCES
    # ============================================================================

    sequence_length = 24

    print(f"\nCreating sequences with length {sequence_length}...")
    X_train_seq, X_train_point, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, X_test_point, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    print(f"Train sequence shapes: X_seq={X_train_seq.shape}, X_point={X_train_point.shape}, y={y_train_seq.shape}")
    print(f"Test sequence shapes: X_seq={X_test_seq.shape}, X_point={X_test_point.shape}, y={y_test_seq.shape}")

    # ============================================================================
    # SPLIT TRAIN/VALIDATION
    # ============================================================================

    val_split = 0.2
    val_size = int(len(X_train_seq) * val_split)

    X_train_seq_final = X_train_seq[:-val_size]
    X_train_point_final = X_train_point[:-val_size]
    y_train_final = y_train_seq[:-val_size]

    X_val_seq = X_train_seq[-val_size:]
    X_val_point = X_train_point[-val_size:]
    y_val = y_train_seq[-val_size:]

    print(f"\nFinal data split:")
    print(f"  Train: {len(X_train_seq_final)} samples")
    print(f"  Validation: {len(X_val_seq)} samples")
    print(f"  Test: {len(X_test_seq)} samples")

    # ============================================================================
    # BUILD AND TRAIN HYBRID MODEL
    # ============================================================================

    print("\n" + "="*80)
    print("BUILDING HYBRID LSTM-XGBOOST MODEL")
    print("="*80)

    num_features = X_train_seq.shape[2]

    hybrid_model = HybridLSTMXGBoost(
        num_features=num_features,
        lstm_hidden=128,
        lstm_layers=2,
        lstm_dropout=0.3
    )

    # Build LSTM
    hybrid_model.build_lstm()

    print(f"\nModel Architecture:")
    print(f"  LSTM Hidden: {hybrid_model.lstm_hidden}")
    print(f"  LSTM Layers: {hybrid_model.lstm_layers}")
    print(f"  LSTM Dropout: {hybrid_model.lstm_dropout}")
    print(f"  LSTM Output Features: {hybrid_model.lstm_model.output_size}")
    print(f"  XGBoost Max Depth: {hybrid_model.xgb_params['max_depth']}")
    print(f"  XGBoost Learning Rate: {hybrid_model.xgb_params['learning_rate']}")

    # Count LSTM parameters
    total_params = sum(p.numel() for p in hybrid_model.lstm_model.parameters())
    print(f"  LSTM Parameters: {total_params:,}")

    # ============================================================================
    # TRAIN LSTM FEATURE EXTRACTOR
    # ============================================================================

    batch_size = 32
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq_final),
        torch.FloatTensor(y_train_final)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_seq),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lstm_history = hybrid_model.train_lstm(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        patience=15
    )

    # ============================================================================
    # EXTRACT LSTM FEATURES
    # ============================================================================

    print("\n" + "="*80)
    print("EXTRACTING LSTM FEATURES")
    print("="*80)

    print("\nExtracting features for training set...")
    lstm_features_train = hybrid_model.extract_lstm_features(X_train_seq_final)
    print(f"LSTM features shape: {lstm_features_train.shape}")

    print("\nExtracting features for validation set...")
    lstm_features_val = hybrid_model.extract_lstm_features(X_val_seq)
    print(f"LSTM features shape: {lstm_features_val.shape}")

    print("\nExtracting features for test set...")
    lstm_features_test = hybrid_model.extract_lstm_features(X_test_seq)
    print(f"LSTM features shape: {lstm_features_test.shape}")

    # Combine LSTM features with original point features
    X_train_combined = np.concatenate([lstm_features_train, X_train_point_final], axis=1)
    X_val_combined = np.concatenate([lstm_features_val, X_val_point], axis=1)
    X_test_combined = np.concatenate([lstm_features_test, X_test_point], axis=1)

    print(f"\nCombined feature shapes:")
    print(f"  Train: {X_train_combined.shape}")
    print(f"  Validation: {X_val_combined.shape}")
    print(f"  Test: {X_test_combined.shape}")

    # ============================================================================
    # TRAIN XGBOOST
    # ============================================================================

    hybrid_model.train_xgboost(
        X_train_combined, y_train_final,
        X_val_combined, y_val
    )

    # ============================================================================
    # EVALUATE MODEL
    # ============================================================================

    # Training set evaluation
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        hybrid_model, X_train_seq_final, X_train_point_final, y_train_final,
        target_scaler, 'Training'
    )

    # Test set evaluation
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        hybrid_model, X_test_seq, X_test_point, y_test_seq,
        target_scaler, 'Testing'
    )

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)

    hybrid_model.save('models')
    print("\nSaved models:")
    print("  models/lstm_extractor.pth")
    print("  models/xgboost_model.json")
    print("  models/architecture.json")

    # Save scalers
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open('models/target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    print("  models/feature_scaler.pkl")
    print("  models/target_scaler.pkl")

    # Save results
    os.makedirs('results', exist_ok=True)

    results = {
        'model_type': 'Hybrid LSTM-XGBoost',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'model_architecture': {
            'lstm_hidden': hybrid_model.lstm_hidden,
            'lstm_layers': hybrid_model.lstm_layers,
            'lstm_dropout': hybrid_model.lstm_dropout,
            'lstm_output_features': hybrid_model.lstm_model.output_size,
            'lstm_params': total_params,
            'xgb_params': hybrid_model.xgb_params,
            'combined_features': X_train_combined.shape[1]
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('results/lstm_xgboost_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved: results/lstm_xgboost_metrics.json")

    # Save LSTM training history
    lstm_history_df = pd.DataFrame(lstm_history)
    lstm_history_df.to_csv('results/lstm_training_history.csv', index=False)
    print("Saved: results/lstm_training_history.csv")

    # Metrics summary
    metrics_summary = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })
    metrics_summary.to_csv('results/lstm_xgboost_metrics_summary.csv', index=False)
    print("Saved: results/lstm_xgboost_metrics_summary.csv")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    print(f"\nSummary:")
    print(f"  Training R²: {train_metrics['R2']:.4f}")
    print(f"  Testing R²: {test_metrics['R2']:.4f}")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")

    print("\n" + "#"*80 + "\n")


if __name__ == '__main__':
    main()
