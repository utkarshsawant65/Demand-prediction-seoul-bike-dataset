"""
LSTM Model for Seoul Bike Sharing Demand Prediction
Implements standard LSTM with both random and temporal splits for comparison with CUBIST
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BikeDataset(Dataset):
    """PyTorch Dataset for bike sharing data"""

    def __init__(self, X, y, seq_length=24):
        """
        Args:
            X: Features array
            y: Target array
            seq_length: Number of previous timesteps to use (lookback window)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # Get sequence of past seq_length timesteps
        X_seq = self.X[idx:idx + self.seq_length]
        # Predict next timestep
        y_target = self.y[idx + self.seq_length]
        return X_seq, y_target

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Take output from last timestep
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out.squeeze()

class SeoulBikeLSTM:
    """LSTM model trainer and evaluator"""

    def __init__(self, seq_length=24, hidden_size=128, num_layers=2,
                 dropout=0.2, learning_rate=0.001, batch_size=64):
        """
        Args:
            seq_length: Number of previous hours to use for prediction
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_data(self, data_path, split_method='random', test_size=0.25):
        """
        Prepare data for LSTM training

        Args:
            data_path: Path to processed CSV file
            split_method: 'random' or 'temporal'
            test_size: Test set proportion (for random split)
        """
        print(f"\n[OK] Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date and hour
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)

        print(f"[OK] Loaded {len(df)} samples")
        print(f"[OK] Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Define features
        target_col = 'Count'
        exclude_cols = ['Date', 'Count', 'Year', 'Month', 'Day', 'DayOfWeek']

        # Separate numeric and categorical features
        numeric_features = ['Hour', 'Temp', 'Hum', 'Wind', 'Visb', 'Dew',
                           'Solar', 'Rain', 'Snow']
        categorical_features = ['Season', 'Holiday', 'Fday', 'WeekStatus', 'DayName']

        # Prepare features
        X_numeric = df[numeric_features].values

        # Encode categorical features
        X_categorical = []
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                X_categorical.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le

        if X_categorical:
            X_categorical = np.hstack(X_categorical)
            X = np.hstack([X_numeric, X_categorical])
        else:
            X = X_numeric

        y = df[target_col].values

        # Store feature names for later
        feature_names = numeric_features + categorical_features
        print(f"[OK] Using {len(feature_names)} features: {feature_names}")

        # Split data based on method
        if split_method == 'random':
            print(f"\n[OK] Creating RANDOM split ({int((1-test_size)*100)}/{int(test_size*100)})")

            # Random split
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=42
            )

            # Sort indices to maintain some temporal order within splits
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            split_info = {
                'method': 'random',
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'test_ratio': test_size
            }

        else:  # temporal
            print(f"\n[OK] Creating TEMPORAL split (9 months train / 3 months test)")

            # Temporal split: first 9 months train, last 3 months test
            split_date = df['Date'].min() + pd.DateOffset(months=9)

            train_mask = df['Date'] < split_date
            test_mask = df['Date'] >= split_date

            print(f"[OK] Training period: {df['Date'].min()} to {split_date - pd.DateOffset(days=1)}")
            print(f"[OK] Testing period: {split_date} to {df['Date'].max()}")

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            split_info = {
                'method': 'temporal',
                'train_samples': train_mask.sum(),
                'test_samples': test_mask.sum(),
                'split_date': split_date.strftime('%Y-%m-%d')
            }

        print(f"[OK] Training samples: {len(X_train)}")
        print(f"[OK] Testing samples: {len(X_test)}")

        # Normalize features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Normalize target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Create datasets
        train_dataset = BikeDataset(X_train_scaled, y_train_scaled, self.seq_length)
        test_dataset = BikeDataset(X_test_scaled, y_test_scaled, self.seq_length)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        print(f"[OK] Created {len(train_dataset)} training sequences")
        print(f"[OK] Created {len(test_dataset)} testing sequences")

        return train_loader, test_loader, X_train_scaled, X_test_scaled, y_train, y_test, split_info

    def build_model(self, input_size):
        """Build LSTM model"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n[OK] Model architecture:")
        print(f"    - Input size: {input_size}")
        print(f"    - Hidden size: {self.hidden_size}")
        print(f"    - Number of layers: {self.num_layers}")
        print(f"    - Dropout: {self.dropout}")
        print(f"    - Total parameters: {total_params:,}")
        print(f"    - Trainable parameters: {trainable_params:,}")

        return self.model

    def train_model(self, train_loader, test_loader, epochs=50, patience=10):
        """Train the LSTM model"""

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        print(f"\n[OK] Starting training for {epochs} epochs (patience={patience})")
        print("="*80)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")

            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/lstm_best.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n[OK] Early stopping at epoch {epoch+1}")
                break

        print("="*80)
        print(f"[OK] Training completed")
        print(f"[OK] Best validation loss: {best_loss:.6f}")

        # Load best model
        self.model.load_state_dict(torch.load('models/lstm_best.pth'))

        return train_losses, val_losses

    def evaluate(self, data_loader, X_scaled, y_true, set_name='Test'):
        """Evaluate model on given dataset"""

        self.model.eval()
        predictions = []

        with torch.no_grad():
            # For proper evaluation, we need to reconstruct sequences
            dataset = BikeDataset(X_scaled, y_true, self.seq_length)

            for i in range(len(dataset)):
                X_seq, _ = dataset[i]
                X_seq = X_seq.unsqueeze(0).to(self.device)  # Add batch dimension

                y_pred = self.model(X_seq)
                predictions.append(y_pred.cpu().numpy())

        predictions = np.array(predictions).flatten()

        # Inverse transform predictions
        predictions_original = self.scaler_y.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

        # Get corresponding true values (skip first seq_length samples)
        y_true_eval = y_true[self.seq_length:]

        # Calculate metrics
        r2 = r2_score(y_true_eval, predictions_original)
        rmse = np.sqrt(mean_squared_error(y_true_eval, predictions_original))
        mae = mean_absolute_error(y_true_eval, predictions_original)
        cv = (rmse / np.mean(y_true_eval)) * 100

        print(f"\n{'='*60}")
        print(f"{set_name.upper()} SET RESULTS")
        print(f"{'='*60}")
        print(f"R²:    {r2:.4f} ({r2*100:.2f}%)")
        print(f"RMSE:  {rmse:.2f} bikes")
        print(f"MAE:   {mae:.2f} bikes")
        print(f"CV:    {cv:.2f}%")
        print(f"{'='*60}")

        metrics = {
            'Set': set_name,
            'R2': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'CV': round(cv, 2)
        }

        return metrics, predictions_original, y_true_eval

def train_lstm_models():
    """Train LSTM models with both random and temporal splits"""

    print("="*80)
    print("LSTM MODEL TRAINING FOR SEOUL BIKE DATA")
    print("="*80)

    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('reports/figures/lstm').mkdir(parents=True, exist_ok=True)

    data_path = 'data/processed/seoul_bike_processed.csv'

    # Hyperparameters
    config = {
        'seq_length': 24,      # Use past 24 hours to predict next hour
        'hidden_size': 128,    # LSTM hidden units
        'num_layers': 2,       # LSTM layers
        'dropout': 0.2,        # Dropout rate
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'patience': 15
    }

    results = {}

    # Train with both split methods
    for split_method in ['random', 'temporal']:
        print(f"\n{'#'*80}")
        print(f"# TRAINING LSTM WITH {split_method.upper()} SPLIT")
        print(f"{'#'*80}")

        # Initialize model
        lstm = SeoulBikeLSTM(**{k: v for k, v in config.items()
                               if k not in ['epochs', 'patience']})

        # Prepare data
        test_size = 0.25 if split_method == 'random' else None
        train_loader, test_loader, X_train, X_test, y_train, y_test, split_info = \
            lstm.prepare_data(data_path, split_method, test_size)

        # Build model
        input_size = X_train.shape[1]
        lstm.build_model(input_size)

        # Train model
        train_losses, val_losses = lstm.train_model(
            train_loader, test_loader,
            epochs=config['epochs'],
            patience=config['patience']
        )

        # Evaluate
        train_metrics, _, _ = lstm.evaluate(train_loader, X_train, y_train, 'Training')
        test_metrics, _, _ = lstm.evaluate(test_loader, X_test, y_test, 'Testing')

        # Save results
        results[split_method] = {
            'train': train_metrics,
            'test': test_metrics,
            'config': config,
            'split_info': split_info
        }

        # Save model
        model_path = f'models/lstm_{split_method}.pth'
        torch.save(lstm.model.state_dict(), model_path)
        print(f"\n[OK] Model saved to {model_path}")

    # Save all results
    results_file = 'results/lstm_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {results_file}")

    # Print comparison
    print(f"\n{'='*80}")
    print("LSTM MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Random Split (Paper Replication)':^40} | {'Temporal Split (Real-World)':^40}")
    print("-"*80)
    print(f"Training R²: {results['random']['train']['R2']:6.4f} ({results['random']['train']['R2']*100:5.2f}%) | "
          f"Training R²: {results['temporal']['train']['R2']:6.4f} ({results['temporal']['train']['R2']*100:5.2f}%)")
    print(f"Testing R²:  {results['random']['test']['R2']:6.4f} ({results['random']['test']['R2']*100:5.2f}%) | "
          f"Testing R²:  {results['temporal']['test']['R2']:6.4f} ({results['temporal']['test']['R2']*100:5.2f}%)")
    print(f"Testing RMSE: {results['random']['test']['RMSE']:6.2f} bikes        | "
          f"Testing RMSE: {results['temporal']['test']['RMSE']:6.2f} bikes")
    print(f"Testing MAE:  {results['random']['test']['MAE']:6.2f} bikes        | "
          f"Testing MAE:  {results['temporal']['test']['MAE']:6.2f} bikes")
    print("="*80)

    return results

if __name__ == "__main__":
    results = train_lstm_models()
