"""
LSTM Model Training for Seoul Bike Data with Feature Engineering
Enhanced version with additional engineered features for better accuracy
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(train_path='data/model_data/train.csv', test_path='data/model_data/test.csv'):
    """Load train and test data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df

def engineer_features(df):
    """Create additional features from existing ones"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    df = df.copy()

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Temporal features
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['DayOfMonth'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for hour (important for temporal patterns)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # Cyclical encoding for day of week
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Cyclical encoding for month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Time of day categories
    df['IsRushHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) |
                        (df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
    df['IsMorning'] = ((df['Hour'] >= 6) & (df['Hour'] <= 11)).astype(int)
    df['IsAfternoon'] = ((df['Hour'] >= 12) & (df['Hour'] <= 17)).astype(int)
    df['IsEvening'] = ((df['Hour'] >= 18) & (df['Hour'] <= 21)).astype(int)

    # Weekend indicator
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Weather interaction features
    df['Temp_Humidity'] = df['Temperature(°C)'] * df['Humidity(%)']
    df['Temp_WindSpeed'] = df['Temperature(°C)'] * df['Wind speed (m/s)']
    df['FeelsLike'] = df['Temperature(°C)'] - (df['Wind speed (m/s)'] * 0.7)  # Simple wind chill

    # Precipitation indicator
    df['HasPrecipitation'] = ((df['Rainfall(mm)'] > 0) | (df['Snowfall (cm)'] > 0)).astype(int)
    df['TotalPrecipitation'] = df['Rainfall(mm)'] + df['Snowfall (cm)']

    # Weather comfort index
    df['ComfortIndex'] = (df['Temperature(°C)'] - df['Humidity(%)'] / 10) * (1 - df['Wind speed (m/s)'] / 10)

    # Visibility categories
    df['GoodVisibility'] = (df['Visibility (10m)'] >= 1500).astype(int)
    df['PoorVisibility'] = (df['Visibility (10m)'] < 1000).astype(int)

    # Solar radiation categories
    df['IsDaytime'] = (df['Solar Radiation (MJ/m2)'] > 0).astype(int)

    # Temperature categories
    df['IsCold'] = (df['Temperature(°C)'] < 0).astype(int)
    df['IsHot'] = (df['Temperature(°C)'] > 30).astype(int)
    df['IsComfortable'] = ((df['Temperature(°C)'] >= 15) & (df['Temperature(°C)'] <= 25)).astype(int)

    # Polynomial features for important variables
    df['Temp_Squared'] = df['Temperature(°C)'] ** 2
    df['Humidity_Squared'] = df['Humidity(%)'] ** 2

    print(f"Created {df.shape[1] - 14} new features")
    print(f"Total features now: {df.shape[1]}")

    return df

def preprocess_data(train_df, test_df):
    """Preprocess data for LSTM training with feature engineering"""

    # Apply feature engineering
    train_processed = engineer_features(train_df)
    test_processed = engineer_features(test_df)

    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)

    # Target column
    target_col = 'Rented Bike Count'

    # Columns to exclude from features
    exclude_cols = [target_col, 'Date']

    # Categorical columns that need encoding
    categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']

    # One-hot encode categorical variables
    for col in categorical_cols:
        if col in train_processed.columns:
            # Get dummies for train
            train_dummies = pd.get_dummies(train_processed[col], prefix=col, drop_first=True)
            train_processed = pd.concat([train_processed.drop(col, axis=1), train_dummies], axis=1)

            # Get dummies for test
            test_dummies = pd.get_dummies(test_processed[col], prefix=col, drop_first=True)
            test_processed = pd.concat([test_processed.drop(col, axis=1), test_dummies], axis=1)

    # Align columns between train and test
    train_cols = set(train_processed.columns)
    test_cols = set(test_processed.columns)

    # Add missing columns to test set
    for col in train_cols - test_cols:
        if col not in exclude_cols:
            test_processed[col] = 0

    # Separate features and target
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

    X_train = train_processed[feature_cols].values
    y_train = train_processed[target_col].values

    X_test = test_processed[feature_cols].values
    y_test = test_processed[target_col].values

    print(f"Features used: {len(feature_cols)}")
    print(f"First 10 features: {feature_cols[:10]}")

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
    """Create sequences for LSTM input"""
    print(f"\nCreating sequences with length {sequence_length}...")

    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

    return X_seq, y_seq

def build_enhanced_lstm_model(input_shape, lstm_units=128, dropout_rate=0.3):
    """Build enhanced LSTM model architecture"""
    print("\n" + "="*80)
    print("BUILDING ENHANCED LSTM MODEL")
    print("="*80)

    model = Sequential([
        # First LSTM layer with more units
        LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Second LSTM layer
        LSTM(lstm_units // 2, activation='tanh', return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Third LSTM layer
        LSTM(lstm_units // 4, activation='tanh'),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Dense layers
        Dense(64, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\nModel Architecture:")
    model.summary()

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=64):
    """Train LSTM model with enhanced callbacks"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    # Create output directories
    os.makedirs('lstm/models', exist_ok=True)
    os.makedirs('lstm/results', exist_ok=True)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        'lstm/models/best_lstm_enhanced_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    # Reduce learning rate when plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

    return model, history

def evaluate_model(model, X, y, target_scaler, set_name='Test'):
    """Evaluate model and calculate metrics"""
    print("\n" + "="*80)
    print(f"{set_name.upper()} SET EVALUATION")
    print("="*80)

    # Make predictions
    y_pred_scaled = model.predict(X, verbose=0)

    # Inverse transform to original scale
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv = (rmse / np.mean(y_true)) * 100
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

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

def save_results(train_metrics, test_metrics, history, feature_cols):
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
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'LSTM with Feature Engineering'
    }

    with open('lstm/results/lstm_enhanced_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Saved: lstm/results/lstm_enhanced_metrics.json")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_mae': history.history['mae'],
        'val_mae': history.history['val_mae']
    })

    history_df.to_csv('lstm/results/training_history_enhanced.csv', index=False)
    print("Saved: lstm/results/training_history_enhanced.csv")

    # Save metrics summary
    metrics_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'R2': [train_metrics['R2'], test_metrics['R2']],
        'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], test_metrics['MAE']],
        'CV': [train_metrics['CV'], test_metrics['CV']],
        'MAPE': [train_metrics['MAPE'], test_metrics['MAPE']]
    })

    metrics_df.to_csv('lstm/results/lstm_enhanced_metrics_summary.csv', index=False)
    print("Saved: lstm/results/lstm_enhanced_metrics_summary.csv")

def main():
    """Main execution function"""
    print("\n" + "#"*80)
    print("# SEOUL BIKE LSTM MODEL TRAINING (ENHANCED WITH FEATURE ENGINEERING) #")
    print("#"*80 + "\n")

    # Hyperparameters
    SEQUENCE_LENGTH = 24  # Use 24 hours of history
    LSTM_UNITS = 128  # Increased from 64
    DROPOUT_RATE = 0.3  # Increased from 0.2
    EPOCHS = 150  # Increased from 100
    BATCH_SIZE = 64  # Increased from 32
    VALIDATION_SPLIT = 0.2

    # Load data
    train_df, test_df = load_data()

    # Preprocess data with feature engineering
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

    # Build model
    input_shape = (SEQUENCE_LENGTH, X_train_seq.shape[2])
    model = build_enhanced_lstm_model(input_shape, LSTM_UNITS, DROPOUT_RATE)

    # Train model
    model, history = train_model(
        model, X_train_final, y_train_final, X_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Evaluate on training set
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        model, X_train_final, y_train_final, target_scaler, 'Training'
    )

    # Evaluate on test set
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        model, X_test_seq, y_test_seq, target_scaler, 'Testing'
    )

    # Save scalers
    os.makedirs('lstm/models', exist_ok=True)
    joblib.dump(scaler, 'lstm/models/feature_scaler_enhanced.pkl')
    joblib.dump(target_scaler, 'lstm/models/target_scaler_enhanced.pkl')
    print("\nSaved scalers:")
    print("  lstm/models/feature_scaler_enhanced.pkl")
    print("  lstm/models/target_scaler_enhanced.pkl")

    # Save final model
    model.save('lstm/models/lstm_enhanced_model.keras')
    print("\nSaved final model:")
    print("  lstm/models/lstm_enhanced_model.keras")

    # Save results
    save_results(train_metrics, test_metrics, history, feature_cols)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  Training R²: {train_metrics['R2']:.4f}")
    print(f"  Testing R²: {test_metrics['R2']:.4f}")
    print(f"  Training RMSE: {train_metrics['RMSE']:.2f}")
    print(f"  Testing RMSE: {test_metrics['RMSE']:.2f}")

    # Load baseline results for comparison
    try:
        with open('lstm/results/lstm_metrics.json', 'r') as f:
            baseline = json.load(f)

        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE MODEL")
        print("="*80)
        print(f"\nBaseline Test R²: {baseline['test_metrics']['R2']:.4f}")
        print(f"Enhanced Test R²: {test_metrics['R2']:.4f}")
        print(f"Improvement: {(test_metrics['R2'] - baseline['test_metrics']['R2'])*100:.2f}%")

        print(f"\nBaseline Test RMSE: {baseline['test_metrics']['RMSE']:.2f}")
        print(f"Enhanced Test RMSE: {test_metrics['RMSE']:.2f}")
        print(f"Improvement: {((baseline['test_metrics']['RMSE'] - test_metrics['RMSE'])/baseline['test_metrics']['RMSE'])*100:.2f}%")
    except:
        print("\nNo baseline results found for comparison")

    print("\n")

if __name__ == "__main__":
    main()
