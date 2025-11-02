"""
TFT Model for Seoul Bike Sharing Demand Prediction
Works with TFT-specific preprocessed data (seoul_bike_tft.csv)
"""

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl  # Use lightning.pytorch instead of pytorch_lightning
from pathlib import Path

class TFTModel:
    def __init__(self):
        self.model = None
        self.training_data = None
        self.validation_data = None
        
    def prepare_data(self, data_path: str, max_encoder_length: int = 24, 
                     max_prediction_length: int = 6):
        """
        Prepare TFT-specific data for training
        
        This method expects data that's already been preprocessed with TFT encodings
        (i.e., seoul_bike_tft.csv created by preprocessing.py --tft)
        """
        print("\n" + "="*80)
        print("PREPARING DATA FOR TFT MODEL")
        print("="*80 + "\n")
        
        # Load TFT-specific preprocessed data
        data = pd.read_csv(data_path)
        
        # Verify required columns exist
        required_cols = ['Date', 'time_idx', 'group', 'Count']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Did you run preprocessing with --tft flag?\n"
                f"Run: python preprocessing.py --tft"
            )
        
        # Convert Date to datetime if needed
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Verify data types
        print("Data types check:")
        numeric_columns = ['Count', 'Temp', 'Hum', 'Wind', 'Visb', 'Dew', 'Solar', 'Rain', 'Snow']
        for col in numeric_columns:
            if col in data.columns:
                if data[col].dtype != 'float32':
                    print(f"  [WARN] {col} is {data[col].dtype}, converting to float32")
                    data[col] = data[col].astype('float32')
                else:
                    print(f"  [OK] {col} is float32")
        
        # Split into train and validation
        train_size = int(len(data) * 0.75)
        train_data = data[:train_size].copy()
        val_data = data.copy()  # Validation uses all data for TimeSeriesDataSet
        
        print(f"\nDataset split:")
        print(f"  Total samples: {len(data)}")
        print(f"  Training samples: {len(train_data)} ({100*len(train_data)/len(data):.1f}%)")
        print(f"  Validation samples: {len(data) - len(train_data)} ({100*(len(data)-len(train_data))/len(data):.1f}%)")
        print(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Define feature groups
        time_varying_known_reals = ['time_idx', 'Hour', 'DayOfWeek', 'Month', 'Day']
        time_varying_known_categoricals = ['Season_encoded', 'Holiday_encoded', 
                                           'Fday_encoded', 'WeekStatus_encoded']
        time_varying_unknown_reals = ['Count', 'Temp', 'Hum', 'Wind', 'Visb', 
                                      'Dew', 'Solar', 'Rain', 'Snow']
        
        # Verify categorical columns are strings (should already be strings from preprocessing)
        for col in time_varying_known_categoricals:
            if col in data.columns:
                if data[col].dtype != 'object':
                    print(f"  [WARN] {col} is not string type, converting...")
                    train_data[col] = train_data[col].astype(str)
                    val_data[col] = val_data[col].astype(str)
        
        print(f"\nFeature configuration:")
        print(f"  Time-varying known reals: {time_varying_known_reals}")
        print(f"  Time-varying known categoricals: {time_varying_known_categoricals}")
        print(f"  Time-varying unknown reals: {time_varying_unknown_reals}")
        
        # Create TimeSeriesDataSet for training
        # Strategy: Use only training data but specify categorical encoders
        # to handle all possible values
        print("\nCreating TimeSeriesDataSet...")

        # Pre-compute all unique categories from full dataset
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        categorical_encoders = {}
        for col in time_varying_known_categoricals:
            if col in data.columns:
                encoder = NaNLabelEncoder(add_nan=True)
                encoder.fit(data[col].values)
                categorical_encoders[col] = encoder
                print(f"  Encoder for {col}: {len(encoder.classes_)} categories")

        training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="Count",
            group_ids=["group"],
            min_encoder_length=max_encoder_length // 2,  # Allow shorter sequences
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,  # Minimum prediction length
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["group"],
                transformation="log1p",  # log1p is more stable than softplus
                center=True
            ),
            categorical_encoders=categorical_encoders,  # Use pre-fitted encoders
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,  # Allow gaps in time series data
        )

        # Create validation dataset using the same encoders
        validation = TimeSeriesDataSet.from_dataset(
            training, val_data, predict=True, stop_randomization=True
        )

        # Create full dataset for later use
        full_data = TimeSeriesDataSet.from_dataset(
            training, data, predict=True, stop_randomization=True
        )
        
        print("\n[OK] Data preparation complete!")
        print(f"  Encoder length: {max_encoder_length} hours")
        print(f"  Prediction length: {max_prediction_length} hours")
        print(f"  Training batches: {len(training)}")
        print(f"  Validation batches: {len(validation)}")
        
        return training, validation, full_data
    
    def create_model(self, training_data: TimeSeriesDataSet, 
                     hidden_size: int = 64,
                     attention_head_size: int = 4,
                     dropout: float = 0.1,
                     learning_rate: float = 0.001):
        """
        Create TFT model with specified configuration
        """
        print("\n" + "="*80)
        print("CREATING TFT MODEL")
        print("="*80 + "\n")
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_size // 2,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        print(f"Model configuration:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Attention heads: {attention_head_size}")
        print(f"  Dropout: {dropout}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Loss function: QuantileLoss")
        
        # Print model summary
        print(f"\nModel summary:")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def train(self, training_data: TimeSeriesDataSet, validation_data: TimeSeriesDataSet,
              max_epochs: int = 50, batch_size: int = 128, gpus: int = 0):
        """
        Train the TFT model
        """
        print("\n" + "="*80)
        print("TRAINING TFT MODEL")
        print("="*80 + "\n")
        
        # Create dataloaders
        train_dataloader = training_data.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation_data.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=0
        )
        
        print(f"Training configuration:")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {'GPU' if gpus > 0 and torch.cuda.is_available() else 'CPU'}")
        print(f"  Train batches per epoch: {len(train_dataloader)}")
        print(f"  Validation batches: {len(val_dataloader)}")
        
        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if gpus > 0 and torch.cuda.is_available() else "cpu",
            devices=gpus if gpus > 0 and torch.cuda.is_available() else "auto",
            gradient_clip_val=0.1,
            enable_checkpointing=True,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min"
                ),
                pl.callbacks.ModelCheckpoint(
                    monitor="val_loss",
                    dirpath="models/checkpoints",
                    filename="tft-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=3,
                    mode="min"
                )
            ]
        )
        
        # Train model
        print("\nStarting training...")
        print("-" * 80)

        # Verify model is a LightningModule
        if self.model is None:
            raise ValueError("Model not created! Call create_model() first.")

        print(f"Model type: {type(self.model)}")
        print(f"Is LightningModule: {isinstance(self.model, pl.LightningModule)}")

        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        print("\n" + "-" * 80)
        print("[OK] Training complete!")

        return trainer

    def evaluate(self, training_data: TimeSeriesDataSet, validation_data: TimeSeriesDataSet):
        """
        Evaluate the trained TFT model and calculate metrics matching CUBIST
        Returns metrics in the same format as CUBIST for fair comparison
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        print("\n" + "="*80)
        print("EVALUATING TFT MODEL")
        print("="*80 + "\n")

        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        print("Loading and preparing data for evaluation...")

        # Load full dataset
        data_df = pd.read_csv('data/processed/seoul_bike_tft.csv')
        train_size = int(len(data_df) * 0.75)

        print(f"  - Total samples: {len(data_df)}")
        print(f"  - Train split: {train_size}")
        print(f"  - Test split: {len(data_df) - train_size}")

        # Make predictions using the dataloaders
        print("\nGenerating predictions...")
        print("  - Training set...")

        train_dataloader = training_data.to_dataloader(
            train=False, batch_size=64, num_workers=0, shuffle=False
        )
        train_predictions_list = []
        train_actuals_list = []

        for batch in train_dataloader:
            # Get predictions for this batch
            # Batch is a tuple: (x_dict, y_tuple)
            x, y = batch
            with torch.no_grad():
                pred = self.model(x)
                # TFT returns an Output object with 'prediction' attribute
                if hasattr(pred, 'prediction'):
                    pred_tensor = pred.prediction
                elif isinstance(pred, dict):
                    pred_tensor = pred['prediction']
                else:
                    pred_tensor = pred

                # Extract median quantile (index 3 out of 7 quantiles)
                if len(pred_tensor.shape) == 3:
                    pred_values = pred_tensor[:, 0, 3]  # [batch, horizon, quantiles]
                else:
                    pred_values = pred_tensor[:, 3]  # [batch, quantiles]

                train_predictions_list.append(pred_values.cpu().numpy())

                # Get actual values from y (which is a tuple of (target, weight))
                actuals = y[0][:, 0].cpu().numpy()  # First element of y, first timestep
                train_actuals_list.append(actuals)

        train_pred = np.concatenate(train_predictions_list)
        train_actual = np.concatenate(train_actuals_list)

        print(f"  - Training predictions: {len(train_pred)} samples")

        # Validation/test set predictions
        # Create a proper test dataset from the full data (not validation_data which is for prediction)
        print("  - Validation/test set...")

        # Create test dataset from scratch using test portion of data
        test_start = train_size
        test_data_df = data_df[data_df['time_idx'] >= test_start].copy()

        # Create a TimeSeriesDataSet for the test period
        test_dataset = TimeSeriesDataSet.from_dataset(
            training_data,
            data_df[data_df['time_idx'] >= test_start - 24],  # Include encoder_length before test period
            predict=False,
            stop_randomization=True
        )

        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=64, num_workers=0, shuffle=False
        )

        val_predictions_list = []
        val_actuals_list = []

        for batch in test_dataloader:
            x, y = batch
            with torch.no_grad():
                pred = self.model(x)
                # TFT returns an Output object with 'prediction' attribute
                if hasattr(pred, 'prediction'):
                    pred_tensor = pred.prediction
                elif isinstance(pred, dict):
                    pred_tensor = pred['prediction']
                else:
                    pred_tensor = pred

                # Extract median quantile
                if len(pred_tensor.shape) == 3:
                    pred_values = pred_tensor[:, 0, 3]
                else:
                    pred_values = pred_tensor[:, 3]

                val_predictions_list.append(pred_values.cpu().numpy())

                # Get actual values from y
                actuals = y[0][:, 0].cpu().numpy()
                val_actuals_list.append(actuals)

        val_pred = np.concatenate(val_predictions_list) if val_predictions_list else np.array([])
        test_actual = np.concatenate(val_actuals_list) if val_actuals_list else np.array([])

        print(f"  - Validation predictions: {len(val_pred)} samples")

        # Calculate metrics for training set
        train_r2 = r2_score(train_actual, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_cv = (train_rmse / np.mean(train_actual)) * 100

        # Calculate metrics for validation set
        val_r2 = r2_score(test_actual, val_pred)
        val_rmse = np.sqrt(mean_squared_error(test_actual, val_pred))
        val_mae = mean_absolute_error(test_actual, val_pred)
        val_cv = (val_rmse / np.mean(test_actual)) * 100

        # Create results dictionary
        train_metrics = {
            'Model': 'TFT_train',
            'R2': round(train_r2, 4),
            'RMSE': round(train_rmse, 2),
            'MAE': round(train_mae, 2),
            'CV': round(train_cv, 2)
        }

        test_metrics = {
            'Model': 'TFT_test',
            'R2': round(val_r2, 4),
            'RMSE': round(val_rmse, 2),
            'MAE': round(val_mae, 2),
            'CV': round(val_cv, 2)
        }

        # Print results
        print("\n" + "="*80)
        print("TFT MODEL PERFORMANCE")
        print("="*80)

        print("\nTRAINING SET RESULTS:")
        print("-"*60)
        for key, value in train_metrics.items():
            print(f"  {key:15s}: {value}")

        print("\nVALIDATION/TEST SET RESULTS:")
        print("-"*60)
        for key, value in test_metrics.items():
            print(f"  {key:15s}: {value}")

        print("\n" + "="*80)
        print("COMPARISON WITH CUBIST:")
        print("Use these metrics to compare with CUBIST model results")
        print("="*80)

        # Save results
        results = {
            'train': train_metrics,
            'test': test_metrics,
            'train_predictions': {'actual': train_actual, 'predicted': train_pred},
            'test_predictions': {'actual': test_actual, 'predicted': val_pred}
        }

        # Save to file
        import json
        results_file = Path('results/tft_results.json')
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'train': train_metrics,
                'test': test_metrics
            }
            json.dump(json_results, f, indent=2)

        print(f"\n[OK] Results saved to {results_file}")

        return train_metrics, test_metrics


def main():
    """
    Main function to run TFT model training
    """
    print("="*80)
    print("TEMPORAL FUSION TRANSFORMER (TFT) MODEL")
    print("Seoul Bike Sharing Demand Prediction")
    print("="*80)
    
    # Check if TFT-specific data exists
    data_path = "data/processed/seoul_bike_tft.csv"
    
    if not Path(data_path).exists():
        print(f"\n[ERROR] TFT-specific data not found at {data_path}")
        print("\nYou need to run preprocessing with --tft flag first:")
        print("  python preprocessing.py --tft")
        print("\nThis will create:")
        print("  - seoul_bike_processed.csv (for CUBIST/RF)")
        print("  - seoul_bike_tft.csv (for TFT) <-- You need this!")
        return
    
    print(f"\n[OK] Loading TFT data from {data_path}...")
    
    # Quick data check
    df = pd.read_csv(data_path)
    print(f"  Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Initialize model
    tft_model = TFTModel()
    
    # Prepare data
    training_data, validation_data, full_data = tft_model.prepare_data(
        data_path=data_path,
        max_encoder_length=24,  # Use 24 hours of history
        max_prediction_length=6  # Predict 6 hours ahead
    )
    
    # Create model
    model = tft_model.create_model(
        training_data=training_data,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        learning_rate=0.001
    )
    
    # Train model
    trainer = tft_model.train(
        training_data=training_data,
        validation_data=validation_data,
        max_epochs=50,
        batch_size=128,
        gpus=0  # Set to 1 if you have a GPU
    )

    # Evaluate model
    train_metrics, test_metrics = tft_model.evaluate(
        training_data=training_data,
        validation_data=validation_data
    )

    print("\n" + "="*80)
    print("TFT MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - models/checkpoints/ (model checkpoints)")
    print("  - results/tft_results.json (evaluation metrics)")
    print("\nYou can now compare these results with CUBIST model!")
    print("="*80)


if __name__ == "__main__":
    main()