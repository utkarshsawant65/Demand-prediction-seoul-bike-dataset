"""
Standalone script to evaluate a trained TFT model
Run this after training is complete to get evaluation metrics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tft_model import TFTModel
import pandas as pd

def main():
    print("="*80)
    print("TFT MODEL EVALUATION")
    print("="*80)

    # Check if data exists
    data_path = "data/processed/seoul_bike_tft.csv"
    if not Path(data_path).exists():
        print(f"\n[ERROR] Data not found at {data_path}")
        print("Run preprocessing first: python src/preprocessing.py --tft")
        return

    print(f"\n[OK] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Initialize model
    tft_model = TFTModel()

    # Prepare data (same as training)
    print("\nPreparing data...")
    training_data, validation_data, full_data = tft_model.prepare_data(
        data_path=data_path,
        max_encoder_length=24,
        max_prediction_length=6
    )

    # Load trained model from checkpoint
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        print(f"\n[ERROR] No checkpoints found at {checkpoint_dir}")
        print("Train the model first: python train_tft.py")
        return

    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        print(f"\n[ERROR] No .ckpt files found in {checkpoint_dir}")
        return

    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n[OK] Loading model from {latest_checkpoint.name}...")

    # Load model
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
    tft_model.model = TemporalFusionTransformer.load_from_checkpoint(str(latest_checkpoint))

    # Evaluate
    train_metrics, test_metrics = tft_model.evaluate(
        training_data=training_data,
        validation_data=validation_data
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
