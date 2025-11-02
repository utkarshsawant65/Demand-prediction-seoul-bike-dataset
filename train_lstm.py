"""
Training script for LSTM models (Random and Temporal splits)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lstm_model import train_lstm_models

if __name__ == "__main__":
    print("Starting LSTM model training...")
    print("This will train two models:")
    print("  1. Random Split (75/25) - Paper replication")
    print("  2. Temporal Split (9mo/3mo) - Real-world scenario")
    print()

    results = train_lstm_models()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nModels saved:")
    print("  - models/lstm_random.pth")
    print("  - models/lstm_temporal.pth")
    print("\nResults saved:")
    print("  - results/lstm_results.json")
    print("="*80)
