"""
Training script for TFT model
Run this after preprocessing the data with preprocessing.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tft_model import main

if __name__ == "__main__":
    main()
