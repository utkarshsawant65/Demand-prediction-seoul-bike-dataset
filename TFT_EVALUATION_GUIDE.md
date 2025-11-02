# TFT Model Evaluation Guide

## How to Check TFT Model Performance

There are **3 ways** to evaluate your TFT model:

---

## Option 1: Automatic Evaluation (After Training)

When you run `python train_tft.py`, the model will **automatically evaluate** after training completes and display:

- **Training Set Metrics**: R2, RMSE, MAE, CV
- **Test Set Metrics**: R2, RMSE, MAE, CV

The results are saved to: `results/tft_results.json`

---

## Option 2: Standalone Evaluation Script

If you've already trained the model and want to re-evaluate:

```bash
python evaluate_tft.py
```

This will:
1. Load the latest checkpoint from `models/checkpoints/`
2. Evaluate on both training and test sets
3. Display metrics in the same format as CUBIST
4. Save results to `results/tft_results.json`

---

## Option 3: Manual Evaluation in Python

```python
from src.tft_model import TFTModel
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

# Initialize and prepare data
tft_model = TFTModel()
training_data, validation_data, full_data = tft_model.prepare_data(
    data_path="data/processed/seoul_bike_tft.csv",
    max_encoder_length=24,
    max_prediction_length=6
)

# Load trained model
tft_model.model = TemporalFusionTransformer.load_from_checkpoint(
    "models/checkpoints/tft-epoch-XX-val_loss-YY.ckpt"
)

# Evaluate
train_metrics, test_metrics = tft_model.evaluate(
    training_data=training_data,
    validation_data=validation_data
)
```

---

## Understanding the Metrics

The TFT model outputs the same metrics as CUBIST for fair comparison:

### Metrics Explained:
- **R2 (R-squared)**: Coefficient of determination (0 to 1, higher is better)
  - Measures how well predictions fit the actual data
  - 1.0 = perfect fit, 0.0 = model is no better than mean

- **RMSE (Root Mean Squared Error)**: Average prediction error (lower is better)
  - In units of bike rentals
  - Penalizes large errors more heavily

- **MAE (Mean Absolute Error)**: Average absolute prediction error (lower is better)
  - In units of bike rentals
  - More robust to outliers than RMSE

- **CV (Coefficient of Variation)**: RMSE as percentage of mean (lower is better)
  - Shows error relative to the scale of the data
  - Useful for comparing across different scales

---

## Example Output

```
================================================================================
TFT MODEL PERFORMANCE
================================================================================

TRAINING SET RESULTS:
------------------------------------------------------------
  Model          : TFT_train
  R2             : 0.9234
  RMSE           : 124.56
  MAE            : 89.32
  CV             : 17.68

VALIDATION/TEST SET RESULTS:
------------------------------------------------------------
  Model          : TFT_test
  R2             : 0.8945
  RMSE           : 156.78
  MAE            : 112.45
  CV             : 22.31

================================================================================
COMPARISON WITH CUBIST:
Use these metrics to compare with CUBIST model results
================================================================================
```

---

## Files and Locations

### Input Files:
- `data/processed/seoul_bike_tft.csv` - Preprocessed data for TFT

### Output Files:
- `models/checkpoints/tft-*.ckpt` - Trained model checkpoints
- `results/tft_results.json` - Evaluation metrics (JSON format)
- `models/checkpoints/lightning_logs/` - Training logs

### Checkpoint Files:
The model saves the best checkpoints based on validation loss:
- `tft-epoch-XX-val_loss-YY.ckpt` where:
  - XX = epoch number
  - YY = validation loss value

---

## Comparing with CUBIST

To compare TFT with CUBIST:

1. **Run CUBIST model** (from your R script)
2. **Run TFT model**: `python train_tft.py`
3. **Compare metrics** side-by-side:

| Metric | CUBIST | TFT | Winner |
|--------|--------|-----|--------|
| R2     | ?      | ?   | Higher is better |
| RMSE   | ?      | ?   | Lower is better |
| MAE    | ?      | ?   | Lower is better |
| CV     | ?      | ?   | Lower is better |

Both models use the **same 75/25 train/test split** for fair comparison!

---

## Troubleshooting

### "No checkpoints found"
- Make sure training completed: `python train_tft.py`
- Check `models/checkpoints/` directory exists

### "Model not trained"
- You must train first before evaluating
- Or load a checkpoint using the manual method

### "Data not found"
- Run preprocessing: `python src/preprocessing.py --tft`
- This creates `seoul_bike_tft.csv`

---

## Quick Reference

```bash
# Full workflow
python src/preprocessing.py --tft    # Prepare data
python train_tft.py                   # Train and evaluate
python evaluate_tft.py                # Re-evaluate anytime

# Check results
cat results/tft_results.json
```
