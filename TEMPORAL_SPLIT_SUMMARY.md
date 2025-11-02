# Temporal Split Model - Quick Summary

## What Was Done

Created a **second CUBIST model** with temporal split (9 months train / 3 months test) to evaluate real-world performance, while **preserving the original random split model** for paper replication.

## Results at a Glance

### Random Split (Paper Replication) - 75/25
- Training R²: **0.9872**
- Testing R²: **0.9395**
- Testing RMSE: **158.17 bikes**
- Files preserved in: `reports/results/` and `models/cubist_seoul_bike.rds`

### Temporal Split (Real-World) - 9mo/3mo
- Training R²: **0.9835**
- Testing R²: **0.7187** ⚠️ (significantly lower)
- Testing RMSE: **345.27 bikes**
- Files in: `reports/results/temporal/` and `models/temporal/cubist_temporal.rds`

## Key Difference: -22% in R²

The temporal split shows **71.87% vs 93.95%** variance explained - a drop of 22 percentage points.

**Why?**
- Random split: Test samples scattered throughout all 12 months (model has seen similar dates)
- Temporal split: Test is Sep-Nov 2018 (Autumn) - **completely unseen season** (training ended in August)

## Critical Finding: Missing Season Problem

| Season | Training | Testing |
|--------|----------|---------|
| Winter | ✓ Dec 2017, Jan-Feb 2018 | - |
| Spring | ✓ Mar-May 2018 | - |
| Summer | ✓ Jun-Aug 2018 | - |
| **Autumn** | **✗ Not included** | **✓ Sep-Nov 2018** |

The model was tested on an **entire season it never saw during training**, explaining the performance drop.

## File Structure

```
seoul-bike-thesis/
├── r/
│   ├── cubist_model.r                    # Original (random split)
│   └── cubist_model_temporal.r           # NEW (temporal split)
├── models/
│   ├── cubist_seoul_bike.rds             # Original model
│   └── temporal/
│       └── cubist_temporal.rds           # NEW temporal model
├── reports/
│   ├── results/
│   │   ├── cubist_metrics_summary.csv    # Original results
│   │   └── temporal/
│   │       ├── cubist_metrics_temporal.csv        # NEW
│   │       ├── cubist_best_params_temporal.csv    # NEW
│   │       └── cubist_predictions_temporal.csv    # NEW
│   └── figures/
│       ├── cubist_*.pdf                  # Original plots
│       └── temporal/
│           └── cubist_*_temporal.pdf     # NEW plots
├── CUBIST_COMPARISON_RANDOM_VS_TEMPORAL.md  # Detailed comparison doc
└── TEMPORAL_SPLIT_SUMMARY.md                # This file
```

## For Your Thesis

### What to Report

1. **Paper Replication** (Random Split):
   - "Our CUBIST implementation achieves R² = 0.9395, closely matching the published result of R² = 0.95"

2. **Real-World Evaluation** (Temporal Split):
   - "However, temporal validation on unseen future data (Sep-Nov 2018) yields R² = 0.7187"
   - "This 22-point drop is due to testing on an unseen season (Autumn) not present in training"

3. **Insight**:
   - "This demonstrates that random split evaluation overestimates real-world performance by 23.5%"
   - "Temporal validation provides more realistic performance expectations for deployment"

### Recommendations

1. **Model Improvement**:
   - Train on at least 1 full year to cover all seasons
   - Use rolling window retraining in production
   - Consider season-specific models or ensemble

2. **Deployment Strategy**:
   - Retrain monthly with sliding 9-12 month window
   - Monitor R² and retrain if drops below 0.75
   - A/B test new models before production deployment

## Next Steps (Optional)

1. **Train TFT with same temporal split** for comparison
2. **Create visualization** showing train/test periods timeline
3. **Analyze residuals** by season to identify specific failure modes
4. **Feature engineering**: Add lagged features and seasonal interactions

## Quick Commands

### Run Random Split Model (already done):
```bash
Rscript r/cubist_model.r
```

### Run Temporal Split Model (already done):
```bash
Rscript r/cubist_model_temporal.r
```

### View Results:
```bash
# Random split
cat reports/results/cubist_metrics_summary.csv

# Temporal split
cat reports/results/temporal/cubist_metrics_temporal.csv
```

---

**Bottom Line**: You now have two CUBIST models:
1. **Random split** (R² = 0.94) - proves your implementation matches the paper ✓
2. **Temporal split** (R² = 0.72) - shows realistic real-world performance ✓

Both models are preserved and ready for your thesis analysis!
