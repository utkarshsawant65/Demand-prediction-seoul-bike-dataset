# CUBIST Comparison Visualizations Guide

## Overview

All visualizations comparing Random Split (Paper Replication) vs Temporal Split (Real-World) CUBIST models have been generated and saved in both PNG and PDF formats.

**Location**: `reports/figures/comparison/`

---

## Generated Visualizations

### 1. R² Comparison Chart
**File**: `cubist_r2_comparison.png/pdf`

**What it shows**:
- Side-by-side comparison of R² scores for both split methods
- Training set (green bars) and Testing set (red bars)
- Blue dashed line showing published paper R² = 0.95
- Yellow annotation highlighting the 23.5% performance drop

**Key Insights**:
- Random Split Test R² = 0.9395 (matches paper closely)
- Temporal Split Test R² = 0.7187 (realistic real-world performance)
- Training performance is nearly identical (~0.98 for both)

**Best for**: Quick visual comparison of model performance, thesis presentations

---

### 2. Comprehensive Metrics Comparison
**File**: `cubist_all_metrics_comparison.png/pdf`

**What it shows**:
- 2×2 grid comparing all four metrics: R², RMSE, MAE, CV
- Each metric shows both training (lighter color) and testing (darker color)
- Direct comparison between Random Split and Temporal Split

**Key Insights**:
- R²: 0.9395 → 0.7187 (drops)
- RMSE: 158.17 → 345.27 bikes (increases by 118%)
- MAE: 87.66 → 245.10 bikes (increases by 180%)
- CV: 22.39% → 42.13% (increases by 88%)

**Best for**: Comprehensive performance analysis, showing multiple angles

---

### 3. Test Set Performance Focus
**File**: `cubist_test_performance_comparison.png/pdf`

**What it shows**:
- Focused comparison of test set performance only
- All four metrics (R², RMSE, MAE, CV) side-by-side
- Direct comparison bars for Random (blue) vs Temporal (red)

**Key Insights**:
- Highlights the dramatic difference in test performance
- Shows that error metrics (RMSE, MAE, CV) more than double
- Useful for understanding deployment implications

**Best for**: Emphasizing real-world vs research performance gap

---

### 4. Timeline Visualization
**File**: `cubist_timeline_comparison.png/pdf`

**What it shows**:
- **Top panel**: Random split - green (training) and red (testing) dots scattered throughout all 12 months
- **Bottom panel**: Temporal split - clear separation at Sep 1, 2018
  - Green dots (training) cover Dec 2017 - Aug 2018
  - Red dots (testing) cover Sep - Nov 2018
- Seasonal annotations (Winter, Spring, Summer in training; Autumn in testing)

**Key Insights**:
- Visual proof that random split distributes samples across entire year
- Temporal split tests on completely unseen future season (Autumn)
- Explains why temporal split has lower performance

**Best for**: Explaining the fundamental difference in evaluation methodology

---

### 5. Performance Drop Analysis
**File**: `cubist_performance_drop.png/pdf`

**What it shows**:
- Horizontal bar chart showing percentage change in each metric
- R²: -23.5% (red bar - performance degradation)
- RMSE: +118.3% (orange bar - error increase)
- MAE: +179.6% (orange bar - error increase)
- CV: +88.2% (orange bar - variability increase)

**Key Insights**:
- Quantifies the exact percentage change for each metric
- R² drops by nearly a quarter
- Error metrics more than double
- Demonstrates significant real-world performance degradation

**Best for**: Highlighting the magnitude of performance change, discussion sections

---

### 6. Summary Dashboard
**File**: `cubist_summary_dashboard.png/pdf`

**What it shows**:
- Comprehensive 6-panel dashboard with:
  1. R² comparison (top left)
  2. Performance summary table (top right)
  3. RMSE comparison (middle left)
  4. MAE comparison (middle center)
  5. CV comparison (middle right)
  6. Key findings text box (bottom)

**Key Insights**:
- All-in-one view of the entire comparison
- Includes detailed textual insights
- Summary table with percentage changes
- Self-contained analysis

**Best for**: Executive summaries, comprehensive thesis figures, presentations

---

## Quick Reference Table

| Visualization | Focus | Use Case | Key Message |
|--------------|-------|----------|-------------|
| **R² Comparison** | R² scores only | Quick comparison | Test R² drops from 0.94 to 0.72 |
| **All Metrics** | R², RMSE, MAE, CV | Comprehensive view | All metrics show degradation |
| **Test Performance** | Test set focus | Real-world impact | Errors double in temporal split |
| **Timeline** | Train/test split | Methodology explanation | Temporal tests on unseen season |
| **Performance Drop** | Percentage changes | Gap quantification | 23.5% R² drop, 118% RMSE increase |
| **Dashboard** | Everything | Full analysis | Complete comparison at a glance |

---

## File Formats

All visualizations are available in two formats:

1. **PNG** (High resolution, 300 DPI)
   - Best for: Word documents, PowerPoint presentations, web viewing
   - File size: ~200-500 KB each

2. **PDF** (Vector format)
   - Best for: LaTeX documents, printing, publications
   - Scalable without quality loss

---

## Usage Recommendations

### For Thesis Document

**Results Section**:
- Include: **R² Comparison** or **All Metrics Comparison**
- Caption: "Comparison of CUBIST model performance between random split (paper replication) and temporal split (real-world scenario). Random split achieves R² = 0.9395 matching the published result, while temporal split yields R² = 0.7187."

**Methodology Section**:
- Include: **Timeline Visualization**
- Caption: "Train/test split strategies. Top: Random split distributes samples throughout the year. Bottom: Temporal split trains on first 9 months and tests on the final 3 months (Autumn), an entirely unseen season."

**Discussion Section**:
- Include: **Performance Drop** or **Summary Dashboard**
- Caption: "Performance degradation analysis showing temporal split reduces R² by 23.5% and increases RMSE by 118.3% compared to random split, highlighting the gap between research benchmarks and real-world deployment."

### For Presentations

**Slide 1 - Main Result**:
- Use: **R² Comparison**
- Message: "Our implementation validates the paper (R² = 0.94) but real-world testing shows lower performance (R² = 0.72)"

**Slide 2 - Why the Difference**:
- Use: **Timeline Visualization**
- Message: "Random split tests on samples from all months; Temporal split tests on completely unseen future season"

**Slide 3 - Full Analysis**:
- Use: **Summary Dashboard**
- Message: "Comprehensive comparison shows systematic performance degradation across all metrics"

### For Research Paper

**Figure 1**:
- **All Metrics Comparison** (2×2 grid)
- Full caption with detailed explanation

**Figure 2**:
- **Timeline Visualization**
- Explains evaluation methodology difference

**Table 1**:
- Extract from Summary Dashboard table
- Quantitative results with percentage changes

---

## Reproducing Visualizations

To regenerate all visualizations:

```bash
python visualize_cubist_comparison.py
```

**Requirements**:
- matplotlib >= 3.10
- seaborn >= 0.13
- pandas >= 2.3
- numpy >= 2.3

**Output**: 12 files (6 PNG + 6 PDF) in `reports/figures/comparison/`

**Time**: ~10-15 seconds

---

## Key Statistics Summary

For quick reference when writing:

| Metric | Random Split | Temporal Split | Change |
|--------|-------------|----------------|--------|
| **Test R²** | 0.9395 | 0.7187 | -23.5% |
| **Test RMSE** | 158.17 bikes | 345.27 bikes | +118.3% |
| **Test MAE** | 87.66 bikes | 245.10 bikes | +179.6% |
| **Test CV** | 22.39% | 42.13% | +88.2% |
| **Train R²** | 0.9872 | 0.9835 | -0.4% |

**Key Insight**: Training performance is nearly identical (~98% R² for both), confirming the difference is purely due to evaluation methodology, not model quality.

---

## Customization

The visualization script (`visualize_cubist_comparison.py`) can be modified to:

1. **Change colors**: Edit color palettes in each function
2. **Add annotations**: Modify text boxes and labels
3. **Adjust layouts**: Change figure sizes and subplot arrangements
4. **Add metrics**: Include additional metrics from results files
5. **Export formats**: Add SVG, EPS, or other formats

---

## Additional Notes

### Why R² is the Focus

R² (coefficient of determination) is emphasized because:
- It's the primary metric in the original paper
- Scale-independent (0-1 range)
- Easily interpretable (% variance explained)
- Widely understood in machine learning

### Why Error Metrics Matter

RMSE and MAE show actual prediction errors in bikes:
- RMSE = 158 bikes (random) vs 345 bikes (temporal)
  - Temporal split averages 187 more bikes of error
  - In a system with mean ~800 bikes, this is significant
- MAE = 88 bikes (random) vs 245 bikes (temporal)
  - Median error is 157 bikes higher in temporal split

### CV (Coefficient of Variation)

CV normalizes error by mean:
- CV = (RMSE / mean) × 100%
- Random: 22.39% - predictions within ~22% of actual
- Temporal: 42.13% - predictions within ~42% of actual
- Higher CV indicates less reliable predictions

---

**Last Updated**: 2025-11-02
**Visualizations Generated**: 6 unique charts (12 files with PNG+PDF)
**Ready for**: Thesis, presentations, publications
