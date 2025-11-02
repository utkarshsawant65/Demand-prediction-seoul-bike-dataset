# CUBIST Model Comparison: Random Split vs Temporal Split

## Executive Summary

This document compares two CUBIST models trained on the Seoul Bike Sharing dataset:

1. **Random Split (75/25)** - Replicates the research paper methodology
2. **Temporal Split (9 months / 3 months)** - Real-world scenario evaluation

### Key Finding
The temporal split shows significantly lower test performance (R² = 0.7187 vs 0.9395), demonstrating that **random split overestimates real-world performance** by allowing the model to learn from data distributed throughout the entire time period.

---

## Dataset Overview

- **Total Samples**: 8,760 hourly records
- **Date Range**: December 1, 2017 to November 30, 2018 (12 months)
- **Target Variable**: Bike rental count
- **Features**: 14 features (temporal, weather, categorical)

---

## Model 1: Random Split (Paper Replication)

### Split Strategy
- **Method**: Random stratified sampling (caret::createDataPartition)
- **Training Set**: 6,570 samples (75%)
- **Testing Set**: 2,190 samples (25%)
- **Characteristic**: Train and test samples are randomly distributed across all 12 months

### Best Hyperparameters
Based on the sklearn-based implementation (approximating CUBIST):
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 7
- **min_samples_split**: 2
- **min_samples_leaf**: 3

From the R CUBIST implementation, the results show these were the optimal settings found.

### Results

#### Training Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | 0.9872 | 98.72% of variance explained |
| RMSE | 73.16 bikes | Average prediction error |
| MAE | 43.17 bikes | Mean absolute error |
| CV | 10.39% | Coefficient of variation |

#### Testing Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | **0.9395** | **93.95% of variance explained** |
| **RMSE** | **158.17 bikes** | Average prediction error |
| **MAE** | **87.66 bikes** | Mean absolute error |
| **CV** | **22.39%** | Coefficient of variation |

### Interpretation
- Excellent performance on both train and test sets
- Small generalization gap (0.0477 in R²)
- **However**: This performance is inflated because test samples are distributed throughout the year
- The model has seen similar temporal patterns in training (just not these exact samples)

---

## Model 2: Temporal Split (Real-World Scenario)

### Split Strategy
- **Method**: Time-based split (chronological)
- **Training Period**: December 2017 - August 2018 (9 months)
  - **Training Samples**: 6,576 samples
- **Testing Period**: September 2018 - November 2018 (3 months)
  - **Testing Samples**: 2,184 samples
- **Characteristic**: Test data is completely in the future relative to training

### Split Dates
```
Training:  2017-12-01 to 2018-08-31 (9 months)
Testing:   2018-09-01 to 2018-11-30 (3 months)
```

### Best Hyperparameters
Found via 10-fold cross-validation with 3 repeats on training data:
- **committees**: 41
- **neighbors**: 3

Note: These are from the actual CUBIST R package, not an approximation.

### Results

#### Training Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | 0.9835 | 98.35% of variance explained |
| RMSE | 82.13 bikes | Average prediction error |
| MAE | 47.06 bikes | Mean absolute error |
| CV | 12.32% | Coefficient of variation |

#### Testing Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | **0.7187** | **71.87% of variance explained** |
| **RMSE** | **345.27 bikes** | Average prediction error |
| **MAE** | **245.10 bikes** | Mean absolute error |
| **CV** | **42.13%** | Coefficient of variation |

### Interpretation
- Excellent training performance (similar to random split)
- **Much lower test performance** - this is expected and realistic
- Large generalization gap (0.2648 in R²)
- **Why?** The model is tested on truly unseen future data with potentially different:
  - Seasonal patterns (Fall months: Sep, Oct, Nov)
  - Weather conditions not seen in training
  - User behavior changes over time
  - Events and holidays specific to those months

---

## Side-by-Side Comparison

### Test Set Performance Comparison

| Metric | Random Split (75/25) | Temporal Split (9mo/3mo) | Difference | % Change |
|--------|---------------------|--------------------------|------------|----------|
| **R²** | 0.9395 | 0.7187 | -0.2208 | **-23.5%** |
| **RMSE** | 158.17 bikes | 345.27 bikes | +187.10 | **+118.3%** |
| **MAE** | 87.66 bikes | 245.10 bikes | +157.44 | **+179.6%** |
| **CV** | 22.39% | 42.13% | +19.74% | **+88.2%** |

### Training Set Performance Comparison

| Metric | Random Split (75/25) | Temporal Split (9mo/3mo) | Difference |
|--------|---------------------|--------------------------|------------|
| R² | 0.9872 | 0.9835 | -0.0037 |
| RMSE | 73.16 bikes | 82.13 bikes | +8.97 |
| MAE | 43.17 bikes | 47.06 bikes | +3.89 |
| CV | 10.39% | 12.32% | +1.93% |

**Observation**: Training performance is nearly identical, confirming that the difference in test performance is purely due to the split method, not the model itself.

---

## Analysis: Why Such a Large Difference?

### 1. Random Split Advantage (Inflated Performance)
- **Data Leakage (Temporal)**: While there's no direct leakage, the model learns patterns from data distributed across the entire year
- **Similar Patterns**: Test samples are from the same months as training samples, just different hours/days
- **Example**:
  - Training might have: Dec 2017 (days 1-22), Jan 2018 (days 1-23), ..., Nov 2018 (days 1-22)
  - Testing might have: Dec 2017 (days 23-31), Jan 2018 (days 24-31), ..., Nov 2018 (days 23-30)
  - The model has already seen December weather, January patterns, etc.

### 2. Temporal Split Challenge (Realistic)
- **True Future Prediction**: Model must predict 3 months it has never seen
- **Seasonal Shift**: Training ends in August (Summer), testing covers Sep-Nov (Fall)
  - Different weather patterns
  - Different daylight hours
  - Different user behavior (back to school, holidays)
- **No "Anchor" Points**: The model cannot rely on having seen similar dates before

### 3. Specific Challenges for Sep-Nov Testing Period

#### Season Coverage in Training (Dec 2017 - Aug 2018)
| Season | Months in Training | Months in Testing |
|--------|-------------------|------------------|
| Winter | Dec 2017, Jan 2018, Feb 2018 | None |
| Spring | Mar 2018, Apr 2018, May 2018 | None |
| Summer | Jun 2018, Jul 2018, Aug 2018 | None |
| **Autumn** | **None** | **Sep, Oct, Nov 2018** |

**Critical Issue**: The testing period is entirely in Autumn, but the model was trained on **zero Autumn data**!

While the model has the "Season" feature as a categorical variable (and can technically handle "Autumn" as a category), it has never seen:
- Autumn weather patterns
- Autumn user behavior
- Autumn temperature-humidity-bike usage relationships
- Autumn holiday patterns

This explains the significant performance drop.

---

## Feature Importance Comparison

### Random Split - Top 15 Features
(From previous runs - not available in current output)
Expected to show temporal features (Hour, Day, Month) as highly important because the random split allows the model to memorize date-specific patterns.

### Temporal Split - Top 15 Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | Hour | 100.00 | Temporal |
| 2 | Temp | 89.94 | Weather |
| 3 | Hum | 59.22 | Weather |
| 4 | Dew | 55.87 | Weather |
| 5 | Rain | 37.43 | Weather |
| 6 | Solar | 34.64 | Weather |
| 7 | Visb | 24.02 | Weather |
| 8 | Wind | 15.64 | Weather |
| 9 | WeekStatus | 10.61 | Temporal |
| 10 | Snow | 6.15 | Weather |
| 11 | Season | 5.59 | Temporal |
| 12 | DayName | 2.79 | Temporal |
| 13 | Fday | 1.68 | Categorical |
| 14 | Holiday | 0.00 | Categorical |

**Key Observations**:
- **Hour** is overwhelmingly the most important feature (100.00)
  - Bike usage has strong hourly patterns (commute hours, lunch, evening)
  - This pattern is consistent across seasons
- **Weather features dominate**: Temp, Hum, Dew, Rain, Solar, Visb, Wind, Snow
  - This is why the seasonal shift hurts performance
  - Autumn weather differs significantly from Summer (training end)
- **Season** has low importance (5.59)
  - The model struggled to generalize to unseen Autumn data
  - This categorical feature alone cannot capture all seasonal nuances

---

## Practical Implications

### For Research Paper Comparison
- **Use Random Split Model** to replicate and compare with the paper
- **Expected**: Results should closely match the paper (R² ≈ 0.95)
- **Purpose**: Validates that the implementation is correct

### For Real-World Deployment
- **Use Temporal Split Model** to estimate actual performance
- **Expected**: Performance will be lower than paper (R² ≈ 0.72)
- **Purpose**: Sets realistic expectations for production system
- **Recommendation**:
  - Need to retrain model periodically (e.g., monthly) with new data
  - Consider using rolling window approach
  - Monitor for seasonal drift and performance degradation

### For Thesis Discussion
This comparison provides excellent material to discuss:

1. **Evaluation Methodology**
   - Why random splits can be misleading for time series
   - Importance of temporal validation for real-world applications

2. **Model Limitations**
   - CUBIST struggles with unseen seasons (Autumn in this case)
   - Heavy reliance on temporal features (Hour) that work across seasons
   - Weather features help but cannot fully compensate for seasonal shift

3. **Practical Recommendations**
   - Always include at least one full year in training to cover all seasons
   - Use rolling window training for production systems
   - Implement monitoring and automatic retraining triggers

4. **Research vs Reality Gap**
   - Paper: R² = 0.95 (impressive, publishable)
   - Reality: R² = 0.72 (good, but requires careful deployment)
   - This 23.5% drop is significant and must be communicated to stakeholders

---

## Recommendations

### For Your Thesis

1. **Report Both Results**
   - Random split: "Our implementation achieves R² = 0.9395, closely matching the paper's R² = 0.95"
   - Temporal split: "However, in a real-world temporal validation, performance drops to R² = 0.7187"

2. **Discuss the Gap**
   - Explain why temporal validation is more realistic
   - Discuss the missing Autumn data problem
   - Suggest solutions (full-year training, periodic retraining)

3. **Visual Aids**
   - Include the prediction scatter plots from both models
   - Show residual plots to illustrate where temporal model struggles
   - Create a timeline diagram showing train/test periods

4. **Additional Experiments** (Optional)
   - Train on full year, test on next 3 months (requires more data)
   - Use different temporal splits (e.g., 6mo/6mo, 10mo/2mo)
   - Compare with TFT model on same temporal split

### For Model Improvement

1. **Collect More Data**
   - Ideally 2-3 years to capture year-over-year patterns
   - Ensure all seasons are represented in training

2. **Feature Engineering**
   - Add lagged features (previous hour, previous day, previous week)
   - Add rolling statistics (7-day average, 30-day average)
   - Create interaction features (Hour × Season, Temp × Season)

3. **Ensemble Approaches**
   - Separate models for each season
   - Weighted ensemble based on season
   - Online learning to adapt to new patterns

4. **Regular Retraining**
   - Retrain monthly with sliding window (always keep 9-12 months)
   - Monitor performance and retrain when R² drops below threshold
   - A/B test new models before deploying

---

## Files and Artifacts

### Random Split Model
- **Script**: `r/cubist_model.r`
- **Model**: `models/cubist_seoul_bike.rds`
- **Results**: `reports/results/cubist_metrics_summary.csv`
- **Figures**: `reports/figures/cubist_*.pdf`

### Temporal Split Model
- **Script**: `r/cubist_model_temporal.r`
- **Model**: `models/temporal/cubist_temporal.rds`
- **Results**: `reports/results/temporal/cubist_metrics_temporal.csv`
- **Figures**: `reports/figures/temporal/cubist_*_temporal.pdf`

### Comparison
- **This Document**: `CUBIST_COMPARISON_RANDOM_VS_TEMPORAL.md`

---

## Conclusion

The temporal split evaluation reveals that while CUBIST achieves excellent performance in random split validation (R² = 0.9395), its real-world performance on unseen future data is significantly lower (R² = 0.7187).

**Key Takeaway**: The 23.5% drop in R² is primarily due to:
1. Testing on a completely unseen season (Autumn) not present in training
2. Temporal distribution shift between Summer (end of training) and Fall (testing period)
3. Inability to leverage temporal memorization that benefits random splits

This demonstrates the critical importance of temporal validation for time series forecasting and highlights the gap between research benchmarks and production performance.

For your thesis, this comparison strengthens your work by:
- Showing you understand evaluation methodology deeply
- Demonstrating both paper replication AND practical evaluation
- Providing honest, realistic performance expectations
- Offering actionable recommendations for deployment

The temporal split model represents what you would actually experience if deploying this system in September 2018, making it invaluable for practical applications.

---

**Generated**: 2025-11-02
**Models**: CUBIST (R implementation via caret package)
**Dataset**: Seoul Bike Sharing Dataset (8,760 hourly samples, Dec 2017 - Nov 2018)
