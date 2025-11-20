# Feature-Engineered Data for Seoul Bike Demand Prediction

This directory contains the feature-engineered datasets ready for model training and testing.

## Files

- **train.csv** - Training dataset (6,840 rows × 66 columns)
- **test.csv** - Testing dataset (1,752 rows × 66 columns)
- **feature_names.txt** - List of all 64 feature names
- **feature_engineering_report.txt** - Detailed report of the feature engineering process

## Data Split

- **Training Period**: 2017-12-08 to 2018-09-18 (before split date)
- **Testing Period**: 2018-09-19 to 2018-11-30 (from split date onwards)
- **Split Ratio**: ~80% train / ~20% test (temporal split)

## Target Variable

- **Column Name**: `Rented Bike Count` and `target` (both contain same values)
- **Type**: Integer (hourly bike rental count)
- **Train Range**: 0 - 3,556 bikes
- **Test Range**: 0 - 3,154 bikes

## Features Summary

Total of **64 features** organized into categories:

### 1. Temporal Features (20)
- Basic temporal: day_of_week, month, day_of_month, is_weekend, is_weekday
- Cyclical encodings: hour_sin/cos, day_of_week_sin/cos, month_sin/cos
- Rush hour indicators: is_morning_rush, is_evening_rush, is_rush_hour
- Time of day categories: time_of_day_night/morning/afternoon/evening
- Work hour indicator: is_work_hour

### 2. Lag Features (8)
- Demand lags: demand_lag_1h, 2h, 3h, 24h, 168h
- Weather lags: temp_lag_1h, humidity_lag_1h, wind_speed_lag_1h

### 3. Rolling Statistics (10)
- Demand rolling: 3h/6h/12h/24h mean, 24h std/min/max
- Temperature rolling: 3h mean/std, 24h mean

### 4. Interaction Features (13)
- Temperature interactions: temp_x_hour, temp_x_is_rush_hour, temp_squared
- Comfort index: is_comfortable_weather
- Weather indicators: has_rain, has_snow, has_precipitation, low_visibility, bad_weather
- Derived metrics: wind_chill, apparent_temp

### 5. Weather Change Features (3)
- Rate of change: temp_change_1h, humidity_change_1h, wind_speed_change_1h

### 6. Categorical Features (5)
- Season dummies: Season_Spring, Season_Summer, Season_Winter (Winter is baseline)
- Binary indicators: is_holiday, is_functioning

### 7. Original Weather Features (8)
- Temperature(°C), Humidity(%), Wind speed (m/s)
- Visibility (10m), Dew point temperature(°C)
- Solar Radiation (MJ/m2), Rainfall(mm), Snowfall (cm)

## Data Quality

✓ **No missing values** in train or test sets
✓ **No infinite values** in train or test sets
✓ **Consistent columns** across train and test
✓ **Temporal ordering** preserved
✓ **No data leakage** - all lag/rolling features created before splitting

## Important Notes

1. **Lag Features Created Before Splitting**: All lag and rolling features were created on the full dataset BEFORE splitting to ensure proper temporal feature creation while avoiding data leakage.

2. **Rolling Windows Use Past Data Only**: All rolling statistics exclude the current hour (using `.shift(1).rolling()`) to prevent information leakage.

3. **Rows Dropped**: 168 rows (1.92%) were dropped due to missing values from lag features (first 168 hours of the dataset).

4. **Cyclical Encoding**: Hour, day_of_week, and month are cyclically encoded using sine/cosine to prevent discontinuity at boundaries (e.g., 23:00 → 00:00).

5. **Target Column**: Both "Rented Bike Count" and "target" columns are included for compatibility. Use either one as your prediction target.

## Usage Example

```python
import pandas as pd

# Load datasets
train = pd.read_csv('data/feature_data/train.csv')
test = pd.read_csv('data/feature_data/test.csv')

# Separate features and target
feature_cols = [col for col in train.columns if col not in ['Rented Bike Count', 'target']]
X_train = train[feature_cols]
y_train = train['target']

X_test = test[feature_cols]
y_test = test['target']

print(f"Training features: {X_train.shape}")
print(f"Testing features: {X_test.shape}")
```

## Generating This Data

To regenerate this feature-engineered data:

```bash
python feature_engineering.py
```

The script will:
1. Load raw data from `data/raw/seoul_bike_data/SeoulBikeData.csv`
2. Create all 64 engineered features
3. Split data temporally (before/after 2018-09-19)
4. Save outputs to `data/feature_data/`

---

**Generated**: 2025-11-18
**Source**: SeoulBikeData.csv (8,760 original rows)
**Script**: feature_engineering.py
