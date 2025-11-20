"""
FEATURE ENGINEERING PIPELINE FOR SEOUL BIKE DEMAND PREDICTION
==============================================================

This script creates a comprehensive set of engineered features for time series
prediction of Seoul bike sharing demand. Features are designed to be used across
all models (LSTM, TCN, GRU, hybrids).

CRITICAL: Lag and rolling features are created on the FULL dataset BEFORE splitting
to ensure proper temporal feature creation while avoiding data leakage.

Author: Feature Engineering Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(data_path):
    """
    Load raw data and prepare for feature engineering.

    Args:
        data_path: Path to SeoulBikeData.csv

    Returns:
        DataFrame with parsed dates and sorted by time
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 70)

    # Load data (handle encoding for special characters like °)
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='latin-1')
    print(f"[OK] Loaded raw data: {df.shape}")

    # Parse date column (format: DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    print(f"[OK] Parsed Date column")

    # Sort by Date and Hour to ensure temporal ordering
    df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
    print(f"[OK] Sorted by Date and Hour")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Store original target
    df['target'] = df['Rented Bike Count']
    print(f"[OK] Saved original target column")

    return df


def create_temporal_features(df):
    """
    Create time-based features including cyclical encodings and rush hour indicators.

    Args:
        df: DataFrame with Date column

    Returns:
        DataFrame with added temporal features
    """
    print("\n" + "=" * 70)
    print("STEP 2: CREATING TEMPORAL FEATURES")
    print("=" * 70)

    # Basic temporal features
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
    print("[OK] Created basic temporal features (day_of_week, month, day_of_month, is_weekend, is_weekday)")

    # Cyclical encoding (prevents discontinuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print("[OK] Created cyclical encodings (hour_sin/cos, day_of_week_sin/cos, month_sin/cos)")

    # Rush hour features (Seoul commute patterns)
    df['is_morning_rush'] = df['Hour'].isin([7, 8, 9]).astype(int)
    df['is_evening_rush'] = df['Hour'].isin([17, 18, 19]).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
    print("[OK] Created rush hour features (is_morning_rush, is_evening_rush, is_rush_hour)")

    # Time of day categories
    df['time_of_day_night'] = df['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df['time_of_day_morning'] = df['Hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)
    df['time_of_day_afternoon'] = df['Hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
    df['time_of_day_evening'] = df['Hour'].isin([18, 19, 20, 21, 22, 23]).astype(int)
    print("[OK] Created time of day categories (night, morning, afternoon, evening)")

    # Work hours
    df['is_work_hour'] = ((df['Hour'] >= 8) & (df['Hour'] <= 18) & (df['is_weekday'] == 1)).astype(int)
    print("[OK] Created work hour indicator")

    print(f"\n  Total temporal features created: 20")

    return df


def create_lag_features(df):
    """
    Create lag features for demand and weather variables.

    CRITICAL: This must be called BEFORE train/test split to avoid data leakage.

    Args:
        df: DataFrame with time series data

    Returns:
        DataFrame with added lag features
    """
    print("\n" + "=" * 70)
    print("STEP 3: CREATING LAG FEATURES (ON FULL DATASET)")
    print("=" * 70)
    print("[WARNING]  IMPORTANT: Creating lags BEFORE splitting to ensure proper temporal features")

    # Demand lag features
    df['demand_lag_1h'] = df['Rented Bike Count'].shift(1)
    df['demand_lag_2h'] = df['Rented Bike Count'].shift(2)
    df['demand_lag_3h'] = df['Rented Bike Count'].shift(3)
    df['demand_lag_24h'] = df['Rented Bike Count'].shift(24)
    df['demand_lag_168h'] = df['Rented Bike Count'].shift(168)
    print("[OK] Created demand lag features (1h, 2h, 3h, 24h, 168h)")

    # Weather lag features
    df['temp_lag_1h'] = df['Temperature(°C)'].shift(1)
    df['humidity_lag_1h'] = df['Humidity(%)'].shift(1)
    df['wind_speed_lag_1h'] = df['Wind speed (m/s)'].shift(1)
    print("[OK] Created weather lag features (temp, humidity, wind_speed)")

    print(f"\n  Total lag features created: 8")

    return df


def create_rolling_features(df):
    """
    Create rolling statistics capturing recent trends.

    CRITICAL: This must be called BEFORE train/test split.
    Rolling windows use only PAST data (not including current hour).

    Args:
        df: DataFrame with time series data

    Returns:
        DataFrame with added rolling features
    """
    print("\n" + "=" * 70)
    print("STEP 4: CREATING ROLLING STATISTICS (ON FULL DATASET)")
    print("=" * 70)
    print("[WARNING]  IMPORTANT: Using only PAST data in rolling windows (not including current hour)")

    # Demand rolling statistics
    df['demand_rolling_3h_mean'] = df['Rented Bike Count'].shift(1).rolling(window=3, min_periods=1).mean()
    df['demand_rolling_6h_mean'] = df['Rented Bike Count'].shift(1).rolling(window=6, min_periods=1).mean()
    df['demand_rolling_12h_mean'] = df['Rented Bike Count'].shift(1).rolling(window=12, min_periods=1).mean()
    df['demand_rolling_24h_mean'] = df['Rented Bike Count'].shift(1).rolling(window=24, min_periods=1).mean()
    df['demand_rolling_24h_std'] = df['Rented Bike Count'].shift(1).rolling(window=24, min_periods=1).std()
    df['demand_rolling_24h_min'] = df['Rented Bike Count'].shift(1).rolling(window=24, min_periods=1).min()
    df['demand_rolling_24h_max'] = df['Rented Bike Count'].shift(1).rolling(window=24, min_periods=1).max()
    print("[OK] Created demand rolling statistics (3h, 6h, 12h, 24h mean/std/min/max)")

    # Temperature rolling statistics
    df['temp_rolling_3h_mean'] = df['Temperature(°C)'].shift(1).rolling(window=3, min_periods=1).mean()
    df['temp_rolling_3h_std'] = df['Temperature(°C)'].shift(1).rolling(window=3, min_periods=1).std()
    df['temp_rolling_24h_mean'] = df['Temperature(°C)'].shift(1).rolling(window=24, min_periods=1).mean()
    print("[OK] Created temperature rolling statistics (3h mean/std, 24h mean)")

    print(f"\n  Total rolling features created: 10")

    return df


def split_train_test(df, split_date='2018-09-19'):
    """
    Split data into training and testing sets based on date.

    Args:
        df: DataFrame with all features
        split_date: Date to split train/test (format: YYYY-MM-DD)

    Returns:
        train_df, test_df
    """
    print("\n" + "=" * 70)
    print("STEP 5: SPLITTING INTO TRAIN/TEST SETS")
    print("=" * 70)

    split_date = pd.to_datetime(split_date)

    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    print(f"[OK] Training set: {train_df.shape[0]} rows")
    print(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"\n[OK] Testing set: {test_df.shape[0]} rows")
    print(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")

    return train_df, test_df


def create_interaction_features(df):
    """
    Create weather interaction and derived features.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with added interaction features
    """
    print("\n" + "=" * 70)
    print("STEP 6: CREATING INTERACTION FEATURES")
    print("=" * 70)

    # Temperature interactions
    df['temp_x_hour'] = df['Temperature(°C)'] * df['Hour']
    df['temp_x_is_rush_hour'] = df['Temperature(°C)'] * df['is_rush_hour']
    df['temp_squared'] = df['Temperature(°C)'] ** 2
    print("[OK] Created temperature interactions (temp_x_hour, temp_x_is_rush_hour, temp_squared)")

    # Comfort index
    df['is_comfortable_weather'] = (
        (df['Temperature(°C)'] > 10) &
        (df['Temperature(°C)'] < 25) &
        (df['Humidity(%)'] < 70)
    ).astype(int)
    print("[OK] Created comfort index (is_comfortable_weather)")

    # Bad weather indicators
    df['has_rain'] = (df['Rainfall(mm)'] > 0).astype(int)
    df['has_snow'] = (df['Snowfall (cm)'] > 0).astype(int)
    df['has_precipitation'] = ((df['has_rain'] == 1) | (df['has_snow'] == 1)).astype(int)
    df['low_visibility'] = (df['Visibility (10m)'] < 1000).astype(int)
    df['bad_weather'] = ((df['has_precipitation'] == 1) | (df['low_visibility'] == 1)).astype(int)
    print("[OK] Created bad weather indicators (has_rain, has_snow, has_precipitation, low_visibility, bad_weather)")

    # Wind chill effect
    df['wind_chill'] = df['Temperature(°C)'] - (df['Wind speed (m/s)'] * 0.7)
    print("[OK] Created wind chill effect")

    # Apparent temperature (feels-like)
    df['apparent_temp'] = df['Temperature(°C)'] - (
        0.4 * (df['Temperature(°C)'] - 10) * (1 - df['Humidity(%)'] / 100)
    )
    print("[OK] Created apparent temperature")

    print(f"\n  Total interaction features created: 13")

    return df


def create_weather_change_features(df):
    """
    Create weather change features (rate of change).

    Args:
        df: DataFrame with lag features

    Returns:
        DataFrame with added change features
    """
    print("\n" + "=" * 70)
    print("STEP 7: CREATING WEATHER CHANGE FEATURES")
    print("=" * 70)

    # 1-hour changes
    df['temp_change_1h'] = df['Temperature(°C)'] - df['temp_lag_1h']
    df['humidity_change_1h'] = df['Humidity(%)'] - df['humidity_lag_1h']
    df['wind_speed_change_1h'] = df['Wind speed (m/s)'] - df['wind_speed_lag_1h']
    print("[OK] Created 1-hour change features (temp, humidity, wind_speed)")

    print(f"\n  Total change features created: 3")

    return df


def create_categorical_encodings(df):
    """
    Encode categorical variables (Seasons, Holiday, Functioning Day).

    Args:
        df: DataFrame with categorical columns

    Returns:
        DataFrame with encoded categorical features
    """
    print("\n" + "=" * 70)
    print("STEP 8: CREATING CATEGORICAL ENCODINGS")
    print("=" * 70)

    # One-hot encode Seasons (drop first to avoid multicollinearity)
    season_dummies = pd.get_dummies(df['Seasons'], prefix='Season', drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)
    print(f"[OK] One-hot encoded Seasons: {list(season_dummies.columns)}")

    # Binary encode Holiday
    df['is_holiday'] = (df['Holiday'] == 'Holiday').astype(int)
    print("[OK] Binary encoded Holiday -> is_holiday")

    # Binary encode Functioning Day
    df['is_functioning'] = (df['Functioning Day'] == 'Yes').astype(int)
    print("[OK] Binary encoded Functioning Day -> is_functioning")

    print(f"\n  Total categorical features created: {len(season_dummies.columns) + 2}")

    return df


def drop_correlated_features(df, threshold=0.90):
    """
    Drop highly correlated features to reduce multicollinearity.

    Args:
        df: DataFrame with all features
        threshold: Correlation threshold for dropping features

    Returns:
        DataFrame with reduced features, list of dropped features
    """
    print("\n" + "=" * 70)
    print("STEP 9: DROPPING HIGHLY CORRELATED FEATURES")
    print("=" * 70)

    # List of numeric columns to check (exclude target and categorical)
    exclude_cols = ['Date', 'Rented Bike Count', 'target', 'Seasons', 'Holiday', 'Functioning Day']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()

    # Find pairs with correlation > threshold
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features to drop
    to_drop = []
    dropped_info = []

    for column in upper_triangle.columns:
        correlated_features = upper_triangle[column][upper_triangle[column] > threshold]
        if len(correlated_features) > 0:
            for corr_feature in correlated_features.index:
                # Prefer to keep Temperature over Dew point temperature
                if 'Dew point temperature' in [column, corr_feature]:
                    drop_col = 'Dew point temperature(°C)'
                    keep_col = 'Temperature(°C)' if column != drop_col else corr_feature
                    if drop_col not in to_drop and drop_col in df.columns:
                        to_drop.append(drop_col)
                        dropped_info.append(f"{drop_col} (corr={corr_matrix.loc[drop_col, keep_col]:.3f} with {keep_col})")

    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[OK] Dropped {len(to_drop)} highly correlated features:")
        for info in dropped_info:
            print(f"  - {info}")
    else:
        print("[OK] No highly correlated features to drop (threshold={threshold})")

    return df, to_drop


def drop_original_columns(df):
    """
    Drop original columns that have been transformed into features.

    Args:
        df: DataFrame with all features

    Returns:
        DataFrame with cleaned columns, list of dropped columns
    """
    print("\n" + "=" * 70)
    print("STEP 10: DROPPING ORIGINAL COLUMNS")
    print("=" * 70)

    # Columns to drop (replaced by engineered features)
    to_drop = ['Date', 'Hour', 'Seasons', 'Holiday', 'Functioning Day']
    existing_to_drop = [col for col in to_drop if col in df.columns]

    df = df.drop(columns=existing_to_drop)
    print(f"[OK] Dropped {len(existing_to_drop)} original columns:")
    for col in existing_to_drop:
        print(f"  - {col}")

    return df, existing_to_drop


def handle_missing_values(df):
    """
    Handle missing values from lag and rolling features.

    Args:
        df: DataFrame with potential missing values

    Returns:
        DataFrame without missing values, number of rows dropped
    """
    print("\n" + "=" * 70)
    print("STEP 11: HANDLING MISSING VALUES")
    print("=" * 70)

    initial_rows = len(df)

    # Check for missing values
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        print(f"[OK] Found missing values in {len(cols_with_missing)} columns:")
        for col, count in cols_with_missing.items():
            print(f"  - {col}: {count} missing ({count/initial_rows*100:.2f}%)")

        # Drop rows with any missing values
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        print(f"\n[OK] Dropped {rows_dropped} rows with missing values ({rows_dropped/initial_rows*100:.2f}%)")
    else:
        print("[OK] No missing values found")
        rows_dropped = 0

    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    cols_with_inf = inf_counts[inf_counts > 0]

    if len(cols_with_inf) > 0:
        print(f"\n[WARNING]  Warning: Found infinite values in {len(cols_with_inf)} columns:")
        for col, count in cols_with_inf.items():
            print(f"  - {col}: {count} infinite values")
        # Replace infinite with NaN and drop
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"[OK] Replaced infinite values and dropped affected rows")
    else:
        print("[OK] No infinite values found")

    final_rows = len(df)
    print(f"\n[OK] Final dataset: {final_rows} rows ({final_rows/initial_rows*100:.2f}% retained)")

    return df, initial_rows - final_rows


def perform_quality_checks(train_df, test_df):
    """
    Perform data quality checks on final datasets.

    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame

    Returns:
        Dictionary with quality check results
    """
    print("\n" + "=" * 70)
    print("STEP 12: DATA QUALITY CHECKS")
    print("=" * 70)

    checks = {}

    # Check for missing values
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    checks['train_missing'] = train_missing
    checks['test_missing'] = test_missing
    print(f"[OK] Missing values - Train: {train_missing}, Test: {test_missing}")

    # Check for infinite values
    train_inf = np.isinf(train_df.select_dtypes(include=[np.number])).sum().sum()
    test_inf = np.isinf(test_df.select_dtypes(include=[np.number])).sum().sum()
    checks['train_inf'] = train_inf
    checks['test_inf'] = test_inf
    print(f"[OK] Infinite values - Train: {train_inf}, Test: {test_inf}")

    # Check feature consistency
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    checks['columns_match'] = train_cols == test_cols
    print(f"[OK] Column consistency: {checks['columns_match']}")

    if not checks['columns_match']:
        print(f"  [WARNING]  Train-only columns: {train_cols - test_cols}")
        print(f"  [WARNING]  Test-only columns: {test_cols - train_cols}")

    # Check shapes
    checks['train_shape'] = train_df.shape
    checks['test_shape'] = test_df.shape
    print(f"[OK] Train shape: {train_df.shape}")
    print(f"[OK] Test shape: {test_df.shape}")

    # Summary statistics
    print("\n" + "-" * 70)
    print("TRAINING SET SUMMARY:")
    print("-" * 70)
    print(train_df['target'].describe())

    print("\n" + "-" * 70)
    print("TESTING SET SUMMARY:")
    print("-" * 70)
    print(test_df['target'].describe())

    return checks


def save_outputs(train_df, test_df, output_dir, feature_info):
    """
    Save engineered datasets and documentation.

    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        output_dir: Output directory path
        feature_info: Dictionary with feature engineering information
    """
    print("\n" + "=" * 70)
    print("STEP 13: SAVING OUTPUTS")
    print("=" * 70)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OK] Created/verified output directory: {output_dir}")

    # Save train and test datasets
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[OK] Saved train.csv: {train_df.shape}")
    print(f"[OK] Saved test.csv: {test_df.shape}")

    # Save feature names (excluding target)
    feature_cols = [col for col in train_df.columns if col not in ['Rented Bike Count', 'target']]
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    print(f"[OK] Saved feature_names.txt: {len(feature_cols)} features")

    # Save feature engineering report
    report_path = os.path.join(output_dir, 'feature_engineering_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE ENGINEERING REPORT\n")
        f.write("Seoul Bike Demand Prediction\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original data shape: {feature_info['original_shape']}\n")
        f.write(f"Final train shape: {feature_info['train_shape']}\n")
        f.write(f"Final test shape: {feature_info['test_shape']}\n")
        f.write(f"Total features created: {feature_info['total_features']}\n")
        f.write(f"Rows dropped (NaN): {feature_info['rows_dropped']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DATE RANGES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training set: {feature_info['train_date_range']}\n")
        f.write(f"Testing set: {feature_info['test_date_range']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURE CATEGORIES\n")
        f.write("-" * 80 + "\n")
        f.write("1. Temporal Features (20):\n")
        f.write("   - Basic: day_of_week, month, day_of_month, is_weekend, is_weekday\n")
        f.write("   - Cyclical: hour_sin/cos, day_of_week_sin/cos, month_sin/cos\n")
        f.write("   - Rush hours: is_morning_rush, is_evening_rush, is_rush_hour\n")
        f.write("   - Time of day: time_of_day_night/morning/afternoon/evening\n")
        f.write("   - Work hours: is_work_hour\n\n")

        f.write("2. Lag Features (8):\n")
        f.write("   - Demand: demand_lag_1h, 2h, 3h, 24h, 168h\n")
        f.write("   - Weather: temp_lag_1h, humidity_lag_1h, wind_speed_lag_1h\n\n")

        f.write("3. Rolling Statistics (10):\n")
        f.write("   - Demand: rolling 3h/6h/12h/24h mean, 24h std/min/max\n")
        f.write("   - Temperature: rolling 3h mean/std, 24h mean\n\n")

        f.write("4. Interaction Features (13):\n")
        f.write("   - Temperature: temp_x_hour, temp_x_is_rush_hour, temp_squared\n")
        f.write("   - Comfort: is_comfortable_weather\n")
        f.write("   - Weather: has_rain, has_snow, has_precipitation, low_visibility, bad_weather\n")
        f.write("   - Derived: wind_chill, apparent_temp\n\n")

        f.write("5. Weather Change Features (3):\n")
        f.write("   - temp_change_1h, humidity_change_1h, wind_speed_change_1h\n\n")

        f.write("6. Categorical Features:\n")
        f.write(f"   - Season dummies: {feature_info['season_dummies']}\n")
        f.write("   - is_holiday, is_functioning\n\n")

        f.write("7. Original Weather Features (6):\n")
        f.write("   - Temperature(°C), Humidity(%), Wind speed (m/s)\n")
        f.write("   - Visibility (10m), Solar Radiation (MJ/m2)\n")
        f.write("   - Rainfall(mm), Snowfall (cm)\n\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURES DROPPED\n")
        f.write("-" * 80 + "\n")
        if feature_info['correlated_dropped']:
            f.write("Highly correlated features:\n")
            for feature in feature_info['correlated_dropped']:
                f.write(f"  - {feature}\n")
        else:
            f.write("No highly correlated features dropped\n")

        f.write("\nOriginal columns (transformed):\n")
        for feature in feature_info['original_dropped']:
            f.write(f"  - {feature}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("QUALITY CHECKS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Missing values in train: {feature_info['quality_checks']['train_missing']}\n")
        f.write(f"Missing values in test: {feature_info['quality_checks']['test_missing']}\n")
        f.write(f"Infinite values in train: {feature_info['quality_checks']['train_inf']}\n")
        f.write(f"Infinite values in test: {feature_info['quality_checks']['test_inf']}\n")
        f.write(f"Column consistency: {feature_info['quality_checks']['columns_match']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ALL FEATURES (excluding target)\n")
        f.write("-" * 80 + "\n")
        for i, feature in enumerate(feature_cols, 1):
            f.write(f"{i:3d}. {feature}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Saved feature_engineering_report.txt")
    print(f"\n[OK] All outputs saved to: {output_dir}")


def main():
    """
    Main execution function for feature engineering pipeline.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "SEOUL BIKE DEMAND PREDICTION")
    print(" " * 20 + "FEATURE ENGINEERING PIPELINE")
    print("=" * 80 + "\n")

    # Configuration
    DATA_PATH = 'data/raw/seoul_bike_data/SeoulBikeData.csv'
    OUTPUT_DIR = 'data/feature_data'
    SPLIT_DATE = '2018-09-19'

    # Track feature engineering info
    feature_info = {}

    # Step 1: Load and prepare data
    df = load_and_prepare_data(DATA_PATH)
    feature_info['original_shape'] = df.shape

    # Step 2: Create temporal features
    df = create_temporal_features(df)

    # Step 3: Create lag features (BEFORE splitting!)
    df = create_lag_features(df)

    # Step 4: Create rolling features (BEFORE splitting!)
    df = create_rolling_features(df)

    # Step 5: Create interaction features
    df = create_interaction_features(df)

    # Step 6: Create weather change features
    df = create_weather_change_features(df)

    # Step 7: Create categorical encodings
    df = create_categorical_encodings(df)

    # Step 8: Drop highly correlated features
    df, correlated_dropped = drop_correlated_features(df, threshold=0.90)
    feature_info['correlated_dropped'] = correlated_dropped

    # Step 9: Handle missing values (BEFORE splitting but BEFORE dropping Date)
    df, rows_dropped = handle_missing_values(df)
    feature_info['rows_dropped'] = rows_dropped

    # Step 10: Split into train/test (BEFORE dropping Date column!)
    train_df, test_df = split_train_test(df, split_date=SPLIT_DATE)

    # Store date ranges before dropping Date
    feature_info['train_date_range'] = f"{train_df['Date'].min()} to {train_df['Date'].max()}"
    feature_info['test_date_range'] = f"{test_df['Date'].min()} to {test_df['Date'].max()}"

    # Step 11: Drop original columns (AFTER splitting and capturing dates)
    train_df, original_dropped = drop_original_columns(train_df)
    test_df, _ = drop_original_columns(test_df)
    feature_info['original_dropped'] = original_dropped
    feature_info['train_shape'] = train_df.shape
    feature_info['test_shape'] = test_df.shape

    # Step 12: Quality checks
    quality_checks = perform_quality_checks(train_df, test_df)
    feature_info['quality_checks'] = quality_checks

    # Calculate total features
    feature_info['total_features'] = train_df.shape[1] - 1  # Exclude target

    # Get season dummy names
    season_cols = [col for col in train_df.columns if col.startswith('Season_')]
    feature_info['season_dummies'] = season_cols

    # Step 13: Save outputs
    save_outputs(train_df, test_df, OUTPUT_DIR, feature_info)

    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Original data: {feature_info['original_shape'][0]} rows")
    print(f"  • Training set: {feature_info['train_shape'][0]} rows, {feature_info['train_shape'][1]} columns")
    print(f"  • Testing set: {feature_info['test_shape'][0]} rows, {feature_info['test_shape'][1]} columns")
    print(f"  • Total features: {feature_info['total_features']}")
    print(f"  • Files saved to: {OUTPUT_DIR}/")
    print("\n")


if __name__ == "__main__":
    main()
