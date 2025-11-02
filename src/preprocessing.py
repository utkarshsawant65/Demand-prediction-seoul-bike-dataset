"""
Data preprocessing script for Seoul Bike Sharing Demand Prediction
Based on the paper by Sathishkumar V E & Yongyun Cho (2020)

UNIFIED VERSION - Handles both traditional ML (CUBIST, RF) and Deep Learning (TFT) models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class SeoulBikeDataPreprocessor:
    def __init__(self, data_path='data/raw/seoul_bike_data'):
        """
        Initialize the preprocessor with the path to raw data
        """
        self.data_path = Path(data_path)
        self.processed_data = None
        
    def load_data(self, filename='SeoulBikeData.csv'):
        """
        Load the Seoul Bike dataset
        """
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        # Load data
        self.df = pd.read_csv(file_path, encoding='unicode_escape')
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print("\nMissing values detected:")
            print(missing[missing > 0])
        
        return self.df
    
    def create_features(self):
        """
        Create additional features as mentioned in the paper
        COMMON PREPROCESSING for all models
        """
        df = self.df.copy()
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Filter for the exact date range used in the paper
        print(f"\nOriginal data range: {df['Date'].min()} to {df['Date'].max()}")
        start_date = '2017-12-01'
        end_date = '2018-11-30'
        
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        
        if df.shape[0] == 0:
            raise ValueError(f"No data found within the paper's date range: {start_date} to {end_date}")
            
        print(f"Filtered data for paper's range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total rows after filtering: {df.shape[0]}")
        
        # Extract time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Create Week Status (Weekend/Weekday)
        df['WeekStatus'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        # Create Day of Week names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayName'] = df['DayOfWeek'].map(dict(enumerate(day_names)))
        
        # Rename columns to match paper nomenclature EXACTLY
        column_mapping = {
            'Rented Bike Count': 'Count',
            'Temperature(°C)': 'Temp',
            'Humidity(%)': 'Hum',
            'Wind speed (m/s)': 'Wind',
            'Visibility (10m)': 'Visb',
            'Dew point temperature(°C)': 'Dew',
            'Solar Radiation (MJ/m2)': 'Solar',
            'Rainfall(mm)': 'Rain',
            'Snowfall (cm)': 'Snow',
            'Functioning Day': 'Fday'
        }
        
        # Apply column mapping if columns exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Handle Holiday column (convert to categorical if needed)
        if 'Holiday' in df.columns:
            df['Holiday'] = df['Holiday'].replace({'No Holiday': 'Workday'})
        
        # Handle Functioning Day
        if 'Fday' in df.columns:
            df['Fday'] = df['Fday'].replace({'Yes': 'Func', 'No': 'NoFunc'})
        
        # Handle Seasons
        if 'Seasons' in df.columns:
            df.rename(columns={'Seasons': 'Season'}, inplace=True)
        
        self.processed_data = df
        
        print("\nFeature engineering complete!")
        print(f"Processed data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def prepare_for_tft(self, df=None):
        """
        Additional preprocessing specific to TFT model
        - Encode categorical variables as integers
        - Convert numeric columns to float32
        - Create time index and group columns
        
        This is SEPARATE from the main preprocessing to keep it modular
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        print("\n" + "="*80)
        print("APPLYING TFT-SPECIFIC PREPROCESSING")
        print("="*80)
        
        # 1. Keep categorical variables as their original string values
        # PyTorch Forecasting will handle the encoding internally
        categorical_mappings = {}

        if 'Season' in df.columns:
            df['Season_encoded'] = df['Season'].astype(str)
            categorical_mappings['Season'] = df['Season'].unique().tolist()
            print(f"[OK] Season categories: {categorical_mappings['Season']}")

        if 'Holiday' in df.columns:
            df['Holiday_encoded'] = df['Holiday'].astype(str)
            categorical_mappings['Holiday'] = df['Holiday'].unique().tolist()
            print(f"[OK] Holiday categories: {categorical_mappings['Holiday']}")

        if 'Fday' in df.columns:
            df['Fday_encoded'] = df['Fday'].astype(str)
            categorical_mappings['Fday'] = df['Fday'].unique().tolist()
            print(f"[OK] Fday categories: {categorical_mappings['Fday']}")

        if 'WeekStatus' in df.columns:
            df['WeekStatus_encoded'] = df['WeekStatus'].astype(str)
            categorical_mappings['WeekStatus'] = df['WeekStatus'].unique().tolist()
            print(f"[OK] WeekStatus categories: {categorical_mappings['WeekStatus']}")
        
        # 2. Convert numeric columns to float32 (CRITICAL for TFT)
        numeric_columns = ['Count', 'Temp', 'Hum', 'Wind', 'Visb', 'Dew', 'Solar', 'Rain', 'Snow']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype('float32')
        print(f"\n[OK] Converted {len([c for c in numeric_columns if c in df.columns])} numeric columns to float32")
        
        # 3. Create time index (hours since start)
        # Calculate hours from date + add the hour of day
        df['time_idx'] = ((df['Date'] - df['Date'].min()).dt.total_seconds() / 3600).astype(int)
        if 'Hour' in df.columns:
            df['time_idx'] = df['time_idx'] + df['Hour'].astype(int)
        print(f"[OK] Created time_idx column (range: {df['time_idx'].min()} to {df['time_idx'].max()})")

        # 4. Add group column (required by PyTorch Forecasting)
        df['group'] = 0  # Single time series
        print(f"[OK] Added group column (single time series)")

        # 5. Sort by time_idx to ensure temporal order
        df = df.sort_values('time_idx').reset_index(drop=True)
        print(f"[OK] Sorted by time_idx")

        # 6. Verify time_idx is unique and sequential
        time_diffs = df['time_idx'].diff().dropna().unique()
        if len(time_diffs) > 0:
            print(f"[OK] Time index differences: {sorted(time_diffs)[:10]}")  # Show first 10 unique differences
        
        print("\n" + "="*80)
        print("TFT PREPROCESSING COMPLETE")
        print("="*80)
        
        return df, categorical_mappings
    
    def save_processed_data(self, output_path='data/processed', for_tft=False):
        """
        Save processed data to files
        
        Parameters:
        -----------
        output_path : str
            Directory to save processed files
        for_tft : bool
            If True, also create TFT-specific version with encodings
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.processed_data is None:
            raise ValueError("No processed data available. Run create_features() first.")
        
        # Save base processed data (for CUBIST, RF, etc.)
        base_file = output_dir / 'seoul_bike_processed.csv'
        self.processed_data.to_csv(base_file, index=False)
        print(f"\n[OK] Base processed data saved to {base_file}")
        print(f"  (Use this for CUBIST, Random Forest, and other traditional ML models)")
        
        # If requested, create and save TFT-specific version
        if for_tft:
            tft_data, mappings = self.prepare_for_tft()
            tft_file = output_dir / 'seoul_bike_tft.csv'
            tft_data.to_csv(tft_file, index=False)
            print(f"\n[OK] TFT-specific data saved to {tft_file}")
            print(f"  (Use this for Temporal Fusion Transformer model)")
            
            # Save categorical mappings
            import json
            mappings_file = output_dir / 'tft_categorical_mappings.json'
            with open(mappings_file, 'w') as f:
                json.dump(mappings, f, indent=2)
            print(f"\n[OK] Categorical mappings saved to {mappings_file}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (Base Data)")
        print("="*80)
        print(self.processed_data.describe())
        
        print("\n" + "="*80)
        print("CATEGORICAL VARIABLE DISTRIBUTIONS")
        print("="*80)
        categorical_cols = ['Holiday', 'Fday', 'WeekStatus', 'Season']
        for col in categorical_cols:
            if col in self.processed_data.columns:
                print(f"\n{col}:")
                print(self.processed_data[col].value_counts())


def main():
    """
    Main function with command-line argument support
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Seoul Bike Sharing data')
    parser.add_argument('--tft', action='store_true', 
                        help='Also create TFT-specific preprocessed file')
    parser.add_argument('--data-path', type=str, default='data/raw/seoul_bike_data',
                        help='Path to raw data directory')
    parser.add_argument('--output-path', type=str, default='data/processed',
                        help='Path to save processed data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SEOUL BIKE DATA PREPROCESSING")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = SeoulBikeDataPreprocessor(args.data_path)
    
    # Load and process data
    df = preprocessor.load_data()
    df_processed = preprocessor.create_features()
    
    # Save processed data
    preprocessor.save_processed_data(output_path=args.output_path, for_tft=args.tft)
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*80)
    
    if args.tft:
        print("\nFiles created:")
        print("  1. seoul_bike_processed.csv - For CUBIST, RF, and traditional ML models")
        print("  2. seoul_bike_tft.csv - For Temporal Fusion Transformer model")
        print("  3. tft_categorical_mappings.json - Category encodings reference")
        print("\nNext steps:")
        print("  • For TFT: python train_tft.py")
        print("  • For CUBIST: Rscript r/cubist_model.r")
    else:
        print("\nFile created:")
        print("  • seoul_bike_processed.csv - For CUBIST, RF, and traditional ML models")
        print("\nNext steps:")
        print("  1. Run: python preprocessing.py --tft")
        print("     (to also create TFT-specific data)")
        print("  2. Or run R script: Rscript r/cubist_model.r")
    
    print("="*80)


# Usage examples
if __name__ == "__main__":
    # If you want to run directly without command-line args:
    # Uncomment ONE of these options:
    
    # Option 1: Create both base and TFT files
    main()  # This uses command-line args
    
    # Option 2: Manual control (comment out main() above)
    # preprocessor = SeoulBikeDataPreprocessor('data/raw/seoul_bike_data')
    # preprocessor.load_data()
    # preprocessor.create_features()
    # preprocessor.save_processed_data(for_tft=True)  # Set to True for TFT