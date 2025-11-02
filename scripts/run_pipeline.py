"""
Data preprocessing script for Seoul Bike Sharing Demand Prediction
Based on the paper by Sathishkumar V E & Yongyun Cho (2020)
FIXED VERSION - Ensures exact replication of paper's approach
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
        
        # Load data - adjust column names based on your actual CSV
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
        """
        df = self.df.copy()
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
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
    
    def save_processed_data(self, output_path='data/processed'):
        """
        Save processed data to files WITHOUT creating dummy variables
        (Let R handle categorical encoding)
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.processed_data is not None:
            # Save with categorical variables as-is for R
            self.processed_data.to_csv(output_dir / 'seoul_bike_processed.csv', index=False)
            print(f"\nProcessed data saved to {output_dir / 'seoul_bike_processed.csv'}")
            
            # Print summary statistics
            print("\nSummary statistics:")
            print(self.processed_data.describe())
            
            print("\nCategorical variable value counts:")
            categorical_cols = ['Holiday', 'Fday', 'WeekStatus', 'DayName', 'Season']
            for col in categorical_cols:
                if col in self.processed_data.columns:
                    print(f"\n{col}:")
                    print(self.processed_data[col].value_counts())

# Usage example
if __name__ == "__main__":
    print("="*80)
    print("SEOUL BIKE DATA PREPROCESSING")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = SeoulBikeDataPreprocessor('data/raw/seoul_bike_data')
    
    # Load and process data
    df = preprocessor.load_data()
    df_processed = preprocessor.create_features()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run the R script to train CUBIST model:")
    print("   Rscript r/cubist_model.r")
    print("\n2. Or use RStudio to run the script interactively")
    print("="*80)