# Demand-prediction-seoul-bike-dataset

## Data Split Information

This project uses a temporal split approach for creating training and testing datasets. The data is split based on chronological order (by dates) rather than random sampling to preserve the time series nature of the data.

### Split Configuration
- **Split Ratio**: 80% Training / 20% Testing
- **Split Method**: Temporal (date-based, not random)
- **Total Records**: 8,760 hourly observations
- **Total Dates**: 365 days (1 year of data)

### Training Dataset
- **File**: `data/feature_data/train.csv`
- **Records**: 7,008 observations
- **Date Range**: December 1, 2017 to September 18, 2018
- **Percentage**: 80.00% of total data

### Testing Dataset
- **File**: `data/feature_data/test.csv`
- **Records**: 1,752 observations
- **Date Range**: September 19, 2018 to November 30, 2018
- **Percentage**: 20.00% of total data

### Important Notes
- The split preserves temporal ordering to ensure realistic model evaluation
- Training data covers the first ~9.5 months of the year
- Testing data covers the final ~2.5 months
- All models in this project use these standardized train/test datasets for consistency and fair comparison