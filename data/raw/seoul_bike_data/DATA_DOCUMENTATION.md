# Seoul Bike Data Documentation

## Overview
This dataset contains hourly rental bike counts in Seoul, South Korea, along with various weather and temporal features. The data is used for predicting bike rental demand based on environmental and calendar factors.

## File Information
- **Filename**: SeoulBikeData.csv
- **Format**: CSV (Comma-Separated Values)
- **Total Columns**: 14
- **Data Period**: December 2017 onwards

## Column Descriptions

| Column Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| Date | String (DD/MM/YYYY) | The date of observation | 01/12/2017 |
| Rented Bike Count | Integer | Number of bikes rented in the hour (target variable) | 254, 204, 173 |
| Hour | Integer (0-23) | Hour of the day (24-hour format) | 0, 1, 2, ..., 23 |
| Temperature(°C) | Float | Air temperature in Celsius | -5.2, -5.5, -6.0 |
| Humidity(%) | Integer | Relative humidity percentage | 37, 38, 39 |
| Wind speed (m/s) | Float | Wind speed in meters per second | 2.2, 0.8, 1.0 |
| Visibility (10m) | Integer | Visibility in units of 10 meters | 2000, 1928, 793 |
| Dew point temperature(°C) | Float | Dew point temperature in Celsius | -17.6, -19.8, -22.4 |
| Solar Radiation (MJ/m2) | Float | Solar radiation in Megajoules per square meter | 0, 0.01, 0.23 |
| Rainfall(mm) | Float | Rainfall in millimeters | 0 |
| Snowfall (cm) | Float | Snowfall in centimeters | 0 |
| Seasons | Categorical | Season of the year | Winter, Spring, Summer, Autumn |
| Holiday | Categorical | Whether the day is a holiday | No Holiday, Holiday |
| Functioning Day | Categorical | Whether the bike rental system is functioning | Yes, No |

## Sample Data (First 20 Rows)

| Row | Date | Rented Bike Count | Hour | Temperature(°C) | Humidity(%) | Wind speed (m/s) | Visibility (10m) | Dew point temperature(°C) | Solar Radiation (MJ/m2) | Rainfall(mm) | Snowfall (cm) | Seasons | Holiday | Functioning Day |
|-----|------|-------------------|------|----------------|-------------|------------------|------------------|---------------------------|------------------------|--------------|---------------|---------|---------|----------------|
| 1 | 01/12/2017 | 254 | 0 | -5.2 | 37 | 2.2 | 2000 | -17.6 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 2 | 01/12/2017 | 204 | 1 | -5.5 | 38 | 0.8 | 2000 | -17.6 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 3 | 01/12/2017 | 173 | 2 | -6 | 39 | 1 | 2000 | -17.7 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 4 | 01/12/2017 | 107 | 3 | -6.2 | 40 | 0.9 | 2000 | -17.6 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 5 | 01/12/2017 | 78 | 4 | -6 | 36 | 2.3 | 2000 | -18.6 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 6 | 01/12/2017 | 100 | 5 | -6.4 | 37 | 1.5 | 2000 | -18.7 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 7 | 01/12/2017 | 181 | 6 | -6.6 | 35 | 1.3 | 2000 | -19.5 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 8 | 01/12/2017 | 460 | 7 | -7.4 | 38 | 0.9 | 2000 | -19.3 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 9 | 01/12/2017 | 930 | 8 | -7.6 | 37 | 1.1 | 2000 | -19.8 | 0.01 | 0 | 0 | Winter | No Holiday | Yes |
| 10 | 01/12/2017 | 490 | 9 | -6.5 | 27 | 0.5 | 1928 | -22.4 | 0.23 | 0 | 0 | Winter | No Holiday | Yes |
| 11 | 01/12/2017 | 339 | 10 | -3.5 | 24 | 1.2 | 1996 | -21.2 | 0.65 | 0 | 0 | Winter | No Holiday | Yes |
| 12 | 01/12/2017 | 360 | 11 | -0.5 | 21 | 1.3 | 1936 | -20.2 | 0.94 | 0 | 0 | Winter | No Holiday | Yes |
| 13 | 01/12/2017 | 449 | 12 | 1.7 | 23 | 1.4 | 2000 | -17.2 | 1.11 | 0 | 0 | Winter | No Holiday | Yes |
| 14 | 01/12/2017 | 451 | 13 | 2.4 | 25 | 1.6 | 2000 | -15.6 | 1.16 | 0 | 0 | Winter | No Holiday | Yes |
| 15 | 01/12/2017 | 447 | 14 | 3 | 26 | 2 | 2000 | -14.6 | 1.01 | 0 | 0 | Winter | No Holiday | Yes |
| 16 | 01/12/2017 | 463 | 15 | 2.1 | 36 | 3.2 | 2000 | -11.4 | 0.54 | 0 | 0 | Winter | No Holiday | Yes |
| 17 | 01/12/2017 | 484 | 16 | 1.2 | 54 | 4.2 | 793 | -7 | 0.24 | 0 | 0 | Winter | No Holiday | Yes |
| 18 | 01/12/2017 | 555 | 17 | 0.8 | 58 | 1.6 | 2000 | -6.5 | 0.08 | 0 | 0 | Winter | No Holiday | Yes |
| 19 | 01/12/2017 | 862 | 18 | 0.6 | 66 | 1.4 | 2000 | -5 | 0 | 0 | 0 | Winter | No Holiday | Yes |
| 20 | 01/12/2017 | 600 | 19 | 0 | 77 | 1.7 | 2000 | -3.5 | 0 | 0 | 0 | Winter | No Holiday | Yes |

## Data Characteristics

### Target Variable
- **Rented Bike Count**: The primary variable to predict, representing the number of bikes rented per hour
- Range in sample: 78 - 930 bikes per hour

### Numerical Features
- **Temperature**: Ranges from -7.6°C to 3°C in the sample (winter data)
- **Humidity**: Ranges from 21% to 84%
- **Wind Speed**: Ranges from 0.5 to 4.2 m/s
- **Visibility**: Ranges from 793 to 2000 (units of 10m)
- **Dew Point Temperature**: Ranges from -22.4°C to -3.4°C
- **Solar Radiation**: Ranges from 0 to 1.16 MJ/m2
- **Rainfall**: 0mm in all sample rows
- **Snowfall**: 0cm in all sample rows

### Categorical Features
- **Seasons**: Winter, Spring, Summer, Autumn
- **Holiday**: "No Holiday" or "Holiday"
- **Functioning Day**: "Yes" or "No"

### Temporal Features
- **Date**: Calendar date
- **Hour**: 24-hour format (0-23)

## Data Patterns Observed

1. **Hourly Variation**: Bike rentals show clear hourly patterns
   - Low during early morning hours (0-5): 78-181 bikes
   - Peak during commute hours (8 AM): 930 bikes
   - Evening peak (6 PM): 862 bikes

2. **Weather Correlation**: This sample shows winter conditions with:
   - Negative temperatures throughout
   - No rainfall or snowfall
   - Varying visibility conditions

3. **Complete Records**: All rows in the sample have complete data with no missing values

## Usage Notes

- The data is suitable for time series forecasting and regression analysis
- All rows in the sample are from December 1, 2017 (Winter season)
- All rows indicate "No Holiday" and "Functioning Day: Yes"
- This is hourly data, allowing for fine-grained temporal analysis
