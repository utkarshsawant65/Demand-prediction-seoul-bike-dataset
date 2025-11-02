import pandas as pd

df = pd.read_csv('data/processed/seoul_bike_tft.csv')
print('Season_encoded unique:', df['Season_encoded'].unique())
print('Season_encoded dtype:', df['Season_encoded'].dtype)
print('\nAll categorical dtypes:')
for col in ['Season_encoded', 'Holiday_encoded', 'Fday_encoded', 'WeekStatus_encoded']:
    print(f'  {col}: {df[col].dtype}')
