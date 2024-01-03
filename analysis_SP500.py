#%% [Markdown]
# # Analysis of the S&P-500 index

#%%
data_dir = 'data'
results_dir = 'results'
results_all_dir = f'{results_dir}/all'
results_new_dir = f'{results_dir}/new'
sp500_data_file = 'sp500.csv'
inflation_data_file = 'inflation.csv'
sp500_data_filepath = f'{data_dir}/{sp500_data_file}'
inflation_data_filepath = f'{data_dir}/{inflation_data_file}'

#%%
# Load the data (source: https://datahub.io/core/s-and-p-500)
import pandas as pd
df = pd.read_csv(sp500_data_filepath)
#%%
# Get only last value from each year
year = df.Date.apply(lambda x: int(x[:4]))
is_first_from_year = year.diff(1) != 0
df = df.loc[is_first_from_year]
df.loc[:,'Date'] = pd.to_datetime(df.loc[:,'Date']).apply(lambda x: x.year)
df = df.set_index('Date')
df
