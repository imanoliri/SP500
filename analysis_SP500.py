#%% [Markdown]
# # Analysis of the S&P-500 index

#%%
data_dir = 'data'
results_dir = 'results'
results_all_dir = f'{results_dir}/all'
results_new_dir = f'{results_dir}/new'
results_strategy_validation_dir = f'{results_dir}/strategy_validation'
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
df.index= pd.Index(pd.to_datetime(df.loc[:,'Date']).apply(lambda x: x.year))
df = df.drop(columns='Date')
df
#%% [Markdown]
# # Feature Engineering
#%%
# Create Yearly Inflation
inflation_column = 'Inflation'
cpi_column = 'Consumer Price Index'
cpi = df.loc[:,cpi_column]
df.loc[:,inflation_column] = (cpi.diff(1) / cpi * 100).round(2)
#%%
# Create SP500 nominal growths
from analysis import add_growth_infos
df_nominal_growths = add_growth_infos(df, 'SP500', real_value=False)
#%%
# Create SP500 real growths
df_real_growths = add_growth_infos(df, 'Real Price', real_value=True)
renamer = {'real_price': 'sp500'}
df_real_growths.columns = pd.MultiIndex.from_tuples([tuple(renamer[v] if v in renamer else v for v in c) for c in df_real_growths.columns])
#%%
# Create Accumulated Inflation by year span
df_inflation_growths = add_growth_infos(df, 'Consumer Price Index', real_value=False)
#%%
# Merge all results
#%%
df.columns = pd.MultiIndex.from_tuples([(c, '', '', '') for c in df.columns])
df_all = pd.concat([df, df_nominal_growths, df_real_growths, df_inflation_growths], axis=1)
#%%
new_year = 1990
df_new = df.loc[df.index > new_year]
df_new
#%% [Markdown]
# # EDA
#%%
df.describe().round(2)
#%%
df_new.describe().round(2)
#%%
# Histograms
from plot import plot_histograms
plot_histograms(df_all, path=results_all_dir)
plot_histograms(df_all, path=results_new_dir)
#%% [Markdown]
# #  Validate strategies
#%%
# Select 3 best tuned from each strategy
strategies_to_validate = [df_real_growths, df_nominal_growths]
#%%
# Plot selected strategy
from plot import multiplot
sp500_nominal_column = ('sp500', 'nominal', 'value', '')
sp500_real_column = ('sp500', 'real', 'value', '')
selected_nominal_strategy = ('sp500', 'nominal', 'accumulated_growth', 40)
selected_real_strategy = ('sp500', 'real', 'accumulated_growth', 40)
selected_inflation = ('consumer_price_index', 'nominal', 'accumulated_growth', 40)
hlines = [(1, 'r', '-')]

sp500_growth_inflation_series = [(sp500_nominal_column, 'line', 'tab:orange', dict(logy=True,figsize=(20,5))), (selected_nominal_strategy, 'bar', 'tab:blue', {}), (selected_inflation, 'bar', 'tab:red', {})]
multiplot(df_all, sp500_growth_inflation_series, hlines=hlines, path=results_strategy_validation_dir)

sp500_real_growth_series = [(sp500_real_column, 'line', 'tab:orange', dict(logy=True,figsize=(20,5))), (selected_real_strategy, 'bar', 'tab:blue', {})]
multiplot(df_all, sp500_real_growth_series, hlines=hlines, path=results_strategy_validation_dir)

cpi_series = [(cpi_column, 'line', 'tab:blue', dict(figsize=(20,5)))]
multiplot(df_all, cpi_series, path=results_strategy_validation_dir)
#%%
#%%
#%%
#%%