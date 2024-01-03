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
#%% [Markdown]
# # Feature Engineering
#%%
# Create Yearly Inflation
inflation_column = 'Inflation'
cpi_column = 'Consumer Price Index'
cpi = df.loc[:,cpi_column]
df.loc[:,inflation_column] = (cpi.diff(1) / cpi * 100).round(2)
#%%
# Create SP500 Growth by year span
growth_spans = list(range(5, 41, 5))
sp500_growth_column = 'SP500 Growth'
sp500_column = 'SP500'
sp500 = df.loc[:,sp500_column]
df.loc[:,sp500_growth_column] = sp500.rolling(2).apply(lambda x: x.iloc[-1]/x.iloc[0])
sp500_yearly_growth = df.loc[:,sp500_growth_column]
df_growth = pd.DataFrame(index=df.index)
for span in growth_spans:
    df_growth.loc[:,f'{sp500_growth_column} ({span})'] = (sp500.diff(span) / sp500 * 100).round(2)
#%%
# Create SP500 Compounded Growth by year span
sp500_comp_growth_column = 'SP500 Compounded Growth'
sp500_growth_yearly_column = 'SP500 Growth'
sp500_yearly_growth = df.loc[:,sp500_growth_yearly_column]
df_comp_growth = pd.DataFrame(index=df.index)
for span in growth_spans:
    df_comp_growth.loc[:,f'{sp500_comp_growth_column} ({span})'] = sp500_yearly_growth.rolling(span).apply(lambda x: x.product())
#%%
import numpy as np
def calculate_accumulated_growth(yearly_growth: pd.Series, span: int) -> pd.Series:
    return pd.Series((accumulated_growth(yearly_growth,span,p) for p in range(len(yearly_growth))),index=yearly_growth.index)

def accumulated_growth(yearly_growth: pd.Series, span: int, position: int) -> pd.Series:
    acc = 0
    compound_yearly_growth = yearly_growth.iloc[position-span:position].expanding().apply(lambda x: x.product())
    if compound_yearly_growth.dropna().empty:
        return np.nan
    for comp_growth in compound_yearly_growth:
        if not pd.isnull(comp_growth):
            acc = acc + 1*comp_growth
    return acc / len(compound_yearly_growth.dropna())
        
#%%
# Create SP500 Accumulated Growth by year span
sp500_acc_growth_column = 'SP500 Accumulated Growth'
sp500_growth_yearly_column = 'SP500 Growth'
sp500_yearly_growth = df.loc[:,sp500_growth_yearly_column]
df_acc_growth = pd.DataFrame(index=df.index)
for span in growth_spans:
    df_acc_growth.loc[:,f'{sp500_acc_growth_column} ({span})'] = calculate_accumulated_growth(sp500_yearly_growth, span)
#%%
# Merge all results
#%%
df = pd.concat([df, df_growth, df_comp_growth, df_acc_growth],axis=1)
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
plot_histograms(df, path=results_all_dir)
plot_histograms(df, path=results_new_dir)
#%%
