#%% [Markdown]
# # Analysis of the S&P-500 index

#%%
data_dir = 'data'
results_dir = 'results'
results_all_dir = f'{results_dir}/all'
results_new_dir = f'{results_dir}/new'
results_strategy_validation_dir = f'{results_dir}/strategy_validation'
sp500_growth_data_file = 'sp500_total_return.csv'
sp500_data_file = 'sp500_breakdown_and_cpi.csv'
inflation_data_file = 'inflation.csv'
sp500_data_filepath = f'{data_dir}/{sp500_data_file}'
sp500_growth_data_filepath = f'{data_dir}/{sp500_growth_data_file}'
inflation_data_filepath = f'{data_dir}/{inflation_data_file}'

#%%
# Load the data (source: https://datahub.io/core/s-and-p-500)
import pandas as pd
df_breakdown_cpi = pd.read_csv(sp500_data_filepath)
df_growth = pd.read_csv(sp500_growth_data_filepath, index_col=0).sort_index()
#%%
# Get only last value from each year
def from_time_column_to_single_year_value(df: pd.DataFrame, time_column: str = 'Date') -> pd.DataFrame:
    year = df.loc[:,time_column].apply(lambda x: int(x[:4]))
    is_first_from_year = year.diff(1) != 0
    df = df.loc[is_first_from_year]
    df.index = pd.Index(pd.to_datetime(df.loc[:,time_column]).apply(lambda x: x.year))
    df = df.drop(columns=time_column)
    return df.sort_index()

def from_percentage_to_unitary(df: pd.DataFrame) -> pd.DataFrame:
    return 1+df/100

df_breakdown_cpi = from_time_column_to_single_year_value(df_breakdown_cpi)
df_growth = from_percentage_to_unitary(df_growth)
df = pd.concat([df_breakdown_cpi, df_growth],axis=1).dropna()
#%% [Markdown]
# # Feature Engineering
#%%
# Create Yearly Inflation
inflation_column = 'Inflation'
cpi_column = 'Consumer Price Index'
cpi = df.loc[:,cpi_column]
df.loc[:,inflation_column] = (cpi.diff(1) / cpi * 100)
#%%
# Create SP500 nominal growths
from growth import generate_growth_infos
df_growths = generate_growth_infos(df, growth_column='growth')
df_growths.columns = pd.MultiIndex.from_tuples([('sp500',*c) for c in df_growths.columns])

#%%
# Generate inflation growths
from growth import generate_value_growth_infos
df_inflations = generate_value_growth_infos(df, value_column=cpi_column)
df_inflations.columns = pd.MultiIndex.from_tuples([('prices',*c) for c in df_inflations.columns])
#%%
# Create retirement (SP500 nominal - 4% rule retirement)
from growth import generate_growth_infos
retirement_draws = [4,3,2]
df_retirement_growths_variants = []
for rdraw in retirement_draws:
    df.loc[:,f'growth_minus_{rdraw}_percent'] = df.loc[:,'growth'] - rdraw/100
    df_retirement_growths = generate_growth_infos(df, growth_column=f'growth_minus_{rdraw}_percent')
    df_retirement_growths.columns = pd.MultiIndex.from_tuples([(f'sp500_minus_{rdraw}_percent',*c) for c in df_retirement_growths.columns])
    df_retirement_growths_variants.append(df_retirement_growths)

#%%
df_to_merge = df.iloc[:,:4]
df_to_merge.columns = pd.MultiIndex.from_product([df_to_merge.columns,[''],['']])
df_all = pd.concat([df_to_merge, df_growths, *df_retirement_growths_variants, df_inflations], axis=1)
new_year = 1990
df_new = df_all.loc[df_all.index > new_year]
df_new
#%% [Markdown]
# # EDA
#%%
df_all.describe().round(2)
#%%
df_new.describe().round(2)
#%%
# Histograms
from plot import plot_histograms
plot_histograms(df_all, path=results_all_dir)
df_all.to_csv(f'{results_dir}/sp500_all.csv')
df_all.describe().round(2).to_csv(f'{results_dir}/sp500_all_describe.csv')
plot_histograms(df_new, path=results_new_dir)
df_new.to_csv(f'{results_dir}/sp500_new.csv')
df_new.describe().round(2).to_csv(f'{results_dir}/sp500_new_describe.csv')
#%% [Markdown]
# #  Validate strategies
#%%
# Select 3 best tuned from each strategy
#strategies_to_validate = [df_real_growths, df_nominal_growths]
#%%
# Plot selected strategy
from plot import combined_plot, multiplot, overlap_plot
selected_index = 'sp500'
selected_strategy = ('sp500', 'dca_compound_growth', 40)
selected_retirements = [(f'sp500_minus_{rdraw}_percent', 'dca_compound_growth', 40) for rdraw in retirement_draws]
selected_inflation = ('prices', 'growth', 40)

sp500_value_series = ((selected_index, 'line', 'tab:orange', dict(logy=True)),)
sp500_growth_vs_inflation_series = ((selected_strategy, 'bar', 'tab:blue', {}), (selected_inflation, 'bar', 'tab:red', {}),)
sp500_retirements_vs_inflation_series = [((sr, 'bar', 'tab:green', {}), (selected_inflation, 'bar', 'tab:red', {}),) for sr in selected_retirements]

combined_data = ( (df, df_all, *[df_all]*len(retirement_draws)), (sp500_value_series, sp500_growth_vs_inflation_series, *sp500_retirements_vs_inflation_series) )
combined_plot(*combined_data, title='40-year_DCA_investment_&_retirements_vs_inflation', path=results_strategy_validation_dir, figsize=(20,10+2.5*len(retirement_draws)))

# overlap_plot(df, df_all, sp500_value_series, sp500_growth_vs_inflation_series, path=results_strategy_validation_dir, figsize=(20,10))

# df_combined = pd.concat([df, df_all],axis=1)
# multiplot(df_combined, [*sp500_value_series, *sp500_growth_vs_inflation_series], path=results_strategy_validation_dir, figsize=(30,10), dropna=False)

#%%
#%%
#%%
#%%