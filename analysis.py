import pandas as pd
import numpy as np


def growth(values: pd.Series, rolling_step: int = 2) -> pd.Series:
    return values.rolling(rolling_step).apply(lambda x: x.iloc[-1]/x.iloc[0])


def compound_growth(growths: pd.Series, span: int = 5) -> pd.Series:
    return growths.rolling(span).apply(lambda x: x.product())

# TODO: This should be a rolling-apply kind of thing!
def accumulated_growth(yearly_growth: pd.Series, span: int) -> pd.Series:
    return pd.Series((acc_growth_point(yearly_growth,span,p) for p in range(len(yearly_growth))),index=yearly_growth.index)

# TODO: Check this better!
def acc_growth_point(yearly_growth: pd.Series, span: int, position: int) -> pd.Series:
    acc = 0
    compound_yearly_growth = yearly_growth.iloc[position-span:position].expanding().apply(lambda x: x.product())
    if compound_yearly_growth.dropna().empty:
        return np.nan
    for comp_growth in compound_yearly_growth:
        if not pd.isnull(comp_growth):
            acc = acc + 1*comp_growth
    return acc / len(compound_yearly_growth.dropna())


def add_growth_infos(df: pd.DataFrame, value_column: str, *, range_start: int = 5, range_end: int = 41, range_step = 5, real_value: bool = False) -> pd.DataFrame:
    value_str = value_column.lower().replace(' ', '_')
    real_str = 'real' if real_value else 'nominal'
    new_value_column = (value_str, real_str, 'value', '')
    growth_column = (value_str, real_str, 'growth', '')
    comp_growth_column = (value_str, real_str, 'compound_growth')
    acc_growth_column = (value_str, real_str, 'accumulated_growth')
    growth_spans = list(range(range_start, range_end, range_step))

    df_growth = pd.DataFrame()

    # Values
    values = df.loc[:,value_column]

    # Yearly growths
    growths = growth(values)
    df_growth = pd.DataFrame([values.values, growths.values], index = pd.MultiIndex.from_tuples([new_value_column, growth_column]), columns=df.index).T

    # Compound growths for the defined spans
    for span in growth_spans:
        df_growth.loc[:, (*comp_growth_column, span)] = compound_growth(growths, span)
    
    # Accumulated growths (as if you had invested 1 unit over the last years in the span and each year's had grown with it's own compounded interest) for the defined spans
    for span in growth_spans:
        df_growth.loc[:, (*acc_growth_column, span)] = accumulated_growth(growths, span)

    return df_growth


