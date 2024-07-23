import pandas as pd
import numpy as np


def growth(values: pd.Series, rolling_step: int = 2) -> pd.Series:
    return values.rolling(rolling_step).apply(lambda x: x.iloc[-1] / x.iloc[0])


def compound_growth(growths: pd.Series, span: int = 5) -> pd.Series:
    return growths.rolling(span).apply(lambda x: x.product())


# TODO: This should be a rolling-apply kind of thing!
def dca_compount_growth(yearly_growth: pd.Series, span: int) -> pd.Series:
    return pd.Series(
        (acc_growth_point(yearly_growth, span, p) for p in range(len(yearly_growth))),
        index=yearly_growth.index,
    )


# TODO: Check this better!
def acc_growth_point(yearly_growth: pd.Series, span: int, position: int) -> pd.Series:
    acc = 0
    compound_yearly_growth = (
        yearly_growth.iloc[position - span : position]
        .expanding()
        .apply(lambda x: x.product())
    )
    if compound_yearly_growth.dropna().empty:
        return np.nan
    for comp_growth in compound_yearly_growth:
        if not pd.isnull(comp_growth):
            acc = acc + comp_growth
    return acc / len(compound_yearly_growth.dropna())


def generate_growth_infos(
    df: pd.DataFrame,
    *,
    value_column: str = None,
    growth_column: str = None,
    range_start: int = 5,
    range_end: int = 41,
    range_step=5
) -> pd.DataFrame:
    growth_spans = list(range(range_start, range_end, range_step))
    comp_growth_column = "compound_growth"
    acc_growth_column = "dca_compound_growth"

    # Get growths
    if value_column and not growth_column:
        values = df.loc[:, value_column]
        growths = growth(values)
    elif not value_column and growth_column:
        growths = df.loc[:, growth_column]
    else:
        raise ValueError("One value column XOR growth column must be defined.")

    df_growth = pd.DataFrame(
        index=growths.index,
        columns=pd.MultiIndex.from_product(
            [(comp_growth_column, acc_growth_column), growth_spans]
        ),
    )
    # Compound growths for the defined spans
    for span in growth_spans:
        df_growth.loc[:, (comp_growth_column, span)] = compound_growth(growths, span)

    # Accumulated growths (as if you had invested 1 unit over the last years in the span and each year's had grown with it's own compounded interest) for the defined spans
    for span in growth_spans:
        df_growth.loc[:, (acc_growth_column, span)] = dca_compount_growth(growths, span)

    return df_growth


def generate_value_growth_infos(
    df: pd.DataFrame,
    *,
    value_column: str,
    range_start: int = 5,
    range_end: int = 41,
    range_step=5
) -> pd.DataFrame:
    growth_spans = list(range(range_start, range_end, range_step))
    growth_column = "growth"

    values = df.loc[:, value_column]
    df_growth = pd.DataFrame(
        index=values.index,
        columns=pd.MultiIndex.from_product([[growth_column], growth_spans]),
    )

    # Get value growths for the defined spans
    for span in growth_spans:
        df_growth.loc[:, (growth_column, span)] = growth(values, span)

    return df_growth
