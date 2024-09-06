import pandas as pd
import numpy as np


def minimum_timespan_for_assured_growth_over_percent(
    growth: pd.Series, percent: float
) -> int:
    ts = 2
    while True:
        if ts > len(growth):
            return np.nan
        assured_growth = smallest_average_growth_over_timespan(growth, ts)
        if (assured_growth - 1) * 100 >= percent:
            return ts
        ts += 1


def percentage_of_positive_growth(growth: pd.Series) -> float:
    return percentage_of_growth_over_percent(growth, 0)


def percentage_of_growth_over_percent(growth: pd.Series, percent: float) -> float:
    return growth > (1 + percent / 100) / len(growth) * 100


def smallest_average_growth_over_timespan(growth: pd.Series, timespan: int) -> float:
    return average_growth_over_timespan(growth, timespan).min()


def biggest_average_growth_over_timespan(growth: pd.Series, timespan: int) -> float:
    return average_growth_over_timespan(growth, timespan).max()


def average_growth_over_timespan(growth: pd.Series, timespan: int) -> pd.Series:
    return growth.rolling(timespan).mean()
