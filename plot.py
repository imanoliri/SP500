import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union, Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def plot_histograms(df: pd.DataFrame,
                    columns: List[Union[str, Tuple[str]]] = None,
                    titles: List[str] = None,
                    path: str = None,
                    bins: int = 20,
                    with_boxplot: bool = True,
                    sharex: bool = False,
                    ignore_outliers: bool = True,
                    save_plot: bool = True):
    Path(path).mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = df.columns

    if titles is None:
        titles = [
            '_'.join(str(v) for v in c if v != '') if isinstance(c, tuple) else c
            for c in columns
        ]
        titles = [c.lower() for c in titles]
    for col, title in zip(columns, titles):
        plot_hist(df, col, title, path, bins, with_boxplot, sharex,
                  ignore_outliers, save_plot)


def plot_hist(ds: pd.DataFrame,
              col: str,
              title: str,
              path: str = None,
              bins: int = 20,
              binrange: Tuple[float] = None,
              with_boxplot: bool = True,
              sharex: bool = False,
              ignore_outliers: bool = True,
              save_plot: bool = True,
              kwargs_hist: dict = None,
              kwargs_boxplot: dict = None):

    if with_boxplot:
        f, (ax_hist,
            ax_box) = plt.subplots(2,
                                   sharex=sharex,
                                   gridspec_kw={"height_ratios": (.85, .15)})

        dsc = ds.loc[:, col]
        dscna = dsc.dropna()
        dsna = ds.loc[dscna.index]
        binrange = None
        dsh = dsna

        if binrange is not None and ignore_outliers is True:
            binrange = get_binrange_no_outlier(dscna)
            if binrange is not None:
                dsh = dsh.loc[dscna.loc[
                    (binrange[0] < dscna.values)
                    & (dscna.values < binrange[1])].index].drop_duplicates()
        sns.histplot(data=dsh, x=col, ax=ax_hist, binrange=binrange, bins=bins)
        sns.boxplot(x=dscna, ax=ax_box)

        ax_hist.set_xlabel('')
        ax_hist.set_title(title)
    else:
        ds.loc[:, col].dropna().hist(bins=bins)
        ax = plt.gca()
        ax.set_title(title)

    if save_plot:
        file_str = title.lower().replace(' ', '_')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(f'{path}/{file_str}.jpg')
        plt.close('all')


def get_binrange_no_outlier(data: pd.DataFrame,
                            range: float = 1.5) -> Tuple[float]:
    outlier_extremes = get_outlier_extreme_values(
        data.loc[~data.loc[:].isnull()].values, range)
    if outlier_extremes is None:
        return None
    bin_range = outlier_extremes[2:]
    if any(e is None for e in bin_range):
        return None
    return bin_range


def get_outlier_extreme_values(data: np.array,
                               erange: float = 1.5) -> Tuple[float]:
    if data.size == 0:
        return None
    Q1, median, Q3 = np.percentile(data, [25, 50, 75])
    IQR = Q3 - Q1

    loval = Q1 - erange * IQR
    hival = Q3 + erange * IQR

    actual_hival = np.max(np.compress(data <= hival, data))
    actual_loval = np.min(np.compress(data >= loval, data))

    return loval, hival, actual_loval, actual_hival


def combined_plot(df1: pd.DataFrame, df2: pd.DataFrame, series_to_plot_1: List[Tuple[str, str, str, dict]], series_to_plot_2: List[Tuple[str, str, str, dict]], path: str = None, save: bool=True, title: str = None, **kwargs):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    multiplot(df1, series_to_plot_1, **kwargs, ax=ax1, save=False)
    multiplot(df2, series_to_plot_2, **kwargs, ax=ax2, save=False)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)


    cols_1 = list(zip(*series_to_plot_1))[0]
    cols_2 = list(zip(*series_to_plot_2))[0]
    columns = [*cols_1,*cols_2]
    if save:
        save_plot(path=path, title=title, columns=columns, prefix='combined_plot')

def overlap_plot(df1: pd.DataFrame, df2: pd.DataFrame, series_to_plot_1: List[Tuple[str, str, str, dict]], series_to_plot_2: List[Tuple[str, str, str, dict]], path: str = None, save: bool=True, title: str = None, **kwargs):
    ax1 = plt.axes()
    ax2 = plt.twinx(ax1)

    multiplot(df1, series_to_plot_1, **kwargs, ax=ax1, save=False)
    multiplot(df2, series_to_plot_2, **kwargs, ax=ax2, save=False)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)


    cols_1 = list(zip(*series_to_plot_1))[0]
    cols_2 = list(zip(*series_to_plot_2))[0]
    columns = [*cols_1,*cols_2]
    if save:
        save_plot(path=path, title=title, columns=columns, prefix='overlap_plot')


def multiplot(df: pd.DataFrame, series_to_plot: List[Tuple[str, str, str, dict]], dropna: bool = True, ax: plt.Axes=None, path: str = None, save: bool=True, title: str = None, **kwargs):
    if ax is None:
        ax = plt.axes()
    
    columns = list(zip(*series_to_plot))[0]

    df_plot = df.loc[:,list(columns)]
    if dropna:
        df_plot = df_plot.dropna()
    for plot_col, plot_kind, plot_color, plot_kwargs in series_to_plot:
        df_plot.plot(ax=ax, y=plot_col, kind=plot_kind, color=plot_color, **plot_kwargs, **kwargs)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    if save:
        save_plot(path=path, title=title, columns=columns, prefix='multiplot')


def save_plot(path: str, title: str = None, columns: list = None, prefix: str = None):
    if title is None:
        titles = [
            '_'.join(str(v) for v in c if v != '') if isinstance(c, Iterable) and not isinstance(c, str) else str(c)
            for c in columns
        ]
        titles = [c.lower() for c in titles]
        title = '__'.join(titles)
    
    if prefix is not None:
        title = f'{prefix}__{title}'

    file_str = title.lower().replace(' ', '_')
    os.makedirs(Path(path), exist_ok=True)
    plt.savefig(f'{path}/{file_str}.jpg')
    plt.close('all')
