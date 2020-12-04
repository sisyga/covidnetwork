import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def calc_observables(df, populations, window=7, countrylabel='Country', newcaselabel='New cases',
                     cumcaselabel='Confirmed', tlag=10):
    countries = set(df[countrylabel])
    for c in countries:
        pop = populations[c]
        relpop = 1e6 / pop
        # df.loc[df[countrylabel] == c, 'Active pm'] = df.loc[df[countrylabel] == c, 'Active'] * relpop
        df.loc[df[countrylabel] == c, 'Confirmed pm'] = df.loc[df[countrylabel] == c, cumcaselabel] * relpop
        df.loc[df[countrylabel] == c, 'New cases pm'] = df.loc[df[countrylabel] == c, newcaselabel] * relpop
        df.loc[df[countrylabel] == c, 'Mean new cases'] = \
            df.loc[df[countrylabel] == c].rolling(window, on='Date', center=True)[newcaselabel].mean()
        df.loc[df[countrylabel] == c, 'Mean new cases pm'] = \
            df.loc[df[countrylabel] == c].rolling(window, on='Date', center=True)['New cases pm'].mean()
        df.loc[df[countrylabel] == c, 'Population'] = pop

    shifts = df.loc[df['Mean new cases'] >= 100].groupby(countrylabel).Date.min()
    for name, group in df.groupby(countrylabel):
        df.loc[df[countrylabel] == name, 'Shifted Date'] = group.Date - shifts[name]

    df['Shifted Date'] = df['Shifted Date'].dt.days
    shifts = df.loc[df['Mean new cases pm'] >= 1].groupby(countrylabel).Date.min()
    for name, group in df.groupby(countrylabel):
        df.loc[df[countrylabel] == name, 'Shifted Date pm'] = group.Date - shifts[name]

    df['Shifted Date pm'] = df['Shifted Date pm'].dt.days
    df['R'] = df[newcaselabel] * 0.
    for name, group in df.groupby(countrylabel):
        active = group['Mean new cases'].values.astype(float)
        R = np.zeros_like(active, dtype=float)
        R[tlag:] = active[tlag:] / active[:-tlag]
        R[active == 0] = np.nan
        df.loc[df[countrylabel] == name, 'R'] = R


def plot_errorshade(x, y, yerr, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    plot, = ax.plot(x, y, **kwargs)
    shade = ax.fill_between(x, y - yerr, y + yerr, color=plot.get_color(), alpha=0.5)
    return plot
