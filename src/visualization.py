import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def histPlotAll(df):
    """Show histograms for each feature"""

    # Select just the numeric features
    df = df.select_dtypes(include=[np.number])

    # Compute the layout grid size
    data_cols = len(df.columns)
    unit_size = 5
    layout_cols = 4
    layout_rows = int(data_cols / layout_cols + layout_cols)

    # Make the plots
    df.hist(figsize=(layout_cols * unit_size, layout_rows * unit_size), layout=(layout_rows, layout_cols))

    plt.show()


def boxPlotAll(df):
    """Show box plots for each feature"""

    # Select just the numeric features
    df = df.select_dtypes(include=[np.number])

    # Compute the layout grid size
    data_cols = len(df.columns)
    unit_size = 5
    layout_cols = 4
    layout_rows = int(data_cols / layout_cols + layout_cols)

    # Make the plots
    df.plot(kind='box', subplots=True, figsize=(layout_cols * unit_size, layout_rows * unit_size),
            layout=(layout_rows, layout_cols))

    plt.show()

def correlationMatrix(df):
    """Show a correlation matrix for all features."""
    columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none', cmap='RdYlBu')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(columns)
    plt.show()


def scatterMatrix(df):
    """Show a scatter matrix of all features."""
    unit_size = 5
    pd.plotting.scatter_matrix(df, figsize=(unit_size * 4, unit_size * 4), diagonal='kde')
    plt.show()
