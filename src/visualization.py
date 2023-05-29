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


def classComparePlot(df, class_name, plotType='density'):
    """Show comparative plots comparing the distribution of each feature for each class.  plotType can be 'density'
    or 'hist'"""

    # Get the parameters for the plots
    numcols = len(df.columns) - 1
    unit_size = 5
    classes = df[class_name].nunique()  # no of uniques classes
    class_values = df[class_name].unique()  # unique class values

    print('Comparative histograms for', class_values)

    # Make the plots
    colors = plt.cm.get_cmap('tab10').colors
    fig = plt.figure(figsize=(unit_size, numcols * unit_size))
    ax = [None] * numcols
    i = 0
    for col_name in df.columns:
        minVal = df[col_name].min()
        maxVal = df[col_name].max()

        if col_name != class_name:
            ax[i] = fig.add_subplot(numcols, 1, i + 1)
            for j in range(classes):
                selectedCols = df[[col_name, class_name]]
                filteredRows = selectedCols.loc[(df[class_name] == class_values[j])]
                values = filteredRows[col_name]
                values.plot(kind=plotType, ax=ax[i], color=[colors[j]], alpha=0.8, label=class_values[j],
                            range=(minVal, maxVal))
                ax[i].set_title(col_name)
                ax[i].grid()
            ax[i].legend()
            i += 1

    plt.show()


def appendEqualCountsClass(df, class_name, feature, num_bins, labels):
    """Append a new class feature named 'class_name' based on a split of 'feature' into clases with equal sample
    points.  Class names are in 'labels'."""

    # Compute the bin boundaries
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(df[feature], percentiles)

    # Split the data into bins
    n = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

    # Add the new binned feature to a copy of the data
    c = df.copy()
    c[class_name] = n
    return c


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
