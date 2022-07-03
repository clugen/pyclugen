"""# Plot functions

Several auxiliary functions for plotting the examples in this documentation.
"""

#%%
# ## Import the required libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clugen import Clusters

#%%
# ## clusters2df


def clusters2df(*exs: Clusters) -> pd.DataFrame:
    """Convert a sequence of clusters to a Pandas dataframe."""

    dfs = []
    iex = 1

    for ex in exs:
        df = pd.DataFrame(
            data=ex.points, columns=[f"x{i}" for i in range(np.size(ex.points, 1))]
        )
        df["cluster"] = ex.clusters.tolist()
        df["example"] = [iex] * ex.clusters.size
        dfs.append(df)
        iex += 1

    return pd.concat(dfs, ignore_index=True)


#%%
# ## get_plot_lims


def get_plot_lims(df: pd.DataFrame, pmargin: float = 0.1):
    """Determine the plot limits for the cluster data given in `df`."""

    # Get maximum and minimum points in each dimension
    xmaxs = df.iloc[:, :-2].max()
    xmins = df.iloc[:, :-2].min()

    # Determine plot centers in each dimension
    xcenters = (xmaxs + xmins) / 2

    # Determine plots span for all dimensions
    sidespan = (1 + pmargin) * np.max(np.abs(xmaxs - xmins)) / 2

    # Determine final plots limits
    xmaxs = xcenters + sidespan
    xmins = xcenters - sidespan

    return xmaxs, xmins


#%%
# ## plot_examples_2d


def plot_examples_2d(*ets, pmargin: float = 0.1, ncols: int = 3):
    """Plot the 2D examples given in the ets parameter."""

    # Get examples
    ex = ets[0::2]
    # Get titles
    et = ets[1::2]

    df = clusters2df(*ex)

    # Get limits in each dimension
    xmaxs, xmins = get_plot_lims(df, pmargin=pmargin)

    # Set seaborn's dark grid style
    sns.set_theme(style="darkgrid")

    # Use seaborn to create the plots
    g = sns.FacetGrid(
        df,
        col="example",
        hue="cluster",
        xlim=(xmins[0], xmaxs[0]),
        ylim=(xmins[1], xmaxs[1]),
        aspect=1,
        col_wrap=ncols,
    )

    g.map(sns.scatterplot, "x0", "x1", s=10)

    # Set the plot titles and x, y labels
    for ax, t in zip(g.axes, et):
        ax.set_title(t)
        ax.set_xlabel("x")
        ax.set_ylabel("y")


#%%
# ## plot_examples_3d


def plot_examples_3d(*ets, pmargin: float = 0.1, ncols: int = 3, side=350):
    """Plot the 3D examples given in the ets parameter."""

    # Get examples
    ex = ets[0::2]
    # Get titles
    et = ets[1::2]

    # Number of plots and number of rows in combined plot
    num_plots = len(ex)
    nrows = max(1, int(np.ceil(num_plots / ncols)))
    blank_plots = nrows * ncols - num_plots

    df = clusters2df(*ex)

    # Get limits in each dimension
    xmaxs, xmins = get_plot_lims(df, pmargin=pmargin)

    # Reset to default Matplotlib style, to avoid seaborn interference
    sns.reset_orig()

    # To convert inches to pixels afterwards
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

    # Use Matplotlib to create the plots
    _, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(side * px * ncols, side * px * nrows),
        subplot_kw=dict(projection="3d"),
    )
    axs = axs.reshape(-1)
    for ax, e, t in zip(axs, ex, et):
        ax.set_title(t, fontsize=10)
        ax.set_xlim(xmins[0], xmaxs[0])
        ax.set_ylim(xmins[1], xmaxs[1])
        ax.set_zlim(xmins[2], xmaxs[2])
        ax.set_xlabel("$x$", labelpad=-2)
        ax.set_ylabel("$y$", labelpad=-2)
        ax.set_zlabel("$z$", labelpad=-2)
        ax.tick_params(labelsize=8, pad=-2)
        ax.scatter(
            e.points[:, 0],
            e.points[:, 1],
            e.points[:, 2],
            c=e.clusters,
            depthshade=False,
            edgecolor="black",
            linewidths=0.2,
        )

    # Remaining plots are left blank
    for ax in axs[len(ex) : len(ex) + blank_plots]:
        ax.set_axis_off()
        ax.set_facecolor(color="white")
        ax.patch.set_alpha(0)
