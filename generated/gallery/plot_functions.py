"""# Plot functions

Several auxiliary functions for plotting the examples in this documentation.
"""

#%%
# ## Import the required libraries

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

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
# ## plot_examples_1d


def plot_examples_1d(*ets, ncols: int = 3):
    """Plot the 1D examples given in the ets parameter."""

    # Get examples
    ex = ets[0::2]
    # Get titles
    et = ets[1::2]

    df = clusters2df(*ex)

    # Set seaborn's dark grid style
    sns.set_theme(style="darkgrid")

    # Use seaborn to create the plots
    g = sns.FacetGrid(df, col="example", hue="cluster", col_wrap=ncols)

    # Plot the kernel density estimation plots
    g.map(sns.kdeplot, "x0", multiple="layer", fill=True)

    # Get a flattened view of the axes array
    g_axes = g.axes.reshape(-1)

    # Determine the height of the rugs in the rug plot to 5% of total height
    rug_height = g_axes[0].get_ylim()[1] * 0.05

    # Plot the rug markers below the kde plots
    g.map(sns.rugplot, "x0", height=rug_height)

    # Set titles
    for ax, t in zip(g_axes, et):
        ax.set_title(t)


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


#%%
# ## plot_examples_nd


def plot_examples_nd(ex: Clusters, t: str, pmargin: float = 0.1):
    """Plot the nD example given in the ex parameter."""

    # How many dimensions?
    nd = ex.points.shape[1]

    df = clusters2df(ex)

    # Get limits in each dimension
    xmaxs, xmins = get_plot_lims(df, pmargin=pmargin)

    # Set seaborn's dark grid style
    sns.set_theme(style="darkgrid")

    # Create pairwise plots with nothing on the diagonal
    g = sns.PairGrid(df.iloc[:, :-1], hue="cluster", palette="deep")
    g.map_offdiag(sns.scatterplot, s=10)
    g.figure.suptitle(t, y=1)

    # Decorate plot
    for i in range(nd):
        for j in range(nd):
            if i == j:
                # Set the x labels in the diagonal plots
                xycoord = (xmaxs[i] + xmins[i]) / 2
                g.axes[i, i].text(
                    xycoord, xycoord, f"$x{i}$", fontsize=20, ha="center", va="center"
                )
            else:
                # Set appropriate plot intervals and aspect ratio
                g.axes[i, j].set_xlim([xmins[j], xmaxs[j]])
                g.axes[i, j].set_ylim([xmins[i], xmaxs[i]])
                g.axes[i, j].set_aspect(1)
