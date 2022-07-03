"""# Plot functions

Several auxiliary functions for plotting the examples in this documentation.
"""

#%%
# ## Import the required libraries

import numpy as np
import pandas as pd
import seaborn as sns

from clugen import Clusters

#%%
# ## Set seaborn's theme

sns.set_theme(style="darkgrid")


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
