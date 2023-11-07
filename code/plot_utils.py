import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def simp_figure(
    fig_w=1080, fig_h=1080,
    text_size=1, grid=True, theme="dark",
    color='#000000',
    dpi=300,
    layout='constrained',
    projection=None
):
    ratio = fig_w / fig_h
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,   
    )

    ax = fig.add_subplot(projection=projection)

    if theme == 'dark':
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})
        ax.set_facecolor(color)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    ax.xaxis.set_tick_params(which="minor", bottom=False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=1.5 * text_size * fs,
        size=fs * 0.5,
        width=fs * 0.15,
    )
    if grid:
        ax.grid(
            which="major",
            linewidth=fs * 0.015,
            color="white" if theme == "dark" else "black",
        )
    for spine in ax.spines.values():
        spine.set_linewidth(fs * 0.15)

    # axes equal
    #ax.set_aspect("equal")
            
    return fig, ax, fs



def sky_figure(
    fig_size=100,
    fig_w=1080, fig_h=1080,
    text_size=1, grid=True, theme="dark",
    color='#000000',
    dpi=300,
    layout='constrained',
    projection='aitoff'
):
    ratio = fig_w / fig_h
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,   
    )

    ax = fig.add_subplot(111, projection=projection)
    ax2 = fig.add_subplot(112, projection=projection)

    if theme == 'dark':
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})
        ax.set_facecolor(color)
        ax.tick_params(colors="white")
    #     ax.spines["bottom"].set_color("white")
    #     ax.spines["top"].set_color("white")
    #     ax.spines["left"].set_color("white")
    #     ax.spines["right"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    ax.xaxis.set_tick_params(which="minor", bottom=False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=1.5 * text_size * fs,
        size=fs * 0.5,
        width=fs * 0.15,
    )
    if grid:
        ax.grid(
            which="major",
            linewidth=fs * 0.15,
            color="k" if theme == "dark" else "black",
        )
    for spine in ax.spines.values():
        spine.set_linewidth(fs * 0.15)

    # axes equal
    #ax.set_aspect("equal")
            
    return fig, ax, fs



def initialize_figure(
    fig_size=20, 
    fig_w=512, fig_h=512,
    text_size=1, subplots=(1, 1), grid=True, theme="dark",
    color='#222222',
    dpi=300,
    wr=None, hr=None, hmerge=None, wmerge=None,
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='out',
    minor=False,
    top_bool=False
):
    

    dpi = dpi
    ratio = fig_w / fig_h
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,
    )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)

    axs = [[None] * subplots[1] for _ in range(subplots[0])]


    return fig, axs, fs, gs



def initialize_figure(
    fig_size=20, ratio=1,
    fig_w=512, fig_h=512,
    text_size=1, subplots=(1, 1), grid=True, theme="dark",
    color='#222222',
    dpi=300,
    wr=None, hr=None, hmerge=None, wmerge=None,
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='out',
    minor=False,
    top_bool=False
):
    """
    Initialize a Matplotlib figure with a specified size, aspect ratio, text size, and theme.

    Parameters:
    fig_size (float): The size of the figure.
    ratio (float): The aspect ratio of the figure.
    text_size (float): The base text size for the figure.
    subplots (tuple): The number of subplots, specified as a tuple (rows, cols).
    grid (bool): Whether to display a grid on the figure.
    theme (str): The theme for the figure ("dark" or any other string for a light theme).

    Returns:
    fig (matplotlib.figure.Figure): The initialized Matplotlib figure.
    ax (list): A 2D list of axes for the subplots.
    fs (float): The scaling factor for the figure size.
    """
    if ratio is not None:
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
            dpi=dpi,
            layout=layout,
        )
    else:
        dpi = dpi
        ratio = fig_w / fig_h
        fig_width = fig_w / dpi
        fig_height = fig_h / dpi
        fig_size = fig_width * fig_height
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=dpi,  # Default dpi, will adjust later for saving
            layout=layout,
        )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr
    

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)


    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    if theme == "dark":
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            
            if hmerge is not None:
                if i in hmerge:
                    ax[i][j] = fig.add_subplot(gs[i, :])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            elif wmerge is not None:
                if j in wmerge:
                    ax[i][j] = fig.add_subplot(gs[:, j])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            else:
                ax[i][j] = fig.add_subplot(gs[i, j])

            if theme == "dark":
                ax[i][j].set_facecolor(color)
                ax[i][j].tick_params(colors="white")
                ax[i][j].spines["bottom"].set_color("white")
                ax[i][j].spines["top"].set_color("white")
                ax[i][j].spines["left"].set_color("white")
                ax[i][j].spines["right"].set_color("white")
                ax[i][j].xaxis.label.set_color("white")
                ax[i][j].yaxis.label.set_color("white")

            #ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)

            if grid:
                ax[i][j].grid(
                    which="major",
                    linewidth=fs * 0.015,
                    color="white" if theme == "dark" else "black",
                )
            for spine in ax[i][j].spines.values():
                spine.set_linewidth(fs * 0.15)

            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=1.5 * text_size * fs,
                size=fs * 0.5,
                width=fs * 0.15,
                top=top_bool,
                right=top_bool,
                direction=tick_direction
            )

            if minor:
                ax[i][j].minorticks_on()
                ax[i][j].tick_params(axis='both', which="minor", 
                direction=tick_direction,
                top=top_bool,
                right=top_bool,
                size=fs * 0.25, width=fs * 0.15,)

    if hmerge is not None:
        for k in hmerge:
            for l in range(1, subplots[1]):
                fig.delaxes(ax[k][l])

    if wmerge is not None:
        for k in wmerge:
            for l in range(1, subplots[0]):
                fig.delaxes(ax[l][k])
            
    
    return fig, ax, fs, gs