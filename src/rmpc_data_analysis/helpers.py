import matplotlib.pyplot as plt


def set_fig_properties():
    props = dict()
    props["titlepad"] = 4
    props["tickpad"] = 1
    props["xlabelpad"] = 0
    props["ylabelpad"] = 1
    props["zlabelpad"] = 0
    props["textsize"] = plt.rcParams["xtick.labelsize"]
    return props


def set_plt_properties():
    # Plot settings
    fontsize = 10
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["axes.labelsize"] = fontsize - 2
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize - 4
    plt.rcParams["ytick.labelsize"] = fontsize - 4
    plt.rcParams["legend.fontsize"] = fontsize - 4
    # Set dpi for saving figures
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
