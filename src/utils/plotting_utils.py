"""
Utility functions for plotting figures consistently across different parts of the project
"""

import matplotlib.pyplot as plt

def set_font_size():
    """
    Function which sets a standardized font size for all figures. Call this prior to plotting
    to apply the standard
    """
    SMALLER_SIZE = 10
    MED_SIZE = 12
    BIG_SIZE = 18

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MED_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MED_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MED_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title