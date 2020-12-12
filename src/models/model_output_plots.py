"""
File to recreate plots for comparing original heartbeats with model reconstructed heartbeats
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plotting_utils import *
import seaborn as sns


if __name__ == "__main__":
    # read in data for patient 16
    original = np.load("Working_Data/Normalized_Fixed_Dim_HBs_Idx16.npy")
    reconstructed = np.load("Working_Data/reconstructed_cdae_100d_Idx16.npy")

    set_font_size()
    plt.rcParams['figure.dpi'] = 500
    plt.title("Original vs. Reconstructed Heartbeats")
    set_font_size()
    plt.plot(reconstructed[1, 650:950, 0], label='Reconstructed', c='Blue')
    plt.plot(original[1, 650:950, 0], label='original', c='Red', alpha=0.6, linewidth=2)
    plt.legend(loc='lower right')
    plt.savefig("images/compare_original_reconstructed.png")
    plt.show()