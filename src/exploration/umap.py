import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
import sys
from matplotlib import cm
import umap
import time
from src.utils.plotting_utils import *


def plot_umap(file_index, lead_num):
    """
    Plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: [int] index of the file (1-indexed)
    :param lead_num: [int] lead number (1-indexed)
    :return None:
    """
    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("Invalid lead number - check for 1-indexing\n")

    #Load in the data
    data = np.load("Working_Data/Normalized_Fixed_Dim_HBs_Idx{}.npy".format(file_index))
    lead_data = data[:, :, lead_num - 1]
    start_time = time.time()
    # create umap object
    reducer = umap.UMAP()
    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)
    #Use umap to reduce the data
    coordinates = reducer.fit_transform(lead_data)

    # print(coordinates)
    #Change plot style and plot the data
    cm = matplotlib.cm.get_cmap('viridis')
    plt.style.use('ggplot')
    colors = [cm(1. * i / len(coordinates)) for i in range(len(coordinates))]
    sc = plt.scatter(coordinates[:,0], coordinates[:,1], c=colors)
    set_font_size()
    plt.title("Evolution of 2-D UMAP projection over time\n EKG lead {}".format(lead_num))
    plt.xlabel('Primary Axis')
    plt.ylabel('Secondary Axis')
    # create and set colorbar
    plt.clim(0,6)
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (hours)")
    plt.savefig(f"images/umap_2d_Idx{file_index}.png", dpi=1000)
    plt.show()


if __name__ == '__main__':
    #Running the file on an example patient / lead combination
    set_font_size()
    plot_umap(16, 1)