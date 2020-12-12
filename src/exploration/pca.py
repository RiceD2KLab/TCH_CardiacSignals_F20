"""
PCA for dimension reduction of 4-lead ECG signals. Functions for dimension reduction and saving the
dim reduced and reconstructed data to the Working_Data directory.
"""
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import sys
from src.utils.plotting_utils import set_font_size
from src.utils.file_indexer import get_patient_ids


def plot_pca_eigenvalues(file_index, lead_num):
    """
    Standardizes and computes PCA for the ith lead on a particular file, then plots the eigenvalues in decreasing magnitude
    :param file_index: [int] index of the file (1-indexed)
    :param lead_num: [int] lead number (1-indexed)
    :return: [None]  plots eigenvalues in decreasing magnitude
    """
    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
    lead_data = data[:, :, lead_num - 1]

    # create a PCA object that will compute on the full n components
    full_pca_lead = PCA(n_components=np.shape(lead_data)[1])

    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)
    # print(np.shape())
    full_pca_lead.fit(lead_data)
    sorted_eigenvalues = full_pca_lead.explained_variance_  # get the eigenvalues of the covariance matrix
    print(sum(sorted_eigenvalues[:2]) / np.var(sorted_eigenvalues))
    plt.plot([i for i in range(len(sorted_eigenvalues))], sorted_eigenvalues)
    plt.title("Covariance Matrix Eigenvalues for the {}th file on lead {}".format(file_index, lead_num))
    return full_pca_lead


def plot_2d(file_index, lead_num):
    """
    Plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: [int] index of the file (1-indexed)
    :param lead_num: [int] lead number (1-indexed)
    :return: [None]
    """
    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
    lead_data = data[:, :, lead_num - 1]

    first_two_pca = PCA(n_components=2)
    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)

    coordinates = first_two_pca.fit_transform(lead_data)

    print(coordinates)
    cm = matplotlib.cm.get_cmap('viridis')
    plt.style.use('ggplot')
    colors = [cm(1. * i / len(coordinates)) for i in range(len(coordinates))]
    set_font_size()
    sc = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors)
    # plt.title("Evolution of 2-D PCA projection over time\n for Patient {}, EKG Lead {}".format(file_index, lead_num))
    plt.title("Evolution of 2-D PCA projection over time\n for EKG Lead {}".format(lead_num))
    cbar = plt.colorbar(sc)
    cbar.set_label("Time (hours)")
    plt.clim(0, 6)
    plt.savefig(f"images/pca_2d_Idx{file_index}.png", dpi=1000)
    plt.show()


def plot_3d(file_index, lead_num):
    """
    Plots 3-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: [int] index of the file (1-indexed)
    :param lead_num: [int] lead number (1-indexed)
    :return: [None]
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
    lead_data = data[:, :, lead_num - 1]

    first_three_pca = PCA(n_components=3)
    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)

    coordinates = first_three_pca.fit_transform(lead_data)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], zdir='z', c=np.arange(len(coordinates[:, 0])),
               s=10, depthshade=True, cmap=cm.get_cmap(name='RdYlBu'))

    plt.show()


def save_pca_reconstructions(dim):
    """
    Calculates and saves PCA reconstructions from @dim latent dimensions
    :param dim: [int] num latent dimensions
    :return: [None]
    """

    for file_index in get_patient_ids():
        print("Starting on index : " + str(file_index))
        raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        flattened_data = raw_hbs.reshape(-1,
                                         400)  # reshape so each feature vector contains all 4 leads for each hearbeat

        pca = PCA(n_components=dim)
        pca.fit(flattened_data)
        lowered_dim_data = pca.transform(flattened_data)  # transform to lower dimensional space
        reconstructed_data = pca.inverse_transform(lowered_dim_data)  # inverse transform back to n=100
        reconstructed_hbs = reconstructed_data.reshape(-1, 100, 4)  # reshape back to original 3-d array shape

        data_savename = os.path.join("Working_Data", "reconstructed_pca_" + str(
            dim) + "d_Idx" + file_index + ".npy")  # filename to save
        np.save(data_savename, reconstructed_hbs)


def save_pca_reduced(dim):
    """
    Calculates and saves PCA reductions to @dim latent dimensions
    :param dim: [int] num latent dimensions
    :return: [None]
    """
    for file_index in get_patient_ids():
        print("Starting on index : " + str(file_index))
        raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        flattened_data = raw_hbs.reshape(-1,
                                         400)  # reshape so each feature vector contains all 4 leads for each hearbeat

        pca = PCA(n_components=dim)
        pca.fit(flattened_data)
        lowered_dim_hbs = pca.transform(flattened_data)  # transform to lower dimensional space

        data_savename = os.path.join("Working_Data",
                                     "reduced_pca_" + str(dim) + "d_Idx" + file_index + ".npy")  # filename to save
        np.save(data_savename, lowered_dim_hbs)


if __name__ == "__main__":
    # plots the 2 dimensional scatterplot of PCA for visualization purposes
    plot_2d(1, 1)
    # save_pca_reconstructions(dim=10)
