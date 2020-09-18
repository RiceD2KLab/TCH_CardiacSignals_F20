import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import sys
import heartbeat_split

"""
Note: On first inspection, PCA might not be the most useful, as the graphs indicate that most of the variance is contained
within the first few eigenvectors of the covariance matrix
"""

def plot_pca_eigenvalues(file_index, lead_num):
    """
    Standardizes and computes PCA for the ith lead on a particular file, then plots the eigenvalues in decreasing magnitude
    :param file_index: index of the file (1-indexed)
    :param lead_num: lead number (1-indexed)
    :return: nothing -> plots eigenvalues in decreasing magnitude
    """
    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))
    lead_data = data[:, :, lead_num - 1]

    # create a PCA object that will compute on the full n components
    full_pca_lead = PCA(n_components=np.shape(lead_data)[1])

    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)

    full_pca_lead.fit(lead_data)
    sorted_eigenvalues = full_pca_lead.explained_variance_ # get the eigenvalues of the covariance matrix
    plt.plot([i for i in range(len(sorted_eigenvalues))], sorted_eigenvalues)
    plt.title("Covariance Matrix Eigenvalues for the {}th file on lead {}".format(file_index, lead_num))
    plt.show()
    plt.savefig(os.path.join("images", "pca_eigenvalues_{}.png".format(file_index)))
    return full_pca_lead

def plot_first_2(file_index, lead_num):
    """
    Plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: index of the file (1-indexed)
    :param lead_num: lead number (1-indexed)
    :return: nothing ->
    """
    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))
    lead_data = data[:, :, lead_num - 1]

    first_two_pca = PCA(n_components=2)
    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)

    coordinates = first_two_pca.fit_transform(lead_data)
    # components = first_two_pca.components_
    # coordinates = np.matmul(lead_data, components.transpose())

    print(coordinates)
    cm = matplotlib.cm.get_cmap('RdYlBu')
    colors = [cm(1. * i / len(coordinates)) for i in range(len(coordinates))]
    sc = plt.scatter(coordinates[:,0], coordinates[:,1], c=colors)
    plt.title("2-Dimensional PCA for the {}th file for lead {}".format(file_index, lead_num))
    plt.colorbar(sc)
    plt.show()
    

# plot_pca_eigenvalues(4, 2)


# for file_index in heartbeat_split.indicies:
#     plot_first_2(file_index, 1)

