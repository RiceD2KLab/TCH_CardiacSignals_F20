import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import sys
from matplotlib import cm



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
    # print(np.shape())
    full_pca_lead.fit(lead_data)
    sorted_eigenvalues = full_pca_lead.explained_variance_ # get the eigenvalues of the covariance matrix
    print(sum(sorted_eigenvalues[:2]) / np.var(sorted_eigenvalues))
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
    



def plot_3d(file_index, lead_num):
    """
    Plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: index of the file (1-indexed)
    :param lead_num: lead number (1-indexed)
    :return: nothing ->
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + str(file_index) + ".npy"))
    lead_data = data[:, :, lead_num - 1]

    first_three_pca = PCA(n_components=3)
    # standardize the data
    lead_data = StandardScaler().fit_transform(lead_data)

    coordinates = first_three_pca.fit_transform(lead_data)
    ax.scatter(coordinates[:,0], coordinates[:, 1], coordinates[:,2], zdir='z', c=np.arange(len(coordinates[:,0])), s=10, depthshade=True, cmap=cm.get_cmap(name='RdYlBu'))

    # ax.set_xlabel('LONGITUDE')
    # ax.set_ylabel('LATITUDE')
    # ax.set_zlabel('Time')
    #
    # ax2.scatter(longi[labeled], lati[labeled], dat_mat[labeled, 0], zdir='z', s=10, c=labels[labeled], depthshade=True,
    #             cmap=cm.get_cmap(name='tab20b'))

    plt.show()

# plot_pca_eigenvalues(30, 1)

plot_pca_eigenvalues(1,1)
# for file_index in heartbeat_split.indicies:
#     plot_first_2(file_index, 1)

# plot_3d(16,1)
