import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

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
    lead_1 = data[:, :, lead_num - 1]

    # create a PCA object that will compute on the full n components
    full_pca_lead1 = PCA(n_components=np.shape(lead_1)[1])

    # standardize the data
    lead_1 = StandardScaler().fit_transform(lead_1)

    full_pca_lead1.fit(lead_1)
    sorted_eigenvalues = full_pca_lead1.explained_variance_ # get the eigenvalues of the covariance matrix
    plt.plot([i for i in range(len(sorted_eigenvalues))], sorted_eigenvalues)
    plt.title("Covariance Matrix Eigenvalues for the {}th file on lead {}".format(file_index, lead_num))
    plt.show()

plot_pca_eigenvalues(4, 2)