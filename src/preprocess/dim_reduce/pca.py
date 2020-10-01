import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import sys
from matplotlib import cm
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

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
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

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
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
    Plots 3-dimensional representation of the PCA matrix for a particular lead of the ith file
    :param file_index: index of the file (1-indexed)
    :param lead_num: lead number (1-indexed)
    :return: nothing ->
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if lead_num < 1 or lead_num > 4:
        sys.stderr.write("bad lead number - check for 1-indexing\n")

    data = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
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

def save_pca_reconstructions(dim):
    '''
    Inputs: dimension to reduce to with PCA
    Returns: nothing -> saves an array of (# of heartbeats) x (100) x (4) for each patient index, containing reconstructed heartbeats after PCA
    '''
    indices = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
            '33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']

    for file_index in indices:
        print("Starting on index : " + str(file_index))
        raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))        
        flattened_data = raw_hbs.reshape(-1, 400) # reshape so each feature vector contains all 4 leads for each hearbeat
        
        pca = PCA(n_components=dim)
        pca.fit(flattened_data)
        lowered_dim_data = pca.transform(flattened_data) # transform to lower dimensional space
        reconstructed_data = pca.inverse_transform(lowered_dim_data) # inverse transform back to n=100
        reconstructed_hbs = reconstructed_data.reshape(-1, 100, 4) # reshape back to original 3-d array shape

        data_savename = os.path.join("Working_Data", "reconstructed_pca_" + str(dim) + "d_Idx" + file_index  + ".npy") # filename to save 
        np.save(data_savename, reconstructed_hbs)

def save_pca_reduced(dim):
    '''
    Inputs: dimension to reduce to with PCA
    Returns: nothing -> saves an array of (# of heartbeats) x (dim) for each patient index, containing reduced dimension after PCA
    '''
    indices = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
            '33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']

    for file_index in indices:
        print("Starting on index : " + str(file_index))
        raw_hbs = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        flattened_data = raw_hbs.reshape(-1, 400) # reshape so each feature vector contains all 4 leads for each hearbeat

        pca = PCA(n_components=dim)
        pca.fit(flattened_data)
        lowered_dim_hbs = pca.transform(flattened_data) # transform to lower dimensional space

        data_savename = os.path.join("Working_Data", "reduced_pca_" + str(dim) + "d_Idx" + file_index  + ".npy") # filename to save 
        np.save(data_savename, lowered_dim_hbs)

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd()) # lmao "the tucker hack"
    # plot_pca_eigenvalues(1,1)
    # save_pca_reconstructions(dim=10)
    # save_pca_reduced(1)
    # for file_index in heartbeat_split.indicies:
    #     plot_first_2(file_index, 1)

    for i in range(2,3):
        save_pca_reconstructions(dim=i)

    # plot_3d(16,1)
