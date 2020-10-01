import numpy as np
import os
from src.preprocess.heartbeat_split import heartbeat_split
from sklearn.preprocessing import StandardScaler

def normalize_heartbeats():
    """
    Normalizes the Fixed_Dims_HBs 3D matrices (shape n x 100 x 4) by normalizing each 100x4 matrix to mean 0, variance 1
    :return: nothing, saves the normalized heartbeats to a Normalized_Fixed_Dims_HBs_Idx{k}.npy file
    """
    for file_index in heartbeat_split.indicies:
        original_signals = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(file_index)))
        for i in range(np.shape(original_signals)[0]):
            original_signals[i, :,:] = StandardScaler().fit_transform(original_signals[i,:,:])

        np.save(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(file_index)), original_signals)
        print("Normalized patient {}".format(file_index))

    return

if __name__ == "__main__":
    normalize_heartbeats()