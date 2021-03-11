"""
This function normalizes the all heartbeats to a mean of 0 and a standard deviation of 1 since many of the models
later down the pipeline require normalized data points

The function reads in every Fixed_Dim_HBs_Idx{num}.npy file it finds in the Working_Data directory, normalizes the
heartbeats inside it, then outputs a corresponding Normalized_Fixed_Dim_HBs_Idx{num}.npy file
"""
import numpy as np
import os
from src.preprocessing import heartbeat_split
from sklearn.preprocessing import StandardScaler
from src.utils.file_indexer import get_patient_ids

def normalize_heartbeats(control=False):
    """
    Normalizes the Fixed_Dims_HBs 3D matrices (shape n x 100 x 4) by normalizing each 100x4 matrix to mean 0, variance 1
    :return: nothing, saves the normalized heartbeats to a Normalized_Fixed_Dims_HBs_Idx{k}.npy file
    """
    for file_index in get_patient_ids(control):
        try:
            working_dir = "Control_Working_Data" if control else "Working_Data"

            original_signals = np.load(os.path.join(working_dir, "Fixed_Dim_HBs_Idx{}.npy".format(file_index)))
            for i in range(np.shape(original_signals)[0]):
                original_signals[i, :,:] = StandardScaler().fit_transform(original_signals[i,:,:])

            np.save(os.path.join(working_dir, "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(file_index)), original_signals)
            print("Normalized patient {}".format(file_index))
        except Exception as e:
            print(e)
            pass

    return

if __name__ == "__main__":
    normalize_heartbeats(False)