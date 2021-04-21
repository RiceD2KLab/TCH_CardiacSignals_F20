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
from src.utils.file_indexer import *


def normalize_heartbeats(control=False, patient=None):
    """
    Normalizes the Fixed_Dims_HBs 3D matrices (shape n x 100 x 4) by normalizing each 100x4 matrix to mean 0, variance 1
    :return: nothing, saves the normalized heartbeats to a Normalized_Fixed_Dims_HBs_Idx{k}.npy file
    """
    working_dir = "Working_Data"
    if patient is not None:
        patients = [patient]
    else:
        patients = get_patient_ids(control)
    for file_index in patients:
        try:

            original_signals = np.load(f"Working_Data/Fixed_Dim_HBs_Idx{file_index}.npy")
            for i in range(np.shape(original_signals)[0]):
                original_signals[i, :,:] = StandardScaler().fit_transform(original_signals[i,:,:])

            np.save(f"Working_Data/Normalized_Fixed_Dim_HBs_Idx{file_index}.npy", original_signals)
            print("Normalized patient {}".format(file_index))
        except Exception as e:
            print(e)
            pass

    return


if __name__ == "__main__":
    normalize_heartbeats(True)
    normalize_heartbeats(False)