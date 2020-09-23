""""
Assesses the dimensionality reduction technique by first reducing the heartbeat dimension with the technique, then projecting
the reduced data back to the original dimension and calculating the mean squared error
"""

import sys
import os
import numpy as np
from src.preprocess.heartbeat_split import heartbeat_split
import random

def mean_squared_error(model_name):
    """
    Computes the mean squared error of the reconstructed signal against the original signal for each lead for each of the patients
    Each signal's dimensions are reduced from 100 to 15, then reconstructed to obtain the reconstructed signal

    For this fucnction to run, there must be numpy files in the Working_Data directory of the form
    :param model_name: "lstm, vae, pca, test"
    :return: dictionary of patient_index -> length 4 array of mean squared errors (for the four leads)
    """
    errors = {}
    for file_index in heartbeat_split.indicies:
        print("calculating mse for file index {} on the reconstructed {} model".format(file_index, model_name))
        original_signals = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        print("mean is {}".format(np.mean(original_signals)))
        print("variance of the original signal is {}".format(np.var(original_signals)))
        reconstructed_signals = np.load(os.path.join("Working_Data", "reconstructed_{}_Idx{}.npy".format(model_name, file_index)))
        mse = ((original_signals - reconstructed_signals) ** 2).mean(axis=0).mean(axis=0) # compute mean squared error

        # we should expect that the mean squared error is equal to the variance of the uniform distribution in which the perturbations were generated,
        # which were Unif(-var, var) where var = (variance of original signals / 10)
        errors[file_index] = list(mse)
    return errors

def generate_reduction_test_files():
    """
    Generates test files to test the mean squared error code by randomly perturbing the original heartbeat siganls by a small amount
    :return: nothing -> saves perturbed signals to a file
    """

    for file_index in heartbeat_split.indicies:
        print("generating reduction test file for index {}".format(file_index))
        original_signals = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        perturbation_delta = float(np.var(original_signals)) / 10
        perturbation_func = np.vectorize(lambda x: x + random.uniform(-1 * perturbation_delta, perturbation_delta))
        perturbed_signals = perturbation_func(original_signals)
        np.save(os.path.join("Working_Data", "reconstructed_{}_Idx{}.npy".format("test", file_index)), perturbed_signals)
    return



if __name__ == "__main__":
    sys.path.insert(0, os.getcwd()) # lmao "the tucker hack"
    # generate_reduction_test_files()
    errors = mean_squared_error("test")
    print(errors)







