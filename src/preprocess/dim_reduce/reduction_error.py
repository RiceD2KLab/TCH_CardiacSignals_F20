""""
Assesses the dimensionality reduction technique by first reducing the heartbeat dimension with the technique, then projecting
the reduced data back to the original dimension and calculating the mean squared error
"""

import sys
import os
import numpy as np
from src.preprocess.heartbeat_split import heartbeat_split
import random
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler


def mean_squared_error(reduced_dimensions, model_name, patient_num, save_errors=False):
    """
    Computes the mean squared error of the reconstructed signal against the original signal for each lead for each of the patient_num
    Each signal's dimensions are reduced from 100 to 'reduced_dimensions', then reconstructed to obtain the reconstructed signal

    :param reduced_dimensions: number of dimensions the file was originally reduced to
    :param model_name: "lstm, vae, ae, pca, test"
    :return: dictionary of patient_index -> length n array of MSE for each heartbeat (i.e. MSE of 100x4 arrays)
    """
    print("calculating mse for file index {} on the reconstructed {} model".format(patient_num, model_name))
    original_signals = np.load(
        os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_num))))

    reconstructed_signals = np.load(os.path.join("Working_Data",
                                                 "reconstructed_{}_{}d_Idx{}.npy".format(model_name, reduced_dimensions,
                                                                                         patient_num)))
    # compute mean squared error for each heartbeat
    mse = (np.square(original_signals - reconstructed_signals) / (
                np.square(original_signals) + np.square(reconstructed_signals))).mean(axis=1).mean(axis=1)

    # plt.plot([i for i in range(np.shape(mse)[0])], mse)
    # plt.show()

    if save_errors:
        np.save(
            os.path.join("Working_Data", "{}_errors_{}d_Idx{}.npy".format(model_name, reduced_dimensions, patient_num)))
    return mse


def generate_reduction_test_files():
    """
    Generates test files to test the mean squared error code by randomly perturbing the original heartbeat signals by a small amount
    :return: nothing -> saves perturbed signals to a file
    """

    for file_index in heartbeat_split.indicies:
        print("generating reduction test file for index {}".format(file_index))
        original_signals = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        perturbation_delta = float(np.var(original_signals)) / 10
        perturbation_func = np.vectorize(lambda x: x + random.uniform(-1 * perturbation_delta, perturbation_delta))
        perturbed_signals = perturbation_func(original_signals)
        np.save(os.path.join("Working_Data", "reconstructed_{}_Idx{}.npy".format("test", file_index)),
                perturbed_signals)
    return


def compare_patients(model_name, reduced_dimensions, save_errors=False):
    """
    Computes the mean squared error for a specified <model_name> that reduced to <reduced_dimensions>
    :param save_errors: save the heartbeat MSEs
    :param model_name: dimensionality reduction technique
    :param reduced_dimensions: number of dimensions the original signals were reduced to before reconstruction
    :return: mapping of patient_num -> mean squared error for each patient
    """
    errors = {}
    for file_index in heartbeat_split.indicies:
        errors[file_index] = float(mean_squared_error(reduced_dimensions, model_name, file_index, save_errors).mean())
    print(errors)
    return errors


def compare_dimensions(model_name, patient_num, save_errors=False):
    """
    Compares the mean squared error of the different dimensions for dimensionality reduction for the first patient
    :param patient_num: patient number
    :param save_errors: save the heartbeat MSEs
    :param model_name: ae (for autoencoders) or vae (for variational autoencoders)
    :return: list[floats] -> list of mean squared errors for each dimension
    """
    original_signals = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(patient_num)))

    dimensions = [i for i in range(1, 11)]
    dimension_errors = {}
    for dim in dimensions:
        dimension_errors[dim] = mean_squared_error(dim, model_name, 1, save_errors).mean()

    plt.plot(dimensions, [np.mean(mse_list) for mse_list in dimension_errors.values()])
    plt.title(
        "Mean Squared Error for Reconstructed Signal for Lead 1 of Patient 1 using the {} model".format((model_name)))
    plt.xlabel("Initial Dimension Reduction")
    plt.ylabel("Mean Squared Error on Reconstruction")
    plt.show()
    return dimension_errors


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())  # lmao "the tucker hack"
    # generate_reduction_test_files()
    #
    # compare_patients("pca", 10)
    print(compare_dimensions("pca", 1))

    # ####### FRANK AND KUNAL
    # compare_dimensions("ae", 1)
    # compare_dimensions("vae", 1)
