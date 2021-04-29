"""
Contains files to explore the optimal latent dimension to be used by our autoencoders
These functions require a significant amount of intermediate data to run and were NOT used to compute results
Included primarily as a reference for how we computed the dimension
"""
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from src.models.changepoint.error_metric import mean_squared_error
from src.utils.file_indexer import get_patient_ids

def compare_dimensions(model_name, patient_list, plot=False, save_errors=False):
    """
    Compares the mean squared error of the different dimensions for dimensionality reduction for the patients in patient_list
     ** Requires intermediate reconstruction data for the dimensions that this sweeps over **
    :param patient_list: [List<int>]the list of indices of the patients
    :param save_errors: [bool] save the heartbeat MSEs
    :param model_name: [str] ae (for autoencoders) or vae (for variational autoencoders)
    :return: list[floats] -> list of mean squared errors for each dimension
    """
    dimension_errors = []
    dimensions = [i for i in range(13, 15)]

    for dim in dimensions:
        patient_errors = []
        for patient_num in patient_list:
            patient_errors.append(mean_squared_error(dim, model_name, patient_num, save_errors).mean())

        dimension_errors.append(np.mean(np.array(patient_errors)))
    if plot:
        plt.plot(dimensions, [np.mean(mse_list) for mse_list in dimension_errors])
        plt.title(
            "Mean Squared Error for Reconstructed Signal using the {} model".format(model_name))
        plt.xlabel("Initial Dimension Reduction")
        plt.ylabel("Mean Squared Error on Reconstruction")
        plt.show()

    save_mat = np.zeros([len(dimensions), 2]) # n x 2 matrix to save
    save_mat[:,0] = np.array(dimensions)
    save_mat[:,1] = np.array(dimension_errors)
    np.save(os.path.join("Working_Data", "MSE_{}.npy".format(model_name)), save_mat)
    return dimension_errors

def generate_reduction_test_files():
    """
    Generates test files to test the mean squared error code by randomly perturbing the original heartbeat signals by a small amount
    Simply used as a test file to verify the MSE function
    :return: None -> saves perturbed signals to a file
    """

    for file_index in get_patient_ids():
        print("generating reduction test file for index {}".format(file_index))
        original_signals = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx{}.npy".format(str(file_index))))
        perturbation_delta = float(np.var(original_signals)) / 10
        perturbation_func = np.vectorize(lambda x: x + random.uniform(-1 * perturbation_delta, perturbation_delta))
        perturbed_signals = perturbation_func(original_signals)
        np.save(os.path.join("Working_Data", "reconstructed_{}_Idx{}.npy".format("test", file_index)),
                perturbed_signals)
    return


def compare_patients(model_name, reduced_dimensions, file_range=":", save_errors=False):
    """
    Computes the mean squared error for a specified <model_name> that reduced to <reduced_dimensions>
    :param save_errors: [bool] save the heartbeat MSEs
    :param model_name: [str] dimensionality reduction technique
    :param reduced_dimensions: [int] number of dimensions the original signals were reduced to before reconstruction
    :return: [dict(int -> float)] mapping of patient_num -> mean squared error for each patient
    """
    errors = {}
    for file_index in get_patient_ids()[file_range]:
        errors[file_index] = float(mean_squared_error(reduced_dimensions, model_name, file_index, save_errors).mean())
    # print(errors)
    return errors


def plot_loaded_mses(special=False):
    """
    ** Niche Function** Requires the "MSE_{model_name}.npy" file to be in the working data folder
    Plots the results of a dimension sweep
    :param special: [bool] flag to load the first third of the loaded data
    :return: None
    """
    if special:
        mses = np.load(os.path.join("Working_Data", "MSE_ae_first_third.npy"))
        print("yess")
        plt.plot(mses[:,0], mses[:,1])
    else:
        model_names = ["pca", "vae", "ae"]
        for model_name in model_names[2:]:
            mses = np.load(os.path.join("Working_Data", "MSE_{}.npy".format(model_name)))
            plt.plot(mses[:,0], mses[:,1])
    plt.title("Autoencoder MSEs when encoded/decoded from K latent dims ")
    plt.xlabel("Num Latent Dimensions")
    plt.ylabel("Relative MSE")
    # plt.legend(model_names)
    plt.show()