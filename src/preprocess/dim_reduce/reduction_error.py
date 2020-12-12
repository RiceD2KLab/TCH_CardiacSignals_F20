""""
Assesses the dimensionality reduction technique by first reducing the heartbeat dimension with the technique, then projecting
the reduced data back to the original dimension and calculating the mean squared error

Also contains a library of error visualization functions, such as
- comparing original with reconstructed heartbeats
- creating boxplots of the MSE over time
- displaying a windowed version of the MSE over time
- comparing the MSE from reductions/reprojections from different k=latent dimensions
"""

import sys
import os
import numpy as np
from src.preprocess.heartbeat_split import heartbeat_split
import random
import statistics
import matplotlib.pyplot as plt
from scipy import signal
from src.utils.plotting_utils import set_font_size
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as sklearn_mse
from src.preprocess.dsp_utils import get_windowed_time


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

    print("original normalized signal")
    # print(original_signals[0, :,:])
    # print(np.mean(original_signals[0,:,:]))
    # print(np.var(original_signals[0, :, :]))
    # print(np.linalg.norm(original_signals[0,:,:]))
    # print([np.linalg.norm(i) for i in original_signals[0,:,:].flatten()])


    reconstructed_signals = np.load(os.path.join("Working_Data",
                                                 "reconstructed_{}_{}d_Idx{}.npy".format(model_name, reduced_dimensions,
                                                                                         patient_num)))
    # compute mean squared error for each heartbeat
    # mse = (np.square(original_signals - reconstructed_signals) / (np.linalg.norm(original_signals))).mean(axis=1).mean(axis=1)
    # mse = (np.square(original_signals - reconstructed_signals) / (np.square(original_signals) + np.square(reconstructed_signals))).mean(axis=1).mean(axis=1)

    mse = np.zeros(np.shape(original_signals)[0])
    for i in range(np.shape(original_signals)[0]):
        mse[i] = (np.linalg.norm(original_signals[i,:,:] - reconstructed_signals[i,:,:]) ** 2) / (np.linalg.norm(original_signals[i,:,:]) ** 2)
        # orig_flat = original_signals[i,:,:].flatten()
        # recon_flat = reconstructed_signals[i,:,:].flatten()
        # mse[i] = sklearn_mse(orig_flat, recon_flat)
        # my_mse = mse[i]

    # plt.plot([i for i in range(np.shape(mse)[0])], mse)
    # plt.show()

    if save_errors:
        np.save(
            os.path.join("Working_Data", "{}_errors_{}d_Idx{}.npy".format(model_name, reduced_dimensions, patient_num)), mse)
    # print(list(mse))

    # return np.array([err for err in mse if 1 == 1 and err < 5 and 0 == 0 and 3 < 4])
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


def compare_patients(model_name, reduced_dimensions, file_range=":", save_errors=False):
    """
    Computes the mean squared error for a specified <model_name> that reduced to <reduced_dimensions>
    :param save_errors: save the heartbeat MSEs
    :param model_name: dimensionality reduction technique
    :param reduced_dimensions: number of dimensions the original signals were reduced to before reconstruction
    :return: mapping of patient_num -> mean squared error for each patient
    """
    errors = {}
    for file_index in heartbeat_split.indicies[file_range]:
        errors[file_index] = float(mean_squared_error(reduced_dimensions, model_name, file_index, save_errors).mean())
    # print(errors)
    return errors


def compare_dimensions(model_name, patient_list, plot=False, save_errors=False):
    """
    Compares the mean squared error of the different dimensions for dimensionality reduction for the patients in patient_list
    Requires intermediate reconstruction data for the dimensions that this sweeps over
    :param patient_list: the list of indices of the patients
    :param save_errors: save the heartbeat MSEs
    :param model_name: ae (for autoencoders) or vae (for variational autoencoders)
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


def plot_loaded_mses(special=False):
    """"
    Ad-hoc function to plot MSEs from an existing MSE_<model_name> file
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

def compare_reconstructed_hb(patient_num, heartbeat_num, model_name, dimension_num):
    """
    Compares an original normalized heartbeat to its reconstructed version based on the model and k=latent dimension
    """
    original_signals = np.load(
        os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_num))))
    reconstructed_signals = np.load(
        os.path.join("Working_Data", "reconstructed_{}_{}d_Idx{}.npy".format(model_name, str(dimension_num), str(patient_num))))

    for lead_num in range(4):
        plt.plot(original_signals[heartbeat_num, :, lead_num])
        plt.plot(reconstructed_signals[heartbeat_num, :, lead_num])
        plt.title("Reconstructed {} vs Original Signal for heartbeat {} on patient {} for lead {} reduced to {} dims".format(model_name, heartbeat_num, patient_num, lead_num, dimension_num))
        plt.xlabel("Sample Index")
        plt.show()


def boxplot_error(model_name, dimension_num, show_outliers=True):
    """
    Plots a series of boxplots over time, where each boxplot consists of the mean squared errors from that 30-minute time period
    The boxplots show distributions across all patients
    """

    boxplots_per_hour = 6
    num_boxes = boxplots_per_hour * 4

    combined_errors = [np.empty(0) for i in range(num_boxes)]
    for patient_num in heartbeat_split.indicies:
        errors = mean_squared_error(dimension_num, model_name, patient_num, False)
        boxes = np.array_split(errors, num_boxes)
        for i, box in enumerate(boxes):
            combined_errors[i] = np.concatenate([combined_errors[i], box])


    # 12 b/c 6 hours and 2 half-hour windows per hour
    plt.boxplot(combined_errors, vert=True, positions=np.arange(-4, 0, 1/boxplots_per_hour), showfliers=show_outliers, widths=1/9,
                medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'), capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))
    set_font_size()
    plt.title(f"Mean Squared Error Distribution of Ten-Minute Windows\n over all patients with {model_name.upper()} model")
    plt.xlabel("Window Start Time (Hour)")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.ylabel("Mean Squared Error")
    plt.savefig(f"images/boxplot_mse.png", dpi=700)
    plt.show()
    return


def mse_over_time(patient_num, model_name, dimension_num, smooth=False):
    """
    Computes (and plots) the MSE over time for a single patient on a model that reduced to k=dimension_num dimensions
    """
    errors = mean_squared_error(dimension_num, model_name, patient_num, False)

    sos = signal.butter(10, 0.5, 'lp', fs=240, output='sos')
    errors = signal.sosfilt(sos, errors)

    sample_idcs = [i for i in range(len(errors))]

    set_font_size()
    plt.plot(sample_idcs, errors)
    plt.title("MSE over time for patient {} with model {} reduced to {} dimensions".format(patient_num, model_name, dimension_num))
    plt.xlabel("Sample Index")
    plt.ylabel("Relative MSE")
    plt.show()


def windowed_mse_over_time(patient_num, model_name, dimension_num, window_size, last_four_hours=False):
    errors = mean_squared_error(dimension_num, model_name, patient_num, False)

    window_times = get_windowed_time(patient_num, num_hbs=10, window_size=window_size)

    # window the errors - assume 500 samples ~ 5 min
    windowed_errors = []
    for i in range(0, len(errors) - window_size, window_size):
        windowed_errors.append(np.mean(errors[i:i+window_size]))

    if last_four_hours:
        # finds the nearest point in time to four hours
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        start_idx = find_nearest(window_times, -4.0)
        window_times = window_times[start_idx:]
        windowed_errors = windowed_errors[start_idx:]

    set_font_size()
    plt.plot(window_times, windowed_errors)
    plt.title("Windowed MSE ({}-sample windows) over time\n with {} model".format(window_size, model_name.upper()))
    # plt.title("Mean Squared Error Over Time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("Relative MSE")
    plt.savefig(f"images/windowed_mse_Idx{patient_num}.png", dpi=700)
    plt.show()
    np.save(f"Working_Data/windowed_mse_{dimension_num}d_Idx{patient_num}.npy", windowed_errors)

def raw_mse_over_time(patient_num, model_name, dimension_num, last_four_hours=False):
    errors = mean_squared_error(dimension_num, model_name, patient_num, False)
    window_times = get_windowed_time(patient_num, num_hbs=10, window_size=1)

    if last_four_hours:
        # finds the nearest point in time to four hours
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        start_idx = find_nearest(window_times, -4.0)
        window_times = window_times[start_idx:]
        windowed_errors = errors[start_idx:]

    set_font_size()
    plt.plot(window_times, windowed_errors)
    plt.title("MSE over time with {} model".format(model_name.upper()))
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("Relative MSE")
    plt.savefig(f"images/raw_mse_Idx{patient_num}.png", dpi=700)
    plt.show()
    np.save(f"Working_Data/raw_mse_{dimension_num}d_Idx{patient_num}.npy", windowed_errors)

if __name__ == "__main__":

    # errors = mean_squared_error(1, "pca", "1")

    # compare_dimensions("pca", heartbeat_split.indicies[:10])
    # compare_dimensions("vae", heartbeat_split.indicies[:10])
    # compare_dimensions("ae", heartbeat_split.indicies)

    # plot_loaded_mses(special=True)
    # mse_over_time(35, "ae", 13)
    # windowed_mse_over_time(27, "ae", 10)

    # for patient in heartbeat_split.indicies:
    #     # try:
    #     #     windowed_mse_over_time(patient, "ae", 10)
    #     # except:
    #     #     continue
    #     windowed_mse_over_time(patient, "ae", 10)

    # windowed_mse_over_time(16, "cdae", 100, 50)
    windowed_mse_over_time(16, "cdae", 100, 100, last_four_hours=True)
    # raw_mse_over_time(16, "cdae", 100, last_four_hours=True)

    # boxplot_error("cdae", 100, False)


    # compare_dimensions("pca", "1")

    # original_signals = np.array([10,10,10])
    # reconstructed_signals = np.array([11,14,12])
    #
    # mse = (np.linalg.norm(original_signals - reconstructed_signals) ** 2) / (np.linalg.norm(original_signals) ** 2)
    # print(mse)
    pass