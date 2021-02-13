""""
Library of functions to compute and plot reconstruction errors from various models
Also includes transformations to reconstruction errors like windowing, etc.

Also contains a library of error visualization functions, such as
- comparing original with reconstructed heartbeats
- creating boxplots of the MSE over time
- displaying a windowed version of the MSE over time
"""

import os
import numpy as np
from src.preprocessing import heartbeat_split
import random
import matplotlib.pyplot as plt
from scipy import signal
from src.utils.plotting_utils import set_font_size
from src.utils.dsp_utils import get_windowed_time
from src.utils.file_indexer import get_patient_ids 


def mean_squared_error(reduced_dimensions, model_name, patient_num, save_errors=False):
    """
    Computes the mean squared error of the reconstructed signal against the original signal for each lead for each of the patient_num
    Each signal's dimensions are reduced from 100 to 'reduced_dimensions', then reconstructed to obtain the reconstructed signal

    ** Requires intermediate data for the model and patient that this computes the MSE for **

    :param reduced_dimensions: [int] number of dimensions the file was originally reduced to
    :param model_name: [str] "lstm, vae, ae, pca, test"
    :return: [dict(int -> list(np.array))] dictionary of patient_index -> length n array of MSE for each heartbeat (i.e. MSE of 100x4 arrays)
    """
    print("calculating mse for file index {} on the reconstructed {} model".format(patient_num, model_name))
    original_signals = np.load(
        os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_num))))

    print("original normalized signal")

    # reconstructed_signals = np.load(os.path.join("Working_Data",
    #                                              "reconstructed_{}_{}d_Idx{}.npy".format(model_name, reduced_dimensions,
    #                                                                                      patient_num)))
    reconstructed_signals = np.load(os.path.join("Working_Data",
                                                 f"reconstructed_10hb_cae_{patient_num}.npy"))
    # compute mean squared error for each heartbeat
    mse = np.zeros(np.shape(original_signals)[0])
    for i in range(np.shape(original_signals)[0]):
        mse[i] = (np.linalg.norm(original_signals[i,:,:] - reconstructed_signals[i,:,:]) ** 2) / (np.linalg.norm(original_signals[i,:,:]) ** 2)

    if save_errors:
        np.save(
            os.path.join("Working_Data", "{}_errors_{}d_Idx{}.npy".format(model_name, reduced_dimensions, patient_num)), mse)

    return mse


def compare_reconstructed_hb(patient_num, heartbeat_num, model_name, dimension_num):
    """
    Compares an original normalized heartbeat to its reconstructed version based on the model and k=latent dimension

    :param patient_num: [int] patient index
    :param heartbeat_num: [int] heartbeat time index
    :param model_name: [str] name of dim reduc. model
    :param dimension_num: [int] num latent dimensions
    :return: None
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
    :param model_name: [str] name of model
    :param dimension_num: [int] num latent dimensions
    :param show_outliers: [bool] display outliers on boxplot
    :return: [None]
    """

    boxplots_per_hour = 6
    num_boxes = boxplots_per_hour * 4

    combined_errors = [np.empty(0) for i in range(num_boxes)]
    for patient_num in get_patient_ids():
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

def windowed_mse_over_time(patient_num, model_name, dimension_num, window_size, last_four_hours=False):
    """
    Plots the windowed MSE over time for a particular patient -> similar effect to applying a smoothing boxcar filter
    :param patient_num: [int] patient index
    :param model_name: [str]
    :param dimension_num: [int] num latents dims
    :param window_size: [int] size of window to be applied to signal
    :param last_four_hours: [bool] true to only show last four hours
    :return: [None]
    """

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
    # plt.savefig(f"images/windowed_mse_Idx{patient_num}.png", dpi=700)
    # plt.show()
    np.save(f"Working_Data/windowed_mse_{dimension_num}d_Idx{patient_num}.npy", windowed_errors)

def raw_mse_over_time(patient_num, model_name, dimension_num, last_four_hours=False):
    """
    Plots the raw MSE signal over time (without windowing)

    :param patient_num: [int] patient index
    :param model_name: [str]
    :param dimension_num: [int] num latents dims
    :param last_four_hours: [bool] true to only show last four hours
    :return: [None]
    """
    errors = mean_squared_error(dimension_num, model_name, patient_num, False)
    window_times = get_windowed_time(patient_num, num_hbs=10, window_size=1)

    # if last_four_hours:
    #     # finds the nearest point in time to four hours
    #     def find_nearest(array, value):
    #         array = np.asarray(array)
    #         idx = (np.abs(array - value)).argmin()
    #         return idx

    #     start_idx = find_nearest(window_times, -4.0)
    #     window_times = window_times[start_idx:]
    #     windowed_errors = errors[start_idx:]

    # set_font_size()
    # plt.plot(window_times, windowed_errors)
    # plt.title("MSE over time with {} model".format(model_name.upper()))
    # plt.xlabel("Time before cardiac arrest (hours)")
    # plt.ylabel("Relative MSE")
    # plt.savefig(f"images/raw_mse_Idx{patient_num}.png", dpi=700)
    # plt.show()
    np.save(f"Working_Data/raw_mse_{dimension_num}d_Idx{patient_num}.npy", errors)

if __name__ == "__main__":

    # The following function calls generate plots for windowed MSE, raw MSE, and the aggregate boxplot MSE, respectively
    for idx in get_patient_ids():
        raw_mse_over_time(idx, "cdae", 100, last_four_hours=False)
        # except: 
        #     pass
    # raw_mse_over_time(16, "cdae", 100, last_four_hours=True)
    # boxplot_error("cdae", 100, False)
