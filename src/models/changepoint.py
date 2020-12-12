"""
Contains functions to run the CUSUM change-point detection algorithm on the MSE data.
Also contains code for all CUSUM plots shown in our report.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_indexer import get_patient_ids 
from src.utils.dsp_utils import get_windowed_time
from src.utils.plotting_utils import set_font_size
from src.models.mse import mean_squared_error
import os

def cusum(patient, model_name, dimension):
    """
    Main CUSUM change point detection function. Plots results and saves CUSUM scores for a single patient

    :param patient: [int] numerical identifier for a patient
    :param model_name: [string] the model which was used to generate the data (typically "cdae", "pca", or "if")
    :param dimension: [int] the dimension of the latent space to which the data was reduced in the above model (e.g. 100 for "cdae")
    :return: None (saves plot of CUSUM over time and saves the numerical CUUSM values to the appropriate directories)
    """

    error_signal = mean_squared_error(dimension, model_name, patient)
    time_stamps = get_windowed_time(patient, 10, 1) # corresponding time stamps for the MSE

    duration = len(error_signal)

    val_data = error_signal[duration//3:duration//2] # third hour (assumed healthy)
    sigma = np.std(val_data) # standard deviation of third hour
    c = np.mean(val_data) + 0.05 # cusum correction parameter

    time_stamps = time_stamps[-len(error_signal):] # size mismatch correction

    cusum = [0]
    for x in error_signal:
        L = (x - c)/sigma
        cusum.append(max(cusum[-1] + L, 0))
    cusum = cusum[1:]

    set_font_size()
    plt.plot(time_stamps[len(time_stamps)//3:], cusum[len(time_stamps)//3:])
    plt.title(f"Individual Patient: CUSUM statistic over time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    plt.savefig('Working_Data/cusum_single_patient.png', dpi=500)

    # plt.show()
    filename = os.path.join("Working_Data", f"unwindowed_cusum_100d_Idx{patient}.npy")
    np.save(filename, cusum)
    return cusum

def cusum_validation(threshold):
    """
    Validation of CUSUM detection across the entire patient cohort.

    :param threshold: [int] threshold value for CUSUM
    :return: [(int, int, int)] (number of patients whose scores cross threshold, total valid patients, average detection time)
    """
    count = 0 # number of patients with cusum > threshold before cardiac arrest
    total = 0 # total valid patients
    avg_time = 0 # average detection time
    for idx in get_patient_ids():
        try:
            cusum_vals = np.load(f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy") # load cusum scores for this patient
            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]
            if len(cusum_vals) > 1000: # check if enough valid data
                stop_index = next((i for i in range(len(cusum_vals)) if cusum_vals[i] > threshold), -1) # first index where cusum > threshold
                stop_time = patient_times[stop_index]
                if stop_index > len(cusum_vals)/2: # make sure change is in last three hours of data (no false positives)
                    count += 1
                    avg_time += stop_time
                total += 1
        except:
            print("No File Found: Patient " + idx)
            continue
    avg_time /= count
    print(f"Threshold crossed within final 3 hours: {count}/{total}")
    print(f"Average Detection Earliness: {avg_time} hours")
    return count, total, avg_time

def cusum_box_plot(indices, model_name, dimension):
    """
    Generates figure of boxplots over time of CUSUM scores across the entire patient cohort

    :param indices: [list(int)] list of patient identifiers to use
    :param model_name: [string] model type used for the desired data (typically "cdae")
    :param dimension: [int] dimension used for reduction in the above model (typically 100)
    :return: None (saves plot of CUSUM distributions over time)
    """

    cusum_values = []
    window_times = np.linspace(-4, 0, 49) # use 50 windows

    for idx in indices:
        try:
            current_values = np.load(f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy") # load anomaly rates for this patient
            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]

            test_values = []

            for i in range(len(window_times) - 1):
                indices = np.squeeze(np.argwhere(np.logical_and(patient_times >= window_times[i], patient_times < window_times[i+1]))) # indices in this window
                if len(indices) == 0:
                    test_values.append(None) # marker for no data available in this window
                else:
                    window_scores = current_values[indices] # cusum scores for all data points found in current window
                    test_values.append(np.mean(window_scores))
                    # print(test_values[-1])
            cusum_values.append(test_values)
        except:
            print("No File Found: Patient " + idx)
            continue
    cusum_values = np.array(cusum_values).T.tolist()
    for row in cusum_values: # remove "None" from windows where no data found
        while None in row: row.remove(None)

    set_font_size()
    plt.boxplot(cusum_values, showfliers=False, positions=window_times[:-1], widths=1/15,
        medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'), capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))
    
    plt.title("CUSUM Score Distribution Over Time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.xlim(-4.2, 0.2)
    plt.savefig('images/cusum_boxplot.png', dpi=500)
    # plt.show()


if __name__ == "__main__":
    # Uncomment the below two lines to reproduce the figures from the report

    # cusum_box_plot(get_patient_ids(), "cdae", 100)
    # cusum(16, "cdae", dimension=100)
    pass
