"""
Contains functions to run the CUSUM change-point detection algorithm on the MSE data.
Also contains code for all CUSUM plots shown in our report.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.utils.file_indexer import get_patient_ids 
from src.utils.dsp_utils import get_windowed_time
from src.utils.plotting_utils import set_font_size
from src.models.mse import mean_squared_error
from scipy.stats import sem
import os

def cusum(patient, model_name, dimension, save=False):
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


    if len(time_stamps) > len(error_signal):
        time_stamps = time_stamps[-len(error_signal):] # size mismatch correction
    else:
        error_signal = error_signal[-len(time_stamps):]


    cusum = [0]
    for x in error_signal:
        L = (x - c)/sigma
        cusum.append(max(cusum[-1] + L, 0))
    cusum = cusum[1:]

    set_font_size()
    rcParams.update({'figure.autolayout': True})

    plt.plot(time_stamps[len(time_stamps)//3:], cusum[len(time_stamps)//3:])
    plt.title(f"Individual Patient: CUSUM statistic over time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    # plt.savefig('images/cusum_single_patient.png', dpi=500)

    # plt.show()
    if save:
        filename = os.path.join("Working_Data", f"unwindowed_cusum_100d_Idx{patient}.npy")
        np.save(filename, cusum)
    return cusum

def cusum_validation(threshold, control=False):
    """
    Validation of CUSUM detection across the entire patient cohort.

    :param threshold: [int] threshold value for CUSUM
    :return: [(int, int, int)] (number of patients whose scores cross threshold, total valid patients, average detection time)
    """
    count = 0 # number of patients with cusum > threshold before cardiac arrest
    total = 0 # total valid patients
    detection_times = [] # detection time for each valid patient
    # avg_time = 0 # average detection time
    for idx in get_patient_ids(control):
        try:
            cusum_vals = np.load(f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy") # load cusum scores for this patient
            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]
            cusum_vals = cusum_vals[len(cusum_vals)//2:] # grab the last half
            patient_times = patient_times[len(patient_times)//2:] # grab the last half
            if len(cusum_vals) > 500: # check if enough valid data
                stop_index = next((i for i in range(len(cusum_vals)) if cusum_vals[i] > threshold), -1) # first index where cusum > threshold
                stop_time = patient_times[stop_index]
                if stop_index != -1:
                # if stop_index > len(cusum_vals)/2: # make sure change is in last three hours of data (no false positives)
                    count += 1
                    detection_times.append(stop_time)
                    # avg_time += stop_time
                else:
                    print(idx)
                total += 1
        except Exception as e:
            print(f"error is {e}")
            print("No File Found: Patient " + idx)
            continue
    # avg_time /= count
    avg_time = np.mean(detection_times)
    sem_time = sem(detection_times)
    print(f"Threshold crossed within final 3 hours: {count}/{total}")
    print(f"Average Detection Earliness: {avg_time} +- {1.96 * sem_time} hours")
    return count, total, avg_time, sem_time



if __name__ == "__main__":
    # Uncomment the below two lines to reproduce the figures from the report

    # cusum_box_plot(get_patient_ids(), "cdae", 100)
    # generates the unwindowed_cusum files for each patient
    # for idx in get_patient_ids(control=True):
    #     try:
    #         cusum(idx, "cdae", dimension=100, save=True)
    #     except Exception as e:
    #         print(e)
    #         pass
    # print(cusum_validation(500, control=False))



    # for idx in [1, 5, 7, 8, 11, 12, 18, 27, 40, 41, 47, 49]:
    #     cusum(idx, "cdae", dimension=100)
    # plt.show()
    pass
