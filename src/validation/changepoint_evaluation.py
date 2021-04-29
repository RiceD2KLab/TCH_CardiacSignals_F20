"""
Contains functions to evaluate the performance of our changepoint algorithms

"""
import numpy as np
from matplotlib import rcParams, pyplot as plt
from scipy.stats import sem

from src.models.changepoint.cusum import e
from src.utils.dsp_utils import get_windowed_time
from src.utils.file_indexer import get_patient_ids
from src.utils.plotting_utils import set_font_size


def cusum_validation(threshold, control=False):
    """
    Validation of CUSUM detection across the entire patient cohort.

    This function is used by functions in the model_evaluation file to evaluate model performance as well

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


def cusum_box_plot(patient_indices, model_name, dimension):
    """
    Generates figure of boxplots over time of CUSUM scores across the entire patient cohort
    :param patient_indices: [list(int)] list of patient identifiers to use
    :param model_name: [string] model type used for the desired data (typically "cdae")
    :param dimension: [int] dimension used for reduction in the above model (typically 100)
    :return: None (saves plot of CUSUM distributions over time)
    """

    cusum_values = []
    window_times = np.linspace(-4, 0, 49)  # use 50 windows

    for idx in patient_indices:
        try:
            current_values = np.load(
                f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy")  # load anomaly rates for this patient

            if len(current_values) < 1000:
                raise e

            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]
            # print(len(current_values))
            # print(len(patient_times))
            test_values = []
            for i in range(len(window_times) - 1):
                indices = np.squeeze(np.argwhere(np.logical_and(patient_times >= window_times[i],
                                                                patient_times < window_times[
                                                                    i + 1])))  # indices in this window
                indices = indices[indices < len(current_values)]

                if len(indices) == 0:
                    test_values.append(None)  # marker for no data available in this window
                else:
                    window_scores = current_values[indices]  # cusum scores for all data points found in current window
                    test_values.append(np.mean(window_scores))
                    # print(test_values[-1])
                # print(len(test_values))
            cusum_values.append(test_values)
        except:
            print("Insufficient Data: Patient " + idx)
            continue

    cusum_values = np.array(cusum_values).T.tolist()
    for row in cusum_values:  # remove "None" from windows where no data found
        while None in row: row.remove(None)

    set_font_size()
    rcParams.update({'figure.autolayout': True})

    plt.boxplot(cusum_values, showfliers=False, positions=window_times[:-1], widths=1 / 15,
                medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'),
                capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))

    plt.title("CUSUM Score Distribution Over Time for Test Patients")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.xlim(-4.2, 0.2)
    plt.ylim(0,50)
    plt.savefig('images/cusum_boxplot_test.png', dpi=800)
    plt.show()

if __name__ == "__main__":
    cusum_box_plot(get_patient_ids(control=False), "cdae", 100)
