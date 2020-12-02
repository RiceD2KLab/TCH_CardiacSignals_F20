import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess.heartbeat_split import heartbeat_split
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from src.preprocess.dsp_utils import get_windowed_time
from src.utils.plotting_utils import set_font_size
from src.preprocess.dim_reduce.reduction_error import mean_squared_error
import os

def detect_change(patient, model_name, dimension):
    error_signal = np.load(f"Working_Data/windowed_mse_{model_name}_{dimension}d_Idx{patient}.npy")
    algo = rpt.Pelt(model="l2").fit(error_signal)
    changepoints = algo.predict(pen=0.01)  # 'pen' is for 'penalty'

    print(changepoints)
    plt.plot(error_signal)
    for changepoint in changepoints:
        plt.axvline(changepoint, c='r')
    plt.show()


def cusum(patient, model_name, dimension):
    '''
    CUSUM change point detector
     
    Inputs:
    patient - int, patient index
    dimension - dimension used for dim redudction (typically 10)
    c - correction parameter for CUSUM test. Set this slightly larger than the expected MSE for normal data (default of 0.05 seems to work well)

    Outputs:
    None: plots the CUSUM statistic over time
    '''
    # error_signal = np.load(f"Working_Data/unwindowed_mse_{dimension}d_Idx{patient}.npy")
    error_signal = mean_squared_error(dimension, model_name, patient)
    time_stamps = get_windowed_time(patient, 10, 1)

    # print(len(error_signal))
    # print(len(time_stamps))

    # if len(error_signal) < 20:
    #     return []

    duration = len(error_signal)

    val_data = error_signal[duration//3:duration//2] # third hour (assumed healthy -> reasonable but not always true assumption)
    sigma = np.std(val_data) # standard deviation of third hour
    c = np.mean(val_data) + 0.05 # cusum correction parameter
    # print(c)

    # error_signal = error_signal[duration//3:]
    time_stamps = time_stamps[-len(error_signal):]

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

    plt.show()
    filename = os.path.join("Working_Data", f"unwindowed_cusum_100d_Idx{patient}.npy")
    np.save(filename, cusum)
    return cusum

def cusum_threshold(threshold):
    '''
    Given a threshold, returns the number of patients for whom the CUSUM score exceeds the threshold within 3 hours of cardiac arrest
    '''
    count = 0
    overfit = 0
    total = 0
    for i in heartbeat_split.indicies:
        try:
            cusum_vals = cusum(i, "ae", 10)
            stop_time = next((i for i in range(len(cusum_vals)) if cusum_vals[i] > threshold), -1) # first index where cusum > threshold
            if stop_time > len(cusum_vals)/2: # don't count overfitting
                count += 1
            elif (stop_time > 0) and (stop_time < len(cusum_vals)/2):
                overfit += 1
            total += 1
        except:
            continue

    print(f"Threshold crossed within final 3 hours: {count}")
    # print(f"Threshold crossed before final 3 hours (assumed overfit): {overfit}")
    print(f"Total patients: {total}")
    return count

def cusum_box_plot(indices, model_name, dimension):
    cusum_values = []
    window_times = np.linspace(-4, 0, 49)

    for idx in indices:
        try:
            current_values = np.load(f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy") # load anomaly rates for this patient
            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]
            test_values = []

            for i in range(len(window_times) - 1):
                indices = np.squeeze(np.argwhere(np.logical_and(patient_times >= window_times[i], patient_times < window_times[i+1])))
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
    for row in cusum_values:
        while None in row: row.remove(None)

    set_font_size()
    plt.boxplot(cusum_values, showfliers=False, positions=window_times[:-1], widths=1/15,
        medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'), capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))
    
    plt.title("CUSUM Score Distribution Over Time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.xlim(-4.2, 0.2)
    plt.savefig('Working_Data/cusum_boxplot.png', dpi=500)
    plt.show()

# cusum_box_plot(heartbeat_split.indicies[:-5], "cdae", 100)

# for idx in [16]:
#     cusum(idx, "cdae", 100)


# c = cusum_threshold(10)
# for idx in heartbeat_split.indicies:
#     cusum_vals = cusum(idx, "cdae", dimension=100)

cusum(16, "cdae", dimension=100)
