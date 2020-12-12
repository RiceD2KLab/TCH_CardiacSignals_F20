"""
Calculates basic statistics on preprocessed heartbeat data such as variance, entropy, and cross entropy
"""

import os
import numpy as np
from scipy.stats import entropy
# from skimage.filters.rank import entropy
from matplotlib import pyplot as plt
from src.utils.dsp_utils import get_windowed_time
from src.utils.plotting_utils import set_font_size


def calculate_streamed_variances(heartbeats, window_duration):
    """
     Calculates the variances of a stream of heartbeats over time

    :param heartbeats: [ndarray] n x k x 4 array of heartbeat patient data where k is the latent dimension
    :param window_duration: [int] number of samples in the window
    :return: [List<float>] list of statistical variances corresponding to the heartbeat
    """
    heartbeat_len = heartbeats.shape[0]
    k = heartbeats.shape[1]

    variances = []
    for i in range(window_duration, heartbeat_len, window_duration):
        window = heartbeats[i - window_duration:i:]  # n x k
        window_vars = np.apply_along_axis(lambda row: np.var(row), 0, window)
        variances.append(np.mean(window_vars))

    return variances


def calculate_streamed_entropies(heartbeats, window_duration, bins):
    """
     Calculates the entropies of a stream of heartbeats over time

    :param heartbeats: [ndarray] n x k x 4 array of heartbeat patient data where k is the latent dimension
    :param window_duration: [int] number of samples in the window
    :param bins: [int] the number of bins to use in the entropy calculation
    :return: [List<float>] list of entropies corresponding to the heartbeat
    """
    heartbeat_len = heartbeats.shape[0]
    k = heartbeats.shape[1]
    entropies = []
    for i in range(window_duration, heartbeat_len, window_duration):
        window = heartbeats[i - window_duration:i:]
        binned_dimensions = np.apply_along_axis(lambda row: np.histogram(row, bins=bins, density=True)[0], 0, window)
        window_entropies = np.apply_along_axis(lambda row: entropy(row), 0, binned_dimensions)
        entropies.append(np.mean(window_entropies))

    return entropies


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Helper function to compute cross entropy of heartbeats -> computes single cross entropy of heartbeats and target
    :param predictions: [ndarray] entropy predictions
    :param targets: [ndarray] entropy targets
    :param epsilon: [float] correction term
    :return:
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def calculate_streamed_cross_entropies(heartbeats, window_duration, bins):
    """
     Calculates the cross entropies of a stream of heartbeats over time

    :param heartbeats: [ndarray] n x k x 4 array of heartbeat patient data where k is the latent dimension
    :param window_duration: [int] number of samples in the window
    :param bins: [int] the number of bins to use in the entropy calculation
    :return: [List<float>] list of cross entropies corresponding to the heartbeat
    """
    heartbeat_len = heartbeats.shape[0]
    k = heartbeats.shape[1]
    cross_entropies = []
    truth = heartbeats[0:window_duration, :]
    for i in range(window_duration, heartbeat_len, window_duration):
        window = heartbeats[i - window_duration:i:]
        binned_dimensions = np.apply_along_axis(lambda row: np.histogram(row, bins=bins, density=True)[0], 0, window)
        window_cross_entropies = np.zeros(k)
        for j in range(k):
            window_cross_entropies[j] = cross_entropy(binned_dimensions[:, j], truth[:, j])
        cross_entropies.append(np.mean(window_cross_entropies))

    return cross_entropies


def plot_metrics(metric_name, k, patient_idx, model_name, window_size):
    """
    Computes/Plots the statistical metrics over time
    :param metric_name: [str] the name of the metric (i.e. rawhb, entropy, variance, cross entropy"
    :param k: [int] num latent dimensions
    :param patient_idx: [int] patient index
    :param model_name: [str]
    :param window_size: [int] size of the window to use if we want to window the statistical metric
    :return:
    """
    if model_name == "rawhb":
        raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
        data = raw_hbs.reshape(-1, raw_hbs.shape[1] * raw_hbs.shape[
            2])  # reshape so each feature vector contains all 4 leads for each hearbeat
    else:
        data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))
        data = data[2, :, :] if model_name == "vae" else data

    if metric_name == "variance":
        variances = calculate_streamed_variances(data, window_size)  # assume 100 bpm with 5 min window = 500 samples

        variances_savename = os.path.join("Working_Data",
                                          "windowed_var_100d_Idx{}.npy".format(patient_idx))
        np.save(variances_savename, variances)

        metrics = [variance if variance < 20 else 0 for variance in variances]
    elif metric_name == "entropy":
        entropies = calculate_streamed_entropies(data, window_size,
                                                 20)  # assume 100 bpm with 5 min window = 500 samples
        metrics = [e if e < 20 else 0 for e in entropies]
    elif metric_name == "cross entropy":
        metrics = calculate_streamed_entropies(data, window_size, 20)

    set_font_size()
    plt.figure()
    window_times = get_windowed_time(patient_idx, 10, window_size)
    plt.plot(window_times, metrics)
    plt.title("{} of 10-heartbeat segments over time".format(metric_name.capitalize(), model_name.upper()))
    plt.ylabel("Variance".format(metric_name))
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.savefig(f"images/{metric_name.capitalize()}_Idx{patient_idx}.png", dpi=700)
    plt.show()


if __name__ == "__main__":
    # plots the windowed variance of the heartbeats over time
    plot_metrics("variance", 100, 16, "rawhb", 50)
