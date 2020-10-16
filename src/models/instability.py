import os
import sklearn
import numpy as np
from scipy.stats import entropy
# from skimage.filters.rank import entropy
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def calculate_streamed_variances(heartbeats, window_duration):
    """

    :param heartbeats: n x k x 4 array of heartbeat patient data where k is the latent dimension
    :param window_duration: number of samples in the window
    :return: list of entropies
    """

    heartbeat_len = heartbeats.shape[1]
    k = heartbeats.shape[2]
    # entropies = []
    scaler = StandardScaler()

    variances = []

    for i in range(window_duration, heartbeat_len):
        window = heartbeats[2, i-window_duration:i, :] # n x k

        # window = scaler.fit_transform(window)
        window_vars = np.apply_along_axis(lambda row: np.var(row), 0, window)

        variances.append(np.mean(window_vars))


    return variances

def plot_variances(k, patient_idx, model_name):
    data = np.load(os.path.join("Working_Data", "reduced_vae_{}d_Idx{}.npy".format(k, patient_idx)))
    variances = calculate_streamed_variances(data, 500) # assume 100 bpm with 5 min window = 500 samples
    variances = [variance if variance < 20 else 0 for variance in variances]

    plt.figure()
    plt.plot(variances)
    plt.title("Variance of 5 minute increments for Patient {} with model {} with latent dim {}".format(patient_idx, model_name, k))
    plt.ylabel("Mean of variances across latent dimensions")
    plt.xlabel("Window End Sample Idx")
    plt.show()

if __name__ == "__main__":

    plot_variances(10, 4, "pca")
    plot_variances(10, 1, "pca")
    plot_variances(10, 4, "vae")
    plot_variances(10, 1, "vae")




