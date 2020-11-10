import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess.heartbeat_split import heartbeat_split


def detect_change(patient, model_name, dimension):
    error_signal = np.load(f"Working_Data/windowed_mse_{model_name}_{dimension}d_Idx{patient}.npy")
    algo = rpt.Pelt(model="l2").fit(error_signal)
    changepoints = algo.predict(pen=0.01)  # 'pen' is for 'penalty'

    print(changepoints)
    plt.plot(error_signal)
    for changepoint in changepoints:
        plt.axvline(changepoint, c='r')
    plt.show()

def cusum(patient, model_name, dimension, c=0.05):
    '''
    CUSUM change point detector
     
    Inputs:
    patient - int, patient index
    model_name - model used for dim reduction (either 'pca' or 'ae')
    dimension - dimension used for dim redudction (typically 10)
    c - correction parameter for CUSUM test. Set this slightly larger than the expected MSE for normal data (default of 0.05 seems to work well)

    Outputs:
    None: plots the CUSUM statistic over time
    '''
    error_signal = np.load(f"Working_Data/windowed_mse_{model_name}_{dimension}d_Idx{patient}.npy")
    cusum = [0]
    for x in error_signal:
        L = x - c
        cusum.append(max(cusum[-1] + L, 0))
    plt.plot(cusum)
    plt.title(f"CUSUM statistic over time for patient {patient}")
    plt.xlabel("Window Index")
    plt.ylabel("CUSUM Score")
    plt.show()


for i in heartbeat_split.indicies:
    try:
        cusum(i, "ae", 10)
    except:
        continue