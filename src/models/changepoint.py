import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def detect_change(patient, model_name, dimension):
    error_signal = np.load(f"Working_Data/windowed_mse_{dimension}d_Idx{patient}.npy")
    algo = rpt.Pelt(model="l2").fit(error_signal)
    changepoints = algo.predict(pen=0.01)  # 'pen' is for 'penalty'

    print(changepoints)
    plt.plot(error_signal)
    for changepoint in changepoints:
        plt.axvline(changepoint, c='r')
    plt.show()


detect_change(5, "ae", 10)