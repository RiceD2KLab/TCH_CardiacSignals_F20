import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from src.preprocess import h5_interface
from src.preprocess.heartbeat_split.noise_filtering import remove_noise
# Parameters ###########################################################################################################
filename = 'Reference_idx_1_Time_block_1.h5'
total_time_min = 300 # in minutes
window_time_min = 5 # in minutes

# Read in the data #####################################################################################################
h5f = h5_interface.readh5(filename)

four_lead, time, heartrate = h5_interface.ecg_np(h5f)
lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

lead1 = remove_noise(time, lead1, plots=True)
# lead2 = remove_noise(time, lead2)
# lead3 = remove_noise(time, lead3)
# lead4 = remove_noise(time, lead4)

def spectral_temporal_analysis(lead_data, lead_label, total_time_min, window_time_min):
    """
    Plots three items
    1. Spectrogram
    2. Difference in power spectrum between each time index, over all time
    3. Spectral entropy against time
    :param lead_data: numpy array (N x 1) of ECG data
    :param lead_label: integer --> either 1, 2, 3, or 4
    :param total_time_min:
    :param window_time_min:
    :return:
    """
    # Parameters #######################################################################################################
    fs = 240  # Hz
    total_time_sec = int(total_time_min * fs * 60)
    window_time_sec = int(window_time_min * fs * 60)

    # STFT #############################################################################################################
    f, t, Zxx = signal.stft(lead_data[-total_time_sec:], fs, nperseg=window_time_sec, noverlap=None)

    # Plot spectrogram #################################################################################################
    plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx)), vmin=np.amin(np.log(np.abs(Zxx))), vmax=np.amax(np.log(np.abs(Zxx))), shading='gouraud')
    plt.title('Lead %d STFT Magnitude (log|F(z)|)' % (lead_label))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [min]')
    plt.colorbar()
    plt.show()

    # Plot spectrum difference #########################################################################################
    Zxx_subtracted = Zxx - np.c_[Zxx[:, 0], Zxx[:, 0:-1]]
    plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx_subtracted)), vmin=np.amin(np.log(np.abs(Zxx_subtracted))), vmax=np.amax(np.log(np.abs(Zxx_subtracted))), shading='gouraud')
    plt.title('Lead %d STFT Change in Magnitude (log|F(z,t)-F(z,t-1)|)' % (lead_label))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [min]')
    plt.colorbar()
    plt.show()

    # Look at spectral entropy #########################################################################################
    spectrogram_shape = np.shape(Zxx)
    spectral_entropy = np.empty((spectrogram_shape[1],))

    for iter in range(0, spectrogram_shape[1]):
        print(iter)
        spectral_entropy[iter,] = entropy(abs(Zxx[:, iter]**2))

    plt.plot(t/60-np.max(t/60), spectral_entropy)
    plt.title('Lead %d Spectral Entropy' % (lead_label))
    plt.ylabel('Entropy of Power Spectrum')
    plt.xlabel('Time [min]')
    plt.show()

spectral_temporal_analysis(lead1, 1, total_time_min, window_time_min)