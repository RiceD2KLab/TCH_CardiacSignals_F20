import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from src.utils import h5_interface
from src.preprocessing.noise_filtering import remove_noise
from src.utils.plotting_utils import set_font_size

# Parameters ###########################################################################################################
filename = 'Reference_idx_16_Time_block_1.h5'
total_time_min = 300 # in minutes
window_time_min = 5 # in minutes

# Read in the data #####################################################################################################
h5f = h5_interface.readh5(filename)

four_lead, time, heartrate = h5_interface.ecg_np(h5f)
lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

lead1 = remove_noise(time, lead1, plots=False)
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

    set_font_size()

    # Plot spectrogram #################################################################################################
    plt.pcolormesh(t/3600-np.max(t/3600), f, (np.abs(Zxx)), vmin=np.amin((np.abs(Zxx))), vmax=np.amax((np.abs(Zxx)))/20, shading='gouraud')
    plt.title('STFT |F(z,t)|', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.ylim([0, 80])
    plt.xlabel('Time (hours)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar()
    plt.savefig('images//frank_stft.png', dpi=500)
    plt.show()

    # Plot spectrum difference #########################################################################################
    # Zxx_subtracted = Zxx - np.c_[Zxx[:, 0], Zxx[:, 0:-1]]
    # plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx_subtracted)), vmin=np.amin(np.log(np.abs(Zxx_subtracted))), vmax=np.amax(np.log(np.abs(Zxx_subtracted))), shading='gouraud')
    # plt.title('Lead %d STFT Change in Magnitude (log|F(z,t)-F(z,t-1)|)' % (lead_label))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [min]')
    # plt.colorbar()
    # plt.show()

    # Look at spectral entropy #########################################################################################
    spectrogram_shape = np.shape(Zxx)
    spectral_entropy = np.empty((spectrogram_shape[1],))

    for iter in range(0, spectrogram_shape[1]):
        print(iter)
        spectral_entropy[iter,] = entropy(abs(Zxx[:, iter]**2))

    plt.plot(t/3600-np.max(t/3600), spectral_entropy)
    plt.title('Spectral Entropy', fontsize=20)
    plt.ylabel('Spectral Entropy', fontsize=14)
    plt.xlabel('Time (hours)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-5, 0)
    plt.savefig('images//frank_spectral_entropy.png', dpi=500)
    plt.show()

spectral_temporal_analysis(lead1, 1, total_time_min, window_time_min)