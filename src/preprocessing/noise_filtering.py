"""
Filter the ECG leads to remove noise

Sources of Noise
1. Baseline wander (remove frequencies below 0.7Hz)
2. High frequency noise (remove frequencies above 50Hz)

Technique: Bandpass Butterworth filter (6 poles, I found this to be most stable type of filter and no. of poles)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from src.utils.plotting_utils import set_font_size
from src.utils import h5_interface


def remove_noise(time, lead_data, plots=False):
    """
    Removes noise from input data. A 6-pole Butterworth bandpass filter with 0.7Hz and 50Hz cutoff frequencies is
    implemented. Filter is implemented twice for zero-phase output.

    :param time: [1xN numpy array] input time data in seconds
    :param lead_data: [1xN numpy array] input lead ECG data
    :param plots: [boolean] set to True if you want to see relevant plots
    :return: [1xN numpy array] lead_data_filtered with noise removed
    """
    lowf = 0.7
    highf = 50
    fs = 240

    b, a = signal.butter(6, [lowf, highf], btype='bandpass', fs=fs)
    lead_data_filtered = signal.filtfilt(b, a, lead_data, padlen=150)

    if plots:
        set_font_size()
        # Transfer function of filter ##################################################################################
        w, h = signal.freqz(b, a, worN=4096, fs=240)

        plt.semilogx(w, abs(h)**2)
        plt.axvline(lowf, color='green')  # cutoff frequency
        plt.axvline(highf, color='green')  # cutoff frequency

        plt.title('Butterworth filter transfer function', fontsize=18)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.xlim(0.1, 120)
        plt.ylabel('Amplitude', fontsize=12)
        plt.yticks(fontsize=10)
        plt.grid(which='both', axis='both')

        plt.savefig('images//butterworth_transfer_function.png', dpi=500)
        plt.show()

        # Original signal spectrum #####################################################################################
        freq = np.linspace(0.0, fs / 2, len(time) // 2)
        lead_spectrum = fft(lead_data)

        plt.loglog(freq[1:len(time) // 2], 2 / len(time) * np.abs(lead_spectrum[1:len(time) // 2]), '-b', alpha=0.7)
        plt.title('ECG Spectrum', fontsize=18)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.xlim(0.1, 120)

        plt.ylabel('Amplitude', fontsize=12)
        plt.yticks(fontsize=10)
        plt.grid(which='both', axis='both')

        # Filtered signal spectrum #####################################################################################
        freq = np.linspace(0.0, fs / 2, len(time) // 2)
        lead_spectrum_filtered = fft(lead_data_filtered)

        plt.loglog(freq[1:len(time) // 2], 2 / len(time) * np.abs(lead_spectrum_filtered[1:len(time) // 2]), '-r', alpha=0.7)
        plt.legend(['Original', 'Filtered'], fontsize=12)
        plt.savefig('images//ECG_spectrum_filtering.png', dpi=500)
        plt.show()

        ###############################################################################################################

    return lead_data_filtered


# # EXAMPLE: Test the noise removal on a patient
# filename = 'Reference_idx_16_Time_block_1.h5'
# h5f = h5_interface.readh5(filename)
#
# four_lead, time, heartrate = h5_interface.ecg_np(h5f)
# lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
# lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]
#
# lead1_filtered = remove_noise(time, lead1, plots=True)














































# def preprocess(filename, curr_index, double_beats=False):
#     curr_index = str(curr_index)
#     print("Starting on file : " + filename)
#
#     h5f = h5_interface.readh5(filename)
#
#     four_lead, time, heartrate = h5_interface.ecg_np(h5f)
#     lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
#     lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]
#
#     pos_sum = dsp_utils.combine_four_lead(four_lead)
#
#     peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate)  # indices on the signal where we found a peak
#     peaks = peaks.astype(int)
#
#     # get the bad "heartbeats"
#     bad_hbs = detect_gaps(pos_sum, peaks)
#
#     # Delete the bad heartbeats
#     if len(bad_hbs) > 0:
#         if heartrate is not None:
#             pos_sum, time, heartrate, lead1, lead2, lead3, lead4 = delete_slices(bad_hbs, len(pos_sum),
#                                                                                  [pos_sum, time, heartrate, lead1,
#                                                                                   lead2, lead3, lead4])
#         else:
#             pos_sum, time, lead1, lead2, lead3, lead4 = delete_slices(bad_hbs, len(pos_sum),
#                                                                       [pos_sum, time, lead1, lead2, lead3, lead4])
#         # refind peaks
#         peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate)  # indices on the signal where we found a peak
#         peaks = peaks.astype(int)
#
#     # hb_lengths = find_lengths(peaks, len(pos_sum), pos_sum = pos_sum)
#     # print(min(hb_lengths), sum(hb_lengths) / len(hb_lengths))
#     # try to refind peaks on long heartbeats, mostly to split missed edge beats
#     # edge beats are peaks on the edge of a dynamic window in dynamic peakfind, they get missed
#     new_peaks = []
#     new_inds = []
#     for i in range(1, len(peaks)):
#         if peaks[i] - peaks[i - 1] > 200:
#             extra_peaks = dsp_utils.get_peaks_dynamic(pos_sum[peaks[i - 1]:peaks[i]], heartrate)
#             if len(extra_peaks) == 1:
#                 if extra_peaks[0] > 40:
#                     new_peaks.append(peaks[i - 1] + extra_peaks[0])
#                     new_inds.append(i)
#     peaks = np.insert(peaks, new_inds, new_peaks)
#
#     # hb_lengths = find_lengths(peaks, len(pos_sum))
#     # print(min(hb_lengths), sum(hb_lengths) / len(hb_lengths))
#     # print(len(peaks))
#     too_longs, peaks = find_long_heartbeats(peaks, len(pos_sum), 200)
#     if too_longs:
#         if heartrate is not None:
#             pos_sum, time, heartrate, lead1, lead2, lead3, lead4 = delete_slices(too_longs, len(pos_sum),
#                                                                                  [pos_sum, time, heartrate, lead1,
#                                                                                   lead2, lead3, lead4])
#         else:
#             pos_sum, time, lead1, lead2, lead3, lead4 = delete_slices(too_longs, len(pos_sum),
#                                                                       [pos_sum, time, lead1, lead2, lead3, lead4])
#     # print(len(peaks), len(pos_sum),peaks[-1])
#     four_lead = np.vstack((lead1, lead2, lead3, lead4))
#     # print(four_lead.shape)
#
#     """
#     #Visual Test for R-Peak identification
#     plt.plot(pos_sum)
#     # plt.vlines(x = peaks, ymin = 0, ymax = 8, colors = "red", linewidth = 2)
#     plt.plot(peaks, pos_sum[peaks], "x")
#     plt.show()
#     """
#
#     # logging setup
#     log_filepath = os.path.join("Working_Data", "Heartbeat_Stats_Idx" + curr_index + ".txt")
#     os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
#     log = open(log_filepath, 'w')
#
#     # Find the lengths of the heartbeats
#     hb_lengths = find_lengths(peaks, four_lead.shape[1])
#
#     # print(sum(hb_lengths) / len(hb_lengths))
#
#     # print(len(hb_lengths), len(hb_lengths[hb_lengths > 150]), len(hb_lengths[hb_lengths > 175]), len(hb_lengths[hb_lengths > 200]))
#
#     # Constant dimension to interpolate heartbeats to
#     maximum_hb_len = 100
#
#     log.write("Average valid heartbeat length : " + str(np.average(hb_lengths)) + "\n")
#     log.write("Total valid heartbeats : " + str(len(peaks)) + "\n")
#     log.write("Total invalid heartbeats : " + str(len(bad_hbs)) + "\n")
#
#     if double_beats:
#         peaks = np.take(peaks, list(range(1, len(peaks), 2)))
#     # Save an array of dimension Num heartbeats x 100 (heartbeat length) x Leads (4)
#     fixed_dimension_hbs = np.zeros((len(peaks) + 1, maximum_hb_len, 4))
#     for lead_num in range(4):
#         # First heartbeat in data
#         fixed_dimension_hbs[0, :, lead_num] = dsp_utils.change_dim(four_lead[lead_num, 0:peaks[0]], maximum_hb_len)
#         # Last heartbeat in data
#         fixed_dimension_hbs[len(peaks), :, lead_num] = dsp_utils.change_dim(four_lead[lead_num, peaks[-1]:],
#                                                                             maximum_hb_len)
#         # iterate through the rest of heartbeats
#         for hb_num, peak in enumerate(peaks[:-1], start=1):
#             individual_hb = four_lead[lead_num, peaks[hb_num - 1]:peaks[hb_num]]
#             try:
#                 fixed_dimension_hbs[hb_num, :, lead_num] = dsp_utils.change_dim(individual_hb, maximum_hb_len)
#             except:
#                 print("interpolation failed on heartbeat:" + str(hb_num))
#             """
#             #Periodic Visual inspection of dimension fixed heartbeat
#             if hb_num % 15000 == 0:
#                 plt.plot(fixed_dimension_hbs[hb_num,:,lead_num])
#                 plt.show()
#             """
#
#     # Save the four lead signals with gaps cut out
#     mod_four_lead_savename = os.path.join("Working_Data", "Mod_Four_Lead_Idx" + curr_index + ".npy")
#     # Save the processed heartbeat arrays
#     data_savename = os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + curr_index + ".npy")
#     # Save the clipped heartrate vector from the ECG machine
#     hr_savename = os.path.join("Working_Data", "Cleaned_HR_Idx" + curr_index + ".npy")
#     # Save the peak indicies
#     peaks_savename = os.path.join("Working_Data", "HB_Peaks_Idx" + curr_index + ".npy")
#     # Save the heartbeat lengths
#     HB_lens_savename = os.path.join("Working_Data", "HB_Lens_Idx" + curr_index + ".npy")
#     # Save the heartbeat timestamps
#     HB_timestamps_savename = os.path.join("Working_Data", "HB_Timestamps_Idx" + curr_index + ".npy")
#
#     np.save(mod_four_lead_savename, four_lead)
#     np.save(data_savename, fixed_dimension_hbs)
#     np.save(hr_savename, heartrate)
#     np.save(peaks_savename, peaks)
#     np.save(HB_lens_savename, hb_lengths)
#     np.save(HB_timestamps_savename, time[peaks])
#     log.close()
#
#
# if __name__ == "__main__":
#     for idx, filename in enumerate(get_filenames()):
#         preprocess(filename, idx, double_beats=False)
