import numpy as np
import matplotlib.pyplot as plt
import os
from src.preprocess import dsp_utils, h5_interface
from scipy import signal


def bandpass_filter(sig, lower, upper):
    '''
    Inputs: signal, cutoff freqs for bandpass filter
    Outputs: filtered signal
    '''
    sos = signal.butter(10, [lower, upper], 'bp', fs=240, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered

def one_sided_filter(sig, cutoff, type):
    '''
    Inputs: signal, cutoff freqs for bandpass filter, type ='lp'or 'hp'
    Outputs: filtered signal
    '''
    sos = signal.butter(10, cutoff, type , fs=240, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered


def peak_height_scatter(patient_num):
    '''
    Inputs: patient index
    Outputs: None, plots a scatter plot of peak height vs. inter-peak interval length

    NOTHING HERE WORKS
    '''
    slice_len = 5*60*240 # first five minutes
    four_lead = np.load(os.path.join("Working_Data", "Four_Leads_Idx{}.npy".format(patient_num)))

    pos_sum = dsp_utils.combine_four_lead(four_lead[:, :slice_len])
    
    peak_indices = np.load(os.path.join("Working_Data", "HB_Peaks_Idx{}.npy".format(patient_num))) # index of peaks
    peak_indices = peak_indices[peak_indices < slice_len]

    hb_lengths = np.load(os.path.join("Working_Data", "HB_Lens_Idx{}.npy".format(patient_num)))[:peak_indices.shape[0]]# length of inter-peak intervals
    hr = 240/hb_lengths # units of beats/second


    avg_hr = np.mean(hr) # units of heartbeats/second

    peak_heights = pos_sum[peak_indices] # height of peaks on combined raw signal
    # peak_heights = pos_sum[peak_indices][:-1] # height of peaks on combined raw signal
    # peak_heights_means = np.convolve(peak_heights, np.ones((2,))/2)[1:-1] # convolve with ones vector to get pairwise means
    
    filtered_peaks = bandpass_filter(peak_heights, avg_hr*0.1, avg_hr*0.5)
    filtered_hr = bandpass_filter(hr, avg_hr*0.1, avg_hr*0.5)
    plt.scatter(filtered_peaks, filtered_hr)
    plt.xlabel("Peak Height")
    plt.ylabel("Heartbeat duration")
    plt.title("Heartbeat duration vs. Peak Height for Patient " + str(patient_num))
    plt.show()

def extract_respiratory_signal(patient_num):
    four_lead = np.load(os.path.join("Working_Data", "Four_Leads_Idx{}.npy".format(patient_num)))
    pos_sum = dsp_utils.combine_four_lead(four_lead)
    peak_indices = np.load(os.path.join("Working_Data", "HB_Peaks_Idx{}.npy".format(patient_num))) # index of peaks

    peak_heights = pos_sum[peak_indices] # height of peaks on combined raw signal
    peak_heights = dsp_utils.change_dim(peak_heights, pos_sum.shape[0])
    respiratory_signal = bandpass_filter(peak_heights, 5/60, 60/60) # band-pass from 5 to 60 breaths per minute
    # plt.plot(peak_heights + 3)
    # plt.plot(pos_sum)
    # plt.plot(peak_indices, pos_sum[peak_indices], "x")
    # plt.show()
    return respiratory_signal


breathing_rate = extract_respiratory_signal(4)