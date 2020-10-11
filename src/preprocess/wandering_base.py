import numpy as np
import matplotlib.pyplot as plt
import os
from src.preprocess import dsp_utils, h5_interface
from scipy import signal

'''
Code for studying the wandering baseline of the ECGs.
This corresponds to the cardio-pulmonary response of each patient (i.e. the Sinus Respiratory Arrythmia)
'''

def bandpass_filter(sig, lower, upper):
    '''
    Butterworth bandpass filter

    Inputs: signal, cutoff freqs for bandpass filter
    Outputs: filtered signal
    '''
    sos = signal.butter(10, [lower, upper], 'bp', fs=240, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered

def one_sided_filter(sig, cutoff, type):
    '''
    Butterworth one-sided (lowpass or highpass) filter

    Inputs: signal, cutoff freqs for filter, type ='lp'or 'hp'
    Outputs: filtered signal
    '''
    sos = signal.butter(10, cutoff, type , fs=240, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered


def peak_height_scatter(patient_num):
    '''
    Inputs: Patient ID Number
    Outputs: None, plots a scatter plot of peak height vs. inter-peak interval length
    '''
    four_lead = np.load(os.path.join("Working_Data", f"Mod_Four_Lead_Idx{patient_num}.npy")) # load 4 lead data
    hr_vec = np.load(os.path.join("Working_Data", f"Cleaned_HR_Idx{patient_num}.npy")) # load heart rate monitor readings

    pos_sum = dsp_utils.combine_four_lead(four_lead) # Take the negative-clipped sum of each lead
    
    peak_indices = np.load(os.path.join("Working_Data", "HB_Peaks_Idx{}.npy".format(patient_num))) # index of peaks found by peak detector

    hb_lengths = np.load(os.path.join("Working_Data", "HB_Lens_Idx{}.npy".format(patient_num))) # length of inter-peak intervals
    instant_hr = 240/hb_lengths # convert to units of beats/second

    hr_vec = hr_vec / 60 # convert bpm to beats/second
    avg_hr = np.mean(hr_vec) # avg heart rate over the whole interval

    peak_heights = pos_sum[peak_indices][1:] # height of peaks on combined raw signal
    
    filtered_peaks = bandpass_filter(peak_heights, avg_hr*0.1, avg_hr*0.5) # bandpass filter to 0.10-0.5 times the average hr
    filtered_hr = bandpass_filter(instant_hr, avg_hr*0.1, avg_hr*0.5) # bandpass filter to 0.10-0.5 times the average hr

    plt.scatter(filtered_peaks, filtered_hr, c = np.arange(len(instant_hr))) # color map from purple -> yellow as you get closer to the cardiac arrest
    plt.xlabel("Peak Height")
    plt.ylabel("Heartbeats per second")
    plt.title("Heartbeat duration vs. Peak Height for Patient " + str(patient_num))
    plt.show()

def extract_respiratory_signal(patient_num):
    '''
    First attempt to extract respiratory signal from ECG signal by bandpass filtering the ECG peak heights 
    within the expected respiratory range

    Inputs: Patient ID number
    Outputs: The estimated respiratory signal over the entire 6 hour period
    '''
    four_lead = np.load(os.path.join("Working_Data", "Four_Leads_Idx{}.npy".format(patient_num))) # load four-lead data
    pos_sum = dsp_utils.combine_four_lead(four_lead) # compute positive sum with neg clipping
    peak_indices = np.load(os.path.join("Working_Data", "HB_Peaks_Idx{}.npy".format(patient_num))) # index of peaks found by peak detector

    peak_heights = pos_sum[peak_indices] # height of peaks on combined raw signal
    peak_heights = dsp_utils.change_dim(peak_heights, pos_sum.shape[0])
    respiratory_signal = bandpass_filter(peak_heights, 5/60, 60/60) # band-pass from 5 to 60 breaths per minute

    # Plotting for visualization
    # plt.plot(peak_heights + 3)
    # plt.plot(pos_sum)
    # plt.plot(peak_indices, pos_sum[peak_indices], "x")
    # plt.show()
    return respiratory_signal


### Testing above functions
#breathing_rate = extract_respiratory_signal(4)
# for i in range(15, 50):
#     try:
#         peak_height_scatter(i)
#     except:
#         continue