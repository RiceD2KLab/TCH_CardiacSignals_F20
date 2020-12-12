import numpy as np 
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def change_dim(data, dim):
	"""
    Interpolates given signal to given dimension
    :param data: [np array[float64]] raw ECG signal 
    :param dim: [int] dimension to be interpolated to
    :return: [np array[float]] interpolated signal
    """
	samp_times = np.linspace(start = 0, stop = len(data) - 1, num = dim, endpoint = True)

	return np.interp(samp_times, np.arange(len(data)), data)

def get_peaks_dynamic(data, heartrate):
	"""
    Checks a list of detected heartbeat slices from ECG signal, and detects flatlines
    :param data: [np array[float64]] raw ECG signal 
    :param heartrate: [np array[int]] heartrate vector from ECG machine
    :return: [list[slice[int:int]]] slices of flatline signal
    """
	interval = 240 * 5 # Use a 2 second window of heartrate information
	peaks = np.zeros((0,))
	if heartrate is None:
		heartrate = np.full((len(data),), 110)
	for i in range(len(data) // interval + 1):
		avg_hr = np.mean(heartrate[i*interval:(i+1)*interval]) # average heart rate (beats/minute) in current window
		if (avg_hr > 0): # If we have valid HR data
			if avg_hr < 80:
				avg_dist = 100
			else:
				avg_dist = (60*240)/avg_hr # average samples/heartbeat in current window
		else: # If we don't have HR data for most of that window, use a default value
			avg_dist = 100

		#window of interval length
		dat_window = data[i*interval:(i+1)*interval]

		if len(dat_window) == 0:
			continue

		window_std = np.std(dat_window)
		window_mean = np.mean(dat_window)
		
		#Next we want to normalize the window between 0-1, don't want the noise spikes
		#(often 10+ times the mean) to hide true peaks. Thus clip to max 5 standard dev
		

		#Clip the window to have maximum of 5 standard deviations
		clipped_window = np.clip(dat_window, a_min = 0, a_max = (window_mean + 5.0 * window_std))

		#avoid dividing by 0 if the window is all 0
		if np.max(np.abs(clipped_window)) == 0:
			continue
		#normalize the window between 0-1
		normed_window = clipped_window / np.max(np.abs(clipped_window))
		#find peaks
		local_peaks, properties = find_peaks( normed_window, distance = 0.8*avg_dist, prominence= .3, wlen=0.25*avg_dist)
	
		#Add peaks
		local_peaks = local_peaks + (i * interval)
		peaks = np.concatenate((peaks, local_peaks))
	return peaks

def combine_four_lead(data):
	"""
    Take absolute value and sums four lead signal
    :param data: [np array[np array[float64]]] raw four lead ECG signal 
    :return: [np array[float64]] sum of absolute valued leads
    """
	pos_sum = np.zeros((data.shape[1],))
	for i in range(4):
		#pos_sum += np.clip(data[i,:], 0, None) #clip negative values
		pos_sum += np.absolute(data[i,:])
	return pos_sum



def get_windowed_time(patient_idx, num_hbs=10, window_size=50):
	"""
	Get time indices which correspond to the first index of each window
	(in units of hours before cardiac arrest)
	:param patient_idx: [int] patient index
	:param num_hbs: [int] number of heartbeats used per data point (usually 10)
	:param window_size: [int] window size used for computing windowed metrics (usually 50)
	
	:return: [list[int]] time indicies spaced by "window_size" indices, in units of hrs before
	cardiac arrest
	"""
	time_stamps = np.load(f"Working_Data/HB_Timestamps_Idx{patient_idx}.npy") # load raw time vector (seconds)
	time_stamps = time_stamps - time_stamps[-1] # convert units to seconds before end of data (negative time)
	time_stamps = time_stamps/3600 # convert to hours
	return time_stamps[0::(window_size*num_hbs)][1:] # corresponding time stamps (x axis values)