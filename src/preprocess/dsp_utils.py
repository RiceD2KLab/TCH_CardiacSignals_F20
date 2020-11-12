import numpy as np 
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

'''
Inputs: ECG Signal, desired dimensions
Outputs: ECG Signal in desired dimension
Map a signal to a specific dimension, linearly interpolating
'''
def change_dim(data, dim):
	samp_times = np.linspace(start = 0, stop = len(data) - 1, num = dim, endpoint = True)

	return np.interp(samp_times, np.arange(len(data)), data)
'''
Inputs: ECG Signal
Outputs: Indicies of peaks
Scans across the signal in intervals. Within each interval finds peaks close to the magnitude of local maximum
'''
def get_peaks(data):
	interval = 240 #sample rate is ~240hz, so interval / 240 = time (seconds)
	peaks = np.zeros((0,))
	for i in range(len(data) // interval):
		maxi = max(data[i*interval:(i+1)*interval])
		local_peaks, properties = find_peaks(data[i*interval:(i+1)*interval], height = maxi * .5, distance = 70)
		local_peaks = local_peaks + (i * interval)
		peaks = np.concatenate((peaks, local_peaks))
	return peaks

'''
Inputs: ECG Signal
Outputs: Indices of peaks
Gets peaks via prominence
'''
def get_peaks_prominence(data):
	peaks, properties = find_peaks(data, distance = 70, prominence=1.0, wlen=50)
	return peaks

'''
Inputs: ECG Signal, heart-rate signal (must be same length)
Outputs: Indices of peaks
Gets peaks via prominence, using known heart-rate information to choose parameters dynamically
'''
def get_peaks_dynamic(data, heartrate, plotting = False):
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

		#For debugging, comment out otherwise
		#ratio = (np.amax(dat_window) - window_mean) / window_std

		"""
		#Check the clipping

		if ratio > 12.0:
			clip = window_mean + 5.0 * window_std
			plt.plot(dat_window)
			plt.hlines(clip, xmin = 0, xmax = 479)
			plt.show()
		"""

		
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
		
		if plotting:
			print(avg_dist)
	
		#Add peaks
		local_peaks = local_peaks + (i * interval)
		peaks = np.concatenate((peaks, local_peaks))
	return peaks

'''
Inputs : 4 x n numpy array, each row is an ECG signal lead
Outputs : Sum of the 4 signals with clipped negative values (set to 0)
'''
def combine_four_lead(data):
	pos_sum = np.zeros((data.shape[1],))
	for i in range(4):
		#pos_sum += np.clip(data[i,:], 0, None) #clip negative values
		pos_sum += np.absolute(data[i,:])
	return pos_sum