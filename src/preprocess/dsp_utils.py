import numpy as np 
from scipy.signal import find_peaks

'''
Inputs: ECG Signal, desired dimensions
Outputs: ECG Signal in desired dimension
Map a signal to a specific dimension, linearly interpolating
'''
def change_dim(data, dim):
	samp_times = np.arange(start = 0, stop = dim-.999999999, step = (dim - 1) / (len(data) - 1))
	resamp_times = np.arange(dim)

	if not (len(samp_times) == len(data)):
		print(len(samp_times), len(data))
	return np.interp(resamp_times, samp_times, data)
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
def get_peaks_dynamic(data, heartrate):
	interval = 240*60*5 # Use a five minute window of heartrate information
	peaks = np.zeros((0,))
	for i in range(len(data) // interval):
		avg_hr = np.mean(heartrate[i*interval:(i+1)*interval]) # average heart rate (beats/minute) in current window
		if (avg_hr > 0): # If we have valid HR data
			avg_dist = (60*240)/avg_hr # average samples/heartbeat in current window
		else: # If we don't have HR data for most of that window, use a default value
			avg_dist = 140
		local_peaks, properties = find_peaks(data[i*interval:(i+1)*interval], distance = 0.5*avg_dist, prominence=1.0, wlen=0.25*avg_dist)
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
		pos_sum += np.clip(data[i,:], 0, None)
	return pos_sum