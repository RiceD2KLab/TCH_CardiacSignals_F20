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
Scans across the entire signal and finds peaks based on their prominence and distance to neighbors
'''
def get_peaks_prominence(data):
	interval = 240*3 #sample rate is ~240hz, so interval / 240 = time (seconds)
	peaks = np.zeros((0,))
	local_peaks, properties = find_peaks(data, distance = 70, prominence=0.5, wlen=50)
	peaks = np.concatenate((peaks, local_peaks))
	return peaks

'''
Inputs : 4 x n numpy array, each row is an ECG signal lead
Outputs : Indicies of peaks from the absolute sum of the four signal leads
'''
def four_lead_peaks(data):
	pos_sum = np.zeros((data.shape[1],))
	for i in range(4):
		pos_sum += np.clip(data[i,:], 0, None)
	return get_peaks(pos_sum)