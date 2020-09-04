import numpy as np 
from scipy.signal import find_peaks

'''
Inputs: ECG Signal, desired dimensions
Outputs: ECG Signal in desired dimension
Map a signal to a specific dimension, linearly interpolating
'''
def raise_dim(data, dim):
	samp_times = np.arange(start = 0, stop = dim-.99, step = (dim - 1) / (len(data) - 1))
	resamp_times = np.arange(dim)

	return np.interp(resamp_times, samp_times, data)
'''
Inputs: ECG Signal
Outputs: Indicies of peaks
Scans across the signal in 2 second windows, collects maximums. Clears the maximums of outliers
and looks for peaks above the minimum maximum value
'''
def get_peaks(data):
	maxes = np.zeros((len(data)//480,))
	for i in range(len(data) // 480):
		maxes[i] = max(data[i*480:(i+1)*480])

	maxes = maxes[abs(maxes - np.mean(maxes)) < .5 * np.std(maxes)]
	peak_height = min(maxes)
	#print("finding peaks above : " + str(peak_height))
	peaks, properties = find_peaks(data, height = 1.3, distance = 80)
	return peaks