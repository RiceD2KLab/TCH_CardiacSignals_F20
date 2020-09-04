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
Scans across the signal in minute intervals. Within each interval finds peaks close to the magnitude of local maximum
'''
def get_peaks(data):
	peaks = np.zeros((0,))
	maxes = np.zeros((len(data)//14400,))
	for i in range(len(data) // 14400):
		maxi = max(data[i*14400:(i+1)*14400])
		local_peaks, properties = find_peaks(data[i*14400:(i+1)*14400], height = maxi * .75, distance = 80)
		peaks = np.concatenate((peaks, local_peaks))
	return peaks