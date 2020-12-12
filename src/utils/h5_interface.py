"""
Functions for working with h5 files

ex. python3 h5_interface.py 1 --Duration 2 --Offset 0
"""

import os
import argparse

import h5py
import numpy as np

import ecg_plot

'''
Inputs : filename of h5 file to be read
Ouputs : h5py "dataset", similar to a dictionary of {labels : numpy array of data}
'''
def readh5(filename):
	filepath = os.path.join("Data_H5_Files", filename)
	return h5py.File(filepath, 'r')
'''
Inputs : h5py dataset of TCH ECG file
Outpus : 4 x n numpy array representing the four lead signals
		 1 x n np array time axis
		 1 x n np array heartrate

Convert h5py dataset to numpy arrays
'''
def ecg_np(dataset, split = False):
	lead_one = (dataset['GE_WAVE_ECG_1_ID'] [()])
	lead_two = (dataset['GE_WAVE_ECG_2_ID'] [()])
	lead_three = (dataset['GE_WAVE_ECG_3_ID'] [()])
	lead_four = (dataset['GE_WAVE_ECG_4_ID'] [()]) 

	if split:
		try:
			return lead_one, lead_two, lead_three, lead_four, dataset['time'][()], dataset['PARM_HR'][()]
		except:
			return lead_one, lead_two, lead_three, lead_four, dataset['time'][()], None
	try:
		return np.vstack((lead_one, lead_two, lead_three, lead_four)), dataset['time'][()], dataset['PARM_HR'][()]
	except:
		return np.vstack((lead_one, lead_two, lead_three, lead_four)), dataset['time'][()], None
'''
Inputs : h5py dataset from one of the TCH files, time length to be plotted (seconds), time offset (seconds)
Outputs : 4 x time_length * 240 numpy array which are the 4 lead signals stacked vertically
'''
def ecg_slice_np(dataset, duration = 2, offset = 0):
	start = int(offset * 240)
	length = int(duration * 240)
	lead_one = (dataset['GE_WAVE_ECG_1_ID'] [start:start + length] [()])
	lead_two = (dataset['GE_WAVE_ECG_2_ID'] [start:start + length] [()])
	lead_three = (dataset['GE_WAVE_ECG_3_ID'] [start:start + length] [()])
	lead_four = (dataset['GE_WAVE_ECG_4_ID'] [start:start + length] [()]) 

	return np.vstack((lead_one, lead_two, lead_three, lead_four))


'''
Inputs : (4,n) numpy array representing the 5 lead ECG
Outputs : Plots the four ECG signals
'''
def four_lead_plot(data):
	ecg_plot.plot(data, sample_rate = 240, title = 'Four Lead ECG', columns = 1)
	ecg_plot.show()

'''
Give an index of one of the TCH h5 ECG files and plot the first two seconds of it.
Optionally provide the duration you want, and the time offset to start at
'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("FileIdx", help="index number of the TCH h5 ECG file to be plotted")
	parser.add_argument("--Duration", type = float, help="Time duration in seconds to be plotted")
	parser.add_argument("--Offset", type = float, help="Time offset of the slice to be plotted")
	args = parser.parse_args()

	filename = "Reference_idx_" + args.FileIdx + "_Time_block_1.h5"
	if args.Duration and args.Offset:
		f = readh5(filename)
		dat = ecg_slice_np(f, args.Duration, args.Offset)
		four_lead_plot(dat)
	else:
		f = readh5(filename)
		dat = ecg_slice_np(f)
		four_lead_plot(dat)