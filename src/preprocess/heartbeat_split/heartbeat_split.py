"""
This Modules performs the basic preprocessing on the raw data:

1. R-Peaks are detected to seperate the signal into individual heartbeats 
2. Heartbeats with less than 5% unique values are clipped
3. Every heartbeat is interpolated to dimension 100
4. Individual interpolated heartbeats are saved

Outputs several processed data files named based off their "index" which is the alphanurmeric
sorted position of the raw data file in the Data_H5_Files directory

Mod_Four_Lead_[idx].npy : 4 x n Numpy array of the four lead signals with lead disconnections clipped out
HB_Peaks_[idx].npy      : (num heartbeats - 1) Numpy vector of the indicies of the R-Peaks in Mod_Four_Lead 
Fixed_Dim_Hbs_[idx].npy : 4 x 100 x (num heartbeats) Numpy array of the four lead signals seperated into 
						  individual heartbeats and interpolated to dimension 100
Cleaned_HR_[idx].npy    : n Numpy vector of the heartrate from monitor, with lead disconnections clipped out
HB_Lens_[idx].npy       : (num heartbeats) Numpy vector of individual heartbeat lengths in samples; before interpolation					  

"""

import numpy as np
import os

import matplotlib.pyplot as plt

from src.utils.file_indexer import get_filenames
from src.preprocess import dsp_utils, h5_interface
from src.preprocess.heartbeat_split.noise_filtering import remove_noise

indicies = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
				'33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']
'''
Function to detect "gaps" in the data signal, which are from the leads being
disconnected. For each detected "heartbeat" (a slice between two R-peaks)

Inputs: 
pos_sum: type: Numpy Array - Signal sum of negative clipped lead signals
peaks  : type: Numpy Array - Indicies of the detected R-peaks in pos_sum

Outputs: 
bad_hbs: type: list of Slices - heartbeats which have less than 5% unique values
'''
def detect_gaps(pos_sum, peaks):
	#Look for gaps (based of unique values)
	bad_hbs = []
	vals = np.unique(pos_sum[0:peaks[0]])
	#leading heartbeat
	if len(vals) < .05 * (peaks[0]):
		bad_hbs = [slice(0, peaks[0])]
	#scan heartbeats
	for i in range(1, len(peaks)):
		vals = np.unique(pos_sum[peaks[i-1]:peaks[i]])
		if len(vals) < .05 * (peaks[i] - peaks[i-1]):
			"""
			#Visual Inspection of bad heartbeat removal
			plt.plot(pos_sum[peaks[i-1]: peaks[i]])
			plt.show()
			"""
			bad_hbs.append(slice(peaks[i-1], peaks[i]))
	vals = np.unique(pos_sum[peaks[-1]:])
	#Trailing heartbeat
	if len(vals) < .05 * (len(pos_sum) - peaks[-1]):
		bad_hbs.append( slice(peaks[-1], len(pos_sum)) )

	return bad_hbs

def find_lengths(peaks, total_len, pos_sum = None):
	hb_lengths = np.zeros((len(peaks),)) 		#array of R-Peak identified heartbeat segment lengths
	for i in range(1, len(peaks)):
		hb_lengths[i] = peaks[i] - peaks[i-1]	#calculate the legnths
		#if peaks[i] - peaks[i-1] < 35:
		#	plt.plot(pos_sum[peaks[i-2]:peaks[i+1]])
		#	plt.show()
	hb_lengths = hb_lengths[1:] 				#remove the leading 0
	np.concatenate( (hb_lengths,np.array([total_len-peaks[-1]]) ) ) #append the last length
	return hb_lengths

def find_long_heartbeats(peaks, total_len, threshold):
	long_slices = []
	rm_idx = []

	removed_lengths = []
	for i in range(1, len(peaks)):
		hb_len = peaks[i] - peaks[i-1]
		if hb_len > threshold:
			long_slices.append(slice(peaks[i-1], peaks[i]))
			removed_lengths.append(hb_len)
			rm_idx.append(i-1)
	if total_len - peaks[-1] > threshold:
		long_slices.append(slice(peaks[-1], total_len))
		peaks = peaks[:-1]
	for i in range(len(rm_idx)):
		peaks[rm_idx[i]+1:] = peaks[rm_idx[i]+1:] - removed_lengths[i]
	return long_slices, np.delete(peaks, rm_idx)

def delete_slices(slices, length, removals):
	ind = np.indices((length,))[0]
	rm = np.hstack(list(ind[i] for i in slices))
	good_indices = sorted(set(ind)-set(rm))

	for i in range(len(removals)):
		 removals[i] = np.take(removals[i], good_indices)
	
	return removals
'''
Inputs: Indicies of the patient files to process
Outputs: Saves multiple files of processed data, in addition to a text file log for each
'''
def preprocess(filename, curr_index, double_beats = False):
	curr_index = str(curr_index)
	print("Starting on file : " + filename)

	h5f = h5_interface.readh5(filename)

	four_lead, time, heartrate = h5_interface.ecg_np(h5f)
	lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
	lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

	# Removing baseline wander and high frequency noise
	lead1 = remove_noise(time, lead1, plots=False)
	lead2 = remove_noise(time, lead2, plots=False)
	lead3 = remove_noise(time, lead3, plots=False)
	lead4 = remove_noise(time, lead4, plots=False)

	pos_sum = dsp_utils.combine_four_lead(four_lead)
	
	peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
	peaks = peaks.astype(int)

	#get the bad "heartbeats"
	bad_hbs = detect_gaps(pos_sum, peaks)

	#Delete the bad heartbeats
	if len(bad_hbs) > 0:
		if heartrate is not None:
			pos_sum, time, heartrate, lead1, lead2, lead3, lead4 = delete_slices(bad_hbs, len(pos_sum), [pos_sum, time, heartrate, lead1, lead2, lead3, lead4])
		else:
			pos_sum, time, lead1, lead2, lead3, lead4 = delete_slices(bad_hbs, len(pos_sum), [pos_sum, time, lead1, lead2, lead3, lead4])
		#refind peaks
		peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
		peaks = peaks.astype(int)
		
	#hb_lengths = find_lengths(peaks, len(pos_sum), pos_sum = pos_sum)
	#print(min(hb_lengths), sum(hb_lengths) / len(hb_lengths))
	#try to refind peaks on long heartbeats, mostly to split missed edge beats
	#edge beats are peaks on the edge of a dynamic window in dynamic peakfind, they get missed
	new_peaks = []
	new_inds = []
	for i in range(1, len(peaks)):
		if peaks[i] - peaks[i - 1] > 200:
			extra_peaks = dsp_utils.get_peaks_dynamic(pos_sum[peaks[i-1]:peaks[i]], heartrate)
			if len(extra_peaks) == 1:
				if extra_peaks[0] > 40:
					new_peaks.append(peaks[i-1] + extra_peaks[0])
					new_inds.append(i)
	peaks = np.insert(peaks, new_inds, new_peaks)

	#hb_lengths = find_lengths(peaks, len(pos_sum))
	#print(min(hb_lengths), sum(hb_lengths) / len(hb_lengths))
	#print(len(peaks))
	too_longs, peaks = find_long_heartbeats(peaks, len(pos_sum), 200)
	if too_longs:
		if heartrate is not None:
				pos_sum, time, heartrate, lead1, lead2, lead3, lead4 = delete_slices(too_longs, len(pos_sum), [pos_sum, time, heartrate, lead1, lead2, lead3, lead4])
		else:
			pos_sum, time, lead1, lead2, lead3, lead4 = delete_slices(too_longs, len(pos_sum), [pos_sum, time, lead1, lead2, lead3, lead4])
	#print(len(peaks), len(pos_sum),peaks[-1])
	four_lead = np.vstack((lead1, lead2, lead3, lead4))
	#print(four_lead.shape)

	"""
	#Visual Test for R-Peak identification
	plt.plot(pos_sum)
	# plt.vlines(x = peaks, ymin = 0, ymax = 8, colors = "red", linewidth = 2)
	plt.plot(peaks, pos_sum[peaks], "x")
	plt.show()
	"""
	
	#logging setup
	log_filepath = os.path.join("Working_Data", "Heartbeat_Stats_Idx" + curr_index + ".txt")
	os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
	log = open(log_filepath, 'w')

	#Find the lengths of the heartbeats
	hb_lengths = find_lengths(peaks, four_lead.shape[1])

	#print(sum(hb_lengths) / len(hb_lengths))

	#print(len(hb_lengths), len(hb_lengths[hb_lengths > 150]), len(hb_lengths[hb_lengths > 175]), len(hb_lengths[hb_lengths > 200]))

	#Constant dimension to interpolate heartbeats to
	maximum_hb_len = 100
	
	log.write("Average valid heartbeat length : " + str(np.average(hb_lengths)) + "\n")
	log.write("Total valid heartbeats : " + str(len(peaks))+ "\n")
	log.write("Total invalid heartbeats : " + str(len(bad_hbs))+ "\n")

	if double_beats:
		peaks = np.take(peaks, list(range(1, len(peaks),2)))
	#Save an array of dimension Num heartbeats x 100 (heartbeat length) x Leads (4)
	fixed_dimension_hbs = np.zeros((len(peaks)+1, maximum_hb_len, 4))
	for lead_num in range(4):
		#First heartbeat in data
		fixed_dimension_hbs[0,:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, 0:peaks[0]], maximum_hb_len)
		#Last heartbeat in data
		fixed_dimension_hbs[len(peaks),:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, peaks[-1]:], maximum_hb_len)
		#iterate through the rest of heartbeats
		for hb_num, peak in enumerate(peaks[:-1], start = 1):
			individual_hb = four_lead[lead_num,peaks[hb_num-1]:peaks[hb_num]]
			try:
				fixed_dimension_hbs[hb_num,:,lead_num] = dsp_utils.change_dim(individual_hb, maximum_hb_len)
			except:
				print("interpolation failed on heartbeat:" + str(hb_num))
			"""
			#Periodic Visual inspection of dimension fixed heartbeat
			if hb_num % 15000 == 0:
				plt.plot(fixed_dimension_hbs[hb_num,:,lead_num])
				plt.show()
			"""

	#Save the four lead signals with gaps cut out
	mod_four_lead_savename = os.path.join("Working_Data", "Mod_Four_Lead_Idx" + curr_index + ".npy")
	#Save the processed heartbeat arrays
	data_savename = os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + curr_index + ".npy")
	#Save the clipped heartrate vector from the ECG machine
	hr_savename = os.path.join("Working_Data", "Cleaned_HR_Idx" + curr_index + ".npy")
	#Save the peak indicies
	peaks_savename = os.path.join("Working_Data", "HB_Peaks_Idx" + curr_index + ".npy")
	#Save the heartbeat lengths
	HB_lens_savename = os.path.join("Working_Data", "HB_Lens_Idx" + curr_index + ".npy")
	#Save the heartbeat timestamps
	HB_timestamps_savename = os.path.join("Working_Data", "HB_Timestamps_Idx" + curr_index + ".npy")
	
	np.save(mod_four_lead_savename, four_lead)
	np.save(data_savename, fixed_dimension_hbs)
	np.save(hr_savename, heartrate)
	np.save(peaks_savename, peaks)
	np.save(HB_lens_savename, hb_lengths)
	np.save(HB_timestamps_savename, time[peaks])
	log.close()
	
if __name__ == "__main__":
	for idx, filename in enumerate(get_filenames()):
		preprocess(filename, idx, double_beats =False)
