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

from src.utils import plotting_utils

indicies = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
				'33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']
'''
Function to detect "gaps" in the data signal, which are from the leads being
disconnected. For each detected "heartbeat" (a slice between two R-peaks)

Inputs: 
signal_data: type: Numpy Array - Signal sum of negative clipped lead signals
peaks  : type: Numpy Array - Indicies of the detected R-peaks in signal_data

Outputs: 
bad_hbs: type: list of Slices - heartbeats which have less than 5% unique values
'''
def detect_gaps(signal_data, peaks):
	#Look for gaps (based of unique values)
	bad_hbs = []
	vals = np.unique(signal_data[0:peaks[0]])
	#leading heartbeat
	if len(vals) < .05 * (peaks[0]):
		bad_hbs = [slice(0, peaks[0])]
	#scan heartbeats
	for i in range(1, len(peaks)):
		vals = np.unique(signal_data[peaks[i-1]:peaks[i]])
		if len(vals) < .05 * (peaks[i] - peaks[i-1]):
			"""
			#Visual Inspection of bad heartbeat removal
			plt.plot(signal_data[peaks[i-1]: peaks[i]])
			plt.show()
			"""
			bad_hbs.append(slice(peaks[i-1], peaks[i]))
	vals = np.unique(signal_data[peaks[-1]:])
	#Trailing heartbeat
	if len(vals) < .05 * (len(signal_data) - peaks[-1]):
		bad_hbs.append( slice(peaks[-1], len(signal_data)) )

	return bad_hbs
'''
Function to get the heartbeat lengths, which is the distance between adjacent
peaks.

Inputs:
peaks:     type: Numpy Array (int) - indicies of the peaks
total_len: type: int 			   - total length of the signal

Outputs:
hb_lengths: type: Numpy Array (int) - lengths of heartbeats
'''
def find_lengths(peaks, total_len, signal_data = None):
	hb_lengths = np.zeros((len(peaks),)) 		#array of R-Peak identified heartbeat segment lengths
	for i in range(1, len(peaks)):
		hb_lengths[i] = peaks[i] - peaks[i-1]	#calculate the legnths
		#if peaks[i] - peaks[i-1] < 35:
		#	plt.plot(signal_data[peaks[i-2]:peaks[i+1]])
		#	plt.show()
	hb_lengths = hb_lengths[1:] 				#remove the leading 0
	np.concatenate( (hb_lengths,np.array([total_len-peaks[-1]]) ) ) #append the last length
	return hb_lengths
'''
Function to get slices of "long" heartbeats

Inputs:
peaks:     type: Numpy Array (int) - indicies of the peaks
total_len: type: int 			   - total length of the signal
threshold: type: int 			   - threshold heartbeat length, above this to be removed

Outputs:
long_slices:   type: Slice (int) - Slices of long heartbeats
revised_peaks: type: Numpy Array - Revised peaks vector corrected for the removals
'''
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

def build_hb_matrix_centered(four_lead, peaks, dimension, plotting = False):
	if len(peaks) % 2 == 0:
		four_lead = four_lead[:,:peaks[-1]]
		peaks = peaks[:-1]
	lead_length = len(four_lead[1,:]) - 1
	#Save an array of dimension Num heartbeats x 200 (heartbeat length) x Leads (4)
	fixed_dimension_hbs = np.zeros(((len(peaks)+3) // 2, dimension * 2, 4))
	for lead_num in range(4):
		#First heartbeat in data
		first_half = dsp_utils.change_dim(four_lead[lead_num, peaks[0] // 10:peaks[0]], dimension)
		second_half = dsp_utils.change_dim(four_lead[lead_num, peaks[0]:peaks[1] - (peaks[1] - peaks[0]) // 10], dimension)
		fixed_dimension_hbs[0,:,lead_num] = np.concatenate((first_half, second_half))
		#Last heartbeat in data
		first_half = dsp_utils.change_dim(four_lead[lead_num, peaks[-2] + (peaks[-1] - peaks[-2]) // 10:peaks[-1]], dimension)
		second_half = dsp_utils.change_dim(four_lead[lead_num, peaks[-1]:lead_length - (lead_length - peaks[-1]) // 10], dimension)
		fixed_dimension_hbs[(len(peaks) + 1) // 2,:,lead_num] = np.concatenate((first_half, second_half))
		
		#iterate through the rest of heartbeats
		for hb_num in range(2, len(peaks) - 1, 2):
			first_half = dsp_utils.change_dim(four_lead[lead_num, peaks[hb_num] + (peaks[hb_num + 1] - peaks[hb_num]) // 10:peaks[hb_num + 1]], dimension)
			second_half = dsp_utils.change_dim(four_lead[lead_num, peaks[hb_num + 1]:peaks[hb_num + 2] - (peaks[hb_num + 2] - peaks[hb_num + 1]) // 10], dimension)
			fixed_dimension_hbs[hb_num // 2,:,lead_num] = np.concatenate((first_half, second_half))
			
			#Periodic Visual inspection of dimension fixed heartbeat
			if plotting and hb_num % 15000 == 0:
				plt.plot(fixed_dimension_hbs[hb_num // 2,:,lead_num])
				plt.title(label = f"Lead {lead_num} hb_num {hb_num // 2}")
				plt.show()
	#plt.plot(fixed_dimension_hbs[-1,:,1])
	#plt.show()
	return fixed_dimension_hbs

def build_hb_matrix(four_lead, peaks, dimension, time, beats_per_vector = 1, plotting = False):
	peaks = np.concatenate((np.array([0]), peaks, np.array([four_lead.shape[1]])))
	if not beats_per_vector == 1:
		num_beats = len(peaks) - 1
		peaks = peaks[num_beats % beats_per_vector:]
	peaks = np.take(peaks, list(range(0, len(peaks),beats_per_vector)))
	#Save an array of dimension Num heartbeats x 100 (heartbeat length) x Leads (4)
	fixed_dimension_hbs = np.zeros((num_beats // beats_per_vector, beats_per_vector * dimension, 4))
	for lead_num in range(4):
		#First heartbeat in data
		#fixed_dimension_hbs[0,:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, 0:peaks[0]], dimension)
		#Last heartbeat in data
		#fixed_dimension_hbs[len(peaks),:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, peaks[-1]:], dimension)
		#iterate through the rest of heartbeats
		for hb_num, start_peak in enumerate(peaks[:-1]):
			individual_hb = four_lead[lead_num,start_peak + 10:peaks[hb_num+1] - 10]

			fixed_dimension_hbs[hb_num,:,lead_num] = dsp_utils.change_dim(individual_hb, beats_per_vector * dimension)
			
			#Periodic Visual inspection of dimension fixed heartbeat
			if plotting and hb_num % 15000 == 0:
				plt.plot(fixed_dimension_hbs[hb_num,:,lead_num])
				plt.show()
	#plt.plot(fixed_dimension_hbs[-1,:,1])
	#plt.show()
	return fixed_dimension_hbs

def load_np(filename):
	print("Opening file : " + filename)

	h5f = h5_interface.readh5(filename)

	lead1, lead2, lead3, lead4, time, heartrate = h5_interface.ecg_np(h5f, split = True)

	four_lead = np.vstack((lead1, lead2, lead3, lead4))

	pos_sum = dsp_utils.combine_four_lead(four_lead)
	'''
	plt.plot(pos_sum)
	plt.title("Example Heartbeat")
	plt.show()
	'''
	return lead1, lead2, lead3, lead4, time, heartrate, pos_sum
def writeout(curr_index, orig_num_hbs, four_lead, fixed_dimension_hbs, heartrate, peaks, hb_lengths, time, percent, prefix = ""):
	#logging setup
	log_filepath = os.path.join("Working_Data", "Heartbeat_Stats_Idx" + curr_index + ".txt")
	os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
	log = open(log_filepath, 'w')
	
	log.write("Average valid heartbeat length : " + str(np.average(hb_lengths)) + "\n")
	log.write("Total valid heartbeats : " + str(len(peaks))+ "\n")
	log.write("Total invalid/removed heartbeats : " + str(orig_num_hbs - len(peaks))+ "\n")
	log.write("Percent unavaliable : " + str(percent))
	#Save the four lead signals with gaps cut out
	mod_four_lead_savename = os.path.join("Working_Data", prefix + "Mod_Four_Lead_Idx" + curr_index + ".npy")
	#Save the processed heartbeat arrays
	data_savename = os.path.join("Working_Data", prefix + "Fixed_Dim_HBs_Idx" + curr_index + ".npy")
	#Save the clipped heartrate vector from the ECG machine
	hr_savename = os.path.join("Working_Data", prefix + "Cleaned_HR_Idx" + curr_index + ".npy")
	#Save the peak indicies
	peaks_savename = os.path.join("Working_Data", prefix + "HB_Peaks_Idx" + curr_index + ".npy")
	#Save the heartbeat lengths
	HB_lens_savename = os.path.join("Working_Data", prefix + "HB_Lens_Idx" + curr_index + ".npy")
	#Save the heartbeat timestamps
	HB_timestamps_savename = os.path.join("Working_Data", prefix + "HB_Timestamps_Idx" + curr_index + ".npy")
	
	np.save(mod_four_lead_savename, four_lead)
	np.save(data_savename, fixed_dimension_hbs)
	np.save(hr_savename, heartrate)
	np.save(peaks_savename, peaks)
	np.save(HB_lens_savename, hb_lengths)
	np.save(HB_timestamps_savename, time[peaks[:-1]])
	log.close()

def preprocess_seperate(filename, curr_index):
	lead1, lead2, lead3, lead4, time, heartrate, pos_sum = load_np(filename)

	peaks1 = (dsp_utils.get_peaks_dynamic(lead1, heartrate)).astype(int)
	peaks2 = (dsp_utils.get_peaks_dynamic(lead2, heartrate)).astype(int)
	peaks3 = (dsp_utils.get_peaks_dynamic(lead3, heartrate)).astype(int)
	peaks4 = (dsp_utils.get_peaks_dynamic(lead4, heartrate)).astype(int)

	mini_len = min(len(peaks1), len(peaks2), len(peaks3), len(peaks4))

	peaks_stack = np.vstack((peaks1[:mini_len], peaks2[:mini_len], peaks3[:mini_len], peaks4[:mini_len]))
	print(len(peaks1), len(peaks2), len(peaks3), len(peaks4))
	stds = np.std(peaks_stack, axis = 0)
	print(max(stds))
	print(peaks_stack[:,mini_len-1])

	
	plt.plot(lead1)
	plt.title(label = f'Patient {curr_index} Lead1')
	plt.vlines(x = peaks1, ymin = -1, ymax = 10, colors = 'r')
	plt.xlim(2e6,2e6 + 600)
	plt.ylim(-4, 4)
	plt.show()
	plt.plot(lead2)
	plt.title(label = f'Patient {curr_index} Lead2')
	plt.vlines(x = peaks2, ymin = -1, ymax = 10, colors = 'r')
	plt.xlim(2e6,2e6 + 600)
	plt.ylim(-4, 4)
	plt.show()
	plt.plot(lead3)
	plt.title(label = f'Patient {curr_index} Lead3')
	plt.vlines(x = peaks3, ymin = -1, ymax = 10, colors = 'r')
	plt.xlim(2e6,2e6 + 600)
	plt.ylim(-4, 4)
	plt.show()
	plt.plot(lead4)
	plt.title(label = f'Patient {curr_index} Lead4')
	plt.vlines(x = peaks4, ymin = -1, ymax = 10, colors = 'r')
	plt.xlim(2e6,2e6 + 600)
	plt.ylim(-4, 4)
	plt.show()
	

	plt.plot(stds)
	plt.show()

def denoise(time, lead1, lead2, lead3, lead4):
	# Removing baseline wander and high frequency noise
	lead1 = remove_noise(time, lead1)
	lead2 = remove_noise(time, lead2)
	lead3 = remove_noise(time, lead3)
	lead4 = remove_noise(time, lead4)
	return lead1, lead2, lead3, lead4, np.vstack((lead1, lead2, lead3, lead4))
'''
Inputs: Indicies of the patient files to process
Outputs: Saves multiple files of processed data, in addition to a text file log for each
'''
def preprocess_sum(filename, curr_index, beats_per_datapoint = 1, file_prefix = ""):
	lead1, lead2, lead3, lead4, time, heartrate, pos_sum = load_np(filename)
	
	peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
	peaks = peaks.astype(int)

	orig_len = len(lead1)
	orig_num_hbs = len(peaks)

	#get the bad "heartbeats"
	gap_beats = detect_gaps(pos_sum, peaks)
	'''
	for gap in gap_beats:
		plt.plot(lead1[gap])
		plt.title("Flatline \"Lead-off\" heartbeat")
		plt.xlabel("Time")
		plt.show()
	'''
	#Delete the bad heartbeats
	if gap_beats:
		if heartrate is not None:
			pos_sum, time, heartrate, lead1, lead2, lead3, lead4 = delete_slices(gap_beats, len(pos_sum), [pos_sum, time, heartrate, lead1, lead2, lead3, lead4])
		else:
			pos_sum, time, lead1, lead2, lead3, lead4 = delete_slices(gap_beats, len(pos_sum), [pos_sum, time, lead1, lead2, lead3, lead4])
		#refind peaks
		peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
		peaks = peaks.astype(int)
		
	lead1, lead2, lead3, lead4, four_lead = denoise(time, lead1, lead2, lead3, lead4)

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
	percent_unavaliable = len(lead1) / orig_len
	#print(four_lead.shape)

	'''
	#Visual Test for R-Peak identification
	plt.plot(pos_sum)
	plt.title("Example Heartbeat")
	#plt.vlines(x = peaks, ymin = 0, ymax = 8, colors = "red", linewidth = 2)
	plt.plot(peaks, pos_sum[peaks], "x")
	plt.show()
	'''
	if not beats_per_datapoint == 2:
		fixed_dimension_hbs = build_hb_matrix(four_lead, peaks, 100, time, beats_per_vector = beats_per_datapoint, plotting = False)
	else:
		fixed_dimension_hbs = build_hb_matrix_centered(four_lead, peaks, 100, plotting = False)

	#Find the lengths of the heartbeats
	hb_lengths = find_lengths(peaks, four_lead.shape[1])

	writeout(str(curr_index), orig_num_hbs, four_lead, fixed_dimension_hbs, heartrate, peaks, hb_lengths, time, percent_unavaliable, prefix = file_prefix)
	
if __name__ == "__main__":
	plotting_utils.set_font_size()
	for idx, filename in zip(indicies, get_filenames()):
		# TODO : Fix this index problem. Need to call resulting files the correct index
		idx = str(idx)
		preprocess_sum(filename, idx, beats_per_datapoint = 2)
		# preprocess_seperate(filename, idx)
