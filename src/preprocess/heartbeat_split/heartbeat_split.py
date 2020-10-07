"""
"""

import numpy as np
import os

import matplotlib.pyplot as plt

from src.preprocess import dsp_utils, h5_interface

indicies = ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
				'33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']
if __name__ == "__main__":
	
	for curr_index in indicies:
		print("Starting on index : " + str(curr_index))
		filename = "Reference_idx_" + curr_index + "_Time_block_1.h5"

		h5f = h5_interface.readh5(filename)

		four_lead, time, heartrate = h5_interface.ecg_np(h5f)
		lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
		lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

		pos_sum = dsp_utils.combine_four_lead(four_lead)
		
		if heartrate is not None:
			peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
		else:
			peaks =  dsp_utils.get_peaks_prominence(pos_sum)
		peaks = peaks.astype(int)

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
				bad_hbs.append(slice(peaks[i-1], peaks[i]))
		vals = np.unique(pos_sum[peaks[-1]:])
		#Trailing heartbeat
		if len(vals) < .05 * (len(pos_sum) - peaks[-1]):
			bad_hbs.append( slice(peaks[-1], len(pos_sum) - 1) )
		#Delete the bad heartbeats
		if len(bad_hbs) > 0:
			ind = np.indices(pos_sum.shape)[0]
			rm = np.hstack(list(ind[i] for i in bad_hbs))
			good_indices = sorted(set(ind)-set(rm))

			pos_sum = np.take(pos_sum, good_indices)
			time = np.take(time, good_indices)

			if heartrate is not None:
				heartrate = np.take(heartrate, good_indices)
				
			lead1 = np.take(lead1, good_indices)
			lead2 = np.take(lead2, good_indices)
			lead3 = np.take(lead3, good_indices)
			lead4 = np.take(lead4, good_indices)
			
			#refind peaks
			if heartrate is not None:
				peaks = dsp_utils.get_peaks_dynamic(pos_sum, heartrate) # indices on the signal where we found a peak
			else:
				peaks =  dsp_utils.get_peaks_prominence(pos_sum)
			peaks = peaks.astype(int)
			
			four_lead = np.vstack((lead1, lead2, lead3, lead4))
		
		"""
		#Visual Test for R-Peak identification
		plt.plot(pos_sum)
		# plt.vlines(x = peaks, ymin = 0, ymax = 8, colors = "red", linewidth = 2)
		plt.plot(peaks, pos_sum[peaks], "x")
		plt.show()
		"""
		
		log_filepath = os.path.join("Working_Data", "Heartbeat_Stats_Idx" + curr_index + ".txt")
		os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
		log = open(log_filepath, 'w')

		hb_lengths = np.zeros((len(peaks),)) 		#array of R-Peak identified heartbeat segment lengths
		for i in range(1, len(peaks)):
			hb_lengths[i] = peaks[i] - peaks[i-1]	#calculate the legnths
		hb_lengths = hb_lengths[1:] 				#remove the leading 0
		np.concatenate( (hb_lengths,np.array([four_lead.shape[1]-peaks[-1]]) ) ) #append the last length


		#Find the maximum length
		maximum_hb_len = 100
		
		log.write("Average heartbeat length before outlier removal : " + str(np.average(hb_lengths)) + "\n")
		log.write("Total valid heartbeats : " + str(len(peaks))+ "\n")
		log.write("Total invalid heartbeats : " + str(len(vals))+ "\n")
		log.write("Average valid heartbeat length : " + str(np.average(hb_lengths))+ "\n")

		#Num heartbeats x Max heartbeat length x Leads (4) 3D array
		fixed_dimension_hbs = np.zeros((len(peaks), maximum_hb_len, 4))
		for lead_num in range(4):
			#First heartbeat in data
			fixed_dimension_hbs[0,:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, 0:peaks[0]], maximum_hb_len)
			#Last heartbeat in data
			fixed_dimension_hbs[len(peaks) - 1,:,lead_num] = dsp_utils.change_dim(four_lead[lead_num, peaks[-1]:], maximum_hb_len)
			#iterate through the rest of heartbeats
			for hb_num, peak in enumerate(peaks[1:-1], start = 1):
				individual_hb = four_lead[lead_num,peaks[hb_num]:peaks[hb_num+1]]
				fixed_dimension_hbs[hb_num,:,lead_num] = dsp_utils.change_dim(individual_hb, maximum_hb_len)
				"""
				#Periodic Visual inspection of dimension fixed heartbeat
				if hb_num % 15000 == 0:
					plt.plot(fixed_dimension_hbs[hb_num,:,lead_num])
					plt.show()
				"""



		data_savename = os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + curr_index + ".npy")
	
		peaks_savename = os.path.join("Working_Data", "HB_Peaks_Idx" + curr_index + ".npy")
		HB_lens_savename = os.path.join("Working_Data", "HB_Lens_Idx" + curr_index + ".npy")
		
		np.save(data_savename, fixed_dimension_hbs)
	
		np.save(peaks_savename, peaks)
		np.save(HB_lens_savename, hb_lengths)
		log.close()
		