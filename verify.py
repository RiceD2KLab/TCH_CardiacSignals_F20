'''
Creates a histogram that compares the detected heartrate versus the given heartrate

This is used as an assessment of the accuracy of our peak detection algorithm 
'''

import numpy as np
import h5_interface
import os
import matplotlib.pyplot as plt


for i in range(1,60):
    try:
        diff_list = []
        heartbeat_peaks = np.load(os.path.join("Working_Data", "HB_Peaks_Idx" + str(i) + ".npy")) # indices of the detected peaks of the heartbeats in the original vector
        heartbeat_lens = np.load(os.path.join("Working_Data", "HB_Lens_Idx" + str(i) + ".npy")) # length of the heartbeat IN SAMPLES (NOT SECONDS) -> vector length is one less than peaks vector
        print("for file {}".format(i))

        h5_file = h5_interface.readh5("Reference_idx_" + str(i) + "_Time_block_1.h5")
        true_heartrate = h5_file["PARM_HR"][()] # turn h5 file into numpy array -> this is the interpolated vector of heartrate vs time

        # go through each detected peak and compare it to the given heartrate at that time
        for j, peak_idx in enumerate(heartbeat_peaks[:-1]):
            true_rate = true_heartrate[peak_idx]
            diff = (60 / true_rate) - (heartbeat_lens[j] / 240)
            diff_list.append(diff)
        plt.hist(diff_list, bins = np.arange(-0.5, 0.5, 0.05), density = True)
        plt.title("Difference in recorded and detected {}".format(i))
        plt.xlabel("Time (sec)")
        plt.ylabel("Number of Heartbeats")
        plt.savefig(os.path.join("Working_Data", "HB_Comparison_histogram_" + str(i) + ".png"))
        plt.close()
    except:
        continue
            
        




        







