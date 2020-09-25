import matplotlib.pyplot as plt
from src.preprocess import dsp_utils, h5_interface

import numpy as np
import os

'''
Inputs : Index of the data to plot
Outputs : Histogram plot of heartbeat lengths
'''


def plot_hb_lengths(idx):
    HB_lens_savename = os.path.join(
        "Working_Data", "HB_Lens_Idx" + str(idx) + ".npy")
    hb_lengths = np.load(HB_lens_savename)
    b = np.asarray([60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
                    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]) / 240
    # b = b / [240 for i in range(len(b))]
    plt.hist(hb_lengths / 240, bins=list(b), density=False)
    plt.title("Heartbeat Length (sec) for File Index {}".format(idx))
    plt.xlabel("Time (sec)")
    plt.ylabel("Number of heartbeats")
    plt.savefig(os.path.join("Working_Data",
                             "HB_histogram_" + str(i) + ".png"))
    plt.close()


def plot_heartbeat_on_signal(i):
    # indices of the detected peaks of the heartbeats in the original vector
    heartbeat_peaks = np.load(os.path.join(
        "Working_Data", "HB_Peaks_Idx" + str(i) + ".npy"))

    # read in the four leads and sum together
    filename = "Reference_idx_" + str(i) + "_Time_block_1.h5"
    h5f = h5_interface.readh5(filename)

    four_lead, time, heartrate = h5_interface.ecg_np(h5f) 
    pos_sum = dsp_utils.combine_four_lead(four_lead)
    print(np.size(pos_sum))
    #############

    # turn h5 file into numpy array -> this is the interpolated vector of heartrate vs time

    plt.plot(time, pos_sum)
    plt.xlim((4400, 4550))
    plt.ylim((-1, 10))
    # off by one error somewhere in these arrays
    plt.scatter(time[heartbeat_peaks], pos_sum[heartbeat_peaks], c='r')
    plt.title("Peak Detection for ECG Signal {} on Lead 1".format(i))
    
    plt.xlabel("time (sec)")
    plt.ylabel("voltage (mV)")
    plt.show()


if __name__ == "__main__":
    # for i in heartbeat_split.indicies:
    #     try:
    #         print(i)
    #         plot_heartbeat_on_signal(i)
    #     except():
    #         continue
    #plot_heartbeat_on_signal("33")
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("FileIdx", help="index number of the TCH h5 ECG file to be plotted")
    args = parser.parse_args()
    i = 11
    filename = "Reference_idx_" + args.FileIdx + "_Time_block_1.h5"
    h5f = h5_interface.readh5(filename)
    four_lead, time, heartrate = h5_interface.ecg_np(h5f) 
    pos_sum = dsp_utils.combine_four_lead(four_lead)
    plt.plot(time, pos_sum)
    plt.show()
    
    # plot_heartbeat_on_signal(21)
