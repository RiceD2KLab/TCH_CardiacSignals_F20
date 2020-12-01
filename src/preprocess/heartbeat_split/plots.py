import matplotlib.pyplot as plt
from src.preprocess import dsp_utils, h5_interface

import numpy as np
import os

from src.utils import plotting_utils
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
    #plt.xlim((4400, 4550))
    plt.ylim((-1, 10))
    # off by one error somewhere in these arrays
    plt.scatter(time[heartbeat_peaks], pos_sum[heartbeat_peaks], c='r')
    plt.title("Peak Detection for ECG Signal {} on Lead 1".format(i))
    
    plt.xlabel("time (sec)")
    plt.ylabel("voltage (mV)")
    plt.show()

def get_percent_missing(idx):
    log_filepath = os.path.join("Working_Data", "Heartbeat_Stats_Idx" + idx + ".txt")
    log = open(log_filepath, "r")
    for num, line in enumerate(log):
        if num == 3:
            label, number = line.split(":")
            number = float(number)
    return number

def percent_missing_boxplot():
    percents = []
    for idx in ['1','4','5','6','7','8','10','11','12','14','16','17','18','19','20','21','22','25','27','28','30','31','32',
                '33','34','35','37','38','39','40','41','42','44','45','46','47','48','49','50','52','53','54','55','56']:
        percents.append(get_percent_missing(idx))
    print(percents)
    plt.boxplot(x = percents)
    plt.title("Valid Signal Availability")
    plt.ylabel("Percent Valid Data")
    plt.show()

if __name__ == "__main__":
    plotting_utils.set_font_size()
    # for i in heartbeat_split.indicies:
    #     try:
    #         print(i)
    #         plot_heartbeat_on_signal(i)
    #     except():
    #         continue
    #plot_heartbeat_on_signal("31")
    percent_missing_boxplot()
    '''
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
    '''
    # plot_heartbeat_on_signal(21)
