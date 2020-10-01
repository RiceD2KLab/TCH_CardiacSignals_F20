import numpy as np
import matplotlib.pyplot as plt
import os
from src.preprocess import dsp_utils, h5_interface


def peak_height_intervals(patient_num):
    '''
    Inputs: patient index
    Outputs: None, plots a scatter plot of peak height vs. inter-peak interval
    '''

    four_lead = np.load(os.path.join("Working_Data", "Four_Leads_Idx{}.npy".format(patient_num)))
    pos_sum = dsp_utils.combine_four_lead(four_lead)
    
    hb_lengths = np.load(os.path.join("Working_Data", "HB_Lens_Idx{}.npy".format(patient_num))) # length of inter-peak intervals
    peak_indices = np.load(os.path.join("Working_Data", "HB_Peaks_Idx{}.npy".format(patient_num))) # index of peaks

    peak_heights = pos_sum[peak_indices] # height of peaks on combined raw signal
    peak_heights_means = np.convolve(peak_heights, np.ones((2,))/2)[1:-1] # convolve with ones vector to get pairwise means
    
    plt.scatter(peak_heights_means, hb_lengths)
    plt.xlabel("Peak Height")
    plt.ylabel("Heartbeat duration")
    plt.title("Heartbeat duration vs. Peak Height for Patient " + str(patient_num))
    plt.show()


peak_height_intervals(6)