import matplotlib.pyplot as plt
import numpy as np

import os

'''
Inputs : Index of the data to plot
Outputs : Histogram plot of heartbeat lengths
'''
def plot_hb_lengths(idx):
	HB_lens_savename = os.path.join("Working_Data", "HB_Lens_Idx" + str(idx) + ".npy")
	hb_lengths = np.load(HB_lens_savename)

	plt.hist(hb_lengths, bins = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180], density = True)
	plt.show()