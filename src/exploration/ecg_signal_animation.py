"""
Generate .gif files for animation of heartbeat and instability detection (for the 5 minute video).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.utils import h5_interface

########################################################################################################################
# Read in the data
filename = 'Reference_idx_16_Time_block_1.h5'
h5f = h5_interface.readh5(filename)

four_lead, time, heartrate = h5_interface.ecg_np(h5f)
lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

########################################################################################################################
# Generate gif for an ECG signal
plt.rcParams["figure.figsize"] = [12, 3]
fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Plot an ECG signal that persists
line, = ax.plot(time[0:3000], lead1[0:3000], 'b-', linewidth=2)
line2, = ax.plot(np.linspace(time[0 + 300], time[0 + 300], 100), np.linspace(-1, 1, 100), color='red')
plt.ylim([-1, 0.8])
plt.xlim([time[0], 5.5])
ax.set_ylabel('Amplitude', fontsize=20)
ax.set_xlabel('Time (sec)', fontsize=20)


# Input function for the updating the gif animation
def update_ecg_gif(i):
    """
    Moves a red vertical line across the x-axis (time)
    :param i: [int] an iterator
    :return: None
    """
    line2.set_xdata(np.linspace(time[i + 300], time[i + 300], 100))


# Save and show the gif
anim = FuncAnimation(fig, update_ecg_gif, frames=np.arange(0, 1000, 2), interval=20)
anim.save('images//ECG.gif', dpi=80, writer='imagemagick')
plt.show()

########################################################################################################################
# Generate gif for an example detection of heartbeat instability
fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Plot a detection metric signal that persists
line, = ax.plot(time[0:300], np.linspace(0, 0, 300), 'g-', linewidth=2)
line2, = ax.plot(np.linspace(time[0 + 300], time[0 + 300], 100), np.linspace(-1, 2, 100), color='red')
plt.ylim([0, 1.2])
plt.xlim([time[0], 5.5])
ax.set_ylabel('Detection Metric', fontsize=20)
ax.set_xlabel('Time (sec)', fontsize=20)


# Input function for the updating the gif animation
def update_detection_gif(i):
    """
    Moves a red vertical line across the x-axis (time), plotting out a sigmoid when instability is detected.
    :param i: [int] an iterator
    :return: None
    """
    time_vector = time[0:i + 300]
    line.set_xdata(time_vector)

    sigmoid = 1 / (1 + np.exp(12 * (-time_vector + 4)))
    line.set_ydata(sigmoid)

    line2.set_xdata(np.linspace(time[i + 300], time[i + 300], 100))


# Save and show the gif
anim = FuncAnimation(fig, update_detection_gif, frames=np.arange(0, 1000, 2), interval=20)
anim.save('images//Detection.gif', dpi=80, writer='imagemagick')
plt.show()

########################################################################################################################
