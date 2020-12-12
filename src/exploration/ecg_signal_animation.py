import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import h5_interface

##### READ IN THE DATA
filename = 'Reference_idx_16_Time_block_1.h5'
h5f = h5_interface.readh5(filename)

four_lead, time, heartrate = h5_interface.ecg_np(h5f)
lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

# plt.plot(time[0:400], lead1[0:400])
# plt.show()

plt.rcParams["figure.figsize"] = [16, 4]
########################################################################################################################
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
#
# # Query the figure's on-screen size and DPI. Note that when saving the figure to
# # a file, we need to provide a DPI for that separately.
# print('fig size: {0} DPI, size in inches {1}'.format(
#     fig.get_dpi(), fig.get_size_inches()))
#
# # Plot a scatter that persists (isn't redrawn) and the initial line.
# #x = np.arange(0, 20, 0.1)
# #ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
# line, = ax.plot(time[0:3000], lead1[0:3000], 'b-', linewidth=2)
# line2, = ax.plot(np.linspace(time[0+300],time[0+300],100), np.linspace(-1,1,100), color='red')
# plt.ylim([-1, 0.8])
# plt.xlim([time[0], 5.5])
# ax.set_ylabel('Amplitude', fontsize=20)
#
# def update(i):
#     label = 'Time (sec)'
#     print(label)
#     # Update the line and the axes (with a new xlabel). Return a tuple of
#     # "artists" that have to be redrawn for this frame.
#     # plt.xlim([min(time[0:i + 600]), max(time[i:i + 600])])
#     # line.set_xdata(time[0:i + 600])
#     # line.set_ydata(lead1[0:i + 600])
#     line2.set_xdata(np.linspace(time[i+300],time[i+300],100))
#     #plt.axvline(time[i+300], color='red')  # cutoff frequency
#
#     ax.set_xlabel(label, fontsize=20)
#     return line, ax
#
# if __name__ == '__main__':
#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 200ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 1000, 2), interval=20)
#     if len(sys.argv) > 1 and sys.argv[1] == 'save':
#         anim.save('ECG.gif', dpi=80, writer='imagemagick')
#     else:
#         anim.save('ECG.gif', dpi=80, writer='imagemagick')
#         # plt.show() will just loop the animation forever.
########################################################################################################################
fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

# Plot a scatter that persists (isn't redrawn) and the initial line.
#x = np.arange(0, 20, 0.1)
#ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))

line, = ax.plot(time[0:300], np.linspace(0, 0, 300), 'g-', linewidth=2)
line2, = ax.plot(np.linspace(time[0+300], time[0+300], 100), np.linspace(-1, 2, 100), color='red')

plt.ylim([0, 1.2])
plt.xlim([time[0], 5.5])

ax.set_ylabel('Detection Metric', fontsize=20)

def update(i):
    label = 'Time (sec)'
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    #plt.xlim([min(time[i:i + 600]), max(time[i:i + 600])])
    #print(len(time[0:i + 300]))

    # if i < 500:
    #     a = (np.linspace(0, 0, 300+i))
    #     line.set_ydata(a)
    # else:
    #     a = (np.linspace(0, 0, 300 + 500))
    #     b = (np.linspace(1, 1, i - 500))
    #     line.set_ydata(np.concatenate((a, b), axis=0))



    #print(np.concatenate((a, b), axis=0))
    #print(np.concatenate(np.linspace(0, 0, 100), np.linspace(0, 0, i), axis=1)) #np.array([1])) ) )

    time_vector = time[0:i + 300]
    line.set_xdata(time_vector)

    sigmoid = 1 / (1 + np.exp(12*(-time_vector+4)))
    line.set_ydata(sigmoid)

    line2.set_xdata(np.linspace(time[i+300],time[i+300],100))

    ax.set_xlabel(label, fontsize=20)
    return line, ax

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, 1000, 2), interval=20)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('Detection.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        anim.save('Detection.gif', dpi=80, writer='imagemagick')
        #plt.show()