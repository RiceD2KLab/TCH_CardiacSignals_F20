import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft

from src.preprocess import h5_interface

fs = 240


# Test the noise removal on patient 1

filename = 'Reference_idx_1_Time_block_1.h5'
h5f = h5_interface.readh5(filename)

four_lead, time, heartrate = h5_interface.ecg_np(h5f)
lead1, lead2, lead3, lead4 = np.vsplit(four_lead, 4)
lead1, lead2, lead3, lead4 = [lead1[0], lead2[0], lead3[0], lead4[0]]

print(np.shape(lead1))
print(np.shape(lead1[0:720000]))


print(len(lead1[-1000000:])/72000)
#print(np.amax(np.log(np.abs(Zxx))))
#print(np.shape(Zxx))


f, t, Zxx = signal.stft(lead1[-1500000:], fs, nperseg=72000, noverlap=None)
plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx)), vmin=np.amin(np.log(np.abs(Zxx))), vmax=np.amax(np.log(np.abs(Zxx))), shading='gouraud')
plt.title('Lead 1 STFT Magnitude (logscale)')
plt.ylabel('Frequency [Hz]')
# plt.xlim(-10, 0)
plt.xlabel('Time [min]')
plt.show()

f, t, Zxx = signal.stft(lead2[-1500000:], fs, nperseg=72000, noverlap=None)
plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx)), vmin=np.amin(np.log(np.abs(Zxx))), vmax=np.amax(np.log(np.abs(Zxx))), shading='gouraud')
plt.title('Lead 2 STFT Magnitude (logscale)')
plt.ylabel('Frequency [Hz]')
# plt.xlim(-10, 0)
plt.xlabel('Time [min]')
plt.show()

f, t, Zxx = signal.stft(lead3[-1500000:], fs, nperseg=72000, noverlap=None)
plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx)), vmin=np.amin(np.log(np.abs(Zxx))), vmax=np.amax(np.log(np.abs(Zxx))), shading='gouraud')
plt.title('Lead 3 STFT Magnitude (logscale)')
plt.ylabel('Frequency [Hz]')
# plt.xlim(-10, 0)
plt.xlabel('Time [min]')
plt.show()

f, t, Zxx = signal.stft(lead4[-1500000:], fs, nperseg=72000, noverlap=None)
plt.pcolormesh(t/60-np.max(t/60), f, np.log(np.abs(Zxx)), vmin=np.amin(np.log(np.abs(Zxx))), vmax=np.amax(np.log(np.abs(Zxx))), shading='gouraud')
plt.title('Lead 4 STFT Magnitude (logscale)')
plt.ylabel('Frequency [Hz]')
# plt.xlim(-10, 0)
plt.xlabel('Time [min]')
plt.show()















# lead1_filtered = remove_noise(time, lead1, plots=True)
# plt.plot(time, lead1, 'k-', label='input')
# plt.plot(time, lead1_filtered, 'c-', linewidth=1.5, label='filtered')
# plt.title('Lead Data')
# plt.xlabel('Time [sec]')
# plt.ylabel('Amplitude')
# plt.legend(loc='best')
# plt.xlim(5, 8)
# plt.ylim(-2, 2)
# plt.show()
