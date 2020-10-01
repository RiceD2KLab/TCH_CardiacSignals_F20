import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())  # lmao "the tucker hack"

file_index = 1
patient_idx = file_index
dim_red = 'pca'
test = np.load(os.path.join("Working_Data", f"reduced_{dim_red}_1d_Idx{str(file_index)}.npy"))
print(test.shape)
plt.figure()
full_data = test.reshape(-1,1)

data_stack = np.hstack([full_data[int(len(full_data)/2):],full_data[:int(len(full_data)/2)]])
print(min(full_data))
print(max(full_data))
plt.hist(data_stack,bins=[-4,-3,-2,-1,0,1,2,3,4,5],stacked=True,label = ['First Half','Second Half'])

plt.title(f"{dim_red.upper()}, Patient # {patient_idx}, All Leads")
plt.legend()
plt.show()