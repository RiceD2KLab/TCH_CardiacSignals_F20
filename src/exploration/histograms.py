import matplotlib.pyplot as plt
import numpy as np
import os




def create_histogram(filename, index,dim_red_method):
    '''
    Creates histograms for data of 1 dimensional dimension reductions of hb data, differentiating between the first and
    second halves of the data to check if anamalous data is visually detectable
    :param filename: [string] A string reference to an npy file with 1 Dimensional data
    :param index: [int] The index of the patient used for reference in the plot title
    :param dim_red_method: The method of dimension reduction used for reference in the plot title
    :return None:
    '''
    patient_data = np.load(filename) #load in patient data
    full_data = patient_data.reshape(-1,1)
    data_stack = np.hstack([full_data[int(len(full_data)/2):],full_data[:int(len(full_data)/2)]]) #split data in half
    plt.figure() #initialize and create histogram
    plt.hist(data_stack,bins=[-4,-3,-2,-1,0,1,2,3,4,5],stacked=True,label = ['First Half','Second Half'])
    plt.title(f"{dim_red_method.upper()}, Patient # {index}, All Leads")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #An example figure created using histograms.py for patient 16 dimension reduced with pca
    dim_red = 'pca'
    file_index = 16
    filename = os.path.join("Working_Data", f"reduced_{dim_red}_1d_Idx{str(file_index)}.npy")

    create_histogram(filename,file_index,dim_red)