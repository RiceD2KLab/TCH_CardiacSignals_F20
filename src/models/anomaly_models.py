import os
import sklearn
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

def train_svm(k, patient_idx, model_name):
    '''
    Trains a one class SVM for the k-dimensional reduced data.

    Inputs:
    k - int, dimension of reduced data to train on
    patient_idx - int, index of patient to train on
    model_name - a string representing a dimensionality reduction method (must be one of 'pca', 'ae', 'vae')

    Outputs:
    A trained One Class SVM for the first third of the data
    '''
    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))
    num_hbs = data.shape[0]
    train_data = data[:num_hbs//3, :] # train on first third of data

    oneclass_svm = svm.OneClassSVM(nu=0.9)
    oneclass_svm.fit(train_data)
    return oneclass_svm # -1 is outlier, 1 is inlier

def isoforest_validate(k, dim, patient_idx, model_name):
    '''
    Performs k-fold validation on normal heartbeat data for an Isolation Forest anomaly detection model. Splits data into k-folds. Tests on k-1 and 
    validates on the other. Reports the percentage of anomalies detected on the validation set.
    This is repeated using each of the folds as the validation set.

    Inputs:
    k - int, number of folds to use in k-fold cross-validation
    dim - int, dimension of reduced data to train on
    patient_idx - int, index of patient to train on
    model_name - a string representing the dimensionality reduction method used (must be one of 'pca', 'ae', 'vae')

    Outputs:
    float, average "false alarm" rate over each of the k-folds
    '''

    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, dim, patient_idx)))
    num_hbs = data.shape[0] # total length of input data
    normal_data = data[:num_hbs//2, :] # use first half of data as train/validate set

    kf = KFold(n_splits=k, shuffle=True)
    anomaly_rate = 0
    for train_index, test_index in kf.split(normal_data): # for each fold

        X_train, X_test = normal_data[train_index], normal_data[test_index]
        isoforest = IsolationForest(n_estimators=1000, max_features=0.35)
        isoforest.fit(X_train)
        labels = isoforest.predict(X_test) # predict on validation set
        num_anomalies = np.count_nonzero(labels == -1) # number of anomalies in validation set
        anomaly_rate += (num_anomalies/X_test.shape[0]) # anomaly rate for the validation set
    anomaly_rate = anomaly_rate / k # compute average anomaly rate
    return anomaly_rate


def train_isoforest(k, patient_idx, model_name):
    # data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))


    raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
    data = raw_hbs.reshape(-1, 400)  # reshape so each feature vector contains all 4 leads for each hearbeat

    num_hbs = data.shape[0]
    train_data = data[:num_hbs//2, :] # train on first third of data

    isoforest = IsolationForest(n_estimators=1000, max_features=0.35)
    isoforest.fit(train_data)
    
    return isoforest # -1 is outlier, 1 is inlier

def anomaly_tracking(k, patient_idx, model_name, detector, window_size):
    # data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))    

    raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
    data = raw_hbs.reshape(-1, 400)  # reshape so each feature vector contains all 4 leads for each hearbeat

    num_hbs = data.shape[0]

    test_data = data
    labels = detector.predict(test_data)
    anomaly_rate = []
    for i in range(0, test_data.shape[0], window_size):
        num_anomalies = np.count_nonzero(labels[i:i+window_size] == -1)
        anomaly_rate.append(num_anomalies/window_size)
    plt.plot(anomaly_rate)
    plt.vlines((num_hbs//2)/window_size, -1, 2, colors='red')
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Window Index")
    plt.ylabel("Percentage of heartbeats classified as anomalies")
    plt.title(f'Isolation Forest: anomaly rate over time for patient {patient_idx}')

    plt.show()
    return anomaly_rate


avg = []
alarm_sounded = 0
for i in [6, 56, 48, 44, 4, 39, 38, 30, 28, 14]:
    try:
        isoforest = train_isoforest(10, i, 'ae')
        anomaly_rate = anomaly_tracking(10, i, 'ae', isoforest, 500)
        if max(anomaly_rate) > 0.5:
            alarm_sounded += 1
        # avg.append(isoforest_validate(5, 10, i, 'ae'))
        # print(avg[-1])
    except:
        continue

# avg = np.mean(avg)
# print(avg)

print(alarm_sounded)