import os
import sklearn
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection


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

def isoforest_validate(k, dim, patient_idx, model_name, params):
    '''
    Performs k-fold validation on normal heartbeat data for an Isolation Forest anomaly detection model. Splits data into k-folds. Tests on k-1 and 
    validates on the other. Reports the percentage of anomalies detected on the validation set.
    This is repeated using each of the folds as the validation set.

    Inputs:
    k - int, number of folds to use in k-fold cross-validation
    dim - int, dimension of reduced data to train on
    patient_idx - int, index of patient to train on
    model_name - a string representing the dimensionality reduction method used (must be one of 'pca', 'ae', 'vae')
    params - a dict with keys for 'n_estimators', 'max_features', and 'contamination', containing hyperparams for isolation forest

    Outputs:
    float, average "false alarm" rate over each of the k-folds
    '''

    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, dim, patient_idx)))
    num_hbs = data.shape[0] # total length of input data
    normal_data = data[:num_hbs//3, :] # use first third of data as train/validate set

    kf = KFold(n_splits=k, shuffle=True)
    anomaly_rate = 0
    for train_index, test_index in kf.split(normal_data): # for each fold

        X_train, X_test = normal_data[train_index], normal_data[test_index]
        isoforest = IsolationForest(n_estimators=params['n_estimators'], max_features=params['max_features'], contamination=params['contamination'])
        isoforest.fit(X_train)
        labels = isoforest.predict(X_test) # predict on validation set
        num_anomalies = np.count_nonzero(labels == -1) # number of anomalies in validation set
        anomaly_rate += (num_anomalies/X_test.shape[0]) # anomaly rate for the validation set
    anomaly_rate = anomaly_rate / k # compute average anomaly rate
    return anomaly_rate


def train_isoforest(k, patient_idx, model_name):
    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))

    # raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
    # data = raw_hbs.reshape(-1, 400)  # reshape so each feature vector contains all 4 leads for each hearbeat

    num_hbs = data.shape[0]
    train_data = data[:num_hbs//3, :] # train on first third of data

    isoforest = IsolationForest(n_estimators=300, max_features=0.6, contamination=0.1)
    isoforest.fit(train_data)
    print("trained")
    return isoforest # -1 is outlier, 1 is inlier

def anomaly_tracking(k, patient_idx, model_name, detector, window_size):
    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))    

    # raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
    # data = raw_hbs.reshape(-1, 400)  # reshape so each feature vector contains all 4 leads for each hearbeat

    num_hbs = data.shape[0]

    test_data = data
    labels = detector.predict(test_data)
    anomaly_rate = []
    for i in range(0, test_data.shape[0], window_size):
        num_anomalies = np.count_nonzero(labels[i:i+window_size] == -1)
        anomaly_rate.append(num_anomalies/window_size)
    plt.plot(anomaly_rate)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Window Index")
    plt.ylabel("Percentage of heartbeats classified as anomalies")
    plt.title(f'Isolation Forest: anomaly rate over time for patient {patient_idx}')
    plt.vlines(num_hbs//(3*window_size), -0.1, 1.1, colors='red')
    plt.show()
    return anomaly_rate

def get_metrics(metric_type, dim, idx, model, window_size, PLOT=False):
    '''
    Returns list containing metric over time for given window size 

    metric_type: Metric type used "isolation_forest"
    dim: dimension of reduced data
    idx: patient index
    model: dimension reduction method used
    window_size: 
    PLOT: bool, whether to show plots
    '''

    anomaly_rate = None
    if metric_type == "isolation_forest":
        isoforest = train_isoforest(dim, idx, model)
        anomaly_rate = anomaly_tracking(dim, idx, model, isoforest, window_size)
    return anomaly_rate


avg = []
for i in range(60):
    try:
        isoforest = train_isoforest(10, i, 'ae')
        anomaly_rate = anomaly_tracking(10, i, 'ae', isoforest, 500)
        # avg.append(isoforest_validate(5, 10, i, 'ae'))
        # print(avg[-1])
    except:
        continue

# best_params = {}
# best_score = 1.0
# for n_estimators in range(100, 1200, 100):
#     for contamination in [0.1, 0.2, 0.3, 0.4, 0.5]:
#         for max_features in np.linspace(0.1, 1.0, 10):
#             params = {'n_estimators': n_estimators, 
#               'contamination': contamination, 
#               'max_features': max_features}
#             score = isoforest_validate(5, 10, 1, 'ae', params)
#             print(score)
#             if score < best_score:
#                 best_score = score
#                 best_params = params
# print(best_params)
# print(best_score)

### {'n_estimators': 300, 'contamination': 0.1, 'max_features': 0.6} yields validation error of 0.09889838824352036

# print(isoforest_validate(5, 10, 1, 'ae', params = {'n_estimators': 300, 
#               'contamination': 0.1, 
#               'max_features': 0.6}))
