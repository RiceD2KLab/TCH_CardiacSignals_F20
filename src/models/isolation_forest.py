import os
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from src.preprocessing import heartbeat_split
from src.utils.plotting_utils import set_font_size
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from src.utils.dsp_utils import get_windowed_time


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

    data = np.load("Working_Data/" + "reduced_{}_{}d_Idx{}.npy".format(model_name, dim, patient_idx))
    num_hbs = data.shape[0] # total length of input data
    normal_data = data[:num_hbs//3, :] # use first third of data as train/validate set

    kf = KFold(n_splits=k, shuffle=True)
    anomaly_rate = 0
    for train_index, test_index in kf.split(normal_data): # for each fold

        X_train, X_test = normal_data[train_index], normal_data[test_index]
        isoforest = IsolationForest(n_estimators=params['n_estimators'], max_features=params['max_features'])
        isoforest.fit(X_train)
        labels = isoforest.predict(X_test) # predict on validation set
        num_anomalies = np.count_nonzero(labels == -1) # number of anomalies in validation set
        anomaly_rate += (num_anomalies/X_test.shape[0]) # anomaly rate for the validation set
    anomaly_rate = anomaly_rate / k # compute average anomaly rate
    print(anomaly_rate)
    return anomaly_rate


def train_isoforest(k, patient_idx, model_name):
    """

    :param k: dimension of data
    :param patient_idx: integer index of patient
    :param model_name: name of model used in filename, ex. ae for autoencoder
    :return: trained isoforest model
    """
    data = np.load(os.path.join("Working_Data", "reduced_{}_{}d_Idx{}.npy".format(model_name, k, patient_idx)))

    # raw_hbs = np.load(os.path.join("Working_Data", "Normalized_Fixed_Dim_HBs_Idx{}.npy".format(str(patient_idx))))
    # data = raw_hbs.reshape(-1, 400)  # reshape so each feature vector contains all 4 leads for each hearbeat

    num_hbs = data.shape[0]
    train_data = data[:num_hbs//3, :] # train on first third of data

    isoforest = IsolationForest(n_estimators=800, max_features=0.5)
    # 800,0.8,0.015

    isoforest.fit(train_data)
    print("trained")
    return isoforest # -1 is outlier, 1 is inlier

def isoforest_hyperparams(n_estimator, contamination, max_features):
    """
    Hyperparameter tuning for isolation forest model
    :param n_estimator: list of potential n_estimator values
    :param contamination: list of potential contamination values (must be 0 to 0.5_
    :param max_features: list of potential max_features
    :return: prints best_params and best score
    """
    # TODO: remove the hardcoded values and put the variables
    best_params = {}
    best_score = 1.0
    for n_estimators in range(700, 1100, 100):
        for max_features in np.linspace(0.1, 1.0, 10):
            params = {'n_estimators': n_estimators,
                'max_features': max_features}
            score = isoforest_validate(5, 100, 1, 'cdae', params)
            print(score)
            if score < best_score:
                best_score = score
                best_params = params
    print(best_params)
    print(best_score)

def anomaly_tracking(k, patient_idx, model_name, detector, window_size):
    """

    :param k: dimension of data
    :param patient_idx: integer index of patient
    :param model_name: name of model used in filename, ex. ae for autoencoder
    :param detector:
    :param window_size: probably need to add a blurb about how to modify this
    :return: anomaly rate
    """
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
    # plt.plot(anomaly_rate)
    # plt.ylim(-0.1, 1.1)
    # plt.rcParams.update({'font_size':30})
    # plt.xlabel("Window Index")
    # plt.ylabel("Percentage of heartbeats classified as anomalies")
    # plt.title(f'Isolation Forest: anomaly rate over time for patient {patient_idx}')
    # plt.vlines(num_hbs//(3*window_size), -0.1, 1.1, colors='red')
    # plt.show()
    return anomaly_rate
# save anomaly rate array as windowed_var_100d_idx{idx}.npy

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



def isoforest_box_plot(indices, model_name, dimension):
    anomaly_rates = []
    window_times = np.linspace(-4, 0, 49)

    for idx in indices:
        try:
            current_values = np.load(f"Working_Data/unwindowed_if_100d_Idx{idx}.npy") # load anomaly rates for this patient
            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)
            test_values = []
            for i in range(len(window_times) - 1):
                indices = np.squeeze(np.argwhere(np.logical_and(patient_times >= window_times[i], patient_times < window_times[i+1])))
                if len(indices) == 0:
                    test_values.append(-1) # marker for no data available in this window
                else:
                    window_labels = current_values[indices] # labels for all data points found in current window
                    test_values.append(np.mean(window_labels))
            anomaly_rates.append(test_values)
        except:
            print("No File Found: Patient " + idx)
            continue
    anomaly_rates = np.array(anomaly_rates).T.tolist()

    for row in anomaly_rates:
        while -1 in row: row.remove(-1)

    set_font_size()
    plt.boxplot(anomaly_rates, showfliers=False, positions=window_times[:-1], widths=1/15,
        medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'), capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))
    
    plt.title("Isolation Forest Anomaly Rate Distribution Over Time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("Anomaly Rate")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.xlim(-4.2, 0.2)
    plt.savefig('images/isoforest_boxplot.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    # isoforest_hyperparams(1, 2, 3)
    isoforest_box_plot(heartbeat_split.indicies, "cdae", 100)
    # avg = []
    # for i in range(60):
    #     try:
    #         isoforest = train_isoforest(100, i, "cdae")
    #         anomaly_rate = anomaly_tracking(100, i, "cdae", isoforest, 50)
    #         # filename = "Working_Data/windowed_if_100d_idx{}_NEW.npy".format(i)
    #         filename = os.path.join("Working_Data", f"windowed_if_100d_Idx{i}.npy")
    #         np.save(filename, anomaly_rate)
    #         print("Done")
    # #         # avg.append(isoforest_validate(5, 10, i, 'ae'))
    # #         # print(avg[-1])
    #     except:
    #         continue





 # 800, 0.015 contamination, 0.8 max features

### {'n_estimators': 300, 'contamination': 0.1, 'max_features': 0.6} yields validation error of 0.09889838824352036

