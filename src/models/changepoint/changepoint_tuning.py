"""
Contains functions to aid in hyperparameter tuning for the changepoint algorithm
The main function is roc_curve, which generates an ROC curve based on the output fo the changepoint algorithm
"""
from src.models.changepoint.cusum import *
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

from src.validation.changepoint_evaluation import cusum_validation


def roc_curve(plot=True, correction=None, annotate=True):
    """
    Plot Receiver Operating Characteristic curve (i.e. true positive vs false positive rate)
    True Positive Rate = True Positives / (True Positives + False Negatives)
    False Positive Rate = False Positives / (False Positives + True Negatives)
    :return: nothing
    """

    # thresholds = list(range(0, 101, 1)) # use this for LSTM
    thresholds = list(range(0, 1000, 5))  # use this for CDAE

    # initialize the true/false postive rates with (1,1) since ROC curves must pass through (0,0) and (1,1)
    true_positive_rates = [1.0]
    false_positive_rates = [1.0]

    # annotations = list(range(0, 105, 5))  # use this for LSTM
    annotations = list(range(0,2600, 200))  # use this for CDAE
    annotation_coords = [] # so we can annotate these points on the scatterplot

    for i in thresholds:

        test_count, test_total, avg_time, sem_time = cusum_validation(i, control=False)

        true_positive = test_count
        false_negative = test_total - test_count

        control_count, control_total, avg_time, sem_time = cusum_validation(i, control=True)
        false_positive = control_count
        true_negative = control_total - control_count

        # transform this into a form usable by sklearn
        # we know that all the test group should be 1s and all the control group should be 0s
        # y_true = np.concatenate(np.ones(test_total), np.zeros(control_total))
        # then we concatenate the actual output
        # y_pred = np.concatenate(np.ones(true_positive), np.zeros(false_negative), np.ones(false_positive), np.zeros(true_negative))

        tpr = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        true_positive_rates.append(tpr)
        false_positive_rates.append(fpr)

        if i in annotations:
            annotation_coords.append((fpr, tpr))

    # append a (1,1) tpr,fpr coordinate so that the tpr, fpr are guaranteed to vary from 0 to 1
    true_positive_rates.append(0.0)
    false_positive_rates.append(0.0)

    if plot:
        plt.figure()
        plt.plot(false_positive_rates, true_positive_rates)
        if annotate:
            plt.scatter(false_positive_rates, true_positive_rates, c=['r'])
            for threshold, coord in zip(annotations, annotation_coords):
                plt.annotate(str(threshold), coord)
        plt.title(f"ROC Curve with c={correction} for Cusum Thresholds from {thresholds[0]} to {thresholds[-1]}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
    #
    # calculate AUC (area under curve)
    auc = metrics.auc(false_positive_rates, true_positive_rates)
    print(f"AUC-ROC score is {auc}")
    print(list(zip(thresholds, true_positive_rates[1:-1], false_positive_rates[1:-1])))
    return auc, true_positive_rates, false_positive_rates


def recall_v_threshold():
    """
    Creates a graph of recall (num detected cardiac arrests / num actual cardiac arrests) vs threshold
    Allows us to visualize true positive rate as a function of threshold
    Use threshold_correction_sweep for more holistic picture
    :return: nothing
    """

    thresholds = list(range(0, 10000, 50))
    recalls = []
    detection_times = []

    for i in thresholds:
        count, total, avg_time, sem_time = cusum_validation(i)
        recalls.append(count / total)
        detection_times.append(avg_time)

    set_font_size()
    plt.plot(thresholds, recalls)
    plt.xlabel(r"CuRE Threshold ($\gamma$)")
    plt.ylabel("Recall")
    plt.show()

    plt.plot(thresholds, detection_times)
    plt.show()

    check_idx = thresholds.index(500)
    print(f"500 threshold detection time is {detection_times[check_idx]}")
    # print(f"500 threshold detection sem is {detection_times[check_idx]}")
    print(f"500 threshold recall is {recalls[check_idx]}")

    return


def threshold_correction_sweep(model_name):
    """
    Performs a sweep over the threshold and correction parameters for changepoint
    Allows us to see model performance as a function of the cusum correction parameter
    @param model_name: the model used (LSTM, conv AE, etc)
    @return: the AUC scores for each of the c values
    """

    all_patients = get_patient_ids(control=False) + get_patient_ids(control=True)

    correction_sweep = np.arange(0, 1, 0.03)
    auc_scores = {}

    for c in correction_sweep:
        for idx in all_patients:
            try:
                cusum(idx, model_name, 100, kl_divergence_timedelay, save=True, correction=c)
            except Exception as e:
                print(e)
                pass

        auc_scores[c] = roc_curve(plot=False)[0]
        print(f"AUC SCORE FOR C = {c} is {auc_scores[c]}")

    return auc_scores


if __name__ == "__main__":
    ## sweep through the correction parameter and save out to a file since this is an expensive computation
    # sweep = threshold_correction_sweep("cdae")
    # print(sweep)
    # with open('Working_Data/cdae_kl_sweep.pickle', 'wb') as handle:
    #     pickle.dump(sweep, handle)


    pass

