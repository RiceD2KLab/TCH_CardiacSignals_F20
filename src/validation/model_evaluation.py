"""
Contains functions to evaluate the performance of our model as a whole

For example, contains functions to plot key ROC curves
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from src.models.changepoint.cusum import calculate_cusum_all_patients
from src.models.changepoint.changepoint_tuning import roc_curve
from src.models.changepoint.error_metric import mean_squared_error, kl_divergence_timedelay


def plot_sweep():
    """
    Plots the AUC vs Cusum correction parameter graph
    Need to run the threshold_sweep() function beforehand to calculate the AUC score distribution
    :return:
    """
    with open('Working_Data/transfer_cdae_kl_sweep.pickle', 'rb') as handle:
        scores = pickle.load(handle)
    plt.figure(dpi=800)
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("CUSUM Correction Parameter")
    plt.ylabel("Area Under Curve")
    plt.title("Area Under Curve vs. Correction Parameter")
    plt.show(dpi=800)

    print(f"argmax correction parameter is {max(scores, key=scores.get)} yielding auc = {max(scores.values())}")


def plot_MSE_transform(patient_id):
    """
    patient_id is a string or integer denoting patient idx
    :return: plots a histogram of MSE data for the last 4 hours, as well as with a natural log transform
    """
    error_signal = mean_squared_error(100, "cdae", patient_id)
    test_error_signal = error_signal[int(len(error_signal)/3)+50:]
    plt.hist(test_error_signal, bins=60)
    plt.xlim(-3, 3)
    plt.xlabel('MSE')
    plt.ylabel('Counts')
    plt.title('MSE (Last 4 Hours): Test Patient '+str(patient_id))
    plt.show()

    plt.hist(np.log(test_error_signal), bins=60)
    plt.xlim(-3, 3)
    plt.xlabel('ln(MSE)')
    plt.ylabel('Counts')
    plt.title('ln(MSE) (Last 4 Hours): Test Patient '+str(patient_id))
    plt.show()

    # plt.hist((test_error_signal)**0.25, bins=60)
    # plt.xlim(-3,3)
    # plt.xlabel('(MSE)^0.25')
    # plt.ylabel('Counts')
    # plt.title('(MSE)^0.25 (Last 4 Hours): Test Patient '+str(idx))
    # plt.show()


def save_roc_curve():
    calculate_cusum_all_patients(0.36, "cdae", kl_divergence_timedelay)
    auc, true_positive_rates, false_positive_rates = roc_curve(True, correction=0.36, annotate=True)
    pairs = np.array([true_positive_rates, false_positive_rates])
    np.save("Working_Data/transfer_cdae_kl_roc.npy", pairs)


def compare_roc_curves():
    """
    Used to generate the figure comparing the best roc curves across the models/error metrics
    ** This function requires that the files defined in the function body exist in the Working_Data folder

    :return: nothing
    """
    plt.figure(dpi=800)
    models = ["cdae", "transfer_cdae", "lstm"]
    error_funcs = ["mse", "kl"]
    legend_items = []
    for model in models:
        for func in error_funcs:
            tpr_fpr = np.load(f"Working_Data/{model}_{func}_roc.npy")
            legend_items.append(f"{model} model with {func} error metric")
            tpr = tpr_fpr[0, :]
            fpr = tpr_fpr[1, :]
            plt.plot(fpr, tpr)

    plt.legend(["CDAE Model With MSE Error Metric",
                "CDAE Model With KL-Div. Error Metric",
                "CDAE Transfer Model With MSE Error Metric",
                "CDAE Transfer Model With KL-Div. Error Metric",
                "LSTM Model With MSE Error Metric",
                "LSTM Model With KL-Div. Error Metric"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Curves for Model/Error Metric Pairs")
    plt.show()


def plot_roc_curve_from_disk():
    """
    Plots an ROC curve if the roc curve is saved out to disk as a zipped list
    :return:
    """
    tpr_fpr = np.load(f"Working_Data/transfer_cdae_kl_roc.npy")
    tpr = tpr_fpr[0, :]
    fpr = tpr_fpr[1, :]
    print(list(zip(tpr, fpr)))
    plt.figure(dpi=500)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("True/False Positive Tradeoff")
    plt.savefig('images/transfer_cdae_kl_roc.png', dpi=1000)
    plt.show()


def compare_fall_spr_semester_results():
    """
    Compares the fall 2020 and spring 2021 results
    :return:
    """
    plt.clf()
    plt.figure(dpi=800)

    # fall semester results
    calculate_cusum_all_patients(0.05, "cdae", mean_squared_error)
    auc, tpr, fpr = roc_curve(plot=False)
    plt.plot(fpr, tpr)

    # spring semster results
    tpr_fpr = np.load(f"Working_Data/transfer_cdae_kl_roc.npy")
    tpr = tpr_fpr[0, :]
    fpr = tpr_fpr[1, :]
    print(list(zip(tpr, fpr)))
    plt.plot(fpr, tpr)

    # plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(["CDAE with MSE", "CDAE (Transfer) with KL-Div."])
    plt.title("Best ROC Curve of Fall 2020 vs. Spr 2021")
    plt.show()
    print(f"fall semester auc is {auc}")
    return


def plot_confusion_matrix():
    plt.figure(dpi=500)
    conf_matrix = np.array([[88, 34],[12, 66]])
    labels = ["Arrest", "No Arrest"]
    # metrics.plot_confusion_matrix(conf_matrix)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(include_values=True, cmap="Blues")
    plt.show()

if __name__ == "__main__":
    # roc_curve(plot=False)
    # cusum_validation(25, control=True)
    # plot_sweep()
    # calculate_cusum_all_patients(0.41, "cdae", mean_squared_error_timedelay)
    # out = roc_curve(True,  correction=0.41, annotate=True)
    # print(out)
    # print(out[0])
    # save_roc_curve()
    # compare_roc_curves()
    # plot_roc_curve_from_disk()
    # plot_confusion_matrix()
    # this compares the roc curves with different correction parameters
    # compare_fall_spr_semester_results()
    pass