from src.models.changepoint import *
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pickle

from src.models.changepoint import calculate_cusum_all_patients
from src.models.mse import mean_squared_error


def cusum_box_plot(patient_indices, model_name, dimension):
    """
    Generates figure of boxplots over time of CUSUM scores across the entire patient cohort
    :param patient_indices: [list(int)] list of patient identifiers to use
    :param model_name: [string] model type used for the desired data (typically "cdae")
    :param dimension: [int] dimension used for reduction in the above model (typically 100)
    :return: None (saves plot of CUSUM distributions over time)
    """

    cusum_values = []
    window_times = np.linspace(-4, 0, 49)  # use 50 windows

    for idx in patient_indices:
        try:
            current_values = np.load(
                f"Working_Data/unwindowed_cusum_100d_Idx{idx}.npy")  # load anomaly rates for this patient

            if len(current_values) < 1000:
                raise e

            patient_times = get_windowed_time(idx, num_hbs=10, window_size=1)[:-1]
            # print(len(current_values))
            # print(len(patient_times))
            test_values = []
            for i in range(len(window_times) - 1):
                indices = np.squeeze(np.argwhere(np.logical_and(patient_times >= window_times[i],
                                                                patient_times < window_times[
                                                                    i + 1])))  # indices in this window
                indices = indices[indices < len(current_values)]

                if len(indices) == 0:
                    test_values.append(None)  # marker for no data available in this window
                else:
                    window_scores = current_values[indices]  # cusum scores for all data points found in current window
                    test_values.append(np.mean(window_scores))
                    # print(test_values[-1])
                # print(len(test_values))
            cusum_values.append(test_values)
        except:
            print("Insufficient Data: Patient " + idx)
            continue

    cusum_values = np.array(cusum_values).T.tolist()
    for row in cusum_values:  # remove "None" from windows where no data found
        while None in row: row.remove(None)

    set_font_size()
    rcParams.update({'figure.autolayout': True})

    plt.boxplot(cusum_values, showfliers=False, positions=window_times[:-1], widths=1 / 15,
                medianprops=dict(color='red', linewidth=2.5), whiskerprops=dict(color='lightgrey'),
                capprops=dict(color='lightgrey'), boxprops=dict(color='lightgrey'))

    plt.title("CUSUM Score Distribution Over Time")
    plt.xlabel("Time before cardiac arrest (hours)")
    plt.ylabel("CUSUM Score")
    plt.xticks(np.arange(-4, 1, 1), np.arange(-4, 1, 1))
    plt.xlim(-4.2, 0.2)
    plt.savefig('images/cusum_boxplot_control.png', dpi=800)
    plt.show()


def recall_v_threshold():
    """
    Creates a graph of recall (num detected cardiac arrests / num actual cardiac arrests) vs threshold
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


def threshold_correction_sweep(model_name):
    """
    Performs a sweep over the threshold and correction parameters for changepoint
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
    ## sweep through the correction parameter and save out to a file since this is an expensive computation
    # sweep = threshold_correction_sweep("cdae")
    # print(sweep)
    # with open('Working_Data/cdae_kl_sweep.pickle', 'wb') as handle:
    #     pickle.dump(sweep, handle)

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
    cusum_box_plot(get_patient_ids(control=True), "cdae", 100)
    pass

