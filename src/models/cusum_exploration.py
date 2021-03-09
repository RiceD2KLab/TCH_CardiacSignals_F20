from changepoint import *
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pickle
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
    # plt.savefig('images/cusum_boxplot.png', dpi=500)
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

def roc_curve(plot=True):
    """
    Plot Receiver Operating Characteristic curve (i.e. true positive vs false positive rate)

    True Positive Rate = True Positives / (True Positives + False Negatives)
    False Positive Rate = False Positives / (False Positives + True Negatives)
    :return: nothing
    """

    thresholds = list(range(0, 4020, 20))
    true_positive_rates = []
    false_positive_rates = []

    annotations = list(range(0, 4000, 200))
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



    if plot:
        plt.figure()
        plt.plot(false_positive_rates, true_positive_rates)
        plt.scatter(false_positive_rates, true_positive_rates, c=['r'])
        for threshold, coord in zip(annotations, annotation_coords):
            plt.annotate(str(threshold), coord)
        plt.title(f"ROC Curve for Cusum Thresholds from {thresholds[0]} to {thresholds[-1]}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
    #
    # calculate AUC (area under curve)
    auc = metrics.auc(false_positive_rates, true_positive_rates)
    print(f"AUC-ROC score is {auc}")
    return auc, true_positive_rates, false_positive_rates


def threshold_correction_sweep():

    all_patients = get_patient_ids(control=False) + get_patient_ids(control=True)

    correction_sweep = np.arange(0, 1, 0.01)
    auc_scores = {}

    for c in correction_sweep:
        for idx in all_patients:
            try:
                cusum(idx, "cdae", dimension=100, save=True, correction=c)
            except Exception as e:
                # print(e)
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
    with open('Working_Data/sweep.pickle', 'rb') as handle:
        scores = pickle.load(handle)
    plt.plot(scores.keys(), scores.values())
    plt.xlabel("CUSUM Correction Parameter")
    plt.ylabel("Area Under Curve")
    plt.title("Area Under Curve vs. Correction Parameter")
    plt.show()

    print(f"argmax correction parameter is {max(scores, key=scores.get)} yielding auc = {max(scores.values())}")

def plot_MSE_transform(patient_id):
    """
    patient_id is a string or integer denoting patient idx
    :return: plots a histogram of MSE data for the last 4 hours, as well as with a natural log transform
    """
    error_signal = mean_squared_error(100, "cdae", patient_id)
    test_error_signal = error_signal[int(len(error_signal)/3)+50:]
    plt.hist(test_error_signal, bins=60)
    plt.xlim(-3,3)
    plt.xlabel('MSE')
    plt.ylabel('Counts')
    plt.title('MSE (Last 4 Hours): Test Patient '+str(idx))
    plt.show()

    plt.hist(np.log(test_error_signal), bins=60)
    plt.xlim(-3,3)
    plt.xlabel('ln(MSE)')
    plt.ylabel('Counts')
    plt.title('ln(MSE) (Last 4 Hours): Test Patient '+str(idx))
    plt.show()

    # plt.hist((test_error_signal)**0.25, bins=60)
    # plt.xlim(-3,3)
    # plt.xlabel('(MSE)^0.25')
    # plt.ylabel('Counts')
    # plt.title('(MSE)^0.25 (Last 4 Hours): Test Patient '+str(idx))
    # plt.show()



if __name__ == "__main__":
    ## sweep through the correction parameter and save out to a file since this is an expensive computation
    # sweep = threshold_correction_sweep()
    # print(sweep)
    # with open('Working_Data/sweep.pickle', 'wb') as handle:
    #     pickle.dump(sweep, handle)


    # roc_curve(plot=False)
    # cusum_validation(25, control=True)
    plot_sweep()

    # this compares the roc curves with different correcton parameters
    # plt.clf()
    # plt.figure()
    # corrections = [0.05, 0.44]
    # for c in corrections:
    #     calculate_cusum_all_patients(c)
    #     auc, tpr, fpr = roc_curve(plot=False)
    #     plt.plot(fpr, tpr)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend(corrections)
    # plt.title("ROC Comparison with tuned vs. untuned correction parameter")
    # plt.show()





