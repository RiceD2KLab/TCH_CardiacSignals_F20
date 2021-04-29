"""
Contains functions to run the CUSUM change-point detection algorithm on the MSE data.
Also contains code for all CUSUM plots shown in our report.
"""

from matplotlib import rcParams
from src.models.changepoint.error_metric import *


def cusum(patient, model_name, dimension, error_function, save=False, correction=0.05, plot=False):
    """
    Main CUSUM change point detection function. Plots results and saves CUSUM scores for a single patient

    @param patient: [int] numerical identifier for a patient
    @param model_name: [string] the model which was used to generate the data (typically "cdae", "pca", or "if")
    @param dimension: [int] the dimension of the latent space to which the data was reduced in the above model (e.g. 100 for "cdae")
    @param error_function: the error function (MSE, wasserstein, kl_divergence) to be used to compute distance
    :return: None (saves plot of CUSUM over time and saves the numerical CUUSM values to the appropriate directories)
    """
    # computes the error function
    error_signal = error_function(dimension, model_name, patient)

    time_stamps = get_windowed_time(patient, 10, 1)  # corresponding time stamps for the MSE

    duration = len(error_signal)

    val_data = error_signal[duration//3:duration//2] # third hour (assumed healthy)
    sigma = np.std(val_data) # standard deviation of third hour
    c = np.mean(val_data) + correction # cusum correction parameter


    if len(time_stamps) > len(error_signal):
        time_stamps = time_stamps[-len(error_signal):] # size mismatch correction
    else:
        error_signal = error_signal[-len(time_stamps):]


    cusum = [0]
    for x in error_signal:
        L = (x - c)/sigma
        cusum.append(max(cusum[-1] + L, 0))
    cusum = cusum[1:]

    set_font_size()
    rcParams.update({'figure.autolayout': True})
    if plot:
        plt.plot(time_stamps[len(time_stamps)//3:], cusum[len(time_stamps)//3:])
        plt.title(f"Individual Patient {patient}: CUSUM statistic over time")
        plt.xlabel("Time before cardiac arrest (hours)")
        plt.ylabel("CUSUM Score")
        # plt.savefig('images/cusum_single_patient.png', dpi=500)

        plt.show()
    if save:
        filename = os.path.join("Working_Data", f"unwindowed_cusum_100d_Idx{patient}.npy")
        np.save(filename, cusum)
    return cusum


def calculate_cusum_all_patients(c, model_name, error_func):
    """
    Recalculates and saves the cusum metric time series over all patients
    :param c: cusum correction term to calculate cusum series with
    :return:
    """
    all_patients = get_patient_ids(control=False) + get_patient_ids(control=True)
    for idx in all_patients:
        try:
            cusum(idx, model_name, 100, error_func, save=True, correction=c, plot=False)
        except Exception as e:
            # print(e)
            pass


if __name__ == "__main__":
    # Uncomment the below two lines to reproduce the figures from the report
    # calculate_cusum_all_patients(0.36, "cdae", kl_divergence_timedelay)
    # cusum_validation(15, control=False)
    # cusum_box_plot(get_patient_ids(), "cdae", 100)
    # generates the unwindowed_cusum files for each patient
    # for idx in get_patient_ids(control=True):
    #     try:
    #         cusum(idx, "cdae", dimension=100, save=False, correction=1.0, plot=True)
    #     except Exception as e:
    #         print(e)
    #         pass
    # print(cusum_validation(15, control=True))

    # for idx in ["C106", "C11", "C214", "C109"]:
    #     print(idx)
    #     try:
    #         cusum(idx, "cae", dimension=100, save=False, correction=0.05, plot=True, timedelay=True)
    #     except Exception as e:
    #         print(e)
    #         pass


    for patient in get_patient_ids(True):
        filename = os.path.join("Working_Data", f"unwindowed_cusum_100d_Idx{patient}.npy")
        try:
            data = np.load(filename)
            plt.plot(data)
            plt.title(f"patient {patient}")
            plt.show()
        except Exception as e:
            continue





    # for idx in [1, 5, 7, 8, 11, 12, 18, 27, 40, 41, 47, 49]:
    #     cusum(idx, "cdae", dimension=100, plot=True)
    # plt.show()

    # error_signal = mean_squared_error(100, "cdae", "C103")
    # print(get_patient_ids(True))

    pass
