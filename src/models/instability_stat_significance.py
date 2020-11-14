# from src.models.instability import  get_metrics
import numpy as np
from scipy.stats import ttest_rel, kstest, kruskal, spearmanr
from src.preprocess.dim_reduce.reduction_error import mean_squared_error
indices = ['1', '4', '5', '6', '7', '8', '10', '11', '12', '14', '16', '17', '18', '19', '20', '21', '22', '25', '27',
           '28', '30', '31', '32',
           '33', '34', '35', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '52', '53',
           '54', '55', '56']


def print_metrics(variance_list):
    '''

    :param variance_list: A lists of lists containing variances for 6 hours worth of heart attack data
    :return:
    '''
    total_valid = 0

    krusk_start_score = 0
    krusk_end_score = 0
    ks_start_score = 0
    ks_end_score = 0
    spearman_start_score = 0
    spearman_end_score = 0

    krusk_start_list = []
    krusk_end_list = []
    ks_start_list = []
    ks_end_list = []
    spearman_start_list = []
    spearman_end_list = []

    for iter, variances in enumerate(variance_list):
        print(f"Computing metrics for {iter + 1}/{len(variance_list)}")
        isValid, krusk_start, krusk_end, ks_start, ks_end, spearman_start, spearman_end = get_metric_differences(variances)
        if isValid:
            total_valid += 1
            krusk_start_score += krusk_start
            krusk_start_list.append(int(krusk_start))
            krusk_end_score += krusk_end
            krusk_end_list.append(int(krusk_end))

            ks_start_score += ks_start
            ks_start_list.append(int(ks_start))
            ks_end_score += krusk_end
            ks_end_list.append(int(ks_end))

            spearman_start_score += spearman_start
            spearman_start_list.append(int(spearman_start))
            spearman_end_score += spearman_end
            spearman_end_list.append(int(spearman_end))




    print(f"Number of patients with Statistical Differences between 1st and 2nd hour using Kruskal = {krusk_start_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 2nd hour using Kruskal as a Boolean, 1 = Different, 0 = Same")
    print(krusk_start_list)
    print(f"Number of patients with Statistical Differences between 1st and 6th hour using Kruskal = {krusk_end_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 6th hour using Kruskal as a Boolean, 1 = Different, 0 = Same")
    print(krusk_end_list)

    print(f"Number of patients with Statistical Differences between 1st and 2nd hour using KS test = {ks_start_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 2nd hour using KS test as a Boolean, 1 = Different, 0 = Same")
    print(ks_start_list)
    print(f"Number of patients with Statistical Differences between 1st and 6th hour using KS test = {ks_end_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 6th hour using KS test as a Boolean, 1 = Different, 0 = Same")
    print(ks_end_list)

    print(f"Number of patients with Statistical Differences between 1st and 2nd hour using Spearman test = {spearman_start_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 2nd hour using KS test as a Boolean, 1 = Different, 0 = Same")
    print(ks_start_list)
    print(f"Number of patients with Statistical Differences between 1st and 6th hour using Spearman test = {spearman_end_score} / {total_valid}")
    print(f"Statistical Differences between 1st and 6th hour using KS test as a Boolean, 1 = Different, 0 = Same")
    print(ks_end_list)

def get_metric_differences(variances):
    '''

    :param variances: A lists containing variances for 6 hours worth of heart attack data
    :return:
    isValid: (Bool) checks if the data is valid
    {krusk,ks}_{start,end}: (Bool) if the {Kruskal Wallace, KS} Test finds a significant difference between the first and {second,sixth} hour
    '''
    first_hour = variances[:int(len(variances) / 6)]
    second_hour = variances[int(len(variances) / 6):2*int( len(variances) / 6)]
    last_hour = variances[(-int(len(variances) / 6)):]

    if len(last_hour) > 0 and len(first_hour) > 0 and len(second_hour) > 0:

        # Calculate kruskal score (compare medians)
        _, pval_start_compare_krusk = kruskal(second_hour, first_hour)
        _, pval_end_compare_krusk = kruskal(last_hour, first_hour)
        krusk_end = True if pval_end_compare_krusk < .05 else False
        krusk_start = True if pval_start_compare_krusk < .05 else False

        # Calculate KS test (compare distributions)
        _, pval_start_compare_ks = kstest(second_hour, first_hour)
        _, pval_end_compare_ks = kstest(last_hour, first_hour)
        ks_end = True if pval_end_compare_ks < .05 else False
        ks_start = True if pval_start_compare_ks < .05 else False

        # Calculate Spearman test
        _, pval_start_compare_spearman = spearmanr(second_hour, first_hour)
        _, pval_end_compare_spearman = spearmanr(last_hour, first_hour)
        spearman_end = True if pval_end_compare_spearman < .05 else False
        spearman_start = True if pval_start_compare_spearman < .05 else False
        isValid = True

    else:
        isValid = False
        krusk_end, krusk_start, ks_end, ks_start, spearman_end, spearman_start = None, None, None, None, None, None

    return isValid, krusk_start, krusk_end, ks_start, ks_end, spearman_start, spearman_end


if __name__ == "__main__":
    # variance_list = []
    # for idx in indices:
    #     print(f"Computing variance for Patient {idx}")
    #
    #     variances = get_metrics("variance", 10, idx, "rawhb", PLOT=False)
    #     variance_list.append(variances)

    mse_list = []
    for idx in indices:
        print(f"computing mse for patient {idx}")
        mse_list.append(np.load(f"Working_Data/windowed_mse_ae_10d_Idx{idx}.npy"))
    print_metrics(mse_list)
#mean_square_error(10, 'ae', patient_num, save_errors=True)