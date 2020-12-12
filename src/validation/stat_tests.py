"""
Contains code for running statistical tests on our various instability metrics.
Used for model validation and comparison across models.
"""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from src.utils.file_indexer import get_patient_ids


def paired_ttest(metric_list, alpha):
    '''
    Conducts a paired t-test on the metric values provided and prints the number of patients with signficant differences
    between the first 90 mins of test data and the last 90 mins of test data

    :param metric_list: [list(list(int))] containing the scores over time for each patient (first two hours are the training data)
    :param alpha: [int] significance level to use (typically 0.05)
    :return: [(int, int)] number of patients with p < alpha and total number of valid patients
    '''
    difference_found = 0
    total_valid = 0
    for idx, mse in enumerate(metric_list):
        # print(f"Computing metrics for {idx+1}/{len(metric_list)}")

        test_data = mse[len(mse)//3:]
        ninety_mins = len(mse) // 12 # number of values in 90 mins of data
        first_ninety = test_data[:ninety_mins]
        next_ninety = test_data[ninety_mins:ninety_mins*2]
        last_ninety = test_data[-ninety_mins:]

        if len(first_ninety)>0 and len(last_ninety)>0:
            total_valid += 1
            score, pval = ttest_rel(first_ninety, last_ninety)
            # print(pval)
            if pval/2 < alpha and score < 0: # if significant difference found
                difference_found += 1
        
    print(f"Number of patients with Statistical Differences between beginning and end of test data = {difference_found} / {total_valid}")
    return difference_found, total_valid

def wilcoxon_test(metric_list, indices, alpha):
    '''
    Conducts a Wilcoxon signed-rank test on the metric values provided and prints the number of patients with signficant differences
    between the first 90 mins of test data and the last 90 mins of test data

    :param metric_list: [list(list(int))] containing the scores over time for each patient (first two hours are the training data)
    :param indices: [list(int)] list of patient indices to check (usually the patients which have existing MSE files)
    :param alpha: [int] significance level to use (typically 0.05)
    :return: [(int, int)] number of patients with p < alpha and total number of valid patients
    '''
    difference_found = 0
    total_valid = 0
    for mse, idx in zip(metric_list, indices):
        test_data = mse[len(mse)//3:]
        if len(test_data) < 10: # check if enough data is available
            print(f"Invalid patient - not enough data: {idx}")

        else:
            ninety_mins = 3*len(mse) // 12 # number of values in 90 mins of data
            print(ninety_mins)
            first_ninety = test_data[:ninety_mins]
            # next_ninety = test_data[ninety_mins:ninety_mins*2]
            last_ninety = test_data[-ninety_mins:]

            if sum(first_ninety - last_ninety) == 0: # unlikely edge case handling where both times have exactly the same value
                last_ninety[-1] += 1e-5 # cheap hack

            total_valid += 1
            score, pval = wilcoxon(first_ninety, last_ninety, alternative="less")
            # print(pval)
            if pval < alpha: # if significant difference found
                difference_found += 1
        
    print(f"Number of patients with Statistical Differences between beginning and end of test data = {difference_found} / {total_valid}")
    return difference_found, total_valid

if __name__ == "__main__":
    # Code below is an exmaple of how to use the functions in this file 

    # indices = get_patient_ids()
    # metric_list = []
    # valid_indices = []
    # for idx in indices:
    #     try:
    #         data = np.load(f"Working_Data/windowed_mse_100d_Idx{idx}.npy")
    #         metric_list.append(data)
    #         valid_indices.append(idx)
    #     except:
    #         print("No File Found: Patient " + idx)
    #         continue
    # sig, total = wilcoxon_test(metric_list, valid_indices, 0.05)
    pass
