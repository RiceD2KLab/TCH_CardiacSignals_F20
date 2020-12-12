import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from src.preprocessing import heartbeat_split


def paired_ttest(mse_list, alpha):
    '''
    Conducts a paired t-test on the MSE values provided and prints the number of patients with signficant differences
    between the first 30 mins of test data and the last 30 mins of test data

    Inputs:
    mse_list - A lists of lists containing the mse over time for each patient (first two hours are the training data)
    alpha - significance level to use
    '''
    difference_found = 0
    total_valid = 0
    for idx, mse in enumerate(mse_list):
        # print(f"Computing metrics for {idx+1}/{len(mse_list)}")

        test_data = mse[len(mse)//3:]
        thirty_mins = len(mse) // 12 # number of values in 30 mins of data
        first_thirty = test_data[:thirty_mins]
        next_thirty = test_data[thirty_mins:thirty_mins*2]
        last_thirty = test_data[-thirty_mins:]

        if len(first_thirty)>0 and len(last_thirty)>0:
            total_valid += 1
            score, pval = ttest_rel(first_thirty, last_thirty)
            # print(pval)
            if pval/2 < alpha and score < 0: # if significant difference found
                # print(f"Patient {idx} showed a significant increase")
                # plt.plot(test_data)
                # plt.show()
                difference_found += 1
        
    print(f"Number of patients with Statistical Differences between beginning and end of test data = {difference_found} / {total_valid}")

def wilcoxon_test(mse_list, indices, alpha):
    '''
    Conducts a Wilcoxon signed-rank test on the MSE values provided and prints the number of patients with signficant differences
    between the first 30 mins of test data and the last 30 mins of test data

    Inputs:
    mse_list - A lists of lists containing the mse over time for each patient (first two hours are the training data)
    indices - the corresponding patient indices
    alpha - significance level to use
    '''
    difference_found = 0
    total_valid = 0
    for mse, idx in zip(mse_list, indices):
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
                # print(f"Patient {idx} showed a significant increase")
                # plt.plot(test_data)
                # plt.show()
                difference_found += 1
        
    print(f"Number of patients with Statistical Differences between beginning and end of test data = {difference_found} / {total_valid}")

if __name__ == "__main__":
    indices = heartbeat_split.indicies
    # print(indices)
    mse_list = []
    valid_indices = []
    for idx in indices:
        try:
            data = np.load(f"Working_Data/windowed_if_100d_Idx{idx}_NEW.npy")
            mse_list.append(data)
            valid_indices.append(idx)
        except:
            print("No File Found: Patient " + idx)
            continue

        # mse_list.append(mse_over_time(idx, 'ae', 10)[:-3])
        # print(len(np.load(f"Working_Data/windowed_mse_100d_Idx{idx}.npy")))
    wilcoxon_test(mse_list, valid_indices, 0.05)
