import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from src.preprocess.dsp_utils import get_windowed_time
from src.preprocess.dim_reduce.reduction_error import mean_squared_error

indices = ['1', '4', '5', '6', '7', '8', '10', '11', '12', '14', '16', '17', '18', '19', '20', '21', '22', '25', '27',
           '28', '31', '32',
           '33', '34', '35', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50'] #, '52', '53',
           #'54', '55', '56']

########################################################################################################################
# THIS IS SOME OLD CODE TO GENERATE THE AE ERRORS. JUST GO ON BOX AND DOWNLOAD THE FOLDERS ten_heartbeat_ae_errors
# AND two_heartbeat_ae_errors TO GET THIS. PUT THESE FOLDERS IN Working_Data.

# for idx in indices:
#    mean_squared_error(10, 'ae', idx, save_errors=True)
#    mean_squared_error(100, 'ae', idx, save_errors=True)
########################################################################################################################
# Spearman test, two heartbeat data
# print(' ')
# print('Two heartbeat data')
# pval = 0.05
#
# rho_list = []
# num_true_tests = 0
#
# for idx in indices:
#     # IMPORT DATA
#     mse = np.load(f"Working_Data/two_heartbeat_ae_errors/ae_errors_10d_Idx{idx}.npy")
#     time_vector = range(0, len(mse))
#
#     # RUN SPEARMAN TEST
#     rho, test_pval = spearmanr(time_vector, mse)
#     stat_significance = True if test_pval < pval else False
#
#     rho_list.append(rho)
#     num_true_tests += stat_significance
#     print('Patient ' + str(idx) + ': ' + str(stat_significance))
#
#
# plt.boxplot(rho_list)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# plt.ylabel('Spearman correlation coef. ')
# plt.ylim((-1, 1))
# plt.title('Two heartbeat data (reduced to 10 dim) \n' + str(num_true_tests) + '/' + str(len(indices)) + ' stat. significant patients')
# plt.show()

########################################################################################################################
# Spearman test, ten heartbeat data
print(' ')
print('Ten heartbeat data')
pval = 0.05

rho_list = []
num_true_tests = 0

for idx in indices:
    # IMPORT DATA
    mse = np.load(f"Working_Data/windowed_mse_100d_Idx{idx}.npy")
    #print(np.shape(mse))
    #print(len(mse))
    time_vector = get_windowed_time(idx, num_hbs=10, window_size=50)
    time_vector = time_vector[-len(mse):]
    #print(np.shape(time_vector))

    stat_significance='NOT RUN'
    if len(mse) != 0:
        # RUN SPEARMAN TEST
        rho, test_pval = spearmanr(time_vector, mse)
        stat_significance = True if test_pval < pval else False

        rho_list.append(rho)
        num_true_tests += stat_significance
    print('Patient ' + str(idx) + ': ' + str(stat_significance))

#print(type(rho_list))
print(rho_list)

rho_list = [x for x in rho_list if str(x) != 'nan']
#plt.figure(figsize=(4,4))
plt.boxplot(rho_list)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Spearman correlation coef.', fontsize=12)
plt.ylim((0, 1))
plt.xlim((0.75, 1.25))
plt.yticks(fontsize=10)
plt.savefig('images//frank_spearman.png')
plt.title('Ten heartbeat data\n' + str(num_true_tests) + '/' + str(36) + ' stat. significant', fontsize=18)
plt.show()
