# Validation
- `stat_tests.py` - Contains code for running statistical tests (Wilcoxon, paired t-test on instability metrics.
Used for model validation and comparison across models.

### Running the Wilcoxon Signed-Rank Test
This test is used to verify an increase in the mean MSE over time. To see how many patients in the cohort demonstrate significant increases in the MSE over time, do the following:

1. Ensure that the MSE for each patient is available in the `Working_Data` directory (see [here](https://github.com/RiceD2KLab/TCH_CardiacSignals_F20/blob/master/src/models/README.md)).
2. Call the function `wilcoxon_test(metric_list, indices, 0.05)`, where `indices` is a list of all patient identifiers for which MSE data is available. (This is given as example code in the `stat_test.py` file)
