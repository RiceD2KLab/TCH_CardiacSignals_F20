# Models

These files work to create the model and instability metrics on the preprocessed data.

## models/autoencoders
The autoencoder is used to learn the patterns of normal heartbeats and then predict abnormal heartbeats to provide an error metric that is later used for determining instability metrics.
- `cdae_timedelay.py` - Best Model: Contains functions to create, train, and save convolutional autoencoders with transfer learning with additional code to load pretrained models
- `conv_denoising.py` - Contains functions to create, train, and save convolutional autoencoder models with additional code to load pretrained models
- `isolation_forest.py` - Contains functions to train an Isolation Forest on reduced heartbeat data. See __main__ function to reproduce the plot in the report
- `LSTMAE.py` - Contains functions to create, train, and save LSTM models
- `patient_split.py` - Helper library used to convert the 6 hour preprocessed data into normal and abnormal data for model training

## models/changepoint
- `changepoint.py` - Contains function to run CUSUM change-point detection on MSE data. See __main__ function to reproduce the two plots in the report.
- `changepoint_tuning.py` - Contains functions to perform hyperparameter changepoint tuning and to compute/visualize overall pipeline performance
- `error_metric.py` - Library of error metrics to compare original and reconstructed heartbeats from the autoencoders

## Other Files
- `simple_variability.py` - Computes the variability of the raw heartbeat data over time. See __main__ function to reproduce the plot in the appendix of the report.
- `reconstructed_plots.py` - Contains code to recreate plot showing the difference between the original and reconstructed heartbeats for a single patient (before running make sure the intermediate data has been generated)
- `isolation_forest.py` - Isolation forests to measure distance between original and reconstructed heartbeats
