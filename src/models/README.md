# Models

These files work to create the model and instability metric on the preprocessed data.

## Helper function
- `patient_split.py` - Patient split is used to convert the 6 hour preprocessed data into normal and abnormal data for model training

## Models
The autoencoder is used to learn the patterns of normal heartbeats and then predict abnormal heartbeats to provide an error metric that is later used for determining instability metrics.
- `conv_denoising.py` - Contains functions to create, train, and save convolutional autoencoder models with additional code to load pretrained models
- `isolation_forest.py` - Contains functions to train an Isolation Forest on reduced heartbeat data. See __main__ function to reproduce the plot in the report

## Instability metrics
 These files use the results from the above models to compute metrics which serve as warnings of imminent cardiac arrests
- `mse.py` - Using the data created by `conv_denoising.py`, functions in this file will compute the MSE between the reconstructed and original data from the autoencoder. 
- `changepoint.py` - Contains function to run CUSUM change-point detection on MSE data. See __main__ function to reproduce the two plots in the report.
- `simple_variability.py` - Computes the variablity of the raw heartbeat data over time. See __main__ function to reproduce the plot in the appendix of the report.

## Plots

- `reconstructed_plots.py` - Contains code to recreate plot showing the difference between the original and reconstructed heartbeats for a single patient 

