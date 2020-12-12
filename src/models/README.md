# Models

These files work to create the model and instability metric on the preprocessed data.

## Helper function
- `patient_split.py` - patient split is used to convert the 6 hour preprocessed data into normal and abnormal data for model training

## Models
The autoencoder is used to learn the patterns of normal heartbeats and then predict abnormal heartbeats to provide an error metric that is later used for determining instability metrics.
- `conv_denoising.py` - contains functions to create, train, and save convolutional autoencoder models with additional code to load pretrained models
- `isolation_forest.py` - 

## Instability metrics
 (somethign here about going from model to instability metrics)
- `mse.py` - 
- `changepoint.py` -  
- `sample_variability.py` - 

## Plots

- `reconstructed_plots.py` - contains code to recreate plot showing the difference between the original and reconstructed heartbeats for a single patient 

