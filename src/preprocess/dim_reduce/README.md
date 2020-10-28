# Dimension Reduction

The following Python files and Jupyter notebooks serve to develop dimensionality reduction techniques for multi-lead ECG signals. Some examples of the strategies we employed were variational autoencoders, determenistic autoencoders, and PCA to reduce dimensionality from 100 --> 10. 

 * `multi_lead_autoencoder.py` trains a single autoencoder for a single patient across the target patients (run on AWS), and saves encoded and reconstructed files
 * `vae.py` trains a single autoencoder for a single patient across all target patients (run on AWS), and saves encoded and reconstructed files
 * `normalize_heartbeats.py` normalizes fixed dimension heartbeats for dimension reduction 
 * `pca.py` runs PCA on original data and reduces data to 1 dimension, saving the encoded and reconstructed files 
 * `reduction_error.py` assesses the dimensionality reduction technique by first reducing the heartbeat dimension with the technique, then projecting
the reduced data back to the original dimension and calculating the mean squared error
 * `umap_viz.py` plots 2-dimen*__________*sional representation of the PCA matrix for a particular lead of the ith file


