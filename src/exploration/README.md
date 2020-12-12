# Data Exploration

The following Python files generate plots for visualizing the ECG data in various ways, as well as visualizing lower 
dimensional representations of the data.

## Raw Data Visualizations 

- `ecg_signal_animation.py`  - Generate .gif files for animation of heartbeat and instability detection (for the 5 
minute video).
- `stft_visualizations.py` - Perform time/frequency domain analysis of ECG leads. Computes short-time Fourier transform,
difference in spectrum for adjacent time windows, and spectral entropy for each window.

## Lower Dimensional Representations
- `pca.py` - runs PCA on original data and reduces data to 1 dimension, saving the encoded and reconstructed files 
- `umap.py` - plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file


