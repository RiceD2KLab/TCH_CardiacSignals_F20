# Data Exploration

The following Python files generate plots for visualizing the ECG data in various ways, as well as visualizing lower 
dimensional representations of the data.

## Raw Data Visualizations 

- `ecg_signal_animation.py`  - Generate .gif files for animation of heartbeat and instability detection (for the 5 
minute video). The .gif plots will plot and save when the script is run. Comment out lines 46-48 and lines 81-83 if 
you do not want to save/show plots.

- `stft_visualizations.py` - Perform time/frequency domain analysis of ECG leads. Computes short-time Fourier transform,
difference in spectrum for adjacent time windows, and spectral entropy for each window. To generate plots shown in the 
report, uncomment the block of code beginning with if __name__ == "__main__": (line 75). This will plot the STFT and 
spectral entropy vs. time for a representative patient.

## Lower Dimensional Representations
- `pca.py` - Performs principle component analysis on original ECG data. Functions are included for plotting covariance
matrix eigenvalues, plotting the first n principle components in a scatterplot (for n=2, 3), and saving reduced data. 
To plot the distribution of data along the first two principal axes (2D scatterplot) for a representative patient and
lead, uncomment the block of code beginning with if __name__ == "__main__": (line 151). 

- `umap.py` - plots 2-dimensional representation of the PCA matrix for a particular lead of the ith file


