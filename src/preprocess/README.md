# Preprocessing

Contains procedures for preprocessing the data, such as peak detection, splitting waveforms into individual heartbeats, 
filtering, and dimension reduction.

`dim_reduce`: contains models like PCA, autoencoders, and variational autoencoders for reducing the dimension of the heartbeat\
`heartbeat_split`: contains code for splitting the waveforms into individual heartbeats and performing peak detection\
`visualization`: code for visualizing some of the unprocessed data/data in the preprocessing stage 

The folder also contains some general utility files for performing signal processing (`dsp_utils.py`), file-io operations (`h5_interface.py`), and 
analyzing wandering baseline/cardio-pulmonary interactions (`wandering_baseline.py`).
