# Preprocessing

Contains procedures for preprocessing the data, such as peak detection, splitting waveforms into individual heartbeats, 
filtering, and dimension reduction.

`dim_reduce`: contains models like PCA, autoencoders, and variational autoencoders for reducing the dimension of the heartbeat\
`heartbeat_split`: contains code for splitting the waveforms into individual heartbeats and performing peak detection\
`visualization`: code for visuaulizing some of the unprocessed data/data in the preprocessing stage 

The folder also contains some general utility files for performing signal processing and file-io operations and one file 
with functions for tracking wandering baseline/cardio-pulmonary interactions.