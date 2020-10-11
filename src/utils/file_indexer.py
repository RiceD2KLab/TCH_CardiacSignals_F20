import os

'''
Function that returns a list of filenames
'''
def get_filenames(start = 0, stop = None):
	dataFilenames = sorted(os.listdir("Data_H5_Files"))
	if stop:
		return dataFilenames[start:stop]
	else:
		return dataFilenames
