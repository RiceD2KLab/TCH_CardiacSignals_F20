import os

"""
Utility functions to help with filenames
"""

original_patient_ids = ['1', '4', '5', '6', '7', '8', '10', '11', '12', '14', '16', '17', '18', '19', '20', '21', '22', '25', '27',
			'28', '30', '31', '32',
			'33', '34', '35', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '52', '53',
			'54', '55', '56']

new_patient_ids = []

def get_filenames(original=True, control=False):
	"""
	Function to get the filenames
	:param start: [int] index to start on
	:param stop: [int] index to stop on
	:param original: [bool] use original set of data
	:returns: [list[string]] filenames
	"""
	# if original:
	#
	# 	filenames = []
	# 	for index in original_patient_ids:
	# 		if int(index) < start:
	# 			continue
	# 		if stop and not int(index) in range(stop):
	# 			break
	# 		filenames.append("Reference_idx_" + index + "_Time_block_1.h5")
	# 	return filenames

	if original:
		dataFilenames = sorted(os.listdir("Data_H5_Files"))
	elif control:
		dataFilenames = sorted(os.listdir("Data_H5_Files_Control"))
	return dataFilenames

def scrape_indices(filenames):
	"""
	Function to get patient IDs from .h5 files listed as 'Reference_idx_ID_Time_block_1.h5'
	:param filenames: [list[string]] filenames of .h5 files
	:returns: [list[string]] corresponding patient IDs
	"""
	ids = []
	for fname in filenames:
		start = fname.find('Reference_idx_') + len('Reference_idx_')
		end = start + fname[start:].find('_')
		ids.append(fname[start:end])
	return ids

def scrape_indices_WD(directory_path, control = False):
	"""
	Function to get valid patient IDs from a directory containing preprocessed files
	based off filenames of form:
	Fixed_Dim_HBs_Idx[C]###.npy
	:param directory_path: String, path to directory
	:returns: [list[string]] corresponding patient IDs
	"""
	files = os.listdir(directory_path)
	file_ids = set([])
	comparisonStr = "IdxC" if control else "Idx"
	for file in files:
		items = file.split("_")
		try:
			if items[3][:len(comparisonStr)] == comparisonStr:
				file_ids.add(items[3][len(comparisonStr):-4])
		except:
			pass
	return list(file_ids)

def get_patient_ids(control=False):
	"""
	Get the patient ids. If there are unique patient indices in the new_patient_id array, then return that id array
	Otherwise, return the original array of patient ids
	"""
	if control:
		return(scrape_indices(get_filenames(original=False, control=True)))
	elif len(new_patient_ids) == 0:
		return original_patient_ids
	else:
		return new_patient_ids

if __name__ == "__main__":
	#print(get_patient_ids(control=True))
	# filenames = (get_filenames(original=False, control=True))
	print(get_patient_ids(True))
	# ids = scrape_indices(filenames)
	# print(len(ids))
	# print(len(original_patient_ids))