import json

import numpy as np
import os

import requests

i = 25;
fixed_four_lead = np.load(os.path.join("Working_Data", "Fixed_Dim_HBs_Idx" + str(i) + ".npy"))
vector_len = fixed_four_lead.shape[1]
num_vectors = 1
new_vector_len = vector_len
if (vector_len < 500):
	num_vectors = 500 // vector_len + 1
	new_vector_len = num_vectors * vector_len
	crimp_len = ((num_vectors * vector_len) % 350) // 2

four_leads = {}
for lead in range(1,5,1):
	average = np.zeros(new_vector_len);
	for sample in range(10):
		for portion in range(num_vectors):
			average[portion * vector_len:(portion+1) * vector_len] += fixed_four_lead[fixed_four_lead.shape[0]-sample*num_vectors-portion-1,:,lead-1]
	average /= 10
	average = average[crimp_len:new_vector_len - crimp_len]
	four_leads["lead" + str(lead)] = list(average)

url = "http://127.0.0.1:5000/4L_data/" + str(i)
with open("four_lead.json", "w") as outfile:
	json.dump(four_leads, outfile,indent = 4)

f = open("four_lead.json", "rb")
x = requests.post(url, files={'four_lead': f})
print(x)
f.close()