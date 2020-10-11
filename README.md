# Detection of Cardiac Electrical Instability Prior to Cardiac Arrest
Rice D2K Lab Fall 2020 project
By: Aneel Damaraju, Chiraag Kaushik, Andrew Pham, Kunal Rai, Tucker Reinhardt, Frank Yang
Mentors: Sebastian Acosta, PhD; Mubbasheer Ahmed, MD; Parag Jain, MD

# The Project 

This main goal of this project is to develop an algorithm for early detection of electrical instability of the heart, specifically in pediatric patients with hypoplastic left heart syndrome. To learn more about this project check out [this link!](https://github.com/RiceD2KLab/TCH_CardiacSignals_F20/blob/master/ReportPDF.pdf)


# To use this Repository

**Ensure that the working directory of any python files are set to the project root**\
This will ensure that package imports work inside the project\
*Pycharm*: Edit the run configuration and set the working directory to the project root\
*Terminal*: From the project root, run `export PYTHONPATH=$PYTHONPATH:'pwd'` to set the working directory, then run the script from the project root (i.e. `python3 src/<package_name>/.../<scriptname>.py`


## Installation

`git clone https://github.com/RiceD2KLab/TCH_CardiacSignals_F20.git` \
`pip3 install -r requirements.txt` \
Download the H5 files folder from the TCH box into the project root. **Rename  to `Data_H5_Files`**

## Running code 

To start running the code in this directory, head to the ``src``directory, for information on preprocessing, modeling and validation. 

# In this directory

* ``ReportPDF.pdf``- Information on this project
* ``requirements.txt`` - Packages required to run this repository
* ``src`` - Directory containing the main files for this project
* ``images`` - Various images created by files in the ``src`` directory
