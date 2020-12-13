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
*Terminal*: From the project root, run `export PYTHONPATH=$PYTHONPATH:'pwd'` to set the working directory, then run the desired script from the project root (i.e. `python3 src/<package_name>/.../<scriptname>.py`


## Installation

`git clone https://github.com/RiceD2KLab/TCH_CardiacSignals_F20.git` \
`pip3 install -r requirements.txt` \
Download the H5 files folder from the TCH box into the project root. **Rename  to `Data_H5_Files`**

## Running code 
We provide the notebook `run.ipynb` that runs a sample patient (id = 16) through our pipeline. Specifically, it runs the
patient through our preprocessing, modeling, and validation sections of the pipeline.\
The notebook defaults to loading a pretrained model to avoid the expensive autoencoder training step. **To get this pretrained model, 
download the pretrained model weights from the TCH Box into the Working_Data directory (detailed instructions inside the notebook)**


For reproducibility, we add notes in the `README` for each module on how to reproduce our figures. Head to the [``src``directory](https://github.com/RiceD2KLab/TCH_CardiacSignals_F20/tree/master/src), for information on preprocessing, modeling and validation.
The following diagram is an overview of the files associated with our data science pipeline
![Data Science Pipeline Overview](images/pipelinediagram.svg) 

# In this directory
* `Data_H5_Files/` - contains the raw ECG signals in the form of H5 files -> contains sample for patient 16
* `images/` - contains plots used for the final report and presentation
*  `src/` - source code for the project
* `Working_Data/` - directory for intermediate data such as normalized heartbeats, reconstructions, etc.
* ``ReportPDF.pdf``- Formal final report for this project
* ``requirements.txt`` - Packages required to run this repository
* `run.ipynb` - notebook for running a single patient through the pipeline, including preprocessing and modeling
