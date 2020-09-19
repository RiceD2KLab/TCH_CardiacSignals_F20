# TCH_CardiacSignals_F20

**Ensure that the working directory of any python files are set to the project root**\
This will ensure that package imports work inside the project


## Installation

`git clone https://github.com/RiceD2KLab/TCH_CardiacSignals_F20.git` \
`pip3 install -r requirements.txt` \
Download the H5 files folder from the TCH box into the project root. **Rename  to `Data_H5_Files`**


## Running Code
To split the ECG leads into individual heartbeats, run `heartbeat_split.py` \
This will split the ECGs for each patient into individual heartbeats, interpolate them, and store them into numpy arrays\
The interpolated hearbeats will be stored into a `Working_Data` directory at the root of the project (ignored by the .gitignore)
python3 h5_interface.py [idx] --Duration [seconds] --Offset [seconds]

idx - number of the h5 file you want to read\
Duration - how many seconds to plot (<10 to fit on screen)\
Offset - How many seconds from the beginning of the signal to splice from\

ex. `python3 h5_interface.py 1 --Duration 4 --Offset 3`

![Example Plot](https://github.com/RiceD2KLab/TCH_CardiacSignals_F20/blob/master/images/idx1d4o3.png)
