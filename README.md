# TCH_CardiacSignals_F20

Clone Directory

pip3 install -r requirements.txt

Download the data from Box, put it in project directory. Rename "Data_H5_Files"

python3 h5_interface.py [idx] --Duration [seconds] --Offset [seconds]

idx - number of the h5 file you want to read
Duration - how many seconds to plot (<10 to fit on screen)
Offset - How many seconds from the beginning of the signal to splice from

ex. $python3 h5_interface.py 1  

![Example Plot](https://github.com/RiceD2KLab/TCH_CardiacSignals_F20/blob/master/images/idx1d4o3.png)
