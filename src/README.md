# Directory Structure

The src directory contains the following subdirectories
* `archive` - contains code for methods that ultimately did not make it into our final report/pipeline
* `exploration` - code to visualize our data and explore the variance in lower dimensional space
* ``models``  - Trains the models that contain the methods to go from processed heartbeats to instability metrics 
* ``preprocess`` - Contains methods to turn raw data into heartbeats interpolated to the same length and perform filtering
* `utils` - utility files such for functions like reading H5 files, standardizing plot font sizes, etc. 
* ``validation`` - Evaluates the performance of our changepoint and model modules 
