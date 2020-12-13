# Directory Structure

The src directory contains three main subdirectories
* `archive` - contains code for methods that ultimately did not make it into our final report/pipeline
* `exploration` - code to visualize our data and explore the variance in lower dimensional space
* ``models``  - Trains the models that contain the methods to go from processesed heartbeats to instability metrics (WIP)
* ``preprocess`` - Contains methods to turn raw data into heartbeats interpolated to the same length and perform filtering
* `utils` - utility files such for functions like reading H5 files, standardizing plot font sizes, etc. 
* ``validation`` - Takes in trained models and runs them on the test data, measuring statistical significance of our model's metrics
