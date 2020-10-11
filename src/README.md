# Directory Structure

The src directory contains three main subdirectories

* ``preprocess`` - Contains methods to turn raw data into heartbeats interpolated to the same length
* ``models``  - Trains the models that contain the methods to go from processesed heartbeats to instability metrics (WIP)
* ``validation`` - Takes in trained models and runs them on the test data, measuring accuracy (WIP)
