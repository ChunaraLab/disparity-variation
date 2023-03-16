# Measuring disparities efficiently

Code for experiments on ACS and BRFSS datasets

## Installation

Install Anaconda environment with pandas, numpy, matplotlib, joblib

Install packages
> pip install folktables torch seaborn scikit-learn

## Run instructions
Run `run.sh` with id of the dataset which corresponds to the 1-indexed item
in the list `DATASET_OPTS` in `main.py`.

For example to run the first dataset id
> bash run.sh 1

## Acknowledgements
We use the excellent `folktables` package to query ACS data.
It is available at [folktables](https://github.com/socialfoundations/folktables).