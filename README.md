# Measuring disparities efficiently

Code for experiments on ACS and BRFSS datasets

## Installation

Code is tested on Python 3.8.

Install Anaconda environment with pandas, numpy, matplotlib, joblib, scikit-learn

Install packages
> pip install folktables torch seaborn

## Datasets
Scripts automatically download and save data for ACS in
a folder names `data`.

For BRFSS, manually download the file from [Kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system).
Save the file `2014.csv` as `2014_diabage_sleeptime.csv` in the `data` folder.

Download the file `us-state-ansi-fips.csv` from this (GitHub)[https://gist.github.com/dantonnoriega/bf1acd2290e15b91e6710b6fd3be0a53] repo into the `data` folder.
This file maps the numeric US FIPS code to readable US Census names of US states.

## Run instructions
Run `run.sh` with id of the dataset which corresponds to the 1-indexed item
in the list `DATASET_OPTS` in `main.py`.

For example to run the first dataset id
> bash run.sh 1

## Acknowledgements
We use the excellent `folktables` package to query ACS data.
It is available at [folktables](https://github.com/socialfoundations/folktables).