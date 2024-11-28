## Setup Instructions
1. All the necessary files are available at `https://github.gatech.edu/gt-omscs-rldm/7642Spring2020sgopal36/project1`
2. `git clone` or download the files to continue with the setup

## Python Env Installation
1. All the files need Python version 3.7 and above
2. The environment file has been exported as `environment.yml` available at `https://github.gatech.edu/gt-omscs-rldm/7642Spring2020sgopal36/`
3. Use `conda env create -f environment. yml` to create the `frozenlake` environment
4. Activate the new environment: `conda activate frozenlake`
5. Incase if you believe a working Python 3.7 environment is setup and reasonably stable, check for the following libraries and their versions.
6. Required libraries: json, pickle, matplotlib, numpy, seaborn, scipy, sklearn

```
jsonschema=3.2.0
pickleshare=0.7.5
matplotlib=3.1.1
numpy=1.17.4
scikit-learn=0.22.1
seaborn=0.10.0
scipy=1.4.1
```

### Folders and Locations
The following folders are required **without which the code would fail**.
1. output
2. files

### Python Code Run
1. Run the python file `project1.py` to run all the experiments.
2. Run the python file `td_sutton.py` to perform the two additional experiments to explain alpha selection.
3. Update the run results in `results.json` if the `generate_ideal_comparison_*` methods need to be executed.

### Variables and Switches for project1.py
1. `USE_SAVED_TRAIN_DATA` - Set this to False to regenerate the training data. A seed is already set in the code as well.
2. `PERFORM_MAIN_RUN` - Set this to True to re-run all the 3 experiments with and without alpha decay.
3. `PERFORM_ADD_RUN` - Set this to True to execute the two additional experiments for multiple training sets and different episode lengths. Note this takes more than couple of hours to execute. 