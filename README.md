# BO4Policy_Search_Plasma
Proposed ACC 2023 Contribution: Towards Personalized Plasma Medicine via Data-efficient, Globally Optimal Adaptation of Fast Deep Learning-based MPC Policies

## Implementation
To run this code on your own device, it is recommended to work within a virtual environment. This repository includes a compiled virtual environment in the directory `BO4PSP` (including packages for simulation and experiments). Otherwise, you may create your own virtual environment and then install the required Python packages by using the command
`pip3 install -r requirements_sim_only.txt` (for simulations only) and `pip3 install -r appj_requirements.txt` (additional packages for experiments with the cold atmospheric plasma jet (CAPJ).

The main file to run simulations with Bayesian optimization is `src/run_sim.py`.

The main file to run experiments with the in-house Mesbah Lab CAPJ testbed is `src/run_exp.py`

Additional details may be found within the `src` folder as well as in commentary within the files.

### Sensitivity Analysis
This project performs a global sensitivity analysis (GSA) on the parameters of a deep neural network (DNN) to the closed-loop metrics of the plasma system (in simulation). For this portion of the project, we interface our Python scripts within MATLAB to take advantage of the established software package [UQLab](https://www.uqlab.com). The main script to run this analysis is `src/sensitivity.m`. To ensure that MATLAB is able to interface with Python, make sure to initialize your MATLAB session with `pyenv('Version', 'PATH_TO_VENV')`, where `PATH_TO_VENV` is the path to the python executable of the virtual environment created for this project. Furthermore, when installing a Python distribution for this purpose, the `--enable-shared` flag must be passed into the configuration for the installation. More details in the following links: 
 * [MathWorks Documentation](https://www.mathworks.com/help/matlab/call-python-libraries.html)
 * [Stack Overflow Answer](https://stackoverflow.com/questions/72730769/pyenv-how-to-install-python-dynamic-shared-library) (links to below)
 * [Pyenv Documentation](https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md#building-with---enable-shared)

(c) 2022 Mesbah Lab

in collaboration with George Makrygiorgos and Ali Mesbah.

Questions regarding this code may be directed to kchan45 (at) berkeley.edu
