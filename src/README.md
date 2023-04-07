**NOTE: This code is provided to reproduce the results obtained in the following paper: .**

# Source Code
This is considered the main working directory to run the code of this project.
The main file to run **simulation-based** studies is `run_sim.py`, and the file(s) to reproduce Fig. xx and Fig. xx is/are `plot_xx.py` and `plot_xx.py`.
The main file to run **real-time experimental** studies is `run_exp.py`.
`run_exp.py` only works when connected to the Mesbah Lab APPJ. The main file to run the **global sensitivity analysis**

Each of the main files above have been written to have the capability to run separate components of this work by modification of particular settings, particularly so for the real-time experimental trials.

## Folder Descriptions
This section briefly describes the purpose of each of the *folders* nested within this directory.
* `config` - contains files that determine the configuration of the system, important for describing the problem formulation of the case study
* `ExperimentalData` - a folder that saves the data collected from the experimental runs with the APPJ testbed; some of this data is redundant as the closed-loop trajectories (for this project) are also saved in the `trials` folder
* `KCutils` - contains the custom helper code for this project; the main focus of the work is on a policy search routine using Bayesian optimization (BO), so the ancillary portions of the work (e.g., scenario-based MPC formulation, neural network building and training, etc.) are placed in helper files.
* `saved` - contains files that save the initial policy used (the policy may change due to stochasticity in training); an alternate version of the code could be adapted such that the initial policy starts from the same file over multiple calls to the script. Additionally contains folders that contain saved plots of data
* `sensitivity` - contains subfolders that consist of collected data after performing sensitivity analyses; to avoid too many files, only data from every 1000th sample is saved; this behavior can be modified by passing in a different value for the keyword argument `mod_save` in the `get_objs` function of the `compute_objs_python.py` file. (The `get_objs` function is called within the MATLAB function `compute_objectives.m`.)
* `trials` - contains folders that consist of the collected data after running the script(s); it is the main source of data for simulation studies and a secondary storage location for experimental studies


## Other Files
This section briefly describes other files that are located within this main working directory.
* `appj_warmup.py` - a Python script for experimental startup. This file can serve two purposes, but the main one is to warm up the plasma jet after greater than 24 hours of inactivity. The secondary purpose is to make sure the asynchronous measurement is working with the connected computer system. (only works when connected to testbed)
* `compute_objectives.m` - a helper function in MATLAB m-code that connects the Python system to the global sensitivity analysis performed in MATLAB
* `compute_objs_python.py` - a Python script that defines a function that wraps around a closed loop simulation of the plasma system to generate the objectives of the work given modified parameters of the DNN-based controller (used for the global sensitivity analysis)
* `exp_data_agg.py` - a Python script that combines and converts collected closed-loop data from the experimental testbed to a set of input-output training data for training a DNN-based controller (used for experimental studies)
* `npy2mat.py` - a custom Python script that converts the data stored in NPY files to MAT file, in case it is necessary to use the data within MATLAB (e.g., generating sets of parameters for sensitivity analysis)
* `plot_data_mobo.py` - a custom Python script for plotting the multi-objective results (i.e., Fig. xx of the paper)
* `plot_end_simdata.py` - a custom Python script for plotting the single objective results (i.e., Fig. xx of the paper)
* `plot_exp_data.py` - a custom Python script for plotting the experimental results (using MOBO) (i.e., Fig. xx of the paper)
* `sensistivity.py` - a MATLAB script that performs a global sensitivity analysis of parameters of the first and last layers of a DNN-based controller to the desired closed-loop metrics explored in this paper
* `spectroscopyLive.py` - a Python script for testing and visualizing the optical emission spectra measurement (only works when connected to testbed)
