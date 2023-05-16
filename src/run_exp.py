'''
main script to run Bayesian optimization for policy search on an atmospheric
pressure plasma jet (APPJ) testbed. A neural network policy is initially learned
using closed-loop data from scenario-based model predictive control (sMPC)

REAL-TIME EXPERIMENT

Requirements:
* Python 3
* CasADi [https://web.casadi.org]
* PyTorch [https://pytorch.org]
* BoTorch [https://botorch.org] and Ax [https://ax.dev]
* Matplotlib [https://matplotlib.org] (for data visualization)
* a variety of other packages are necessary for communication with the testbed,
  please see the import list for these requirements

Copyright (c) 2022 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''

# import Python packages
import sys
sys.dont_write_bytecode = True
import os
from datetime import datetime
# import 3rd party packages
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_MOO_EHVI
from ax.modelbridge.modelbridge_utils import observed_hypervolume

# packages relevant to testbed (data acquisition)
from seabreeze.spectrometers import Spectrometer, list_devices
import time
import serial
import cv2
import asyncio
import scipy.io as sio

# import custom packages
from config.multistage import get_prob_info_exp
from config.reference_signal import myRef_CEM
from KCutils.controller import MultiStageMPC
from KCutils.neural_network import DNN, SimpleDNN
import KCutils.APPJPythonFunctions as appj
from KCutils.experiments import Experiment
import KCutils.plot_utils as pu
import KCutils.pytorch_utils as ptu

STARTUP_DUTY_CYCLE = 100 # default duty cycle
STARTUP_POWER = 2.0 # default power
STARTUP_FLOW = 3.0 # default flow rate

## NOTE: This script is designed to be used in several aspects of the work. As
# such, it is recommended that the user follows a particular order to obtain the
# desired results, which involves running this script several times with
# different purposes:
# 0. All triggers (options that should have boolean values) should begin with False
# 1. collect open loop data to identify a nominal LTI to be used in designing a controller; this is achieved by setting collect_open_loop_data=True
# 2. tune the controller (reset collect_open_loop_data=False, use_hw=False, use_dnn=False), adjust the controller tuning parameters in prob_info; examine the closed-loop profiles
# 3. collect closed-loop training data for the purpose of dnn approximation. Keeping the same conditions as above, set collect_trn_data=True; alternatively, manually adjust
# 4. train a DNN to approximate the controller by using the data generated in the previous step (reset above to False, train_dnn=True)
# 5. perform BO experiments initializing with the DNN trained in the previous step (reset all above to False)

### user inputs/options
ts = 0.5 # sampling time, value given by model identification from APPJ (overwritten below)
Nsim = int(2*60/ts)   # experiment horizon (for CEM dose delivery, this is the maximum time steps to consider)
Ts0_des = 37.0  # desired initial surface temperature to make consistent experiments
coolDownDiff = 1 # degrees to subtract from desired surface temperature for cooldown
warmUpDiff = 1 # degrees to subtract from desired surface temperature for warming up
model_file = '../model/APPJmodel_2022_09_22_17h28m06s.mat' # filename of identified model for plasma jet (to be used in controller)
# switch to collect open loop data of the APPJ system; it is recommended to collect
# open loop data to learn a new LTI model of the system in case the APPJ has been
# adjusted/moved/altered since the last model was identified
collect_open_loop_data = False
use_hw = False # whether to use a hardware (embedded) controller
use_dnn = True # whether to use an approximate controller (approximated by DNN)
run_test = False # option to run test or run full BO experiments
Nrep = 1 # number of replicates to repeat the experiment

## problem options
population_K = 0.5 # assumed "population" value for K
patient_K = 0.55 # individiual patient value for K
population_Tmax = 45.0 # assumed "population" temperature constraint
patient_Tmax = 44.5 # individual patient temperature constraint

## options related to the policy approximation by DNN
collect_trn_data = False # option to collect closed-loop data for training the policy (True to collect training data; False otherwise)
train_dnn = True # option to generate training data for the DNN (True to generate training data (specify number of samples to generate in second line), False to load data from a file (specify filename in the next line))
dnn_trn_file = './saved/dnn_trn_data_exp_data.npy' # filename for storing the training data for the policy approximation
ntr = int(5e3) # number of samples to generate for training an approximate policy; data generated is saved with the filename f'./saved/dnn_trn_data_{ntr}samp.npy'
# training parameters for customizing the training (in development)
fit_params = {}
fit_params['epochs'] = 5000
# DNN structure
H = 7 # number of nodes
L = 4 # number of layers
activation = 'relu' # activation function (currently, only ReLU is supported in this project)
initial_dnn_file = '' # initial parameters for the DNN policy

## options related to policy search using Bayesian optimization
policy_search = True     # option to perform policy search (True to perform policy search, False otherwise (if False, script stops after approximate controller has been tested))
random_search = False # option to use Random Search to modify the policy parameters (True for random search, False for Bayesian optimization)
MOO = True # option to perform a multi-objective (Bayesian) optimization routine (True for multi-objective, False for single objective)
n_mc = 1 # number of Monte Carlo samples of the full Bayesian optimization routine
n_bo_iter = 15 # number of iterations to update the policy
Nreps_per_iteration = 1 # number of repetitions of closed-loop runs using the same BO-suggested parameters per iteration
objective1_threshold = 100.0 # objective threshold for the first objective
objective2_threshold = 20.0 # objective threshold for the second objective

## options related to visualization of the simulations, note that the entire script will run before displaying the figures
plot_initial_smpc = False # option to plot the trajectories of just the sMPC with no mismatch between the true and assumed dose parameter (True to plot, False otherwise)
plot_some_training_data = False # option to visualize a section of the training data generated for training the DNN (True to plot, False otherwise)
plot_initial_dnn = False # option to plot the trajectories of the DNN controller with mismatch between the true and assumed dose parameter (True to plot, False otherwise)
Fontsize = 14 # default font size for plots
Lwidth = 3 # default line width for plots


### SETUP: do not edit below this line, otherwise the reproducibility of the results is not guaranteed
## setup for establishing plotting defaults
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

## setup for establishing PyTorch defaults
tkwargs = {
    'dtype': torch.double,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

if not collect_open_loop_data:
    # load the normalization factor for intensity used in model generation
    model = sio.loadmat(model_file)
    dataInfo = model['dataInfo'].item()
    I_NORMALIZATION = 1/float(dataInfo[4])
    print(I_NORMALIZATION)

date = datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')
print(f'Timestamp for save files: {date}')
# set up Experimental Data folder
directory = os.getcwd()
saveDir = directory+"/ExperimentalData/"+date+"/"
print(f'\nData will be saved in the following directory:\n {saveDir}')

################################################################################
# PROBLEM SETUP
################################################################################
if not collect_open_loop_data:
    # get problem information. Problem information is loaded from the
    # KCutils.multistage file. This file provides problem-specific information, which
    # may include system size and bounds, functions for evaluating the physical
    # system, controller parameters, etc.
    prob_info = get_prob_info_exp(Kcem=population_K, Tmax=population_Tmax, model_file=model_file, ts=ts)
    # a second problem info is created to establish a mismatch between the assumed
    # dose parameter value (derived from population data) and the "true" dose
    # parameter value of a particular "patient"
    prob_info2 = get_prob_info_exp(Kcem=patient_K, Tmax=patient_Tmax, model_file=model_file, ts=ts)

    xss = prob_info['xss']
    uss = prob_info['uss']
    xssp = prob_info['xssp']
    ussp = prob_info['ussp']
    x_max = prob_info['x_max']
    u_min = prob_info['u_min']
    u_max = prob_info['u_max']

################################################################################
# EXPERIMENT SETUP
################################################################################
# configure run options for gathering data
runOpts = appj.RunOpts()
runOpts.collectData = True
runOpts.collectEntireSpectra = True
runOpts.collectOscMeas = False
runOpts.collectSpatialTemp = False
runOpts.saveSpectra = True
runOpts.saveOscMeas = False
runOpts.saveSpatialTemp = False
runOpts.tSampling = ts

# connect to/open connection to devices in setup
# Arduino
arduinoAddress = appj.getArduinoAddress(os="ubuntu")
print("Arduino Address: ", arduinoAddress)
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
s = time.time()
# Oscilloscope
oscilloscope = appj.Oscilloscope()       # Instantiate object from class
instr = oscilloscope.initialize()	# Initialize oscilloscope
# Spectrometer
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])
spec.integration_time_micros(12000*6)
# Thermal Camera
dev, ctx = appj.openThermalCamera()
print("Devices opened/connected to sucessfully!")

devices = {}
devices['arduinoPI'] = arduinoPI
devices['arduinoAddress'] = arduinoAddress
devices['instr'] = instr
devices['spec'] = spec

# send startup inputs
time.sleep(2)
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
input("Ensure plasma has ignited and press Return to begin.\n")

## Startup asynchronous measurement
if os.name == 'nt':
    ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(ioloop)
else:
    ioloop = asyncio.get_event_loop()
# run once to initialize measurements
prevTime = (time.time()-s)*1e3
tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
print('measurement devices ready!')
s = time.time()

# let APPJ run for a bit
STARTUP_POWER = 1.5
STARTUP_FLOW = 1.5
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
time.sleep(0.5)

w8 = input("Wait 5 min? [y,n]\n")
if w8 == 'y':
    print("Waiting 5 minutes to ensure roughly steady plasma startup...\n")
    time.sleep(60)
    print("4 minutes left...")
    time.sleep(60)
    print("3 minutes left...")
    time.sleep(60)
    print("2 minutes left...")
    time.sleep(60)
    print("1 minute left...")
    time.sleep(60)
else:
    time.sleep(5)

# wait for cooldown
appj.sendInputsArduino(arduinoPI, 0.0, 0.0, STARTUP_DUTY_CYCLE, arduinoAddress)
arduinoPI.close()
while appj.getSurfaceTemperature() > Ts0_des-coolDownDiff:
    time.sleep(runOpts.tSampling)
    print('cooling down ...')
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
time.sleep(2)
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
# wait for surface to reach desired starting temp
while appj.getSurfaceTemperature() < Ts0_des-warmUpDiff:
    time.sleep(runOpts.tSampling)
    print('warming up ...')

prevTime = (time.time()-s)*1e3
# get initial measurements
tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
if runOpts.collectData:
    thermalCamOut = tasks[0].result()
    Ts0 = thermalCamOut[0]
    specOut = tasks[1].result()
    if collect_open_loop_data:
        I0 = specOut[0]
    else:
        I0 = specOut[0]*I_NORMALIZATION
    oscOut = tasks[2].result()
    arduinoOut = tasks[3].result()
    outString = "Measured Outputs: Temperature: %.2f, Intensity: %.2f" % (Ts0, I0)
    print(outString)
else:
    Ts0 = 37
    I0 = 100

s = time.time()

arduinoPI.close()

########################## create controller ###############################
if not collect_open_loop_data:
    if use_hw:
        c = None
        print('Not implemented yet!')
        raise
    elif use_dnn:
        c = DNN()
        if collect_trn_data:
            c = MultiStageMPC(prob_info)
            c.get_mpc()
            print('Nrep has been changed to 10 to collect closed loop data.')
            Nrep = 10
            print('Once data has been collected, data should be processed for training.')
        else:
            c.load_data(dnn_trn_file)
            in_data = c.inputs
            out_data = c.outputs

        if plot_some_training_data:
            in_range = c.input_max - c.input_min
            out_range = c.output_max - c.output_min
            T_in_data = ((in_data+1)*in_range[:,None]/2+c.input_min[:,None])[2,:]
            fig = plt.figure(figsize=(8,2))
            ax = fig.add_subplot(111)
            ax.plot(T_in_data[2*60:6*60])
            plt.draw()

        if train_dnn:
            initial_dnn_file = f'./saved/{date}_initial_policy_info_exp.npy'
            c.build_neural_network_pytorch(H=H, L=L, activation=activation)
            net = c.train_neural_network_pytorch(fit_params=fit_params,
                                                 save_file=initial_dnn_file)

            # evaluate initial DNN model
            X = ptu.from_numpy(c.inputs.T)
            Y = ptu.from_numpy(c.outputs.T)
            net.eval()
            with torch.no_grad():
                pred = net(X)
            max_deltas = torch.abs(0.99*Y)
            abs_deltas = torch.abs(pred - Y)
            check = abs_deltas < max_deltas
            acc = ptu.to_numpy(torch.sum(check, dim=0)/ntr)
            print(f'The training accuracy with 99% tolerance was {acc*100.0}% for each output (power, flow rate), respectively.')
        else:
            c = SimpleDNN(initial_dnn_file)

        _ = c.create_casadi_model()
    else:
        c = MultiStageMPC(prob_info)
        _, _, _ = c.get_mpc()
        # run an open loop simulation to test
        res, feas = c.solve_mpc()
        # print(res)
        # print(feas)

if any([collect_trn_data, collect_open_loop_data, run_test]):
    ################################################################################
    ## Begin Experiment: Experiment with generated hardware code
    ################################################################################
    exp = Experiment(Nsim, saveDir)

    if Nrep>1 and not use_hw:
        rng = np.random.default_rng(0)
        Ts0_vec = rng.uniform(36.0,40.0,size=(Nrep,))
    for i in range(Nrep):
        # connect to Arduino
        arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
        s = time.time()
        print('Pausing for cooldown...')
        time.sleep(2)
        if Nrep>1 and not use_hw:
            Ts0_des = Ts0_vec[i]
            print(f'Desired Starting Temperature: {Ts0_des}\n')
        while appj.getSurfaceTemperature() > Ts0_des-coolDownDiff:
            time.sleep(runOpts.tSampling)
        appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
        devices['arduinoPI'] = arduinoPI

        if collect_open_loop_data:
            # create input sequences
            uvec1 = np.linspace(1.5,3.5,5) # for power
            uvec2 = np.linspace(3.5,7.5,9) # for flow rate
            uu1,uu2 = np.meshgrid(uvec1,uvec2)
            uvec1 = uu1.reshape(-1,)
            uvec2 = uu2.reshape(-1,)
            rng = np.random.default_rng(0)
            rng.shuffle(uvec1)
            pseq = np.copy(uvec1)
            pseq = np.insert(pseq,0,[0.0,2.5,2.5,2.5])
            rng.shuffle(uvec2)
            qseq = np.copy(uvec2)
            qseq = np.insert(qseq,0,[0.0,5.0,5.0,5.0])
            print(pseq)
            print(qseq)

            pseq = np.repeat(pseq, 30/runOpts.tSampling).reshape(-1,)
            qseq = np.repeat(qseq, 30/runOpts.tSampling).reshape(-1,)
            print(pseq.shape[0])

            prevTime = (time.time()-s)*1e3
            exp_data = exp.run_open_loop(ioloop,
                                         power_seq=pseq,
                                         flow_seq=qseq,
                                         runOpts=runOpts,
                                         devices=devices,
                                         prevTime=prevTime)
        else:
            # prob_info['y0'] = np.ravel([Ts0,I0])
            if plot_initial_dnn:
                exp.load_prob_info(prob_info2)
            else:
                exp.load_prob_info(prob_info)

            prevTime = (time.time()-s)*1e3
            exp_data = exp.run_closed_loop_mpc(c, ioloop,
                                               runOpts=runOpts,
                                               devices=devices,
                                               prevTime=prevTime,
                                               CEM=True,
                                               hw_flag=use_hw,
                                               dnn_flag=use_dnn,
                                               I_NORM=I_NORMALIZATION)
            if plot_initial_smpc:
                fig_objs = pu.plot_data_from_dict(exp_data, prob_info, CEM=True)
            elif run_test:
                fig_objs = pu.plot_data_from_dict(exp_data, prob_info, CEM=True)
            elif plot_initial_dnn:
                fig_objs = pu.plot_data_from_dict(exp_data, prob_info2, CEM=True)

        arduinoPI.close()
# user input to continue the script. The purpose of this user input is to ensure
# the initial approximate policy is sufficient to continue to optimization.
continue_with_bo = input('Continue with BO? [Y/n]\n')
if collect_open_loop_data:
    continue_with_bo = 'n'
if continue_with_bo in ['Y', 'y']:
    ################################################################################
    # POLICY SEARCH with BAYESIAN OPTIMIZATION
    ################################################################################
    def evaluate(parameters, init_file='', save_prefix='', Nrep=1, no_noise=False,
                 Nsim=60, use_hw=False, use_dnn=False, devices={}, prevTime=0.0,
                 I_NORM=1, Ts0_des=37.0, coolDownDiff=1, prob_info={}):
        # define a function which encapsulates a closed-loop run of the system;
        # the evaluate function is the expensive function to query from in order
        # to obtain closed-loop metrics
        sdnn = SimpleDNN(init_file)
        n_out = sdnn.outputs.shape[0]
        H = sdnn.H
        Wn = np.asarray([parameters[f'W{i}'] for i in range(n_out*H)])
        bn = np.asarray([parameters[f'b{i}'] for i in range(n_out)])
        Wn = Wn.reshape(n_out,H)
        bn = bn.reshape(n_out,)
        sdnn.modify_last_layer([Wn,bn], save_file=f'./trials/{date}/dnns/{save_prefix}_dnn_exp.npy')

        # patient has different Kcem value compared to what was trained initially
        exp = Experiment(Nsim, saveDir+save_prefix+'/')
        exp.load_prob_info(prob_info)

        if Nrep>1 and not use_hw:
            rng = np.random.default_rng(0)
            Ts0_vec = rng.uniform(35.0,39.0,size=(Nrep,))
        # repeat the simulation Nrep
        obj1s = np.zeros((Nrep,))
        obj2s = np.zeros((Nrep,))
        d = {}
        for i in range(Nrep):
            # connect to Arduino
            arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
            s = time.time()
            print('Pausing for cooldown...')
            time.sleep(2)
            if Nrep>1 and not use_hw:
                Ts0_des = Ts0_vec[i]
                print(f'Desired Starting Temperature: {Ts0_des}\n')
            while appj.getSurfaceTemperature() > Ts0_des-coolDownDiff:
                time.sleep(runOpts.tSampling)
            appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
            devices['arduinoPI'] = arduinoPI
            prevTime = (time.time()-s)*1e3

            exp_data = exp.run_closed_loop_mpc(c, ioloop,
                                               runOpts=runOpts,
                                               devices=devices,
                                               prevTime=prevTime,
                                               CEM=True,
                                               hw_flag=use_hw,
                                               dnn_flag=use_dnn,
                                               I_NORM=I_NORM)

            st = exp_data['CEM_stop_time']
            CEM = np.ravel(exp_data['CEMsim'][:,:st])
            Ref = np.ravel(exp_data['Yrefsim'][:,:st])
            T = np.ravel(exp_data['Tsave'][:st])
            Tmax = prob_info2['x_max'][0]+prob_info2['xss'][0]

            CEMobj = np.sum((CEM-Ref)**2)
            Tobj = np.sum((T[T>Tmax] - Tmax)**2)
            s = {}
            s['exp_data'] = exp_data
            s['params'] = [Wn, bn]
            s['obj_val'] = {'CEMobj': CEMobj, 'Tobj': Tobj}
            d[f'rep{i}'] = s
            obj1s[i] = CEMobj
            obj2s[i] = Tobj
        # closed loop data is saved as a dictionary of dictionaries (keys are
        # 'rep{#}'). The nested dictionary contains the sim_data, the parameters
        # by BO, and the objective value(s).
        np.save(f'./trials/{date}/cl_data/{save_prefix}_exp_data.npy', d, allow_pickle=True)

        if Nrep > 1:
            # compute stats for multiple repetitions
            obj1 = np.mean(obj1s)
            obj1_ste = np.std(obj1s)/np.sqrt(Nrep)
            obj2 = np.mean(obj2s)
            obj2_ste = np.std(obj2s)/np.sqrt(Nrep)

            obj1 = torch.tensor(obj1, **tkwargs)
            obj2 = torch.tensor(obj2, **tkwargs)
            obj1_ste = torch.tensor(obj1_ste, **tkwargs)
            obj2_ste = torch.tensor(obj2_ste, **tkwargs)

            return {'obj1': (obj1.item(), obj1_ste.item()),
                    'obj2': (obj2.item(), obj2_ste.item())}

        elif Nrep == 1 and no_noise:
            # no noise (if function is deterministic)
            obj1 = np.ravel(obj1s)
            obj2 = np.ravel(obj2s)

            obj1 = torch.tensor(obj1, **tkwargs)
            obj2 = torch.tensor(obj2, **tkwargs)

            return {'obj1': (obj1.item(), 0.0), 'obj2': (obj2.item(), 0.0)}
        else:
            # otherwise ask BO algorithm to estimate the noise
            obj1 = np.ravel(obj1s)
            obj2 = np.ravel(obj2s)

            obj1 = torch.tensor(obj1, **tkwargs)
            obj2 = torch.tensor(obj2, **tkwargs)

            return {'obj1': (obj1.item(), None), 'obj2': (obj2.item(), None)}


    n_out = out_data.shape[0]
    n_params = n_out * (H+1)
    print(f'Number of Parameters: {n_params}')
    # last layer weights: 0:n_out*H -> Wn, n_out*H: -> bn
    initial_parameters = np.concatenate(((c.W[-1]).flatten(), (c.b[-1]).flatten()))
    assert len(initial_parameters) == n_params
    r = 2.0
    lower_bounds = [1/r * initial_parameters[i] if initial_parameters[i]>0 else r * initial_parameters[i] for i in range(n_params)]
    upper_bounds = [r * initial_parameters[i] if initial_parameters[i]>0 else 1/r * initial_parameters[i] for i in range(n_params)]

    param_keys = [*[f'W{i}' for i in range(n_out*H)],
                  *[f'b{j}' for j in range(n_out)]]
    initial_parameters = [float(initial_parameters[i]) for i in range(n_params)]
    initial_parameters = dict(zip(param_keys, initial_parameters))

    if policy_search:
        # create directories for saving files
        os.makedirs(f'./trials/{date}', exist_ok=True)
        os.makedirs(f'./trials/{date}/dnns', exist_ok=True)
        os.makedirs(f'./trials/{date}/cl_data', exist_ok=True)

        # set up parameter generation strategy for BO
        if random_search:
            gs = GenerationStrategy(steps=[GenerationStep(Models.SOBOL, num_trials=-1)])
        else:
            if MOO:
                bo_model = Models.MOO
            else:
                bo_model = Models.GPEI

            gs = GenerationStrategy(
                steps = [
                # Bayesian optimization step (requires data some form of
                # starting data and learns from all data available at the time
                # of each new candidate generation call)
                GenerationStep(
                    model = bo_model,
                    num_trials = -1,
                ),
                ]
            )

        # set up objective(s)
        if MOO:
            objectives = {
                    'obj1': ObjectiveProperties(minimize=True, threshold=objective1_threshold),
                    'obj2': ObjectiveProperties(minimize=True, threshold=objective2_threshold),
                    }
        else:
            objectives = {'obj1': ObjectiveProperties(minimize=True)}

        # set up parameter(s)
        parametersW = [
            {'name': f'W{i}',
             'type': 'range',
             'bounds': [float(lower_bounds[i]), float(upper_bounds[i])],
             'value_type': 'float',
            } for i in range(n_out*H)
        ]
        parametersb = [
            {'name': f'b{i}',
             'type': 'range',
             'bounds': [float(lower_bounds[i+n_out*H]), float(upper_bounds[i+n_out*H])],
             'value_type': 'float',
            } for i in range(n_out)
        ]

        # pack away testbed-experiment-relevant infomation for portability
        exp_settings = {'devices': devices,
                        'Nsim': Nsim,
                        'use_hw': use_hw,
                        'use_dnn': use_dnn,
                        'I_NORM': I_NORMALIZATION,
                        'Ts0_des': Ts0_des,
                        'coolDownDiff': coolDownDiff,
                        'prob_info': prob_info2,
                        }

        ## use Bayesian optimization to adjust policy
        hv_mc = []
        for n in range(n_mc):
            ## setup experiments for Bayesian optimization
            mobo_save_filepath = f'./trials/{date}/exp_ax_client_snapshot{n}.json'

            # set a random seed/state for repeatability
            rs = int(42*(n+1234))

            ax_client = AxClient(random_seed=42, generation_strategy=gs)
            ax_client.create_experiment(
                name = f'bo_policy_search_trial{n}',
                parameters = [*parametersW, *parametersb],
                objectives = objectives,
                overwrite_existing_experiment = True,
                is_test = False,
            )

            # attach initial trial/data
            ax_client.attach_trial(parameters=initial_parameters)
            raw_data = evaluate(initial_parameters,
                                Nrep=Nreps_per_iteration,
                                init_file=initial_dnn_file,
                                save_prefix=f'trial{n}_iter{0}',
                                **exp_settings)
            ax_client.complete_trial(trial_index=0, raw_data=raw_data)
            ax_client.save_to_json_file(mobo_save_filepath)

            if MOO:
                hv_list = np.zeros((n_bo_iter,))

            for i in range(n_bo_iter):
                parameters, trial_index = ax_client.get_next_trial()
                raw_data = evaluate(parameters,
                                    Nrep=Nreps_per_iteration,
                                    init_file=initial_dnn_file,
                                    save_prefix=f'trial{n}_iter{i+1}',
                                    **exp_settings)
                ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

                if MOO:
                    try:
                        # fit a GP-based model in order to calculate hypervolume
                        dummy_model = get_MOO_EHVI(
                                        experiment=ax_client.experiment,
                                        data=ax_client.experiment.fetch_data(),
                                        )
                        hv_list[i] = observed_hypervolume(modelbridge=dummy_model)
                    except Exception as e:
                        print(e)
                        print('Failed to compute hypervolume')
                        hv_list[i] = 0.0
                    hv_mc.append(hv_list)

                ax_client.save_to_json_file(mobo_save_filepath)

            if not MOO:
                # if single objective, plot the closed-loop trajectories using
                # the best parameters suggested by BO
                best_parameters, values = ax_client.get_best_parameters()
                print(best_parameters)

                sdnn = SimpleDNN(initial_dnn_file)
                n_out = sdnn.outputs.shape[0]
                H = sdnn.H
                Wn = np.asarray([best_parameters[f'W{i}'] for i in range(n_out*H)])
                bn = np.asarray([best_parameters[f'b{i}'] for i in range(n_out)])
                Wn = Wn.reshape(n_out,H)
                bn = bn.reshape(n_out,)
                sdnn.modify_last_layer([Wn,bn], save_file=f'./trials/{date}/dnns/best_dnn.npy')

                # patient has different Kcem value compared to what was trained initially
                exp = Experiment(Nsim, saveDir+'final/')
                exp.load_prob_info(prob_info2)

                exp_data2 = exp.run_closed_loop(sdnn, CEM=True)
                st2 = exp_data2['CEM_stop_time']
                CEMplot2 = np.ravel(exp_data2['CEMsim'][:,:st2])
                Tplot2 = exp_data2['Tsave'][:st2] + xssp[0]

                fig = plt.figure(figsize=(10,5))
                ax = fig.add_subplot(221)
                ax.axhline(Yrefplot[0], color='r', linestyle='--', label='Target Reference')
                ax.plot(np.arange(len(CEMplot2))*ts, CEMplot2, ':', label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('CEM')
                ax.legend(loc='lower right')

                ax = fig.add_subplot(222)
                ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
                ax.plot(np.arange(len(Tplot2))*ts, Tplot2, '--', label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Surface Temperature ($^\circ$C)')

                ax = fig.add_subplot(223)
                ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(st2-1)*ts, exp_data2['Psave'][:(st2-1)]+uss[0], label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Power (W)')

                ax = fig.add_subplot(224)
                ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(st2-1)*ts, exp_data2['qSave'][:(st2-1)]+uss[1], label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Flow Rate (SLM)')
                plt.draw()

        if MOO:
            # if multi-objective, just plot the expected hypervolume improvement
            hv_mc = np.vstack(hv_mc)
            mean_hv = np.nanmean(hv_mc, axis=0).reshape(-1,1)
            std_hv = np.nanstd(hv_mc, axis=0).reshape(-1,1)
            ste_hv = std_hv/np.sqrt(n_mc)

            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            ax.plot(np.arange(n_bo_iter), mean_hv)
            ax.fill_between(
                            np.arange(n_bo_iter),
                            np.ravel(mean_hv-ste_hv),
                            np.ravel(mean_hv+ste_hv),
                            alpha=0.2,
            )
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
            plt.draw()

            s = {}
            s['hypervolumes'] = hv_mc
            s['hv_mean'] = mean_hv
            s['hv_ste'] = ste_hv
            s['hv_std'] = std_hv
            np.save(f'./trials/{date}/hypervolume_stats.npy',s,allow_pickle=True)

else:
    print('Did not perform Bayesian optimization.')
plt.show()

# reconnect Arduino
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
devices['arduinoPI'] = arduinoPI

# turn off plasma jet (programmatically)
appj.sendInputsArduino(arduinoPI, 0.0, 0.0, STARTUP_DUTY_CYCLE, arduinoAddress)
arduinoPI.close()
print(f"Experiments complete at {datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')}!\n"+
    "################################################################################################################\n"+
    "IF FINISHED WITH EXPERIMENTS, PLEASE FOLLOW THE SHUT-OFF PROCEDURE FOR THE APPJ\n"+
    "################################################################################################################\n")
