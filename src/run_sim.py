'''
main script to run Bayesian optimization for policy search on an atmospheric
pressure plasma jet (APPJ) system in silico. A neural network policy is
initially learned using closed-loop data from scenario-based model predictive
control (sMPC)

Requirements:
* Python 3
* CasADi [https://web.casadi.org]
* PyTorch [https://pytorch.org]
* BoTorch [https://botorch.org] and Ax [https://ax.dev]
* Matplotlib [https://matplotlib.org] (for data visualization)

Copyright (c) 2021 Mesbah Lab. All Rights Reserved.

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
# import custom packages
from config.multistage import get_prob_info
from KCutils.controller import MultiStageMPC
from KCutils.simulation import Simulation
from KCutils.neural_network import DNN, SimpleDNN
import KCutils.pytorch_utils as ptu

### user inputs/options
ts = 0.5 # sampling time, value given by model identification from APPJ (overwritten below)
Nsim = int(2*60/ts)   # Simulation horizon (for CEM dose delivery, this is the maximum simulation steps to consider)

## problem options
population_K = 0.5 # assumed "population" value for K
patient_K = 0.55 # individiual patient value for K
population_Tmax = 45.0 # assumed "population" temperature constraint
patient_Tmax = 45.5 # individual patient temperature constraint

## options related to the policy approximation by DNN
open_loop_training = False  # option to perform generate open-loop or closed-loop data for training the policy (True for open-loop training data; False for closed-loop training data)
train_dnn = False # option to generate training data for the DNN (True to generate training data (specify number of samples to generate in second line), False to load data from a file (specify filename in the next line))
dnn_trn_file = './saved/dnn_trn_data_5000samp.npy' # filename for storing the training data for the policy approximation
ntr = int(5e3) # number of samples to generate for training an approximate policy; data generated is saved with the filename f'./saved/dnn_trn_data_{ntr}samp.npy'
# training parameters for customizing the training (in development)
fit_params = {}
fit_params['epochs'] = 5000
# DNN structure
H = 7 # number of nodes
L = 4 # number of layers
activation = 'relu' # activation function (currently, only ReLU is supported in this project)
dnn_sensitivity_analysis = False
sensitivity_option = 'first-last' # select one from ['all', 'half', 'first-last']
Nreps_per_sensitivity = 3 # number of repetitions of closed-loop simulations (to obtain noisy closed-loop metrics)
if dnn_sensitivity_analysis:
    if sensitivity_option == 'all':
        layers_to_analyze = list(range(L))
        flattened_nodes_step = 1
    elif sensitivity_option == 'half':
        layers_to_analyze = list(range(0,L,2))
        flattened_nodes_step = 2
    elif sensitivity_option == 'first-last':
        layers_to_analyze = [0,L]
        flattened_nodes_step = 1
    else:
        print('Unsupported sensitivity option. Please select on from [''all'', ''half'', ''first-last''].')
        raise

## options related to policy search using Bayesian optimization
policy_search = True     # option to perform policy search (True to perform policy search, False otherwise (if False, script stops after approximate controller has been tested))
random_search = False # option to use Random Search to modify the policy parameters (True for random search, False for Bayesian optimization)
MOO = True # option to perform a multi-objective (Bayesian) optimization routine (True for multi-objective, False for single objective)
n_mc = 5 # number of Monte Carlo samples of the full Bayesian optimization routine
n_bo_iter = 50 # number of iterations to update the policy
Nreps_per_iteration = 3 # number of repetitions of closed-loop runs using the same BO-suggested parameters per iteration

## options related to visualization of the simulations, note that the entire script will run before displaying the figures
plot_initial_smpc = True # option to plot the trajectories of just the sMPC with no mismatch between the true and assumed dose parameter (True to plot, False otherwise)
plot_some_training_data = False # option to visualize a section of the training data generated for training the DNN (True to plot, False otherwise)
plot_initial_dnn = False # option to plot the trajectories of the DNN controller with mismatch between the true and assumed dose parameter (True to plot, False otherwise)
Fontsize = 14 # default font size for plots
Lwidth = 3 # default line width for plots

model_file = '../model/APPJmodel_2022_09_22_17h28m06s.mat' # filename of identified model for plasma jet (to be used in controller)


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

################################################################################
# SIMULATION SETUP
################################################################################
date = datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')

# get problem information. Problem information is loaded from the
# config.multistage file. This file provides problem-specific information, which
# may include system size and bounds, functions for evaluating the physical
# system, controller parameters, etc.
prob_info = get_prob_info(Kcem=population_K, Tmax=population_Tmax)
# prob_info = get_prob_info_exp(Kcem=population_K, Tmax=45, model_file=model_file) # for testing model for experiments
# a second problem info is created to establish a mismatch between the assumed
# dose parameter value (derived from population data) and the "true" dose
# parameter value of a particular "patient"
prob_info2 = get_prob_info(Kcem=patient_K, Tmax=patient_Tmax)
# prob_info2 = get_prob_info_exp(Kcem=patient_K, Tmax=45, model_file=model_file) # for testing model for experiments

ts = prob_info['ts']
xss = prob_info['xss']
uss = prob_info['uss']
xssp = prob_info['xssp']
ussp = prob_info['ussp']
x_max = prob_info['x_max']
u_min = prob_info['u_min']
u_max = prob_info['u_max']

########################## create multistage mpc ###############################
c = MultiStageMPC(prob_info)
_, _, _ = c.get_mpc()

# run an open loop simulation to test
res, feas = c.solve_mpc()
# print(res)
# print(feas)

## run closed loop simulation using MPC
sim = Simulation(Nsim)
sim.load_prob_info(prob_info)
sim_data = sim.run_closed_loop(c, CEM=True, multistage=True)
print('Simulation Data Keys: ', [*sim_data])

Yrefsim = np.ravel(sim_data['Yrefsim'])
CEMsim = np.ravel(sim_data['CEMsim'])
st = sim_data['CEM_stop_time']
ctime = sim_data['ctime'][:st]
Xsim = sim_data['Xsim']
print('Total Runtime: ', np.sum(ctime))
print('Average Runtime: ', np.mean(ctime))

CEMplot = CEMsim[:st+1]
Yrefplot = Yrefsim[:st+1]
Tplot = Xsim[0,:st+1] + xssp[0]

if plot_initial_smpc:
    ## plot outputs
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(221)
    ax.plot(np.arange(len(CEMplot))*ts, Yrefplot, 'k--')
    ax.plot(np.arange(len(CEMplot))*ts, CEMplot)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CEM')

    ax = fig.add_subplot(222)
    ax.plot(np.arange(len(CEMplot))*ts, (x_max[0]+xss[0])*np.ones_like(Tplot), 'k--')
    ax.plot(np.arange(len(CEMplot))*ts, Tplot)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Surface Temperature (degC)')

    ## plot inputs
    ax = fig.add_subplot(223)
    ax.step(np.arange(st-1)*ts, sim_data['Usim'][0,:(st-1)]+ussp[0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power')

    ax = fig.add_subplot(224)
    ax.step(np.arange(st-1)*ts, sim_data['Usim'][1,:(st-1)]+ussp[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flow Rate')
    plt.draw()

############## train a neural network to approximate control law ###############
c_approx = DNN()
if train_dnn:
    ## get training data
    if open_loop_training:
        in_data, out_data = c_approx.get_training_data_open_loop(c, Nsamp=ntr, mpc_type='multistage')
    else:
        in_data, out_data = c_approx.get_training_data_closed_loop(c, Nsamp=ntr, mpc_type='multistage', Nsim=100)
    s = {}
    s['inputs'] = c_approx.inputs
    s['outputs'] = c_approx.outputs
    s['input_min'] = c_approx.input_min
    s['input_max'] = c_approx.input_max
    s['output_min'] = c_approx.output_min
    s['output_max'] = c_approx.output_max
    np.save(f'./saved/dnn_trn_data_{ntr}samp.npy', s , allow_pickle=True)
else:
    c_approx.load_data(dnn_trn_file)
    in_data = c_approx.inputs
    out_data = c_approx.outputs

if plot_some_training_data:
    in_range = c_approx.input_max - c_approx.input_min
    out_range = c_approx.output_max - c_approx.output_min
    T_in_data = ((in_data+1)*in_range[:,None]/2+c_approx.input_min[:,None])[2,:]
    fig = plt.figure(figsize=(8,2))
    ax = fig.add_subplot(111)
    ax.plot(T_in_data[2*60:6*60])
    plt.draw()

initial_dnn_file = f'./saved/{date}_initial_policy_info.npy'
c_approx.build_neural_network_pytorch(H=H, L=L, activation=activation)
net = c_approx.train_neural_network_pytorch(fit_params=fit_params,
                                            save_file=initial_dnn_file)

# evaluate initial DNN model
X = ptu.from_numpy(c_approx.inputs.T)
Y = ptu.from_numpy(c_approx.outputs.T)
net.eval()
with torch.no_grad():
    pred = net(X)
max_deltas = torch.abs(0.99*Y)
abs_deltas = torch.abs(pred - Y)
check = abs_deltas < max_deltas
acc = ptu.to_numpy(torch.sum(check, dim=0)/ntr)
print(f'The training accuracy with 99% tolerance was {acc*100.0}% for each output (power, flow rate), respectively.')

netca = c_approx.create_casadi_model()

print('Simulation with DNN (no mismatch): ')
sim_data1 = sim.run_closed_loop(c_approx, CEM=True)
print('Simulation Data Keys: ', [*sim_data1])
st1 = sim_data1['CEM_stop_time']
ctime1 = sim_data1['ctime'][:st1]
Xsim1 = sim_data1['Xsim']
print('Total Runtime: ', np.sum(ctime1))
print('Average Runtime: ', np.mean(ctime1))

CEMplot1 = np.ravel(sim_data1['CEMsim'][:,:st1])
Yrefplot1 = np.ravel(sim_data1['Yrefsim'][:,:st1])
Tplot1 = Xsim1[0,:st1] + xssp[0]

## plot results - the results of a simulation using the approximate controller
# (DNN) WITHOUT mismatch are plotted to visually examine the degree of fit between
# the DNN and the sMPC
len_list = [st, st1]
max_idx = np.argmax(len_list)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(221)
ax.axhline(Yrefplot[0], color='r', linestyle='--', label='Target Reference')
ax.plot(np.arange(len(CEMplot))*ts, CEMplot, label='sMPC')
ax.plot(np.arange(len(CEMplot1))*ts, CEMplot1, '--', label='DNN')
ax.set_xlabel('Time (s)')
ax.set_ylabel('CEM')
ax.legend(loc='lower right')

ax = fig.add_subplot(222)
ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
ax.plot(np.arange(len(Tplot))*ts, Tplot, label='sMPC')
ax.plot(np.arange(len(Tplot1))*ts, Tplot1, '--', label='DNN')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Surface Temperature ($^\circ$C)')

ax = fig.add_subplot(223)
ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[0], label='sMPC')
ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][0,:(st1-1)]+uss[0], label='DNN')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power (W)')

ax = fig.add_subplot(224)
ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][1,:(st-1)]+uss[1], label='sMPC')
ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][1,:(st1-1)]+uss[1], label='DNN')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow Rate (SLM)')
plt.draw()

if dnn_sensitivity_analysis:
    # create directories for saving files
    os.makedirs(f'./sensitivity/{date}', exist_ok=True)
    os.makedirs(f'./sensitivity/{date}/dnns', exist_ok=True)
    os.makedirs(f'./sensitivity/{date}/cl_data', exist_ok=True)

    n_in = len(in_range)
    n_out = len(out_range)

    layers_W = [(c_approx.W[l]).flatten() for l in layers_to_analyze]
    W_sensitivity = [W[::flattened_nodes_step] for W in layers_W]
    layers_b = [(c_approx.b[l]).flatten() for l in layers_to_analyze]
    b_sensitivity = [b[::flattened_nodes_step] for b in layers_b]

    initial_parameters = np.concatenate((W_sensitivity, b_sensitivity))
    n_params = len(initial_parameters)
    print(f'Number of Parameters: {n_params}')
    r = 2.0
    lower_bounds = [1/r * initial_parameters[i] if initial_parameters[i]>0 else r * initial_parameters[i] for i in range(n_params)]
    upper_bounds = [r * initial_parameters[i] if initial_parameters[i]>0 else 1/r * initial_parameters[i] for i in range(n_params)]

    n_intervals = 5
    for p in range(n_params):
        print(initial_parameters[p])
        search_vals = np.geomspace(lower_bounds[p], upper_bounds[p], num=n_intervals, base=r)
        print(search_vals)

        if sensitivity_option == 'all':
            param_num = p
        elif sensitivity_option == 'half':
            param_num = p*2
        elif sensitivity_option == 'first-last':
            if p < n_in*H:
                # first layer weight matrix
                param_num = p
            elif p < n_in*H + H**2*(L-1):
                # last layer weight matrix and first layer bias
                param_num = p + H**2*(L-1)
            else:
                # last layer bias
                param_num = p + (L-1)*H**2 + H*n_out + H + (L-1)*H
        else:
            print('Unsupported sensitivity option.')
            raise

        for s,val in enumerate(search_vals):
            sdnn = SimpleDNN(initial_dnn_file)
            sdnn.modify_nth_weight(param_num, val, save_file=f'./sensitivity/{date}/dnns/param{param_num}/dnn{s}.npy')

            obj1s = np.zeros((Nreps_per_sensitivity,))
            obj2s = np.zeros((Nreps_per_sensitivity,))
            d = {}
            for i in range(Nreps_per_sensitivity):
                sim.run_closed_loop(sdnn, CEM=True, rand_seed=i*5678)

                st = sim_data['CEM_stop_time']
                CEM = np.ravel(sim_data['CEMsim'][:,:st])
                Ref = np.ravel(sim_data['Yrefsim'][:,:st])
                T = np.ravel(sim_data['Ysim'][0,:st])
                Tmax = prob_info2['x_max'][0]

                CEMobj = np.sum((CEM-Ref)**2)
                Tobj = np.sum((T[T>Tmax] - Tmax)**2)
                s = {}
                s['sim_data'] = sim_data
                # s['params'] = [Wn, bn]
                s['obj_val'] = {'CEMobj': CEMobj, 'Tobj': Tobj}
                d[f'rep{i}'] = s
                obj1s[i] = CEMobj
                obj2s[i] = Tobj
            # closed loop data is saved as a dictionary of dictionaries (keys are
            # 'rep{#}'). The nested dictionary contains the sim_data, the parameters
            # by BO, and the objective value(s).
            np.save(f'./sensitivity/{date}/cl_data/param{param_num}/sim_data_{s}.npy', d, allow_pickle=True)


print('Simulation with DNN (with mismatch, before BO): ')
sim.load_prob_info(prob_info2)
sim_data1 = sim.run_closed_loop(c_approx, CEM=True)
print('Simulation Data Keys: ', [*sim_data1])
st1 = sim_data1['CEM_stop_time']
ctime1 = sim_data1['ctime'][:st1]
Xsim1 = sim_data1['Xsim']
print('Total Runtime: ', np.sum(ctime1))
print('Average Runtime: ', np.mean(ctime1))

CEMplot1 = np.ravel(sim_data1['CEMsim'][:,:st1])
Yrefplot1 = np.ravel(sim_data1['Yrefsim'][:,:st1])
Tplot1 = Xsim1[0,:st1] + xssp[0]

if plot_initial_dnn:
    ## plot results - this plotting for the DNN is different than previous. These
    # show the results of a DNN on a system WITH mismatch.
    len_list = [st, st1]
    max_idx = np.argmax(len_list)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(221)
    ax.axhline(Yrefplot[0], color='r', linestyle='--', label='Target Reference')
    ax.plot(np.arange(len(CEMplot))*ts, CEMplot, label='sMPC')
    ax.plot(np.arange(len(CEMplot1))*ts, CEMplot1, '--', label='DNN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CEM')
    ax.legend(loc='lower right')

    ax = fig.add_subplot(222)
    ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
    ax.plot(np.arange(len(Tplot))*ts, Tplot, label='sMPC')
    ax.plot(np.arange(len(Tplot1))*ts, Tplot1, '--', label='DNN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Surface Temperature ($^\circ$C)')

    ax = fig.add_subplot(223)
    ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
    ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
    ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[0], label='sMPC')
    ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][0,:(st1-1)]+uss[0], label='DNN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')

    ax = fig.add_subplot(224)
    ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
    ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
    ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[1], label='sMPC')
    ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][0,:(st1-1)]+uss[1], label='DNN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flow Rate (SLM)')
    plt.draw()

# user input to continue the script. The purpose of this user input is to ensure
# the initial approximate policy is sufficient to continue to optimization.
continue_with_bo = input('Continue with BO? [Y/n]\n')
if continue_with_bo in ['Y', 'y']:
    ################################################################################
    # POLICY SEARCH with BAYESIAN OPTIMIZATION
    ################################################################################
    def evaluate(parameters, init_file='', save_prefix='', Nreps=1, no_noise=False):
        # define a function which encapsulates a closed-loop run of the system
        sdnn = SimpleDNN(init_file)
        n_out = sdnn.outputs.shape[0]
        H = sdnn.H
        Wn = np.asarray([parameters[f'W{i}'] for i in range(n_out*H)])
        bn = np.asarray([parameters[f'b{i}'] for i in range(n_out)])
        Wn = Wn.reshape(n_out,H)
        bn = bn.reshape(n_out,)
        sdnn.modify_last_layer([Wn,bn], save_file=f'./trials/{date}/dnns/{save_prefix}_dnn.npy')

        # patient has different Kcem value compared to what was trained initially
        sim = Simulation(Nsim)
        sim.load_prob_info(prob_info2)

        # repeat the simulation Nreps
        obj1s = np.zeros((Nreps,))
        obj2s = np.zeros((Nreps,))
        d = {}
        for i in range(Nreps):
            sim_data = sim.run_closed_loop(sdnn, CEM=True, rand_seed2=i*987)

            st = sim_data['CEM_stop_time']
            CEM = np.ravel(sim_data['CEMsim'][:,:st])
            Ref = np.ravel(sim_data['Yrefsim'][:,:st])
            T = np.ravel(sim_data['Ysim'][0,:st])
            Tmax = prob_info2['x_max'][0]

            CEMobj = np.sum((CEM-Ref)**2)
            Tobj = np.sum((T[T>Tmax] - Tmax)**2)
            s = {}
            s['sim_data'] = sim_data
            s['params'] = [Wn, bn]
            s['obj_val'] = {'CEMobj': CEMobj, 'Tobj': Tobj}
            d[f'rep{i}'] = s
            obj1s[i] = CEMobj
            obj2s[i] = Tobj
        # closed loop data is saved as a dictionary of dictionaries (keys are
        # 'rep{#}'). The nested dictionary contains the sim_data, the parameters
        # by BO, and the objective value(s).
        np.save(f'./trials/{date}/cl_data/{save_prefix}_sim_data.npy', d, allow_pickle=True)

        if Nreps > 1:
            # compute stats for multiple repetitions
            obj1 = np.mean(obj1s)
            obj1_ste = np.std(obj1s)/np.sqrt(Nreps)
            obj2 = np.mean(obj2s)
            obj2_ste = np.std(obj2s)/np.sqrt(Nreps)

            obj1 = torch.tensor(obj1, **tkwargs)
            obj2 = torch.tensor(obj2, **tkwargs)
            obj1_ste = torch.tensor(obj1_ste, **tkwargs)
            obj2_ste = torch.tensor(obj2_ste, **tkwargs)

            return {'obj1': (obj1.item(), obj1_ste.item()),
                    'obj2': (obj2.item(), obj2_ste.item())}

        elif Nreps == 1 and no_noise:
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
    initial_parameters = np.concatenate(((c_approx.W[-1]).flatten(), (c_approx.b[-1]).flatten()))
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

        ## use Bayesian optimization to adjust policy
        hv_mc = []
        for n in range(n_mc):
            ## setup experiments for Bayesian optimization
            mobo_save_filepath = f'./trials/{date}/ax_client_snapshot{n}.json'

            # set a random seed/state for repeatability
            rs = int(42*(n+1234))

            if random_search:
                gs = GenerationStrategy(steps=[GenerationStep(Models.SOBOL, num_trials=-1)])
            else:
                if MOO:
                    bo_model = Models.MOO
                else:
                    bo_model = Models.GPEI

                gs = GenerationStrategy(
                    steps = [
                    # # 1. Initialization step (does not require pre-exiting data and
                    # # is well-suited for initial sampling of the search space)
                    # GenerationStep(
                    #     model = Models.SOBOL,
                    #     num_trials = 1,
                    #     max_parallelism = 5,
                    #     min_trials_observed = 1,
                    #     model_kwargs = {'seed': rs},
                    #     model_gen_kwargs = {},
                    # ),
                    # 2. Bayesian optimization step (requires data obtained from
                    # previous phase and learns from all data available at the time
                    # of each new candidate generation call)
                    GenerationStep(
                        model = bo_model,
                        num_trials = -1,
                    ),
                    ]
                )

            if MOO:
                objectives = {
                        'obj1': ObjectiveProperties(minimize=True, threshold=120.0),
                        'obj2': ObjectiveProperties(minimize=True, threshold=20.0),
                        }
            else:
                objectives = {'obj1': ObjectiveProperties(minimize=True)}

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
                                Nreps=Nreps_per_iteration,
                                init_file=initial_dnn_file,
                                save_prefix=f'trial{n}_iter{0}')
            ax_client.complete_trial(trial_index=0, raw_data=raw_data)
            ax_client.save_to_json_file(mobo_save_filepath)

            if MOO:
                hv_list = np.zeros((n_bo_iter,))

            for i in range(n_bo_iter):
                parameters, trial_index = ax_client.get_next_trial()
                raw_data = evaluate(parameters,
                                    Nreps=Nreps_per_iteration,
                                    init_file=initial_dnn_file,
                                    save_prefix=f'trial{n}_iter{i+1}')
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
                sim = Simulation(Nsim)
                sim.load_prob_info(prob_info2)

                sim_data2 = sim.run_closed_loop(sdnn, CEM=True)
                st2 = sim_data2['CEM_stop_time']
                CEMplot2 = np.ravel(sim_data2['CEMsim'][:,:st2])
                Tplot2 = sim_data2['Xsim'][0,:st2] + xssp[0]

                len_list = [st, st1, st2]
                max_idx = np.argmax(len_list)

                fig = plt.figure(figsize=(10,5))
                ax = fig.add_subplot(221)
                ax.axhline(Yrefplot[0], color='r', linestyle='--', label='Target Reference')
                ax.plot(np.arange(len(CEMplot))*ts, CEMplot, label='sMPC')
                ax.plot(np.arange(len(CEMplot1))*ts, CEMplot1, '--', label='DNN (before BO)')
                ax.plot(np.arange(len(CEMplot2))*ts, CEMplot2, ':', label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('CEM')
                ax.legend(loc='lower right')

                ax = fig.add_subplot(222)
                ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
                ax.plot(np.arange(len(Tplot))*ts, Tplot, label='sMPC')
                ax.plot(np.arange(len(Tplot1))*ts, Tplot1, '--', label='DNN (before BO)')
                ax.plot(np.arange(len(Tplot2))*ts, Tplot2, '--', label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Surface Temperature ($^\circ$C)')

                ax = fig.add_subplot(223)
                ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[0], label='sMPC')
                ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][0,:(st1-1)]+uss[0], label='DNN (before BO)')
                ax.step(np.arange(len_list[2]-1)*ts, sim_data2['Usim'][0,:(st2-1)]+uss[0], label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Power (W)')

                ax = fig.add_subplot(224)
                ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][1,:(st-1)]+uss[1], label='sMPC')
                ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][1,:(st1-1)]+uss[1], label='DNN (before BO)')
                ax.step(np.arange(len_list[2]-1)*ts, sim_data2['Usim'][1,:(st2-1)]+uss[1], label='DNN (after BO)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Flow Rate (SLM)')
                plt.draw()

                s = {}
                s['sim_data'] = sim_data
                s['sim_data1'] = sim_data1
                s['sim_data2'] = sim_data2
                np.save(f'./trials/{date}/trial{n}_profiles_compare.npy',s,allow_pickle=True)

        if MOO:
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

print('\n-------------------------------------------------')
print('Completed Simulations!')
print('-------------------------------------------------\n')
