'''
script to plot the data gethered by running run_sim.py. This script will plot
the trajectories (sMPC (no mismatch), DNN before BO (with mismatch), DNN after
BO (with mismatch)) and the objective (convergence) values of BO. THIS SCRIPT
WORKS FOR SINGLE OBJECTIVE BAYESIAN OPTIMIZATION TRIALS ONLY

Requirements:
* Python 3
* Matplotlib [https://matplotlib.org] (for data visualization)

Copyright (c) 2021 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''




import numpy as np
import matplotlib.pyplot as plt
from config.multistage import get_prob_info


plot_individual_trial_trajectories = False
plot_individual_trial_objectives = True

Fontsize = 14 # default font size for plots
Lwidth = 3 # default line width for plots

# set some defaults for plots
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs
colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:pink','tab:red']
line_styles = ['-', '--', ':', '-.']

date = '2022_08_30_09h10m47s'
# date = '2022_08_30_11h20m00s'
# date = '2022_08_30_11h44m12s'
ntrials = 5

prob_info = get_prob_info()
prob_info2 = get_prob_info(Kcem=0.6)

ts = prob_info['ts']
xss = prob_info['xss']
uss = prob_info['uss']
xssp = prob_info['xssp']
ussp = prob_info['ussp']
x_max = prob_info['x_max']
u_min = prob_info['u_min']
u_max = prob_info['u_max']

if plot_individual_trial_trajectories:
    for n in range(ntrials):
        filename = f'./trials/{date}/trial{n}_profiles_compare.npy'

        s = np.load(filename, allow_pickle=True).item()
        sim_data = s['sim_data']
        sim_data1 = s['sim_data1']
        sim_data2 = s['sim_data2']

        sim_datas = [sim_data, sim_data1, sim_data2]
        stop_times = [sdata['CEM_stop_time'] for sdata in sim_datas]
        max_idx = np.argmax(stop_times)
        labels = ['sMPC (no mismatch)', 'DNN (before BO)', 'DNN(after BO)']
        CEMplots = [np.ravel(sdata['CEMsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]
        Tplots = [sdata['Ysim'][0,:st]+xssp[0] for sdata,st in zip(sim_datas,stop_times)]
        Pplots = [sdata['Usim'][0,:(st-1)]+uss[0] for sdata,st in zip(sim_datas,stop_times)]
        qplots = [sdata['Usim'][1,:(st-1)]+uss[1] for sdata,st in zip(sim_datas,stop_times)]
        Yrefs = [np.ravel(sdata['Yrefsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(221)
        ax.plot(np.arange(stop_times[max_idx])*ts, Yrefs[max_idx], color='r', linestyle='--', label='Target Reference')
        for CEMplot,label,ls in zip(CEMplots,labels,line_styles):
            ax.plot(np.arange(len(CEMplot))*ts, CEMplot, linestyle=ls, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CEM')
        ax.legend(loc='lower right')

        ax = fig.add_subplot(222)
        ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
        for Tplot,label,ls in zip(Tplots,labels,line_styles):
            ax.plot(np.arange(len(Tplot))*ts, Tplot, linestyle=ls, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surface Temperature ($^\circ$C)')

        ax = fig.add_subplot(223)
        ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
        ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
        for Pplot,label,ls in zip(Pplots,labels,line_styles):
            ax.step(np.arange(len(Pplot))*ts, Pplot, linestyle=ls, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')

        ax = fig.add_subplot(224)
        ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
        ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
        for qplot,label,ls in zip(qplots,labels,line_styles):
            ax.step(np.arange(len(qplot))*ts, qplot, linestyle=ls, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flow Rate (SLM)')
        plt.draw()

## plot Bayesian optimization results
import torch
from ax.storage.json_store.load import load_experiment
from ax.service.ax_client import AxClient
from ax.service.utils.report_utils import exp_to_df

obj_vals = []
for n in range(ntrials):
    filename = f'./trials/{date}/ax_client_snapshot{n}.json'
    ax_client = AxClient.load_from_json_file(filepath=filename)
    df = ax_client.get_trials_data_frame()
    outcomes = torch.tensor(df['obj1'].values)
    trial_numbers = df.trial_index.values

    (values,indicies) = torch.cummin(outcomes,dim=0)
    obj_vals.append(values)

    if plot_individual_trial_objectives:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(211)
        ax.plot(trial_numbers, outcomes, 'x', label='Observed')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.legend(loc='upper right')

        ax = fig.add_subplot(212)
        ax.plot(trial_numbers, values, '-', color=colors[0], label='Incumbent')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.legend(loc='upper right')
        plt.draw()
        plt.savefig(f'./saved/{date}-results/{date}_objective_trial{n}.png')

if not plot_individual_trial_objectives:
    all_objs = torch.stack(obj_vals)
    mean_objs = torch.mean(all_objs,dim=0)
    std_objs = torch.std(all_objs,dim=0)
    ste_objs = std_objs/torch.sqrt(torch.tensor(ntrials))

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(trial_numbers, mean_objs, 'b-', label='Incumbent')
    ax.fill_between(
                    trial_numbers,
                    mean_objs - ste_objs,
                    mean_objs + ste_objs,
                    alpha = 0.2,
                    color = colors[0]
    )
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    plt.draw()


plt.show()
