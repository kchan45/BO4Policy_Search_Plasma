'''
script to plot the data gethered by running run_sim.py. This script will plot
the trajectories (sMPC (no mismatch), DNN before BO (with mismatch), DNN after
BO (with mismatch)) and the objective (convergence) values of BO. THIS SCRIPT
WORKS FOR MULTI-OBJECTIVE BAYESIAN OPTIMIZATION TRIALS ONLY

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
from matplotlib.cm import ScalarMappable
from config.multistage import get_prob_info

plot_individual_trial_trajectories = False
plot_individual_trial_pareto = False
plot_hypervolume = True

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

# date = '2022_09_06_11h21m50s'
# date = '2022_09_09_12h23m08s'
# date = '2022_09_09_13h23m27s'
# date = '2022_09_15_16h31m29s'
# date = '2022_09_16_08h34m16s'
date = '2022_09_16_13h47m02s'
# date = '2022_09_16_18h12m18s'

n_mc = 5
n_cl_runs = 50+1
Nreps_per_iteration = 3
n_ste = 2

# CHANGE TO VALUES USED IN SIMULATION/EXPERIMENT
# prob_info = get_prob_info(Kcem=0.5,Tmax=45) # population
prob_info2 = get_prob_info(Kcem=0.55,Tmax=44.5) # patient

ts = prob_info2['ts']
xss = prob_info2['xss']
uss = prob_info2['uss']
xssp = prob_info2['xssp']
ussp = prob_info2['ussp']
x_max = prob_info2['x_max']
u_min = prob_info2['u_min']
u_max = prob_info2['u_max']

if plot_individual_trial_trajectories:
    for n in range(n_mc):
        fig = plt.figure(figsize=(10,5))
        ax11 = fig.add_subplot(221)
        ax12 = fig.add_subplot(222)
        ax21 = fig.add_subplot(223)
        ax22 = fig.add_subplot(224)

        for i in range(0,n_cl_runs,10):
            filename = f'./trials/{date}/cl_data/trial{n}_iter{i}_sim_data.npy'

            s = np.load(filename, allow_pickle=True).item()
            sim_data = s['rep0']
            sim_data = sim_data['sim_data']

            sim_datas = [sim_data]
            stop_times = [sdata['CEM_stop_time'] for sdata in sim_datas]
            max_idx = np.argmax(stop_times)
            labels = [f'Iteration {i}']
            CEMplots = [np.ravel(sdata['CEMsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]
            Tplots = [sdata['Ysim'][0,:st]+xssp[0] for sdata,st in zip(sim_datas,stop_times)]
            Pplots = [sdata['Usim'][0,:(st-1)]+uss[0] for sdata,st in zip(sim_datas,stop_times)]
            qplots = [sdata['Usim'][1,:(st-1)]+uss[1] for sdata,st in zip(sim_datas,stop_times)]
            Yrefs = [np.ravel(sdata['Yrefsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]

            for CEMplot,label,ls in zip(CEMplots,labels,line_styles):
                ax11.plot(np.arange(len(CEMplot))*ts, CEMplot, linestyle=ls, label=label, alpha=0.2)

            for Tplot,label,ls in zip(Tplots,labels,line_styles):
                ax12.plot(np.arange(len(Tplot))*ts, Tplot, linestyle=ls, label=label, alpha=0.2)

            for Pplot,label,ls in zip(Pplots,labels,line_styles):
                ax21.step(np.arange(len(Pplot))*ts, Pplot, linestyle=ls, label=label, alpha=0.2)

            for qplot,label,ls in zip(qplots,labels,line_styles):
                ax22.step(np.arange(len(qplot))*ts, qplot, linestyle=ls, label=label, alpha=0.2)

        ax11.axhline(Yrefs[max_idx][0], color='r', linestyle='--', label='Target Reference')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('CEM')
        ax11.legend(loc='lower right', fontsize='small')

        ax12.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Surface Temperature ($^\circ$C)')

        ax21.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
        ax21.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
        ax21.set_xlabel('Time (s)')
        ax21.set_ylabel('Power (W)')

        ax22.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
        ax22.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
        ax22.set_xlabel('Time (s)')
        ax22.set_ylabel('Flow Rate (SLM)')

        plt.draw()

## plot Bayesian optimization results
import torch
from ax.storage.json_store.load import load_experiment
from ax.service.ax_client import AxClient
from ax.service.utils.report_utils import exp_to_df
from KCutils.simulation import Simulation
from KCutils.neural_network import SimpleDNN

from ax import Metric
from ax import Objective
from ax import ObjectiveThreshold
from ax import ComparisonOp
from ax import MultiObjective
from ax import MultiObjectiveOptimizationConfig


if not plot_individual_trial_pareto:
    fig = plt.figure(figsize=(4,3.5),layout='tight',dpi=300)
    ax = fig.add_subplot(111)

hv_all = []
for n in range(n_mc):
    filename = f'./trials/{date}/ax_client_snapshot{n}.json'
    ax_client = AxClient.load_from_json_file(filepath=filename)
    df = ax_client.get_trials_data_frame()
    outcomes = torch.tensor(df[['obj1', 'obj2']].values)
    trial_numbers = df.trial_index.values

    ref_point = torch.tensor([-100.0,-20.0]) # should be set to same values as objective thresholds

    if plot_individual_trial_pareto:
        fig = plt.figure(figsize=(4,3.5),layout='tight',dpi=300)
        ax = fig.add_subplot(111)
        ax.set_xlabel('Dose Delivery Measure')
        ax.set_ylabel('Temperature Constraint Measure')

    sc = ax.scatter(outcomes[:,0], outcomes[:,1], alpha=0.9, c='tab:blue')

    if plot_individual_trial_pareto:
        fig.canvas.draw_idle()
        fig.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_trial{n}.png')
    elif n == n_mc-1:
        ax.set_xlabel('Dose Delivery Measure')
        ax.set_ylabel('Temperature Constraint Measure')
        ax.set_ylim([-3,25])
        ax.set_xlim([40,120])
        fig.canvas.draw_idle()
        ax.set_title('(b)')
        fig.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_all.png')
        print('saved figure')

    hv_trace = ax_client.get_trace(experiment=ax_client.experiment)
    hv_all.append(hv_trace)

    m1 = Metric(name='obj1')
    m2 = Metric(name='obj2')
    obj1 = Objective(metric=m1, minimize=True)
    obj2 = Objective(metric=m2, minimize=True)
    obj1_th = ObjectiveThreshold(metric=m1, bound=80.0, relative=False, op=ComparisonOp.LEQ)
    obj2_th = ObjectiveThreshold(metric=m2, bound=5.0, relative=False, op=ComparisonOp.LEQ)

    mo = MultiObjective(objectives=[obj1,obj2])

    opt_config = MultiObjectiveOptimizationConfig(objective=mo, objective_thresholds=[obj1_th, obj2_th])

    pareto_params = ax_client.get_pareto_optimal_parameters(
                                    # optimization_config=opt_config,
                                    use_model_predictions=False,
                                    )
    pareto_trials = list(pareto_params.keys())
    pareto_trials.sort()
    print(pareto_trials)

    fig2 = plt.figure(figsize=(12,6),layout='tight')
    ax11 = fig2.add_subplot(221)
    ax12 = fig2.add_subplot(222)
    ax21 = fig2.add_subplot(223)
    ax22 = fig2.add_subplot(224)

    count = 0
    for i in range(0,10,2):
        sub_pareto_trials = np.asarray(pareto_trials)[np.asarray(pareto_trials)<=i]
        print(sub_pareto_trials)

        if len(sub_pareto_trials) > 0:
            objs1 = []
            objs2 = []
            for j in sub_pareto_trials:
                params, stats = pareto_params[j]
                means, covs = stats
                objs1.append(means['obj1'])
                objs2.append(means['obj2'])

            objs1 = np.asarray(objs1)
            objs2 = np.asarray(objs2)
            min_idxs = np.argmin(objs2)
            red_obj1 = objs1[min_idxs]
            min_val = np.amin(red_obj1)
            best_idx = int(np.asarray(sub_pareto_trials)[objs1==min_val])

            (optimal_params, optimal_stats) = pareto_params[best_idx]
            # print(optimal_params)
            optimal_objs, optimal_covs = optimal_stats
            print(optimal_objs)

        else:
            best_idx = 0

        print(best_idx)

        filename = f'./trials/{date}/cl_data/trial{n}_iter{best_idx}_sim_data.npy'
        s = np.load(filename, allow_pickle=True).item()

        sim_datas = [s[f'rep{r}']['sim_data'] for r in range(Nreps_per_iteration)]
        stop_times = [sdata['CEM_stop_time'] for sdata in sim_datas]
        max_idx = np.argmax(stop_times)
        label = f'Iteration {i}'

        CEMplots = [np.ravel(sdata['CEMsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]
        Tplots = [sdata['Ysim'][0,:st]+xssp[0] for sdata,st in zip(sim_datas,stop_times)]
        Pplots = [sdata['Usim'][0,:(st-1)]+uss[0] for sdata,st in zip(sim_datas,stop_times)]
        qplots = [sdata['Usim'][1,:(st-1)]+uss[1] for sdata,st in zip(sim_datas,stop_times)]

        CEMplot = np.zeros((stop_times[max_idx],Nreps_per_iteration)); CEMplot[:] = np.nan
        Tplot = np.zeros((stop_times[max_idx],Nreps_per_iteration)); Tplot[:] = np.nan
        Pplot = np.zeros((stop_times[max_idx]-1,Nreps_per_iteration)); Pplot[:] = np.nan
        qplot = np.zeros((stop_times[max_idx]-1,Nreps_per_iteration)); qplot[:] = np.nan
        for r,st in zip(range(Nreps_per_iteration),stop_times):
            CEMplot[:st,r] = CEMplots[r]
            Tplot[:st,r] = Tplots[r]
            Pplot[:(st-1),r] = Pplots[r]
            qplot[:(st-1),r] = qplots[r]

        CEMmean = np.nanmean(CEMplot, axis=1); CEMstd = np.nanstd(CEMplot, axis=1)
        Tmean = np.nanmean(Tplot, axis=1); Tstd = np.nanstd(Tplot, axis=1)
        Pmean = np.nanmean(Pplot, axis=1); Pstd = np.nanstd(Pplot, axis=1)
        qmean = np.nanmean(qplot, axis=1); qstd = np.nanstd(qplot, axis=1)
        assert CEMplot.shape[0] == stop_times[max_idx]

        ax11.plot(np.arange(len(CEMmean))*ts, CEMmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
        ax11.fill_between(np.arange(len(CEMmean))*ts,
                            CEMmean-n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                            CEMmean+n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                            alpha=0.2,
                         )
        ax12.plot(np.arange(len(Tmean))*ts, Tmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
        ax12.fill_between(np.arange(len(Tmean))*ts,
                            Tmean-n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                            Tmean+n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                            alpha=0.2,
                         )
        ax21.step(np.arange(len(Pmean))*ts, Pmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
        # ax21.fill_between(np.arange(len(Pmean))*ts,
        #                     Pmean-n_ste*Pstd/np.sqrt(Nreps_per_iteration),
        #                     Pmean+n_ste*Pstd/np.sqrt(Nreps_per_iteration),
        #                     alpha=0.2,
        #                  )
        ax22.step(np.arange(len(qmean))*ts, qmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
        # ax22.fill_between(np.arange(len(qmean))*ts,
        #                     qmean-n_ste*qstd/np.sqrt(Nreps_per_iteration),
        #                     qmean+n_ste*qstd/np.sqrt(Nreps_per_iteration),
        #                     alpha=0.2,
        #                  )
        Yrefs = [np.ravel(sdata['Yrefsim'][:,:st]) for sdata,st in zip(sim_datas,stop_times)]
        count += 1

    ax11.axhline(Yrefs[max_idx][0], color='r', linestyle='--', label='Target Reference')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('CEM')
    ax11.legend(loc='lower right', fontsize='small')

    ax12.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Surface Temperature ($^\circ$C)')

    ax21.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
    ax21.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
    ax21.set_xlabel('Time (s)')
    ax21.set_ylabel('Power (W)')

    ax22.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
    ax22.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
    ax22.set_xlabel('Time (s)')
    ax22.set_ylabel('Flow Rate (SLM)')

    fig2.canvas.draw_idle()
    fig2.savefig(f'./saved/{date}-results/{date}_trajectories_stats_nste{n_ste}_trial{n}.png')

    fig3 = plt.figure(figsize=(5,4),layout='tight')
    ax2 = fig3.add_subplot(111)
    cm = plt.cm.get_cmap('cool')
    ax2.scatter(outcomes[:i,0], outcomes[:i,1], alpha=0.9, c=trial_numbers[:i], cmap=cm)
    norm = plt.Normalize(trial_numbers[:i].min(), trial_numbers[:i].max())
    sm = ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    cbar = fig3.colorbar(sm)
    cbar.ax.set_title('Iteration')
    ax2.set_xlabel('Dose Delivery Measure')
    ax2.set_ylabel('Temperature Constraint Measure')
    # ax2.set_xlim([40,120])
    fig3.canvas.draw_idle()
    fig3.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_trial{n}_up2iter{i}.png')

if plot_hypervolume:
    hvs = np.vstack(hv_all)
    mean_hv = np.nanmean(hvs, axis=0).reshape(-1,1)
    std_hv = np.nanstd(hvs, axis=0).reshape(-1,1)
    ste_hv = std_hv/np.sqrt(n_mc)

    fig4 = plt.figure(figsize=(4,3.5),layout='tight',dpi=300)
    ax = fig4.add_subplot(111)
    ax.plot(np.arange(n_cl_runs), mean_hv)
    ax.fill_between(
                    np.arange(n_cl_runs),
                    np.ravel(mean_hv-n_ste*ste_hv),
                    np.ravel(mean_hv+n_ste*ste_hv),
                    alpha=0.2,
    )
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title('(a)')
    fig4.canvas.draw_idle()
    fig4.savefig(f'./saved/{date}-results/{date}_hypervolume.png')

plt.show()
