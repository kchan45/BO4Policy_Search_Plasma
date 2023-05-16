'''
script to plot the data gethered by running run_exp.py. This script will plot
a variety of figures for the data. In its unmodified state, it will plot Fig. 3
and Fig. 4 of the cited paper as well as the trajectories of the experimental
data stored in the published results. Modifications to the settings of this code
may be done to achieve different visualizations of the data.
THIS SCRIPT WORKS FOR EXPERIMENTAL TRIALS ONLY

Requirements:
* Python 3
* Matplotlib [https://matplotlib.org] (for data visualization)
* PyTorch [https://pytorch.org]
* Ax [https://ax.dev]

Copyright (c) 2023 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly J. Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''

import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib

import KCutils.plot_utils as pu
from config.multistage import get_prob_info_exp

plot_compare = False # plot a comparison of a smpc, dnn with no mismatch, dnn with mismatch
plot_bo = True
plot_mean = False
plot_individual_trial_pareto = False
plot_hypervolume = True
save_figs = False

model_file = '../model/APPJmodel_2022_09_22_17h28m06s.mat' # filename of identified model for plasma jet (to be used in controller)
ts = 0.5

Fontsize = 14 # default font size for plots
Lwidth = 2 # default line width for plots

lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

# root directory for all experimental data in this project
root = './ExperimentalData/'
Nrep = 1

## problem options: ENSURE THESE ARE THE SAME AS THOSE USED DURING EXPERIMENTS
population_K = 0.5 # assumed "population" value for K
patient_K = 0.55 # individiual patient value for K
population_Tmax = 45.0 # assumed "population" temperature constraint
patient_Tmax = 44.5 # individual patient temperature constraint

# get problem information. Problem information is loaded from the
# KCutils.multistage file. This file provides problem-specific information, which
# may include system size and bounds, functions for evaluating the physical
# system, controller parameters, etc.
prob_info = get_prob_info_exp(Kcem=population_K, Tmax=population_Tmax, model_file=model_file, ts=ts)
# a second problem info is created to establish a mismatch between the assumed
# dose parameter value (derived from population data) and the "true" dose
# parameter value of a particular "patient"
prob_info2 = get_prob_info_exp(Kcem=patient_K, Tmax=patient_Tmax, model_file=model_file, ts=ts)

xss = prob_info2['xss']
uss = prob_info2['uss']
xssp = prob_info2['xssp']
ussp = prob_info2['ussp']
x_max = prob_info2['x_max']
u_min = prob_info2['u_min']
u_max = prob_info2['u_max']


if plot_compare:
    fig_objs = {}
    # plot dnn run (with mismatch)
    folder = '2022_09_23_11h33m59s'
    for exp_num in range(Nrep):
        filename = root+folder+f'/Backup/Experiment_{exp_num}.npy'

        fig_objs = pu.plot_data_from_file(filename, prob_info2, CEM=True, fig_objs=fig_objs, label='dnn (with mismatch)')

    # plot smpc run
    folder = '2022_09_22_21h33m53s'
    for exp_num in range(9):
        filename = root+folder+f'/Backup/Experiment_{exp_num+2}.npy'

        fig_objs = pu.plot_data_from_file(filename, prob_info, CEM=True, fig_objs=fig_objs, label='smpc')

    # plot dnn run (no mismatch)
    folder = '2022_09_23_11h19m30s'
    for exp_num in range(Nrep):
        filename = root+folder+f'/Backup/Experiment_{exp_num}.npy'

        fig_objs = pu.plot_data_from_file(filename, prob_info, CEM=True, fig_objs=fig_objs, label='dnn (no mismatch)')

    plt.show()

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

m1 = Metric(name='obj1')
m2 = Metric(name='obj2')
obj1 = Objective(metric=m1, minimize=True)
obj2 = Objective(metric=m2, minimize=True)
obj1_th = ObjectiveThreshold(metric=m1, bound=120.0, relative=False, op=ComparisonOp.LEQ)
obj2_th = ObjectiveThreshold(metric=m2, bound=8.0, relative=False, op=ComparisonOp.LEQ)

mo = MultiObjective(objectives=[obj1,obj2])

opt_config = MultiObjectiveOptimizationConfig(objective=mo, objective_thresholds=[obj1_th, obj2_th])



if plot_bo:
    folder = '2022_09_23_13h58m20s'

    import os
    fig_save_folder = f'./saved/{folder}-results/'
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder, exist_ok=True)

    n_mc = 1
    n_iter = 15+1
    iters_to_plot = list(range(0,n_iter,1))
    labels = [f'Iteration {iter}' for iter in iters_to_plot]
    Nreps_per_iteration = 3
    Nsim = int(2*60/ts)
    n_ste = 2
    line_styles = ['-', '--', ':', '-.']

    for n in range(n_mc):
        fig_objs = pu.plot_core(prob_info2, CEM=True)
        fig = fig_objs['fig']
        (ax11, ax12, ax21, ax22) = fig_objs['axes']

        count = 0
        for i,label in zip(iters_to_plot,labels):

            stack_CEM = np.zeros((Nreps_per_iteration,Nsim))
            stack_CEM[:] = np.nan
            stack_T = np.zeros((Nreps_per_iteration,Nsim))
            stack_T[:] = np.nan
            stack_P = np.zeros((Nreps_per_iteration,Nsim))
            stack_P[:] = np.nan
            stack_q = np.zeros((Nreps_per_iteration,Nsim))
            stack_q[:] = np.nan
            stack_st = []

            for rep in range(Nreps_per_iteration):
                datafile = root+folder+f'/trial{n}_iter{i}/Backup/Experiment_{rep}.npy'
                exp_data = np.load(datafile, allow_pickle=True).item()

                stack_st.append(exp_data['CEM_stop_time'])
                stack_CEM[rep,:stack_st[-1]] = np.ravel(exp_data['CEMsim'])[:stack_st[-1]]
                stack_T[rep,:stack_st[-1]] = np.ravel(exp_data['Tsave'])[:stack_st[-1]]
                stack_P[rep,:(stack_st[-1]-1)] = np.ravel(exp_data['Psave'])[:(stack_st[-1]-1)]
                stack_q[rep,:(stack_st[-1]-1)] = np.ravel(exp_data['qSave'])[:(stack_st[-1]-1)]

            if plot_mean:
                CEMmean = np.nanmean(stack_CEM, axis=0); CEMstd = np.nanstd(stack_CEM, axis=0)
            else:
                CEMmin = stack_CEM[np.argmin(stack_st),:]; CEMmax = stack_CEM[np.argmax(stack_st),:]
                CEMmed = np.nanmedian(stack_CEM, axis=0)
            Tmean = np.nanmean(stack_T, axis=0); Tstd = np.nanstd(stack_T, axis=0)
            Pmean = np.nanmean(stack_P, axis=0); Pstd = np.nanstd(stack_P, axis=0)
            qmean = np.nanmean(stack_q, axis=0); qstd = np.nanstd(stack_q, axis=0)

            if plot_mean:
                ax11.plot(np.arange(len(CEMmean))*ts, CEMmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
                ax11.fill_between(np.arange(len(CEMmean))*ts,
                                  CEMmean-n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                                  CEMmean+n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                                  alpha=0.2,
                                  )
            else:
                ax11.plot(np.arange(len(CEMmed))*ts, CEMmed, linestyle=line_styles[count%4], label=label, alpha=0.8)
                ax11.fill_between(np.arange(len(CEMmed))*ts,
                                  CEMmin,
                                  CEMmax,
                                  alpha=0.2,
                                  )
            ax12.plot(np.arange(len(Tmean))*ts, Tmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
            ax12.fill_between(np.arange(len(Tmean))*ts,
                                Tmean-n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                                Tmean+n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                                alpha=0.2,
                             )
            ax21.step(np.arange(len(Pmean))*ts, Pmean, linestyle=line_styles[count%4], label=label, alpha=0.8)

            ax22.step(np.arange(len(qmean))*ts, qmean, linestyle=line_styles[count%4], label=label, alpha=0.8)

            count += 1

        ax11.legend(loc='lower right', fontsize='small')
        plt.tight_layout()
        plt.draw()
        if save_figs:
            fig.savefig(f'./saved/{folder}-results/{folder}_observed_treatments_up2iter{i}.png')

    if not plot_individual_trial_pareto:
        fig = plt.figure(figsize=(8,8),dpi=300)
        ax = fig.add_subplot(111)

    hv_all = []
    date = folder
    for n in range(n_mc):
        filename = f'./trials/{date}/exp_ax_client_snapshot{n}.json'
        ax_client = AxClient.load_from_json_file(filepath=filename)
        df = ax_client.get_trials_data_frame()
        outcomes = torch.tensor(df[['obj1', 'obj2']].values)
        trial_numbers = df.trial_index.values
        print(trial_numbers)
        print(df['obj2'].values)

        ref_point = torch.tensor([-100.0,-20.0]) # should be set to same values as objective thresholds

        if plot_individual_trial_pareto:
            fig = plt.figure(figsize=(8,8),dpi=300)
            ax = fig.add_subplot(111)
            ax.set_xlabel('Dose Delivery Measure')
            ax.set_ylabel('Temperature Constraint Measure')

        sc = ax.scatter(outcomes[:,0], outcomes[:,1], alpha=0.9, c='tab:blue')

        if plot_individual_trial_pareto:
            fig.canvas.draw_idle()
            if save_figs:
                fig.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_trial{n}.png')
        elif n == n_mc-1:
            ax.set_xlabel('Dose Delivery Measure')
            ax.set_ylabel('Temperature Constraint Measure')
            # ax.set_xlim([50,100])
            fig.canvas.draw_idle()
            if save_figs:
                fig.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_all.png')

        hv_trace = ax_client.get_trace(experiment=ax_client.experiment)#, optimization_config=opt_config)
        hv_all.append(hv_trace)

        pareto_params = ax_client.get_pareto_optimal_parameters(
                                        optimization_config=opt_config,
                                        use_model_predictions=False,
                                        )
        pareto_trials = list(pareto_params.keys())
        pareto_trials.sort()
        print(pareto_trials)
        print(outcomes[pareto_trials])

        fig2 = plt.figure(figsize=(12,6),dpi=300,layout='tight')
        ax11 = fig2.add_subplot(221)
        ax12 = fig2.add_subplot(222)
        ax21 = fig2.add_subplot(223)
        ax22 = fig2.add_subplot(224)
        fig5 = plt.figure(figsize=(3,2),dpi=300,layout='tight')
        ax5 = fig5.add_subplot(111)

        ax11.axhline(1.5, color='k', linestyle='--', label='Target Reference')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('CEM (min)')

        ax12.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
        ax12.text(0.99, 0.985, 'Constraint', color='r', fontsize='small',
                horizontalalignment='right', verticalalignment='top',
                transform=ax12.transAxes)
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Surface Temperature ($^\circ$C)')

        bbox_dict = dict(facecolor='w', edgecolor='w', boxstyle="Square,pad=0.0")
        ax21.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
        # ax21.text(0.99, 0.99, 'Maximum', color='r', fontsize='small',
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax21.transAxes, bbox=bbox_dict)
        ax21.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
        # ax21.text(0.01, 0.01, 'Minimum', color='r', fontsize='small',
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax21.transAxes, bbox=bbox_dict)
        ax21.set_xlabel('Time (s)')
        ax21.set_ylabel('Power (W)')

        ax22.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
        # ax22.text(0.01, 0.99, 'Maximum', color='r', fontsize='small',
        #         horizontalalignment='left', verticalalignment='top',
        #         transform=ax22.transAxes, bbox=bbox_dict)
        ax22.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
        # ax22.text(0.99, 0.01, 'Minimum', color='r', fontsize='small',
        #         horizontalalignment='right', verticalalignment='bottom',
        #         transform=ax22.transAxes, bbox=bbox_dict)
        ax22.set_xlabel('Time (s)')
        ax22.set_ylabel('Flow Rate (SLM)')

        count = 0
        prev_best_idx = 0
        for i in iters_to_plot:
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
            if i == 0 or best_idx != prev_best_idx:
                plot_switch = True
            else:
                plot_switch = False

            if plot_switch:
                filename = f'./trials/{date}/cl_data/trial{n}_iter{best_idx}_exp_data.npy'
                s = np.load(filename, allow_pickle=True).item()

                exp_datas = [s[f'rep{r}']['exp_data'] for r in range(Nreps_per_iteration)]
                stop_times = [edata['CEM_stop_time'] for edata in exp_datas]
                max_idx = np.argmax(stop_times)
                label = f'Iteration {i}'
                print(stop_times)

                CEMplots = [np.ravel(sdata['CEMsim'][:,:st]) for sdata,st in zip(exp_datas,stop_times)]
                Tplots = [sdata['Tsave'][:st] for sdata,st in zip(exp_datas,stop_times)]
                Pplots = [sdata['Psave'][:(st-1)] for sdata,st in zip(exp_datas,stop_times)]
                qplots = [sdata['qSave'][:(st-1)] for sdata,st in zip(exp_datas,stop_times)]

                CEMplot = np.zeros((stop_times[max_idx],Nreps_per_iteration)); CEMplot[:] = 1.5
                Tplot = np.zeros((stop_times[max_idx],Nreps_per_iteration)); Tplot[:] = np.nan
                Pplot = np.zeros((stop_times[max_idx]-1,Nreps_per_iteration)); Pplot[:] = np.nan
                qplot = np.zeros((stop_times[max_idx]-1,Nreps_per_iteration)); qplot[:] = np.nan
                for r,st in zip(range(Nreps_per_iteration),stop_times):
                    CEMplot[:st,r] = CEMplots[r]
                    Tplot[:st,r] = Tplots[r]
                    Pplot[:(st-1),r] = Pplots[r]
                    qplot[:(st-1),r] = qplots[r]

                if plot_mean:
                    CEMmean = np.nanmean(CEMplot, axis=1); CEMstd = np.nanstd(CEMplot, axis=1)
                else:
                    # print(stop_times)
                    # CEMmin = CEMplot[:,np.argmin(stop_times)]; CEMmax = CEMplot[:,np.argmax(stop_times)]
                    med_st = np.median(stop_times)
                    CEMmed = np.squeeze(CEMplot[:,np.argwhere(stop_times==med_st)])
                    CEMmed = np.nanmedian(CEMplot,axis=1)
                    CEMmin = np.nanmin(CEMplot, axis=1); CEMmax = np.nanmax(CEMplot,axis=1)
                Tmean = np.nanmean(Tplot, axis=1); Tstd = np.nanstd(Tplot, axis=1)
                Pmean = np.nanmean(Pplot, axis=1); Pstd = np.nanstd(Pplot, axis=1)
                qmean = np.nanmean(qplot, axis=1); qstd = np.nanstd(qplot, axis=1)
                assert CEMplot.shape[0] == stop_times[max_idx]

                if plot_mean:
                    ax11.plot(np.arange(len(CEMmean))*ts, CEMmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
                    ax11.fill_between(np.arange(len(CEMmean))*ts,
                                        CEMmean-n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                                        CEMmean+n_ste*CEMstd/np.sqrt(Nreps_per_iteration),
                                        alpha=0.2,
                                     )
                else:
                    ax11.plot(np.arange(len(CEMmed))*ts, CEMmed, linestyle=line_styles[count%4], label=label, alpha=0.8)
                    ax11.fill_between(np.arange(len(CEMmed))*ts,
                                      CEMmax,
                                      CEMmin,
                                      alpha=0.2,
                                      )
                ax12.plot(np.arange(len(Tmean))*ts, Tmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
                ax12.fill_between(np.arange(len(Tmean))*ts,
                                    Tmean-n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                                    Tmean+n_ste*Tstd/np.sqrt(Nreps_per_iteration),
                                    alpha=0.2,
                                 )
                time_vec = np.arange(len(Tmean))*ts
                plot_range = range(40,61)
                ax5.plot(time_vec[plot_range], Tmean[plot_range], linestyle=line_styles[count%4], label=label, alpha=0.8)
                ax5.fill_between(time_vec[plot_range],
                                    Tmean[plot_range]-n_ste*Tstd[plot_range]/np.sqrt(Nreps_per_iteration),
                                    Tmean[plot_range]+n_ste*Tstd[plot_range]/np.sqrt(Nreps_per_iteration),
                                    alpha=0.2,
                                 )
                ax21.step(np.arange(len(Pmean))*ts, Pmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
                ax22.step(np.arange(len(qmean))*ts, qmean, linestyle=line_styles[count%4], label=label, alpha=0.8)
                Yrefs = [np.ravel(sdata['Yrefsim'][:,:st]) for sdata,st in zip(exp_datas,stop_times)]
                plot_switch = False
                count += 1
            prev_best_idx = best_idx

        ax11.legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        fig2.canvas.draw_idle()
        if save_figs:
            fig2.savefig(f'./saved/{date}-results/{date}_trajectories_stats_nste{n_ste}_trial{n}.png')

        ax5.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
        fig5.canvas.draw_idle()
        if save_figs:
            fig5.savefig(f'./saved/{date}-results/{date}_temp_zoom_stats_nste{n_ste}_trial{n}.png')

        fig3 = plt.figure(figsize=(5,4),dpi=300)
        ax2 = fig3.add_subplot(111)
        q = len(trial_numbers[:i])
        cm = matplotlib.colormaps.get_cmap('Purples').resampled(q)
        ax2.scatter(outcomes[:i,0], outcomes[:i,1], alpha=0.9, c=trial_numbers[:i], cmap=cm, edgecolors=cm(q))
        norm = plt.Normalize(trial_numbers[:i].min(), trial_numbers[:i].max())
        sm = ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
        cbar = fig3.colorbar(sm)
        cbar.ax.set_title('Iteration')
        ax2.set_xlabel('Dose Delivery Measure')
        ax2.set_ylabel('Temperature Constraint Measure')
        # ax2.set_xlim([40,120])
        plt.tight_layout()
        fig3.canvas.draw_idle()
        if save_figs:
            fig3.savefig(f'./saved/{date}-results/{date}_observed_pareto_front_trial{n}_up2iter{i}.png')

    if plot_hypervolume:
        hvs = np.vstack(hv_all)
        mean_hv = np.nanmean(hvs, axis=0).reshape(-1,1)
        std_hv = np.nanstd(hvs, axis=0).reshape(-1,1)
        ste_hv = std_hv/np.sqrt(n_mc)

        fig4 = plt.figure(figsize=(4,4),dpi=300)
        ax = fig4.add_subplot(111)
        ax.plot(np.arange(n_iter), mean_hv)
        ax.fill_between(
                        np.arange(n_iter),
                        np.ravel(mean_hv-ste_hv),
                        np.ravel(mean_hv+ste_hv),
                        alpha=0.2,
        )
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        fig4.canvas.draw_idle()
        plt.tight_layout()
        if save_figs:
            fig4.savefig(f'./saved/{date}-results/{date}_hypervolume.png')


    plt.show()
