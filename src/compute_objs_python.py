
import numpy as np

from KCutils.simulation import Simulation
from KCutils.neural_network import SimpleDNN
import KCutils.controller # this import is needed to ensure simulation class runs
from config.multistage import get_prob_info


ts = 0.5
Nsim = int(2*60/ts)

## problem options
population_K = 0.5 # assumed "population" value for K
patient_K = 0.55 # individiual patient value for K
population_Tmax = 45.0 # assumed "population" temperature constraint
patient_Tmax = 44.5 # individual patient temperature constraint

initial_dnn_file = f'./saved/2022_09_16_13h47m02s_initial_policy_info.npy'

# get problem information. Problem information is loaded from the
# KCutils.multistage file. This file provides problem-specific information, which
# may include system size and bounds, functions for evaluating the physical
# system, controller parameters, etc.
prob_info = get_prob_info(Kcem=population_K, Tmax=population_Tmax)

# a second problem info is created to establish a mismatch between the assumed
# dose parameter value (derived from population data) and the "true" dose
# parameter value of a particular "patient"
prob_info2 = get_prob_info(Kcem=patient_K, Tmax=patient_Tmax)


def get_objs(W1, b1, Wend, bend, n, Nreps=3, mod_save=1000):
    n = int(n)
    Nreps = int(Nreps)

    sim = Simulation(Nsim)
    sim.load_prob_info(prob_info2)

    c = SimpleDNN(initial_dnn_file)
    new_weights = [[np.squeeze(W1), np.squeeze(Wend)], [np.squeeze(b1), np.squeeze(bend)]]
    if n%mod_save == 0:
        c.modify_first_and_last_layers(new_weights, save_file=f'./sensitivity/dnns/sample{n}.npy')
    else:
        c.modify_first_and_last_layers(new_weights)

    # repeat the simulation Nreps
    obj1s = np.zeros((Nreps,))
    obj2s = np.zeros((Nreps,))
    d = {}
    for i in range(Nreps):
        sim_data = sim.run_closed_loop(c, CEM=True, rand_seed2=i*987)

        st = sim_data['CEM_stop_time']
        CEM = np.ravel(sim_data['CEMsim'][:,:st])
        Ref = np.ravel(sim_data['Yrefsim'][:,:st])
        T = np.ravel(sim_data['Ysim'][0,:st])
        Tmax = prob_info2['x_max'][0]

        CEMobj = np.sum((CEM-Ref)**2)
        Tobj = np.sum((T[T>Tmax] - Tmax)**2)
        s = {}
        s['sim_data'] = sim_data
        s['params'] = new_weights
        s['obj_val'] = {'CEMobj': CEMobj, 'Tobj': Tobj}
        d[f'rep{i}'] = s
        obj1s[i] = CEMobj
        obj2s[i] = Tobj

    # closed loop data is saved as a dictionary of dictionaries (keys are
    # 'rep{#}'). The nested dictionary contains the sim_data, the parameters
    # by BO, and the objective value(s).
    if n%mod_save == 0:
        np.save(f'./sensitivity/cl_data/sample{n}_sim_data.npy', d, allow_pickle=True)

    if Nreps > 1:
        # compute stats for multiple repetitions
        obj1 = np.mean(obj1s)
        obj1_ste = np.std(obj1s)/np.sqrt(Nreps)
        obj2 = np.mean(obj2s)
        obj2_ste = np.std(obj2s)/np.sqrt(Nreps)
    else:
        # otherwise return the value
        obj1 = np.ravel(obj1s)
        obj2 = np.ravel(obj2s)

    if obj2 <= 0:
        obj2 = 1e-10

    return obj1, obj2
# test
# print(get_objs(np.zeros((7,3)), np.zeros((7,1)), np.zeros((2,7)), np.zeros((2,1)),5))
