# neural network approximation of a controller
#
# Requirements:
# * Python 3
# * CasADi [https://web.casadi.org]
# * Tensorflow2 [https://www.tensorflow.org/install] OR
#   MATLAB Engine [https://www.mathworks.com/help/matlab/matlab-engine-for-python.html]
#
# Copyright (c) 2022 Mesbah Lab. All Rights Reserved.
# Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import sys
sys.dont_write_bytecode = True
import numpy as np
import casadi as cas
from numpy.random import default_rng
import torch
from torch import nn
import torch.optim as optim
import scipy.io as io
import os

import KCutils.pytorch_utils as ptu

# Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class DNN():
    """docstring for DNN."""

    def __init__(self):
        super(DNN, self).__init__()
        self.inputs = None
        self.outputs = None
        self.train_method = None

    def get_training_data_open_loop(self, c, Nsamp=300, mpc_type='multistage'):

        # extract relevant problem information
        nx = c.prob_info['nx']
        nu = c.prob_info['nu']
        nyc = c.prob_info['nyc']
        x_min = c.prob_info['x_min'].reshape(-1,1)
        x_max = c.prob_info['x_max'].reshape(-1,1)
        rand_seed = c.prob_info['rand_seed']
        if self.mpc_type == 'multistage' or self.mpc_type == 'economic':
            myref = c.prob_info['myref']
            cem_ref = myref(0)
            if self.mpc_type == 'multistage':
                Wset = c.prob_info['Wset']

        # create a random number generator (RNG); use the same seed for
        # consistent training data
        if rand_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(rand_seed)

        # generate a set of feasible inputs and outputs for training (open loop training data)
        input_data = np.empty((nx+nyc,Nsamp))
        output_data = np.empty((nu,Nsamp))
        for i in range(Nsamp):
            c.reset_initial_guesses()

            if mpc_type == 'nominal' or mpc_type == 'offsetfree':
                rand_x = rng.random(size=(nx,1)) * (x_max-x_min) + x_min
                rand_yref = rng.random(size=(nyc,1)) * (x_max[0,:]-x_min[0,:]) + x_min[0,:]
                c.set_parameters([rand_x, rand_yref])

                input_data[:,i] = np.ravel(np.concatenate((rand_x, rand_yref)))

            elif mpc_type == 'economic' or mpc_type == 'multistage':
                rand_x = rng.random(size=(nx,1)) * (x_max-x_min) + x_min
                rand_cem0 = rng.random(size=(nyc,1)) * cem_ref

                if mpc_type == 'multistage':
                    c.set_parameters([rand_x, cem_ref, rand_cem0, Wset])
                else:
                    c.set_parameters([rand_x, cem_ref, rand_cem0])

                input_data[:,i] = np.ravel(np.concatenate((rand_x, rand_cem0)))

            else:
                print('MPC type not supported!')

            res, _ = c.solve_mpc()
            output_data[:,i] = res['U']

        # scale data for training
        self.input_min = np.min(input_data, axis=1)
        self.input_max = np.max(input_data, axis=1)
        self.output_min = np.min(output_data, axis=1)
        self.output_max = np.max(output_data, axis=1)
        in_range = self.input_max - self.input_min
        out_range = self.output_max - self.output_min
        self.inputs = 2*(input_data - self.input_min[:,None])/(in_range[:,None]) - 1
        self.outputs = 2*(output_data - self.output_min[:,None])/(out_range[:,None]) - 1

        self.n_in = self.inputs.shape[0]
        self.n_out = self.outputs.shape[0]

        # # plot data to visualize it
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(input_data[0,:], 'k.', label='Ts0')
        # plt.plot(input_data[1,:], 'k.', label='I0')
        # plt.plot(input_data[2,:], 'b.', label='yref')
        # plt.legend()

        return self.inputs, self.outputs

    def get_training_data_closed_loop(self, c, Nsamp=300, mpc_type='nominal', Nsim=60):
        nx = c.prob_info['nx']
        nu = c.prob_info['nu']
        ny = c.prob_info['ny']
        nyc = c.prob_info['nyc']
        nw = c.prob_info['nw']
        nv = c.prob_info['nv']
        rand_seed = c.prob_info['rand_seed']

        if mpc_type == 'multistage' or mpc_type == 'economic':
            myref = c.prob_info['myref']
            if mpc_type == 'multistage':
                Wset = c.prob_info['Wset']

        x_min = c.prob_info['x_min'].reshape(-1,1)
        x_max = c.prob_info['x_max'].reshape(-1,1)
        w_min = c.prob_info['w_min'].reshape(-1,1)
        w_max = c.prob_info['w_max'].reshape(-1,1)

        fp = c.prob_info['fp']
        hp = c.prob_info['hp']

        # create a random number generator (RNG); use the same seed for
        # consistent training data
        if rand_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(rand_seed)

        Ncl = int(np.ceil(Nsamp/Nsim))

        CL_training_data = []
        for i in range(Ncl):
            # initialize container variables to store simulation data
            States = np.zeros((Nsim+1,nx))
            Outputs = np.zeros((Nsim+1,ny))
            Inputs = np.zeros((Nsim,nu))
            Disturbances = np.zeros((Nsim,nw))
            Objective = np.zeros((Nsim,1))
            Reference = np.zeros((Nsim,nyc))

            if mpc_type == 'economic' or mpc_type == 'multistage':
                CEMadd = c.prob_info['CEMadd']
                CEM = np.zeros((Nsim+1,1))

            # randomly initialize an initial state(s) and reference point
            Xcl = rng.random(size=(nx,1)) * (x_max-x_min) + x_min
            if mpc_type == 'nominal' or mpc_type == 'offsetfree':
                Yrefcl = rng.random(size=(nyc,1)) * (x_max[0,:]-x_min[0,:]) + x_min[0,:]
            elif mpc_type == 'economic' or mpc_type == 'multistage':
                CEM[0,:] = 0.0

            States[0,:] = np.ravel(Xcl)
            c.reset_initial_guesses()

            for k in range(Nsim):

                if mpc_type == 'nominal' or mpc_type == 'offsetfree':
                    Reference[k,:] = np.ravel(Yrefcl)
                    c.set_parameters([States[k,:], Yrefcl])
                elif mpc_type == 'economic':
                    Reference[k,:] = myref(k)
                    c.set_parameters([States[k,:], Reference[k,:], CEM[k,:]])
                elif mpc_type == 'multistage':
                    Reference[k,:] = myref(k)
                    c.set_parameters([States[k,:], Reference[k,:], CEM[k,:], Wset])

                res, f = c.solve_mpc()
                # simulate system and update state
                Inputs[k,:] = res['U']
                Objective[k,:] = res['J']
                Disturbances[k,:] = np.ravel(rng.random(size=(nw,1)) * (w_max-w_min) + w_min)
                States[k+1,:] = np.ravel(fp(States[k,:],Inputs[k,:],Disturbances[k,:]).full())
                Outputs[k+1,:] = np.ravel(hp(States[k,:],np.zeros((nv,1))))

                if mpc_type == 'economic' or mpc_type == 'multistage':
                    CEM[k+1,:] = CEM[k,:] + np.ravel(CEMadd(Outputs[k+1,:]).full())

            # store data in a dictionary
            data_dict = {}
            data_dict["States"] = States
            data_dict["Outputs"] = Outputs
            data_dict["Inputs"] = Inputs
            data_dict["Disturbances"] = Disturbances
            data_dict["Objective"] = Objective
            data_dict["Reference"] = Reference
            if mpc_type == 'economic' or mpc_type == 'multistage':
                data_dict["CEM"] = CEM

            # add data to list
            CL_training_data.append(data_dict)

        # gather all training data
        data_in = []
        data_out = []
        for data_dict in CL_training_data:

            if mpc_type == 'nominal' or mpc_type == 'offsetfree':
                data_in.append(np.vstack((data_dict["States"][:-1].T,data_dict["Reference"].T)))
            elif mpc_type == 'economic' or mpc_type == 'multistage':
                data_in.append(np.vstack((data_dict["States"][:-1].T,data_dict["CEM"][:-1].T)))

            data_out.append(data_dict["Inputs"].T)

        data_in = np.hstack(data_in)
        data_out = np.hstack(data_out)

        data_in = data_in[:, :Nsamp]
        data_out = data_out[:, :Nsamp]

        # scale data for training
        self.input_min = np.min(data_in, axis=1)
        self.input_max = np.max(data_in, axis=1)
        self.output_min = np.min(data_out, axis=1)
        self.output_max = np.max(data_out, axis=1)
        in_range = self.input_max - self.input_min
        out_range = self.output_max - self.output_min
        self.inputs = 2*(data_in - self.input_min[:,None])/(in_range[:,None]) - 1
        self.outputs = 2*(data_out - self.output_min[:,None])/(out_range[:,None]) - 1
        self.n_in = self.inputs.shape[0]
        self.n_out = self.outputs.shape[0]

        # # plot data to visualize it
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(data_in[0,:], 'r.', label='dnn_input1')
        # plt.plot(data_in[1,:], 'k.', label='dnn_input2')
        # plt.plot(data_in[2,:], 'b.', label='dnn_input3')
        # plt.legend()
        # plt.draw()

        return self.inputs, self.outputs

    def load_data(self,filename):
        split_filename = filename.split('.')
        extension = split_filename[-1]
        if extension == 'npy':
            s = np.load(filename, allow_pickle=True)
            s = s.item()
        elif extension == 'mat':
            s = io.loadmat(filename)
        else:
            print('Unsupported file extension. Please use either npy or mat')
            raise

        self.inputs = s['inputs']
        self.outputs = s['outputs']
        self.input_min = s['input_min']
        self.input_max = s['input_max']
        self.output_min = s['output_min']
        self.output_max = s['output_max']
        self.n_in = self.inputs.shape[0]
        self.n_out = self.outputs.shape[0]

    def build_neural_network_pytorch(self, H, L, activation='relu', inputs=None, outputs=None):

        if self.inputs is None:
            if inputs is None:
                print('No input data found in this object, and no input data was passed into this method. Please run the train_neural_network_keras() method to generate training data or pass in input data!')
                raise
            else:
                self.inputs = inputs
        if self.outputs is None:
            if outputs is None:
                print('No output data found in this object, and no output data was passed into this method. Please run the train_neural_network_keras() method to generate training data or pass in output data!')
                raise
            else:
                self.outputs = outputs
        self.H = H
        self.L = L
        self.activation = activation

        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        in_size = self.inputs.shape[0]
        out_size = self.outputs.shape[0]

        layers = []
        for _ in range(L):
            curr_layer = nn.Linear(in_size, H)
            layers.append(curr_layer)
            layers.append(activation)
            in_size = H

        last_layer = nn.Linear(in_size, out_size)
        layers.append(last_layer)
        layers.append(nn.Identity())

        self.model = nn.Sequential(*layers)
        print(self.model)

    def train_neural_network_pytorch(self,
                                     fit_params={'epochs':5000},
                                     optim_kwargs={},
                                     checkpoint_dir=None,
                                     split_test=False,
                                     save_file=None
                                     ):

        loss_fcn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), **optim_kwargs)

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                        os.path.join(checkpoint_dir, 'checkpoint')
                        )
            self.model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        data_inputs = ptu.from_numpy(self.inputs.T)
        data_targets = ptu.from_numpy(self.outputs.T)
        if split_test:
            Nsamp = self.inputs.shape[1]
            test_split_frac = 0.8
            test_split_idx = int(test_split_frac*Nsamp)
            train_inputs = data_inputs[:,:test_split_idx]
            train_targets = data_targets[:,:test_split_idx]
            train_inputs = data_inputs[:,test_split_idx:]
            train_targets = data_targets[:,test_split_idx:]
        else:
            train_inputs = data_inputs
            train_targets = data_targets

        running_loss = 0.0
        for epoch in range(fit_params['epochs']):

            optimizer.zero_grad()

            predictions = self.model(train_inputs)
            loss = loss_fcn(predictions, train_targets)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if (epoch+1)%100 == 0:
                print(f'Epoch: {epoch+1}     Loss: {running_loss/100:.3f}')
                running_loss = 0.0
        print('Finished Training')

        # get weights from trained model
        parameters = self.model.parameters()
        parameters = list(parameters)
        self.W = [ptu.to_numpy(parameters[i]) for i in range(0,len(parameters),2)]
        self.b = [ptu.to_numpy(parameters[i]) for i in range(1,len(parameters),2)]
        self.train_method = 'pytorch'

        if save_file:
            s = {}
            s['H'] = self.H
            s['L'] = self.L
            s['activation'] = self.activation
            s['W'] = self.W
            s['b'] = self.b
            s['inputs'] = self.inputs
            s['outputs'] = self.outputs
            s['input_min'] = self.input_min
            s['input_max'] = self.input_max
            s['output_min'] = self.output_min
            s['output_max'] = self.output_max
            s['train_method'] = self.train_method
            np.save(save_file, s, allow_pickle=True)

        return self.model

    # def train_neural_network_matlab(self, H, L, activation="poslin"):
    #     import matlab
    #     import matlab.engine # matlab engine import
    #     # start/connect to matlab engine
    #     # eng = matlab.start_matlab() # start new matlab engine
    #     # eng = matlab.connect_matlab() # connect to existing matlab engine
    #     future = matlab.engine.start_matlab(background=True)
    #     eng = future.result()

    #     data_in = matlab.double(self.inputs.tolist())
    #     data_out = matlab.double(self.outputs.tolist())
    #     Wm, bm = eng.trainDNN(data_in, data_out, self.H, self.L, activation, nargout=2)

    #     W = [np.asarray(a) for a in Wm]
    #     b = [np.asarray(a) for a in bm]
    #     # print(W)
    #     # print(b)
    #     self.W = W
    #     self.b = b
    #     self.train_method = 'matlab'

    # def hyperparameter_BO_matlab(self, H_max=None, L_max=None):
    #     import matlab
    #     import matlab.engine # matlab engine import
    #     # start/connect to matlab engine
    #     # eng = matlab.start_matlab() # start new matlab engine
    #     # eng = matlab.connect_matlab() # connect to existing matlab engine
    #     future = matlab.engine.start_matlab(background=True)
    #     eng = future.result()

    #     if H_max is None:
    #         H_max = self.H
    #     if L_max is None:
    #         L_max = self.L

    #     data_in = matlab.double(self.inputs.tolist())
    #     data_out = matlab.double(self.outputs.tolist())
    #     bestH, bestL = eng.hpBO(data_in, data_out, H_max, L_max, nargout=2)
    #     self.H = int(bestH)
    #     self.L = int(bestL)
    #     print('Optimal number of NODES: ', self.H)
    #     print('Optimal number of LAYERS: ', self.L)
    #     return self.H, self.L

    # def build_neural_network_keras(self, H, L, activation="relu", inputs=None, outputs=None):
    #     import tensorflow as tf
    #     from tensorflow import keras
    #     from tensorflow.keras import layers

    #     if self.inputs is None:
    #         if inputs is None:
    #             print('No input data found in this object, and no input data was passed into this method. Please run the train_neural_network_keras() method to generate training data or pass in input data!')
    #             raise
    #         else:
    #             self.inputs = inputs
    #     if self.outputs is None:
    #         if outputs is None:
    #             print('No output data found in this object, and no output data was passed into this method. Please run the train_neural_network_keras() method to generate training data or pass in output data!')
    #             raise
    #         else:
    #             self.outputs = outputs

    #     self.H = H
    #     self.L = L

    #     # build keras model of DNN
    #     model = keras.Sequential()
    #     # specify initializers for semi-consistent training/fitting
    #     model.add(layers.InputLayer(input_shape=self.n_in))

    #     # hidden layers
    #     for i in range(self.L):
    #         model.add(layers.Dense(self.H, activation=activation,
    #                                 name='hidden_layer'+str(i)))

    #     # output layer
    #     model.add(layers.Dense(self.n_out, activation='linear',
    #                             name='output_layer'))

    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    #     # compile model
    #     model.compile(
    #         optimizer = optimizer,
    #         loss = tf.keras.losses.MeanSquaredError(),
    #         metrics = ['accuracy'],
    #     )

    #     self.model = model
    #     return model

    # def train_neural_network_keras(self, fit_params={'epochs':5000, 'batch_size':32, 'validation_split':0.3}, verbose=1, cbs=True):
    #     import tensorflow as tf
    #     from tensorflow import keras
    #     from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    #     model = self.model

    #     if cbs:
    #         es = EarlyStopping(
    #                 monitor='val_loss',
    #                 mode='min',
    #                 verbose=verbose,
    #                 patience=5)

    #         tmp_save_file = 'saved/best_model.h5'
    #         mc = ModelCheckpoint(tmp_save_file, monitor='val_loss',
    #                              mode='min', verbose=verbose, save_best_only=True)

    #         # fit model
    #         history = model.fit(self.inputs.T, self.outputs.T,
    #                             **fit_params, callbacks=[es, mc], verbose=verbose)

    #         # # plot training history
    #         # import matplotlib.pyplot as plt
    #         # plt.figure()
    #         # plt.plot(history.history['accuracy'], label='train')
    #         # plt.plot(history.history['val_accuracy'], label='validation')
    #         # plt.legend()
    #         # plt.xlabel('Epoch')
    #         # plt.ylabel('Accuracy')
    #         # plt.show()

    #         model = keras.models.load_model('saved/best_model.h5')
    #         if verbose == 1:
    #             model.summary()
    #             model.evaluate(self.inputs.T, self.outputs.T)
    #     else:
    #         model.fit(self.inputs.T, self.outputs.T,
    #                     **fit_params, verbose=verbose)

    #     self.model = model

    #     # get weights from model
    #     weights = self.model.get_weights()
    #     self.W = weights[::2]
    #     self.b = weights[1::2]
    #     self.train_method = 'keras'
    #     return model

    def create_casadi_model(self):
        """
        method to create a casadi function representation of a trained neural
        network for faster and more portable evaluation of the neural network;
        currently assumes ReLU activation function only
        """
        n_in = self.inputs.shape[0]
        in_range = self.input_max - self.input_min
        out_range = self.output_max - self.output_min

        # "manually" evaluate the DNN using CasADi symbolics
        x = cas.SX.sym('x', n_in)
        xs = 2*(x - self.input_min)/(in_range) - 1

        if self.train_method == 'keras':
            z = cas.fmax(cas.mtimes(self.W[0].T,xs) + self.b[0],0)
            for i in range(1, self.L):
                z = cas.fmax(cas.mtimes(self.W[i].T,z) + self.b[i],0)
            z = cas.mtimes(self.W[-1].T,z) + self.b[-1]
        elif self.train_method == 'matlab' or self.train_method == 'pytorch':
            z = cas.fmax(cas.mtimes(self.W[0],xs) + self.b[0],0)
            for i in range(1, self.L):
                z = cas.fmax(cas.mtimes(self.W[i],z) + self.b[i],0)
            z = cas.mtimes(self.W[-1],z) + self.b[-1]
        else:
            print('No Neural Network was trained. Please run one of the training methods before creating the CasADi Function version.')
            raise

        us = (z+1)/2*out_range + self.output_min

        # create and save a CasADi function
        netca = cas.Function('netca', [x], [us])
        self.netca = netca
        return netca

    # def clear_model_keras(self):
    #     from tensorflow.keras import backend as K
    #     from tensorflow.python.framework import ops

    #     del self.model
    #     # clear keras session
    #     K.clear_session()
    #     ops.reset_default_graph()

    # def load_saved_model_keras(self, save_loc):
    #     from tensorflow import keras
    #     self.model = keras.models.load_model(save_loc)
    #     # get weights from model
    #     weights = self.model.get_weights()
    #     self.W = weights[::2]
    #     self.b = weights[1::2]
    #     self.L = len(self.W)-1
    #     self.H = self.W[1].shape[0]
    #     self.train_method = 'keras'
    #     self.create_casadi_model()

class SimpleDNN():
    """
    SimpleDNN - simple DNN model based in CasADi functions to train DNNs in
    black-box manner (e.g., using BO).
    """

    def __init__(self, filename):
        super(SimpleDNN, self).__init__()

        s = np.load(filename, allow_pickle=True)
        s = s.item()

        self.H = s['H']
        self.L = s['L']
        self.activation = s['activation']
        self.W = s['W']
        self.b = s['b']
        self.inputs = s['inputs']
        self.outputs = s['outputs']
        self.input_min = s['input_min']
        self.input_max = s['input_max']
        self.output_min = s['output_min']
        self.output_max = s['output_max']
        self.train_method = s['train_method']

        self.netca = None

    def create_casadi_model(self):
        """
        method to create a casadi function representation of a trained neural
        network for faster and more portable evaluation of the neural network;
        currently assumes ReLU activation function only
        """
        n_in = self.inputs.shape[0]
        in_range = self.input_max - self.input_min
        out_range = self.output_max - self.output_min

        # "manually" evaluate the DNN using CasADi symbolics
        x = cas.SX.sym('x', n_in)
        xs = 2*(x - self.input_min)/(in_range) - 1

        if self.train_method == 'keras':
            z = cas.fmax(cas.mtimes(self.W[0].T,xs) + self.b[0],0)
            for i in range(1, self.L):
                z = cas.fmax(cas.mtimes(self.W[i].T,z) + self.b[i],0)
            z = cas.mtimes(self.W[-1].T,z) + self.b[-1]
        elif self.train_method == 'matlab' or self.train_method == 'pytorch':
            z = cas.fmax(cas.mtimes(self.W[0],xs) + self.b[0],0)
            for i in range(1, self.L):
                z = cas.fmax(cas.mtimes(self.W[i],z) + self.b[i],0)
            z = cas.mtimes(self.W[-1],z) + self.b[-1]
        else:
            print('No Neural Network was trained. Please run one of the training methods before creating the CasADi Function version.')
            raise

        us = (z+1)/2*out_range + self.output_min

        # create and save a CasADi function
        netca = cas.Function('netca', [x], [us])
        self.netca = netca
        return netca


    def modify_lth_layer(self, l, new_weights, save_file=None):
        '''
        method to modify the l-th layer of a DNN. l should be a value between 0
        and L+1; a value of -1 for l modifies the last layer

        Input(s):
        --------
        l               the layer to be modified
        new_weights     the new weights to be used to modify the DNN, should be a
                        two-element list of the new weight matrix of the layer and
                        the new bias vector of the layer
        save_file       a filename if the DNN is desired to be saved

        Output(s):
        ----------
        netca           a CasADi function that may be used for a forward evaluation
                        of the DNN
        '''
        newW = new_weights[0]
        newb = new_weights[1]

        self.W[l] = newW
        self.b[l] = newb

        if save_file:
            self.save_dnn_info_to_file(save_file)

        return self.create_casadi_model()

    def modify_last_layer(self, new_weights, save_file=None):
        '''
        method to specifically modify the last layer

        see the modify_lth_layer function for more details
        '''
        return self.modify_lth_layer(-1, new_weights, save_file=save_file)
        # Wn = new_weights[0]
        # bn = new_weights[1]
        #
        # self.W[-1] = Wn
        # self.b[-1] = bn
        #
        # if save_file:
        #     self.save_dnn_info_to_file(save_file)
        #
        # return self.create_casadi_model()
    def modify_first_layer(self, new_weights, save_file=None):
        '''
        method to specifically modify the first layer

        see the modify_lth_layer function for more details
        '''
        return self.modify_lth_layer(0, new_weights, save_file=save_file)

    def modify_all_layers(self, new_weights, save_file=None):
        '''
        method to modify the all layers of a DNN.

        Input(s):
        --------
        new_weights     the new weights to be used to modify the DNN, should be a
                        two-element list of the new weight matrices and
                        the new bias vectors
        save_file       a filename if the DNN is desired to be saved

        Output(s):
        ----------
        netca           a CasADi function that may be used for a forward evaluation
                        of the DNN
        '''
        newW = new_weights[0]
        newb = new_weights[1]
        for i,Wi in enumerate(newW):
            self.W[i] = Wi
        for i,bi in enumerate(newb):
            self.b[i] = bi

        if save_file:
            self.save_dnn_info_to_file(save_file)

        return self.create_casadi_model()

    def modify_first_and_last_layers(self, new_weights, save_file=None):
        '''
        method to modify the first and last layers of the DNN together; this
        method is equivalent to using modify_first_layer and modify_last_layer
        together, but without the extra saving step

        see modify_lth_layer for more details
        '''
        newW = new_weights[0]
        newb = new_weights[1]
        self.W[0] = newW[0]
        self.W[-1] = newW[1]
        self.b[0] = newb[0]
        self.b[-1] = newb[1]

        if save_file:
            self.save_dnn_info_to_file(save_file)

        return self.create_casadi_model()

    def modify_nth_weight(self, n, weight_val, save_file=None):

        n_in = len(self.input_max)
        n_out = len(self.output_max)

        W_sizes = [W.size for W in self.W]
        W_cumsum = np.cumsum(W_sizes)

        b_sizes = [b.size for b in self.b]
        b_cumsum = np.cumsum(b_sizes)

        if n < n_in*self.H + (self.L-1)*self.H**2 + self.H*n_out:
            W_idx = 0
            for i in W_cumsum:
                if i > n:
                    W_idx += 1
                else:
                    break
            W_to_modify = self.W[W_idx]
            W_to_modify[n-(W_cumsum[W_idx-1])] = weight_val
            self.W[W_idx] = W_to_modify
        else:
            b_idx = 0
            for i in b_cumsum:
                if i > n:
                    b_idx += 1
                else:
                    break
            b_to_modify = self.b[b_idx]
            b_to_modify[n-(np.sum(W_cumsum)+b_cumsum[b_idx-1])] = weight_val
            self.b[b_idx] = b_to_modify

        if save_file:
            self.save_dnn_info_to_file(save_file)

        return self.create_casadi_model()

    def save_dnn_info_to_file(self, save_file):

        s = {}
        s['H'] = self.H
        s['L'] = self.L
        s['activation'] = self.activation
        s['W'] = self.W
        s['b'] = self.b
        s['inputs'] = self.inputs
        s['outputs'] = self.outputs
        s['input_min'] = self.input_min
        s['input_max'] = self.input_max
        s['output_min'] = self.output_min
        s['output_max'] = self.output_max
        s['train_method'] = self.train_method
        np.save(save_file, s, allow_pickle=True)
