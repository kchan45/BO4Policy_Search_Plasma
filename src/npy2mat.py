'''
extra script to convert numpy data file that stores DNN information (i.e., its
parameters) to a mat file

Requirements:
* Python 3

Copyright (c) 2021 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''

import numpy as np
import scipy.io as sio

# filename of the npy file
initial_dnn_file = './saved/2022_09_16_13h47m02s_initial_policy_info.npy'
new_filename = initial_dnn_file[8:-4]
print(f'New filename is: {new_filename}.mat')

saved_dnn_info = np.load(initial_dnn_file, allow_pickle=True).item()

L = saved_dnn_info['L']

layers_W = [(saved_dnn_info['W'][l]).flatten() for l in range(L+1)]
layers_b = [(saved_dnn_info['b'][l]).flatten() for l in range(L+1)]

saved_dnn_info['W_flattened'] = np.concatenate(layers_W)
saved_dnn_info['b_flattened'] = np.concatenate(layers_b)
saved_dnn_info['all_weights_W_then_b'] = np.concatenate((saved_dnn_info['W_flattened'],saved_dnn_info['b_flattened']))
alternate_Wb = [np.concatenate((W,b)) for W,b in zip(layers_W,layers_b)]
saved_dnn_info['all_weights_alt_Wb'] = np.concatenate(alternate_Wb)

sio.savemat(f'./saved/mat_files/{new_filename}.mat', saved_dnn_info)
