import pyopencl as cl
import numpy as np
import os
import global_params as gp
import scipy.io as sio


def display_mat_keys(file_path, key):
    mat_contents = sio.loadmat(file_path)
    print(mat_contents.keys())
    if key in mat_contents:
        print(f"Value for key '{key}': {mat_contents[key]}")
    else:
        print(f"Key '{key}' not found in the .mat file")

if __name__ == '__main__':
    directory = 'Joch_data_given'
    file_name = 'TwoChoiceTrajectoriesDensity_000.mat'  # Replace with your actual .mat file name
    file_path = os.path.join(directory, file_name)
    display_mat_keys(file_path,'r_step')