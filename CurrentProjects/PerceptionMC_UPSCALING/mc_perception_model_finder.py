from scipy.optimize import minimize
from mc_perception_frwd_cl import calc_next_state
import mc_perception_main as gp
import os
import csv
import numpy as np
import scipy.io as sio
import time


rmax = gp.MAXTOP
emax = gp.MAXBOT

def count_transitions(data):
    count = {}
    for i in range(len(data[0]) - 1):

        indices = (
        data[0][i], data[1][i], data[2][i], data[3][i], data[0][i + 1], data[1][i + 1], data[2][i + 1], data[3][i + 1])
        if indices not in count:
            count[indices] = 0
        count[indices] += 1
    return count

def cl_likelyhood(params7, count, prog_path):

    L = 0

    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = params7
    params = (halpha, ha, halpha, ha, hgamma, hc, hgamma, hc, kcoop, kcomp, kdu, kud, kx)

    p_arr_cache = {}

    for idx,val in count.items():

        state = tuple(idx[:4])
        next_state = tuple(idx[4:])
        state_key = f"{state}"

        if state_key in p_arr_cache:
            p_arr = p_arr_cache[state_key]
        else:
            p_arr = calc_next_state(params, state, prog_path)
            try:
                p_arr /= np.sum(p_arr)
            except ZeroDivisionError:
                continue
            p_arr = p_arr.reshape((rmax, rmax, emax, emax))
            p_arr_cache[state_key] = p_arr

        p_val = p_arr[next_state]

        if p_val != 0 and p_val != np.nan:
            L += val * np.log(p_val)

    print('Likelyhood: ', -1 * L /np.sum(list(count.values())))
    return -L / np.sum(list(count.values()))

def maximize_likelyhood(count,initial,prog_path):

    maxL = minimize(cl_likelyhood, initial, args=(count,prog_path), method='Nelder-Mead')
    return maxL.x
def load_mat_data(file_path):
    mat_contents = sio.loadmat(file_path)
    a = mat_contents['r_li'][0]
    b = mat_contents['r_li'][1]
    c = mat_contents['e_li'][0]
    d = mat_contents['e_li'][1]
    return a, b, c, d

def save_inferred_model_csv(I_test,  params):


    params_names = ['hgamma', 'hc', 'halpha', 'ha', 'kcoop', 'kcomp', 'kdu', 'kud', 'kx']
    dir_path = 'Infered_parameters'
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f'{I_test}_inferred_params.csv')

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['I',params_names])
        writer.writerow([I_test, list(params)])


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    perception_cl_prog  = os.path.join(current_dir, 'mc_perception_opencl.cl')

    directory = 'Joch_data_given'
    I_test = '100'

    counts_file = os.path.join(directory, f'counts_{I_test}.npy')
    if os.path.exists(counts_file):
        count = np.load(counts_file, allow_pickle=True).item()
        print('LOADED COUNTS FROM FILE')
    else:
        joch_a, joch_b, joch_c, joch_d = load_mat_data(f'Joch_data_given/TwoChoiceTrajectoriesDensity_{I_test}.mat')
        count = count_transitions((joch_a, joch_b, joch_c, joch_d))
        np.save(counts_file, count)
        print('FOUND COUNTS AND SAVED')
    #print(count)

    t0 = time.time()
    #initial_guess= (-8.96733557, -7.73231853, -6.01935508 ,-0.99322105,  4.7228139 ,  1.98114397 ,6.05944224 , 0.29747507 , 1.53067954)#old
    #initial_guess=(-11.45954105,- 9.81027345, - 10.15358925,- 1.49456199,  0.93641602, 1.79710763, 2.86152824, 0.11585655, 0.56313622)#000   .013
    #initial_guess= (-11.2481983, - 10.05704405,- 8.5134479, - 3.30299464 ,  0.79097099,  1.89522803, 2.70952583 , 0.11745314 , 0.6743023)#025#.01295
    #initial_guess= (-11.03212318, - 10.28098307, - 8.55471608 ,- 3.29921075,  0.80347528, 1.85855315,  2.67696223,  0.11808324,  0.70043714)# 050 .0132
    initial_guess = ( -10.57359064 ,- 10.74478163, - 8.46817699, - 3.50334181,   0.81771474,  1.87695072,   2.6787412, 0.12152318,    0.71633869)#100 .01913

    #Save initial guess in a csv file with the names of parameters, and I_test, and data length.


    save_inferred_model_csv(I_test, initial_guess)
    #max_params = maximize_likelyhood(count, initial_guess,perception_cl_prog)
    #print(max_params,'time:',time.time()-t0)


