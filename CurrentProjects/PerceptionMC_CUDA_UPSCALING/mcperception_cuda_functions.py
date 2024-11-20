from numba import cuda
import numpy as np
import math
import global_params as gp
from numba import config

config.DEBUG = False


RMax = gp.MAXTOP
EMax = gp.MAXBOT
M = gp.M_step

@cuda.jit(device=True)
def stirling(n):
    if n == 0:
        return 0
    else:
        return n*math.log(n) - n + 0.5*math.log(2*math.pi*n)

@cuda.jit(device=True)
def ln_comb(n,l):
    n = float(n)
    l = float(l)
    if n == 0:
        return 0
    elif n == l or l == 0:
        return stirling(math.ceil(n/2)) + stirling(math.floor(n/2)) - stirling(n)
    else:
        return (stirling(math.ceil(n/2)) + stirling(math.floor(n/2)) - stirling(l)- stirling(n-l))


@cuda.jit(device=True)
def calc_H(params, state, action):
    hap,ham, hbp,hbm, hcp,hcm, hdp,hdm,kcoop,kcomp,kdu,kud,kx = params
    A,B,C,D = state
    (la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m) = action
    la_p, lb_p, lc_p, ld_p = int(la_p), int(lb_p), int(lc_p), int(ld_p)
    la_m, lb_m, lc_m, ld_m = int(la_m), int(lb_m), int(lc_m), int(ld_m)
    hap, ham, hbp, hbm, hcp, hcm, hdp, hdm = map(float, (hap, ham, hbp, hbm, hcp, hcm, hdp, hdm))
    kcoop, kcomp, kdu, kud, kx = map(float, (kcoop, kcomp, kdu, kud, kx))


    H = (hap * (la_p-((RMax-A)/2)) + hbp* (lb_p - ((RMax-B)/2)) + hcp*(lc_p-((EMax-C))/2) + hdp*(ld_p - (EMax-D)/2) + ham * (la_m-A/2) + hbm*(lb_m-B/2) + hcm*(lc_m-C/2)+hdm*(ld_m-D/2) + kcoop*((la_p-la_m)*A - (RMax*A/2) + (lb_p-lb_m)*B- (RMax*B/2)) + kcomp*((la_m-la_p)*B -(RMax*B/2) + (lb_m-lb_p)*A - (RMax*A/2)) + kdu*((la_p-la_m)*C - (C*RMax/2) +(lb_p-lb_m)*D -(D*RMax/2)) + kud*(A*(lc_m-lc_p)-(A*EMax/2)+B*(ld_m-ld_p)-(B*EMax/2))+kx*((lb_m-lb_p)*C-(C*RMax/2)+(la_m-la_p)*D-(D*RMax/2)))
    combs = (ln_comb(RMax-1-A,la_p) + ln_comb(RMax-1-B,lb_p) +  ln_comb(A,la_m) + ln_comb(B,lb_m) +ln_comb(EMax-1-C,lc_p) + ln_comb(EMax-1-D,ld_p)+ ln_comb(C,lc_m) + ln_comb(D,ld_m))
    H+= combs
    return H


### MAIN FUNCTION TO CALL ###
@cuda.jit
def calc_probability_next_state(arr, params, state):
    arr_index = cuda.grid(1)
    A,B,C,D = state

    ### Next state index converting ### [A,B,C,D] -> [A_p,B_p,C_p,D_p]
    A_p = arr_index/(RMax*EMax*EMax)
    B_p = arr_index/(EMax*EMax)%RMax
    C_p = arr_index/(EMax)%EMax
    D_p = arr_index%EMax

    ### Probability Calculation ###
    loop_sum = 0
    for la_m in range(0,A+1): ## la minus, 0- A
        for lb_m in range(0,B+1): ##lb minus, 0- B
            for lc_m in range(0,M+1):
                for ld_m in range(0,M+1):
                    la_p = A_p - A - la_m
                    lb_p = B_p - B - lb_m
                    lc_p = C_p - C - lc_m
                    ld_p = D_p - D - ld_m
                    action = (la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m)
                    if la_p >= 0 and lb_p >= 0 and lc_p >= 0 and ld_p >= 0:
                        if la_p <= RMax-1-A and lb_p <= RMax-1-B and lc_p <= M and ld_p <= M:
                            loop_sum += math.exp(calc_H(params,state,action))

    arr[arr_index] = loop_sum
    return


def get_prob_array(params, state):
    prob_next_state = np.zeros(RMax*RMax*EMax*EMax, dtype=np.float64)
    mem_array = cuda.to_device(prob_next_state)
    threads_per_block =128

    blocks_per_grid = (prob_next_state.size + threads_per_block - 1) // threads_per_block
    params_device = cuda.to_device(np.array(params, dtype=np.float64))
    state_device = cuda.to_device(np.array(state, dtype=np.float64))

    print(mem_array.dtype, mem_array.shape)
    print(params_device.dtype, params_device.shape)
    print(state_device.dtype, state_device.shape)


    calc_probability_next_state[blocks_per_grid, threads_per_block](mem_array, params_device, state_device)
    prob_next_state = mem_array.copy_to_host()
    prob_next_state /= np.sum(prob_next_state)
    prob_next_state = prob_next_state.reshape((RMax,RMax,EMax,EMax))

    return prob_next_state
