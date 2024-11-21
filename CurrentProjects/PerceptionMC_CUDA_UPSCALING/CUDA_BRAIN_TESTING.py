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
        return stirling(math.ceil(n / 2)) + stirling(math.floor(n / 2)) - stirling(l)- stirling(n - l)


@cuda.jit(device=True)
def calc_H(hap,ham,hbp,hbm,hcp,hcm,hdp,hdm,kcoop,kcomp,kdu,kud,kx, A,B,C,D, la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m):
    H = (hap * (la_p-((RMax-A)/2)) + hbp * (lb_p - ((RMax-B)/2)) + hcp * (lc_p - (EMax - C) / 2) + hdp * (ld_p - (EMax - D) / 2) + ham * (la_m - A / 2) + hbm * (lb_m - B / 2) + hcm * (lc_m - C / 2) + hdm * (ld_m - D / 2) + kcoop * ((la_p - la_m) * A - (RMax * A / 2) + (lb_p - lb_m) * B - (RMax * B / 2)) + kcomp * ((la_m - la_p) * B - (RMax * B / 2) + (lb_m - lb_p) * A - (RMax * A / 2)) + kdu * ((la_p - la_m) * C - (C * RMax / 2) + (lb_p - lb_m) * D - (D * RMax / 2)) + kud * (A * (lc_m - lc_p) - (A * EMax / 2) + B * (ld_m - ld_p) - (B * EMax / 2)) + kx * ((lb_m - lb_p) * C - (C * RMax / 2) + (la_m - la_p) * D - (D * RMax / 2)))
    combs = (ln_comb(RMax-1-A,la_p) + ln_comb(RMax-1-B,lb_p) +  ln_comb(A,la_m) + ln_comb(B,lb_m) +ln_comb(EMax-1-C,lc_p) + ln_comb(EMax-1-D,ld_p)+ ln_comb(C,lc_m) + ln_comb(D,ld_m))
    H+= combs
    return H


### MAIN FUNCTION TO CALL ###
@cuda.jit
def calc_probability_next_state(arr, params, state):
    arr_index = cuda.grid(1)

    A = int(state[0])
    B = int(state[1])
    C = int(state[2])
    D = int(state[3])

    hap = params[0]
    ham = params[1]
    hbp = params[2]
    hbm = params[3]
    hcp = params[4]
    hcm = params[5]
    hdp = params[6]
    hdm = params[7]
    kcoop = params[8]
    kcomp = params[9]
    kdu = params[10]
    kud = params[11]
    kx = params[12]


    ### Next state index converting ### [A,B,C,D] -> [A_p,B_p,C_p,D_p]
    A_p:int = int(arr_index)/(RMax*EMax*EMax)
    B_p:int = int(arr_index)/(EMax*EMax)%RMax
    C_p:int = int(arr_index) / EMax % EMax
    D_p:int = int(arr_index)%EMax

    ### Probability Calculation ###
    loop_sum = 0
    for la_m in range(0,A+1): ## la minus, 0- A
        for lb_m in range(0,B+1): ##lb minus, 0- B
            for lc_m in range(0,M+1):
                for ld_m in range(0,M+1):
                    la_p:int = A_p - A - la_m
                    lb_p:int = B_p - B - lb_m
                    lc_p:int = C_p - C - lc_m
                    ld_p:int = D_p - D - ld_m
                    if la_p >= 0 and lb_p >= 0 and lc_p >= 0 and ld_p >= 0:
                        if la_p <= RMax-1-A and lb_p <= RMax-1-B and lc_p <= M and ld_p <= M:
                            loop_sum += math.exp(calc_H(hap,ham,hbp,hbm,hcp,hcm,hdp,hdm,kcoop,kcomp,kdu,kud,kx,A,B,C,D,la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m))

    arr[arr_index] = loop_sum



def get_prob_array(params, state):
    prob_next_state = np.zeros(RMax*RMax*EMax*EMax, dtype=np.float64)
    mem_array = cuda.to_device(prob_next_state)
    threads_per_block =128

    blocks_per_grid = (prob_next_state.size + threads_per_block - 1) // threads_per_block
    params_device = cuda.to_device(params)
    state_device = cuda.to_device(state)

    print(mem_array.dtype, mem_array.shape)
    print(params_device.dtype, params_device.shape)
    print(state_device.dtype, state_device.shape)

    calc_probability_next_state[blocks_per_grid, threads_per_block](
        mem_array, params_device, state_device
    )
    prob_next_state = mem_array.copy_to_host()
    prob_next_state /= np.sum(prob_next_state)
    prob_next_state = prob_next_state.reshape((RMax,RMax,EMax,EMax))

    return prob_next_state

print(cuda.current_context().get_memory_info())
free_mem = cuda.current_context().get_memory_info()[0]
print(f"Free Memory: {free_mem / (1024**2):.2f} MB")
print(free_mem//8 )
