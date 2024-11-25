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
    n = float(n)
    if n == 0:
        return 0
    else:
        return n*math.log(n) - n + 0.5*math.log(2*math.pi*n)

@cuda.jit(device=True)
def ln_comb(n,l):
    a = int(n)
    b = int(l)
    n = float(n)
    l = float(l)
    if a == 0:
        return 0
    elif a == b or b == 0:
        #return stirling(n) - stirling(l) - stirling(n-l)
        return stirling(math.ceil(n/2)) + stirling(math.floor(n/2)) - stirling(n)
    else:
        #return stirling(n) - stirling(l) - stirling(n-l)
        return stirling(math.ceil(n / 2)) + stirling(math.floor(n / 2)) - stirling(l) - stirling(n - l)


@cuda.jit(device=True)
def calc_H(params, state, action):
    hap,ham,hbp,hbm,hcp,hcm,hdp,hdm, kcoop,kcomp,kdu,kud,kx = float(params[0]),float(params[1]),float(params[2]),float(params[3]),float(params[4]),float(params[5]),float(params[6]),float(params[7]),float(params[8]),float(params[9]),float(params[10]),float(params[11]),float(params[12])
    #hap,ham,hbp,hbm,hcp,hcm,hdp,hdm, kcomp,kcoop,kdu,kud,kx = float(params[0]),float(params[1]),float(params[2]),float(params[3]),float(params[4]),float(params[5]),float(params[6]),float(params[7]),float(params[8]),float(params[9]),float(params[10]),float(params[11]),float(params[12])

    A,B,C,D = float(state[0]),float(state[1]),float(state[2]),float(state[3])
    (la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m) = action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7]
    (la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m) = float(la_p), float(lb_p), float(lc_p), float(ld_p), float(la_m), float(lb_m), float(lc_m), float(ld_m)

    # la_p, lb_p, lc_p, ld_p = float(la_p), float(lb_p), float(lc_p), float(ld_p)
    # la_m, lb_m, lc_m, ld_m = float(la_m), float(lb_m), float(lc_m), float(ld_m)

    H = (hap*la_p + hbp*lb_p + hcp*lc_p + hdp*ld_p + ham*la_m + hbm*lb_m + hcm*lc_m + hdm*ld_m + kcoop*((la_p-la_m)*A + B*(lb_p - lb_m))+ kcomp*((la_m-la_p)*B + A*(lb_m - lb_p)) + kdu*((la_p-la_m)*C+ D*(lb_p-lb_m)) + kud*((lc_m-lc_p)*A +B*(ld_m- ld_p)) + kx*(lb_m-lb_p)*C + kx*(la_m-la_p)*D)

    #H = (hap * (la_p-((RMax-A)/2)) + hbp * (lb_p - ((RMax-B)/2)) + hcp*(lc_p-(EMax-C) / 2) + hdp*(ld_p-(EMax-D)/2) + ham * (la_m - A / 2) + hbm*(lb_m- B/2) + hcm*(lc_m- C/2) + hdm*(ld_m - D/2) + kcoop*((la_p-la_m)*A - (RMax*A/2) + (lb_p -lb_m)*B- (RMax*B/2)) + kcomp*((la_m-la_p)*B -(RMax*B/2) + (lb_m - lb_p) * A - (RMax * A / 2)) + kdu * ((la_p - la_m) * C - (C * RMax / 2) + (lb_p - lb_m) * D - (D * RMax / 2)) + kud * (A * (lc_m - lc_p) - (A * EMax / 2) + B * (ld_m - ld_p) - (B * EMax / 2)) + kx * ((lb_m - lb_p) * C - (C * RMax / 2) + (la_m - la_p) * D - (D * RMax / 2)))
    #H = (hap * (la_p-((RMax-A)/2)) + hbp * (lb_p - ((RMax-B)/2)) + hcp*(lc_p-(EMax-C) / 2) + hdp*(ld_p-(EMax-D)/2) + ham * (la_m - A / 2) + hbm*(lb_m- B/2) + hcm*(lc_m- C/2) + hdm*(ld_m - D/2) + kcoop*((la_p-la_m)*A - (RMax*A/2) + (lb_p -lb_m)*B- (RMax*B/2)) + kcomp*((la_m-la_p)*B -(RMax*B/2) + (lb_m - lb_p) * A - (RMax*  A / 2)) + kdu * ((la_p - la_m)*  C - (C*RMax    /2) +(lb_p -  lb_m)*  D - (D*RMax    /2)) + kud*  (A*  (lc_m - lc_p) - (A * EMax / 2) + B*  (ld_m - ld_p) - (B*EMax    /2))+  kx * ((lb_m - lb_p) * C - (C*RMax /   2) + (la_m - la_p) * D - (D * RMax / 2)))
    combs = (ln_comb(RMax-1-A,la_p) + ln_comb(RMax-1-B,lb_p) +  ln_comb(A,la_m) + ln_comb(B,lb_m) +ln_comb(EMax-1-C,lc_p) + ln_comb(EMax-1-D,ld_p)+ ln_comb(C,lc_m) + ln_comb(D,ld_m))
    H+= combs
    # if math.fabs(H) > 307:
    #     print(H, la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m)
    return H


### MAIN FUNCTION TO CALL ###
@cuda.jit
def calc_probability_next_state(arr, params:tuple[float], state:tuple[float]):
    arr_index = cuda.grid(1)
    A,B,C,D = int(state[0]),int(state[1]),int(state[2]),int(state[3])
    arr_index = int(arr_index)
    ### Next state index converting ### [A,B,C,D] -> [A_p,B_p,C_p,D_p]
    A_p:int = int(arr_index) // (RMax*EMax*EMax)
    B_p:int = int(arr_index) // (EMax*EMax)%RMax
    C_p:int = int(arr_index) // EMax % EMax
    D_p:int = int(arr_index) % EMax
    #print(arr_index, A_p,B_p,C_p,D_p)

    ### Probability Calculation ###
    loop_sum = 0
    e_max = 700

    for la_m in range(0,A+1): ## la minus, 0- A
        for lb_m in range(0,B+1): ##lb minus, 0- B
            for lc_m in range(0,M+1):
                for ld_m in range(0,M+1):
                    la_p:int = A_p - A + la_m
                    lb_p:int = B_p - B + lb_m
                    lc_p:int = C_p - C + lc_m
                    ld_p:int = D_p - D + ld_m
                    action = (la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m)
                    if la_p >= 0 and lb_p >= 0 and lc_p >= 0 and ld_p >= 0:
                        #if la_p <= RMax-1-A and lb_p <= RMax-1-B and lc_p <= M and ld_p <= M:
                        if la_p <= M and lb_p <= M and lc_p <= M and ld_p <= M:
                            #print(A_p,B_p,C_p,D_p,':',la_p,lb_p,lc_p,ld_p,la_m,lb_m,lc_m,ld_m)
                            e = calc_H(params,state,action) - e_max
                            loop_sum += math.exp(e)

    arr[arr_index] = loop_sum



def get_prob_array(params, state):
    prob_next_state = np.zeros(RMax*RMax*EMax*EMax, dtype=np.float64)
    mem_array = cuda.to_device(prob_next_state)
    threads_per_block =256

    blocks_per_grid = (prob_next_state.size + threads_per_block - 1) // threads_per_block
    params_device = cuda.to_device(np.array(params, dtype=np.float64))
    #params_device = cuda.to_device(params)
    state_device = cuda.to_device(np.array(state, dtype=np.int32))

    calc_probability_next_state[blocks_per_grid, threads_per_block](mem_array, params_device, state_device)
    prob_next_state = mem_array.copy_to_host()

    total_sum = np.sum(prob_next_state)
    if total_sum == 0:
        raise ValueError("Sum of probabilities is zero; check calc_H and loops.")
    prob_next_state /= total_sum

    prob_next_state = prob_next_state.reshape((RMax,RMax,EMax,EMax))

    return prob_next_state

#get_prob_array((1,1,1,1,1,1,1,1,1,1,1,1,1),(1,1,1,1))