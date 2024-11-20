import numpy as np
import math
import pandas as pd
import random
import os
from mcperception_cuda_functions import calc_probability_next_state,get_prob_array
import global_params as gp
from matplotlib import pyplot as plt

RMax = gp.MAXTOP
EMax = gp.MAXBOT
T_total = gp.TMax


def get_Pij(params):
    Pij = np.zeros((RMax,RMax,EMax,EMax,RMax,RMax,EMax,EMax))
    for A in range(0,RMax):
        for B in range(0,RMax):
            for C in range(0,EMax):
                for D in range(0,EMax):
                    state = (A,B,C,D)
                    P_next = get_prob_array(params, state)

                    Pij[A,B,C,D,:,:,:,:] = P_next

    return Pij

def nextState(pij,state):
    A,B,C,D = state
    prob_next_state = pij[A,B,C,D,:,:,:,:]
    randnum = random.random()
    pshape = prob_next_state.shape
    flattened_parr = prob_next_state.reshape(-1)
    cumsum_parr = np.cumsum(flattened_parr)
    index = np.searchsorted(cumsum_parr,randnum)
    if index < len(cumsum_parr):
        a_next,b_next,c_next,d_next = np.unravel_index(index, pshape)
        nextState = (a_next,b_next,c_next,d_next)
    return nextState

def sim_forward(start_state, pij, T_total):
    a0,b0,c0,d0 = start_state
    A,B,C,D, = [a0],[b0],[c0],[d0]
    t =0
    while t<=T_total:
        a_next,b_next,c_next,d_next = nextState(pij, (A[-1],B[-1],C[-1],D[-1]))
        A.append(a_next)
        B.append(b_next)
        C.append(c_next)
        D.append(d_next)
        t+=1
        print(t)

    return A,B,C,D

if __name__ == '__main__':
    epsilon1,epsilon2 = 0,0
    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = (-8.39780022, -8.31815575, -6.24283186, -0.62797361, 4.64786633, 2.1348466, 6.06874194, 0.29438665, 1.62159095)

    params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud, kx)
    #
    start = (0,0,0,0)
    Pij = get_Pij(params)

    A,B,C,D = sim_forward(start, Pij, T_total)

    ts= np.linspace(0,T_total,len(A))
    plt.plot(ts,A,linewidth=1,c='b')
    plt.plot(ts,B,linewidth=1,c='r')
    plt.xlabel("T",fontsize=15)
    plt.ylabel("# Activated",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-.1,4.5)
    plt.show()
    # # #
    plt.figure()
    plt.plot(ts, C, linewidth=1, c='b')
    plt.plot(ts, D, linewidth=1, c='r')
    plt.xlabel("T" , fontsize=15)
    plt.ylabel("# Activated", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()