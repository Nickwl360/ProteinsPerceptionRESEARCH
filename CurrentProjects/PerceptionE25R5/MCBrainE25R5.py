import numpy as np
import matplotlib.pyplot as plt
import math
#from MCBrain2layer import *
from scipy.optimize import minimize
import pyopencl as cl
import scipy.io
from collections import defaultdict

# directory = 'JochNE25NR5Counts'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# constants
MAXTOP = 6
MAXBOT = 26
PendStateE25R5File = '/Users/Nick/PycharmProjects/Researchcode (1) (1)/CurrentProjects/PerceptionE25R5/MCBrainTransitionGPU.cl'

#############JOCHEN DATA  TRAJS############
set = 0  #######DT = .001
jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE25NR5.mat')
Rkij = jochdatafix['R_kij']
Ekij = jochdatafix['E_kij']
lencount= 100_000_000
dataa=Rkij[0,:lencount,set]/(MAXTOP-1)
datab=Rkij[1,:lencount,set]/(MAXTOP-1)
datac=Ekij[0,:lencount,set]/(MAXBOT-1)
datad=Ekij[1,:lencount,set]/(MAXBOT-1)

def countbrainSparced(dataa, datab, datac, datad):
    # Initialize a dictionary to store the counts of transitions
    count = defaultdict(int)
    dataa = np.asarray(dataa)
    datab = np.asarray(datab)
    datac = np.asarray(datac)
    datad = np.asarray(datad)

    for i in range(lencount - 1):
        print(i)
        # Create a tuple representing the current and next state
        current_state = (int(dataa[i]), int(datab[i]), int(datac[i]), int(datad[i]))
        next_state = (int(dataa[i + 1]), int(datab[i + 1]), int(datac[i + 1]), int(datad[i + 1]))

        # Create a tuple representing the transition
        transition = (current_state, next_state)

        # Increment the count of this transition
        count[transition] += 1

    return count


def CalcPEndState(params, ns, prog_path):
    na,nb,nc,nd = ns
    halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    with open(prog_path, "r") as f:
        program_source = f.read()
    print('read')
    program = cl.Program(ctx, program_source).build()
    print('built')
    NextPmnop = np.zeros(((MAXTOP*MAXTOP*MAXBOT*MAXBOT)),dtype=np.float64)
    Pmnop_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, NextPmnop.nbytes)
    global_size = ((MAXTOP*MAXTOP*MAXBOT*MAXBOT),)
    calc = program.compute_Pmnop
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pmnop_buf, na,nb,nc,nd,halpha, hbeta, hgamma, hdelta,ha,hb,hc,hd, kcomp, kcoop,kdu,kud,kx)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, NextPmnop, Pmnop_buf)
    return NextPmnop

def calcPTransitiongpu(params,start,end):
    na1, nb1, nc1, nd1 = start
    na2, nb2, nc2, nd2 = end
    #calculate probability based on ^
    Praw = CalcPEndState(params,start,PendStateE25R5File)
    Preshape= Praw.reshape((MAXTOP,MAXTOP,MAXBOT,MAXBOT))
    PendstateNormal= Preshape/(np.sum(Preshape))
    #THIS IS WHERE YOU CAN SAVE AND CHECK IF ALREADY MADE#####################################
    ptrans = PendstateNormal[na2,nb2,nc2,nd2]
    #save P(start)
    #find prob i->j from P(start)
    return ptrans

def get_non_zero_indices_and_values(arr, index=()):
    indices_values = []
    if isinstance(arr, np.ndarray):
        non_zero_indices = np.transpose(np.nonzero(arr))
        for idx in non_zero_indices:
            indices_values.append((index + tuple(idx), arr[tuple(idx)]))
    else:
        for i, sub_arr in enumerate(arr):
            indices_values.extend(get_non_zero_indices_and_values(sub_arr, index + (i,)))
    return indices_values

def brainlikelyhood(params9, countsdict):
    hgamma,hc,halpha,ha,kcoop,kcomp,kdu,kud,kx = params9
    epsilon2=0
    params = (halpha, ha, halpha, ha , hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)

    likelyhood = 0
    for transition, count in countsdict.items():
        start,end = transition
        prob= calcPTransitiongpu(params,start,end)
        likelyhood += count*np.log(prob)
    length= 100_000_000
    val= -1*likelyhood/length
    print('Likelyhood: ', val)
    return val
def minlikely(countsdict):
    #SEED FROM NE11NR4 RESULTS
    initialdt001I0625=(-8.9673,-7.7323,-6.01935,-0.9932,4.722814,1.981144,6.05944,0.29747,1.53068)

    #MINIMIZE
    maxL = minimize(brainlikelyhood, initialdt001I0625, args=(countsdict,), method='Powell' ,tol= 1e-9)
    maxparams = maxL.x

    return maxparams

if __name__ == "__main__":

    ####SAVINGCOUNTS#####################
    # count = countbrainSparced(dataa, datab, datac, datad)
    # count_array = np.array(list(count.items()), dtype=object)
    # np.save('E25R5DT001I0625JochCounts.npy', count_array)

    ###########LOADINGCOUTNS###################
    loaded_count_array = np.load('E25R5DT001I0625JochCounts.npy', allow_pickle=True)
    loaded_counts_dict = defaultdict(int, dict(loaded_count_array))
    print(loaded_counts_dict)

    ##########INFERENCE THINGS#################
    # params = minlikely(loaded_counts_dict)
    # print(params, 'max likelyhood: ', brainlikelyhood(params,count))





