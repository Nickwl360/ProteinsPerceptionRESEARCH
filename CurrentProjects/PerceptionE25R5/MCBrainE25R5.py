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
PendStateE25R5File = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/CurrentProjects/PerceptionE25R5/MCBrainTransitionGPU.cl'

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
def organize_counts_by_start_state(countsdict):
    start_state_dict = defaultdict(list)
    for (start, end), count in countsdict.items():
        start_state_dict[start].append((end, count))
    return start_state_dict
def CalcPEndState(params, ns, prog_path):
    na,nb,nc,nd = ns
    halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    platform = cl.get_platforms()
    device = platform.get_devices()
    with open(prog_path, "r") as f:
        program_source = f.read()
    program = cl.Program(ctx, program_source).build()

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
def calcPTransitiongpu(params,start):
    na1, nb1, nc1, nd1 = start
    #calculate probability based on ^
    Praw = CalcPEndState(params,start,PendStateE25R5File)
    Preshape= Praw.reshape((MAXTOP,MAXTOP,MAXBOT,MAXBOT))
    PendstateNormal= Preshape/(np.sum(Preshape))
    # RETURN THE TOTAL TRANSITION PROB AS AN ARRAY
    return PendstateNormal
def brainlikelyhood(params9, countsdict):
    hgamma,hc,halpha,ha,kcoop,kcomp,kdu,kud,kx = params9
    epsilon2=0
    params = (halpha, ha, halpha, ha , hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)

    #ONLY LOOK THROUGH UNIQUE STARTS
    start_state_dict = organize_counts_by_start_state(countsdict)

    likelyhood = 0

    for start, transitions in start_state_dict.items():
        prob_array = calcPTransitiongpu(params, start)  # Calculate prob once for each start state
        for end, count in transitions:
            prob = prob_array[end]
            if not np.isnan(prob) and prob != 0:
                likelyhood += count * np.log(prob)
            #print(start, transitions, end, count, prob, likelyhood)
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
    print('loadedcounts')

    ##########INFERENCE THINGS#################
    params = minlikely(loaded_counts_dict)
    print(params, 'max likelyhood: ', brainlikelyhood(params,loaded_counts_dict))

    ########################TESTINGGPUS###############
    # (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)= (-8.96733557, -7.73231853, -6.01935508 ,-0.99322105,  4.7228139 ,  1.98114397 ,6.05944224 , 0.29747507 , 1.53067954)
    # epsilon1, epsilon2 = 0, 0
    # params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    # p = calcPTransitiongpu(params,(0,0,0,0))
    # print(p)


