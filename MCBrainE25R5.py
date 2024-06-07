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
PendStateE25R5File = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MCBrainTransitionGPU.cl'

#############JOCHEN DATA  TRAJS############
# set = 0  #######DT = .001
# jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE25NR5.mat')
# Rkij = jochdatafix['R_kij']
# Ekij = jochdatafix['E_kij']
# lencount= 50_000_000
# dataa=Rkij[0,lencount,set]/(MAXTOP-1)
# datab=Rkij[1,lencount,set]/(MAXTOP-1)
# datac=Ekij[0,lencount,set]/(MAXBOT-1)
# datad=Ekij[1,lencount,set]/(MAXBOT-1)





def countbrain(dataa, datab, datac, datad):
    count = np.zeros((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    lencount = np.len(dataa)
    for i in range(0, lencount - 1):
        print(i)
        count[int(dataa[i])][int(datab[i])][int(datac[i])][int(datad[i])][int(dataa[i + 1])][int(datab[i + 1])][
            int(datac[i + 1])][int(datad[i + 1])] += 1
    return count


def countbrain2_0(dataa, datab, datac, datad):
    # Initialize a dictionary to store the counts of transitions
    count = defaultdict(int)
    lencount = len(dataa)

    for i in range(lencount - 1):
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

def brainlikelyhood(params9, counts):
    hgamma,hc,halpha,ha,kcoop,kcomp,kdu,kud,kx = params9
    epsilon2=0
    params = (halpha, ha, halpha, ha , hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    non_zero_indices_values = get_non_zero_indices_and_values(counts)
    #non_zero_indices_values = counts

    likelyhood = 0
    for idx, value in non_zero_indices_values:
        start = idx[0],idx[1],idx[2],idx[3]
        end = idx[4],idx[5],idx[6],idx[7]
        prob= calcPTransitiongpu(params,start,end)
        likelyhood += value*np.log(prob)
    length= 100_000_000
    val= -1*likelyhood/length
    print('Likelyhood: ', val)
    return val

def minlikely(counts):
    #SEED FROM NE11NR4 RESULTS
    initialdt001I0625=(-8.9673,-7.7323,-6.01935,-0.9932,4.722814,1.981144,6.05944,0.29747,1.53068)

    ###ALL REST
    maxL = minimize(brainlikelyhood, initialdt001I0625, args=(counts,), method='Powell' ,tol= 1e-9)

    maxparams = maxL.x
    return maxparams

if __name__ == "__main__":
    #count = countbrain(dataa, datab, datac, datad)

    #np.save('JochDt001I1Counts',count)
    #np.save('JochDt001I0625Counts',count)
    #np.save('JochDt001I375Counts',count)
    #np.save('JochDt001I6875Counts',count)
    #np.save('Jochdt001I0625NE25NR5counts',count)
    #np.save('Jochdt001I0625NE25NR5counts1/2',count)

    #np.save('Jochdt001I1NE25NR5counts',count)
    #np.save('Jochdt001I375NE25NR5counts',count)
    #np.save('Jochdt001I6875NE25NR5counts',count)


    #count= np.load('JochDt001I0625Counts.npy')
    #count=np.load('JochDt001I375Counts.npy')
    #count=np.load('JochDt001I6875Counts.npy')
    epsilon1, epsilon2 = 0,0
    #params= (-8.39780022, -8.31815575, -6.24283186, -0.62797361, 4.64786633, 2.1348466, 6.06874194, 0.29438665, 1.62159095)
    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = (-8.39780022, -8.31815575, -6.24283186, -0.62797361, 4.64786633, 2.1348466, 6.06874194, 0.29438665, 1.62159095)

    params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    p = calcPTransitiongpu(params, (0, 0, 0, 0), (0, 1, 1, 1))

    #
    #count = np.load('Jochdt001I6875NE25NR5counts.npy')
    # count = np.load('Jochdt001I0625NE25NR5counts.npy')

    # params = minlikely(count)
    # print(params, 'max likelyhood: ', brainlikelyhood(params,count))
    #
    #epsilon1,epsilon2=0,0

    #Strong infered calcedhs dt01
    #paramsBEST= (-6.00432578 ,-6.03486149, -3.80626702 , 5.35061176 , 6.4340547  , 3.67099913 ,8.6066445  , 0.28647123 , 2.42905138)

    #weak infered calcedhas dt01
    #paramsBEST=(-6.58151137, -5.4149016 , -7.32583317, 11.50703375,  8.02477748,  7.15952577, 11.87505467,  0.31408652,  5.4198252 )

    #strong infered dt001BEST
    #paramsBEST = (-8.28640719, -8.42166917, -6.27641919, -0.61254556, 4.71567638, 2.15322098, 6.15779811, 0.29307981, 1.63738961)

    #print(brainlikelyhood(paramsBEST,count))



