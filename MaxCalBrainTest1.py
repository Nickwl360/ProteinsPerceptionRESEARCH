import numpy as np
import matplotlib.pyplot as plt
import math
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.optimize import minimize
from EQUILIBRIUMFINDER import *

Pijkl_prog = 'brainpijkl.cl'
# constants
Tmax = 1000
MAX=25
#TESTINGRESHAPE
# rng = np.random
# a = np.array([0,1,  2, 3,    4, 5, 6, 7,    8,9,10,11,   12,13,14,15])
# a = a.reshape((2,2,2,2))
# print(a)
# print(a[0,0,0,1])

#needs(max,max,max,max)
def renormalize(P):
    # normalizefactors = np.zeros((MAX+1,MAX+1),dtype=np.float64)
    # for i in range(MAX+1):
    #     for k in range(MAX+1):
    #         for j in range(MAX+1):
    #             for l in range(MAX+1):
    #                 normalizefactors[i][k] += P[i][k][j][l]
    # normalizefactors[normalizefactors==0]=1
    # for i in range(MAX+1):
    #     for k in range(MAX+1):
    #         for j in range(MAX+1):
    #             for l in range(MAX+1):
    #                 P[i][k][j][l] /= normalizefactors[i][k]
    normalizefactors = np.sum(P, axis=(2, 3))  # Sum over the end state indices
    # Ensure that the denominator is not zero to avoid division by zero
    normalizefactors[normalizefactors == 0] = 1.0
    # Normalize the transition matrix
    P /= normalizefactors[:, :, np.newaxis, np.newaxis]
    # This works, I double checked with the old method
    return P
def run_program(params, prog_path):
    halpha, ha,hbeta,hb, kaa, kab = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    with open(prog_path, "r") as f:
        program_source = f.read()

    program = cl.Program(ctx, program_source).build()

    Pijkl = np.zeros(((MAX+1)**4),dtype=np.float64)
    Pijkl_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Pijkl.nbytes)
    global_size = ((MAX+1)**4,)
    calc = program.compute_Pijkl
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    event = calc(queue, global_size, None, Pijkl_buf, halpha,ha,hbeta,hb,kaa,kab)
    queue.flush()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, Pijkl, Pijkl_buf, wait_for=[event])
    queue.flush()
    queue.finish()

    return Pijkl

def nextNs(NA, NB, Parr):
    probs = Parr[NA, NB, :, :]
    randnum = np.random.random()
    SUM = 0
    NAj = 0
    NBl = 0
    for i in range(MAX+1):
        for j in range(MAX+1):
            SUM += probs[i][j]
            if SUM > randnum:
                NAj = i
                NBl = j
                return NAj,NBl


def simulation(Nstart,Tmax,P):
    NA = Nstart[0]
    NB = Nstart[1]
    t = 0
    A = [NA]
    B = [NB]

    Pno=P
    while t < Tmax:
        NA, NB = nextNs(NA, NB, Pno)
        t += 1
        A.append(NA)
        B.append(NB)

    return A, B
def getequilibrium(P):

    Pfix = np.reshape(P, ((MAX + 1) ** 2, (MAX + 1) ** 2))
    Plarge = np.linalg.matrix_power(Pfix,10000)
    Plarge = Plarge.transpose()
    evalues, evectors = np.linalg.eig(Plarge)
    threshold = 1e-6
    indices_close_to_1 = [i for i, eigenvalue in enumerate(evalues) if abs(eigenvalue - 1) < threshold]
    eigenvectors_close_to_1 = [evectors[:, i] for i in indices_close_to_1]

    for i in indices_close_to_1:
        state = eigenvectors_close_to_1[i]
        state = np.real(state)
        state = state.reshape((MAX+1,MAX+1))
        state = state.transpose()
        plt.matshow(state, cmap='viridis', origin='lower')
        plt.xlabel('NAJ')
        plt.ylabel('NBL')
        plt.colorbar()
        plt.show()
        print(eigenvectors_close_to_1[i])
    return #equilsquare
def getequilib(Pnorm):
    na, nb = (0,0)
    # initial = np.zeros(((MAX+1)**2))
    # initial[int((MAX+1)**2/2)]=1
    #initial[0]=1
    # Pfix = np.reshape(Pnorm,((MAX+1)**2,(MAX+1)**2))
    # plarge = np.linalg.matrix_power(Pfix,1000000)
    # pmnoequ = np.matmul(initial,plarge)
    # pequshape = np.reshape(pmnoequ,(MAX+1,MAX+1))
    pequshape= EquilibriumMatrix(Pnorm,(MAX+1)**2,((MAX+1)**2,(MAX+1)**2),(MAX+1,MAX+1))
    pequshape=pequshape.transpose()
    for i in range(MAX+1):
        na += i*np.sum(pequshape[i, :])
        nb += i*np.sum(pequshape[:, i])
    na/= np.sum(pequshape)
    nb/=np.sum(pequshape)
    return na, nb
def calcaminusb1(halpha,hbeta):
    ha = halpha
    hb = hbeta
    e1 = halpha-hbeta
    (ha, hA , kcoop, kcomp) = ( ha,-1*ha, 1,2)
    params1 = (halpha, hA, hb, hA + e1, kcoop, kcomp)
    Pmnop = run_program(params1, Pijkl_prog)
    pmnopreshape = Pmnop.reshape((MAX+1,MAX+1,MAX+1,MAX+1))
    pmnopnormal = renormalize(pmnopreshape)
    a,b = getequilib(pmnopnormal)

    return (a-b)/(MAX)


############TESTSPACE###########
# initial=(0,24) #Tmax = 3000
# epsilon=.5
# (halpha, ha, ka, kb) = (.0, .0, .1, .1)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
#
# pijkl = run_program(params,Pijkl_prog)
# Pijklreshape = pijkl.reshape((MAX + 1, MAX + 1, MAX + 1, MAX + 1))
# Ptrans= renormalize(Pijklreshape)

# ##EQUILIBRI###########################
# Pequilibrium = EquilibriumMatrix(Ptrans,(MAX+1)**2,MAX+1)
# plt.matshow(Pequilibrium, cmap='viridis', origin='lower')
# plt.xlabel('NAJ')
# plt.ylabel('NBL')
# plt.colorbar()
# plt.show()

####FORWARD##############################
# #
# As,Bs = simulation( initial,Tmax,Ptrans)
# ts= np.linspace(0,Tmax,len(As))
# plt.plot(ts,As,linewidth=.25,c='b')
# plt.plot(ts,Bs,linewidth=.25,c='r')
# plt.title("blue=A, red=B, full transition matrix")
# plt.show()
#
# AAvg = np.mean(As)
# BAvg = np.mean(Bs)
#
# Astd = np.std(As)
# Bstd = np.std(Bs)
#
# print('avgs: ',AAvg,BAvg)
# print('stds: ' ,Astd,Bstd)