import numpy as np
import matplotlib.pyplot as plt
import math
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.optimize import minimize
from numba import jit

Pkl_prog = 'mcbraintime.cl'

# constants
Tmax = 1000
MAX=25
rng = np.random

def runtime_program(params, prog_path,NA,NB):
    halpha, ha, hbeta,hb, kaa, kab = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    with open(prog_path, "r") as f:
        program_source = f.read()

    program = cl.Program(ctx, program_source).build()

    NextPkl = np.zeros(((MAX+1)**2),dtype=np.float64)
    Pkl_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, NextPkl.nbytes)
    global_size = ((MAX+1)**2, )
    calc = program.compute_Pkl
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pkl_buf, halpha, ha, hbeta, hb, kaa, kab,NA,NB)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, NextPkl, Pkl_buf)
    return NextPkl



def nextNs(Parr):

    randnum = rng.random()
    SUM = 0
    NAj = 0
    NBl = 0
    for i in range(MAX+1):
        for j in range(MAX+1):
            SUM += Parr[i][j]
            if SUM > randnum:
                NAj = i
                NBl = j
                SUM = -100

    return NAj, NBl

def simulation(Nstart,params,Tmax):
    NA = Nstart[0]
    NB = Nstart[1]
    t = 0
    A = [NA]
    B = [NB]

    while t < Tmax:

        #print(params)
        nextpkl = runtime_program(params,Pkl_prog,NA,NB)
        normal = 0
        for index, value in enumerate(nextpkl):
            normal += value
        for i in range(len(nextpkl)):
            nextpkl[i] /= normal
        pklreshape= nextpkl.reshape((MAX+1,MAX+1))
        NA, NB = nextNs(pklreshape)
        t += 1
        A.append(NA)
        B.append(NB)

    return A, B
# halph,ha,hbeta,kaa,kab
# halpha-hbeta = epsilon
# hB - hA = epsilon
# #
initial=(0,0)
epsilon=0
(halpha, ha, ka, kb) = (.0, .0, .0, .0)
params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)

As,Bs = simulation(initial,params,Tmax)
ts= np.linspace(0,Tmax,len(As))
plt.plot(ts,As,linewidth=.25,c='b')
plt.plot(ts,Bs,linewidth=.25,c='r')
plt.title("blue=A, red=B, just forward")
plt.show()