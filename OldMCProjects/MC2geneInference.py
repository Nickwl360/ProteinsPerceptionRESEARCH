import numpy as np
import matplotlib.pyplot as plt
import math
import pyopencl as cl
import pyopencl.array as cl_array
from scipy.optimize import minimize

Pijkl_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/togglePijklcalc.cl'

# constants
Tmax = 1000000
M = 31
MAX = 40
rng = np.random.default_rng(100)
# dataa = np.load('2geneA.npy')
# datab = np.load('2geneB.npy')

mike= np.load('../ToggleSwitchTrajectory.npy')
mikea=  mike[0,:]
mikeb = mike[1,:]

def countmatrix(dataa,datab):
    count = np.zeros((MAX, MAX, MAX, MAX),dtype=int)
    for i in range(0, len(datab) - 1):
        count[int(dataa[i])][int(datab[i])][int(dataa[i+1])][int(datab[i + 1])] += 1
    return count
def countmatrix300(dataa,datab):
    count = np.zeros((MAX, MAX,MAX,MAX),dtype=int)
    for i in range(0, (len(dataa) - 1) // 300):
        if i * 300 < len(dataa) and (i * 300 + 300) < len(dataa):
            count[int(dataa[i * 300])][int(datab[i * 300])][int(dataa[i*300+300])][int(datab[i*300+300])] += 1
    return count

# with np.printoptions(threshold=np.inf):
#     print(countmatrix(dataa,datab))

def likelyhood(params, counts):
    Pijkl = run_program(params, Pijkl_prog)  # GPU CALC PART
    Pijklreshape = Pijkl.reshape((40, 40, 40, 40))
    Pijklnorm= renormalize(Pijklreshape)
    Pijkl300 = np.linalg.matrix_power(Pijklnorm, 300)

    # with np.printoptions(threshold=np.inf):
    #     print(Pijklreshape)
    L = 0
    for i in range(0, MAX):
        for k in range(0,MAX):
            for j in range(0, MAX):
                for l in range(0, MAX):
                    if counts[i][k][j][l] != 0 and Pijkl300[i][k][j][l]!= 0:
                        L += counts[i][k][j][l] * np.log(Pijkl300[i][k][j][l])
    print(L)
    return -1 * L


def minlikely(counts):
    initial = (0.259 , 1.526 , -0.034, -0.244)
    bounds = [(0.01,1),(0.1,2),(-.1,0.1),(-.9,0.1)]
    maxL = minimize(likelyhood, initial, args=(counts,), method='Nelder-Mead', bounds=bounds, tol= 1e-5)
    maxparams = maxL.x
    return maxparams
def renormalize(Pijkl):
    normalizefactors = np.zeros((MAX,MAX),dtype=np.float64)
    for i in range(MAX):
        for k in range(MAX):
            for j in range(MAX):
                for l in range(MAX):
                    normalizefactors[i][k] += Pijkl[i][k][j][l]
    for i in range(MAX):
        for k in range(MAX):
            for j in range(MAX):
                for l in range(MAX):
                    Pijkl[i][k][j][l] /= normalizefactors[i][k]
    return Pijkl
def run_program(params, prog_path):
    halpha, ha, kaa, kab = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    with open(prog_path, "r") as f:
        program_source = f.read()

    program = cl.Program(ctx, program_source).build()

    Pijkl = np.zeros((40**4),dtype=np.float64)
    Pijkl_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Pijkl.nbytes)
    global_size = (40**4,)
    calc = program.compute_Pijkl
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pijkl_buf, halpha,ha,kaa,kab)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, Pijkl, Pijkl_buf)
    return Pijkl

#count = countmatrix(dataa,datab)
# count300 = countmatrix300(mikea,mikeb)
# print(minlikely(count300))
Pijkl = (run_program((0.0066657 ,  2.     ,     0.03904631 ,-0.50370036), Pijkl_prog))
Pijklreshape=Pijkl.reshape((40,40,40,40))
Pijklnormal = renormalize(Pijklreshape)
with np.printoptions(threshold=np.inf):
    print(Pijklnormal)
np.save("kernelpijkl", Pijklnormal)
