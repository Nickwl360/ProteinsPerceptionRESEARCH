import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.colors as colors
import pandas as pd
import itertools
import time
from SCDcalc import *
#from DMCalcs import SCDMCalc as pyscd

#SCDM_lowsalt_prog = '/Users/Austin/PycharmProjects/{GHOSH}/SCDM_lowsalt_program.cl'
SCDM_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/SCDM_program.cl'


def iter_ptm(ptm,num_choices=2):
    return [ptm_combo for ptm_combo in itertools.combinations(ptm, num_choices)]


def q_list(seq):
    N = len(seq)
    q_temp = np.zeros([N,1])
    for n in range(0, N):
        if seq[n] == 'K' or seq[n] == 'R':
            q_temp[n] = 1.0
        elif seq[n] == 'E' or seq[n] == 'D':
            q_temp[n] = -1.0
        elif seq[n] == 'X':
            q_temp[n] = -2.0
    return q_temp


def run_program(seq, prog_path):
    print(f'Starting Protein')
    scdm_prog = open(prog_path).read()

    ctx = cl.create_some_context()

    queue = cl.CommandQueue(ctx)
    # mf = cl.mem_flags

    N = len(seq)
    # print(N)
    # print(len(list(range(idr_range[0]-1,idr_range[1]+1))))
    # charge_seq = q_list(seq)

    q_array = np.empty((N*N)).astype(np.float32)
    cl_q_array = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, q_array.nbytes)

    charge_list = q_list(seq).astype(np.float32)
    cl_charge_list = cl.array.to_device(queue, charge_list)

    program = cl.Program(ctx, scdm_prog).build()
    q_array_res = program.SCDM_ij
    q_array_res.set_scalar_arg_dtypes([np.int32, None, None])
    q_array_res(queue, (N, N), None, N, cl_q_array, cl_charge_list.data)

    queue.finish()
    cl.enqueue_copy(queue, q_array, cl_q_array)

    q_temp_arr = q_array
    SCDM_array = np.reshape(q_temp_arr, (N,N))

    for i in range(N):
        for j in range(i):
            SCDM_array[i, j] /= (i-j)
            SCDM_array[j, i] = 0.0

    return SCDM_array

RAM1 = 'DDRKRRRQHGQLWFPEGFKVSEASKKKRREDLEKTVVQELTWPALLANKESQTERNDLLLLGDFKDGEPNGMALDSMHVPAGPMFRDEQDARWDQHKDQD'
RAM1_SCDM = run_program(RAM1,SCDM_prog)
#returns 100x100


RAM1_SCDMflat = [item for sublist in RAM1_SCDM for item in sublist]

print(len(RAM1_SCDMflat))  # Check the N of the flattened list
#returns 100?

#RAM1_SCDMpy = pyscd.calcSCDM(RAM1)
#RAM1_SCDMpyflat = [item for sublist in RAM1_SCDMpy for item in sublist]
#print(len(RAM1_SCDMpyflat))

#compare = [a - b for a, b in zip(RAM1_SCDMflat, RAM1_SCDMpyflat)]

plt.plot(RAM1_SCDMflat)
plt.title("GPU-Python SCDM RAM1")
plt.show()


#IDPs = getseq("xij_test_seqs.xlsx")
#calc1= run_program(IDPs[0], SCDM_prog)

#plt.imshow(calc1, cmap='bwr')  # Set the desired color map
#plt.title("SCDM")
#plt.colorbar()  # Add a color bar
#plt.show()  # Display the color map
