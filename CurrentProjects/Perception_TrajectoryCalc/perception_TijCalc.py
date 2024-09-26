import numpy as np
import pyopencl as cl
from perception_Traj_init import MAXTOP

def runtime_program(params,ll, prog_path):
    ll1,ll2 = ll
    halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    with open(prog_path, "r") as f:
        program_source = f.read()
    print('read')
    program = cl.Program(ctx, program_source).build()
    print('built')
    NextPmnop = np.zeros(((MAXTOP*MAXTOP)**2),dtype=np.float64)
    Pmnop_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, NextPmnop.nbytes)

    global_size = ((MAXTOP*MAXTOP)**2,)
    calc = program.compute_Pmnop
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pmnop_buf, ll1,ll2, halpha, hbeta, hgamma, hdelta,ha,hb,hc,hd, kcomp, kcoop,kdu,kud,kx)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, NextPmnop, Pmnop_buf)
    return NextPmnop
def renormalize(Tij):
    # Create a matrix to store the normalization factors
    normalizefactors = np.sum(Tij, axis=(2, 3))  # Sum over the end state indices
    # Ensure that the denominator is not zero to avoid division by zero
    normalizefactors[normalizefactors == 0] = 1.0
    # Normalize the transition matrix
    Tij /= normalizefactors[:, :, np.newaxis, np.newaxis]
    #This works, I double checked with the old method

    return Tij


