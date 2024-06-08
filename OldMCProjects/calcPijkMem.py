import numpy as np
import pyopencl as cl

MAX=75
Pijk_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MemPijk.cl'


def run_Memprogram(params, prog_path):
    halpha, ha, ka, km = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    with open(prog_path, "r") as f:
        program_source = f.read()

    program = cl.Program(ctx, program_source).build()

    Pijk = np.zeros((MAX**3),dtype=np.float64)
    Pijk_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Pijk.nbytes)
    global_size = (MAX**3,)
    calc = program.compute_Pijk
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pijk_buf, halpha,ha,ka,km)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, Pijk, Pijk_buf)
    return Pijk