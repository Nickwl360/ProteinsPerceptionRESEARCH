import pyopencl as cl
import numpy as np

context = cl.create_some_context()
queue = cl.CommandQueue(context)


device = cl.get_platforms()[0].get_devices()[0]
print(f"Max workgroup size: {device.max_work_group_size}")
print(f"Max memory alloc size: {device.max_mem_alloc_size}")
print(f"Global memory size: {device.global_mem_size}")
print(f"Local memory size: {device.local_mem_size}")

kernel_code = """
__kernel void add_one(__global float *arr) {
    int i = get_global_id(0);
    arr[i] += 1.0f;
}
"""

program = cl.Program(context, kernel_code).build()
array = np.random.rand(10).astype(np.float32)

mf = cl.mem_flags
buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=array)

program.add_one(queue, array.shape, None, buffer)
cl.enqueue_copy(queue, array, buffer)
print(array)

