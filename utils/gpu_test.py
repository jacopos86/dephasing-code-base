import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply(float *dest, float *a, float *b) {
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
    }
""")
multiply = mod.get_function("multiply")

a = np.random.randn(500).astype(np.float32)
b = np.random.randn(500).astype(np.float32)

dest = np.zeros_like(a)
multiply(
    drv.Out(dest), drv.In(a), drv.In(b),
    block=(500,1,1), grid=(1,1))
print(dest)