import numpy as np
from pydephasing.common.print_objects import print_2D_matrix, print_1D_array
from pydephasing.global_params import GPU_ACTIVE

if GPU_ACTIVE:
    import pycuda.driver as cuda

#  handler to reshape and transform GPU arrays

#  set GPU array structure

class GPU_ARRAY:
    def __init__(self, array, dtype=np.double):
        self.shape = array.shape
        self.cpu_array = array.astype(dtype)
        self.dtype = dtype
        self.gpu_array = None
    def to_gpu(self, allocate_only=False):
        '''transfer CPU array to GPU memory'''
        if self.gpu_array is None:
            self.gpu_array = cuda.mem_alloc(self.cpu_array.nbytes)
        if not allocate_only:
            cuda.memcpy_htod(self.gpu_array, self.cpu_array)
        return self.gpu_array
    def from_gpu(self):
        '''retrieve data from GPU back to CPU'''
        result = np.empty_like(self.cpu_array)
        cuda.memcpy_dtoh(result, self.gpu_array)
        return result
    def length(self):
        return np.int32(len(self.cpu_array))
    def print_array(self):
        if len(self.shape) == 1:
            print_1D_array(self.cpu_array)
        elif len(self.shape) == 2:
            print_2D_matrix(self.cpu_array)
    def free(self):
        if self.gpu_array is not None:
            self.gpu_array.free()
            self.gpu_array = None