import numpy as np
from pathlib import Path
from parallelization.GPU_arrays_handler import GPU_ARRAY
from pydephasing.global_params import GPU_ACTIVE, CUDA_SOURCE_DIR
if GPU_ACTIVE:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

# GPU class
class GPU_obj:
    def __init__(self, block_dim, grid_dim, device_id=0):
        self.device_id = device_id
        self.ctx = cuda.Device(device_id).make_context()
        self.BLOCK_SIZE = np.array(block_dim)
        self.GRID_SIZE = np.array(grid_dim)
        max_threads = cuda.Device(device_id).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        if np.prod(self.BLOCK_SIZE) > max_threads:
            raise ValueError(f"GPU_BLOCK_SIZE {np.prod(self.BLOCK_SIZE)} exceeds device max of {max_threads} threads per block")
    def get_device_module(self, src_file):
        gpu_src = Path(CUDA_SOURCE_DIR+src_file).read_text()
        dev_func = SourceModule(gpu_src, options=["-I"+CUDA_SOURCE_DIR])
        return dev_func
    def set_grid_info(self):
        self.nthr_block = self.BLOCK_SIZE[0]*self.BLOCK_SIZE[1]*self.BLOCK_SIZE[2]
        self.nblocks = self.GRID_SIZE[0]*self.GRID_SIZE[1]
        self.gpu_size = self.nthr_block * self.nblocks
        self.block = (int(self.BLOCK_SIZE[0]),int(self.BLOCK_SIZE[1]),int(self.BLOCK_SIZE[2]))
        self.grid = (int(self.GRID_SIZE[0]),int(self.GRID_SIZE[1]))
    def distribute_data_on_grid(self, data):
        data = np.asarray(data)
        len_data = len(data)
        # divide data in approx. equal parts
        lengths = np.full(self.gpu_size, len_data // self.gpu_size, dtype=np.int32)
        lengths[:len_data % self.gpu_size] += 1
        assert np.sum(lengths) == len_data
        # compute initial index for each thread
        init_index = np.zeros(self.gpu_size, dtype=np.int32)
        init_index[1:] = np.cumsum(lengths[:-1])
        for i in range(self.gpu_size-1):
            assert init_index[i]+lengths[i] == init_index[i+1]
        return GPU_ARRAY(init_index, np.int32), GPU_ARRAY(lengths, np.int32)
    def cleanup(self):
        '''clean up Cuda context'''
        if self.ctx is not None:
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None
#