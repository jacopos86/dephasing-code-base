import numpy as np
import random
import pycuda.gpuarray as gpuarray
from pydephasing.GPU_arrays_handler import GPU_ARRAY

# GPU class
class GPU_obj:
    def __init__(self, block_dim, grid_dim):
        self.BLOCK_SIZE = np.array(block_dim)
        self.GRID_SIZE = np.array(grid_dim)
    def set_grid_info(self):
        self.nthr_block = self.BLOCK_SIZE[0]*self.BLOCK_SIZE[1]*self.BLOCK_SIZE[2]
        self.nblocks = self.GRID_SIZE[0]*self.GRID_SIZE[1]
        self.gpu_size = self.nthr_block * self.nblocks
        self.block = (int(self.BLOCK_SIZE[0]),int(self.BLOCK_SIZE[1]),int(self.BLOCK_SIZE[2]))
        self.grid = (int(self.GRID_SIZE[0]),int(self.GRID_SIZE[1]))
    def split_data_on_grid(self, list_data):
        data = list(np.array(list_data))
        # divide data in approx. equal parts
        lengths = np.zeros(self.nblocks, dtype=np.int32)
        lengths[:] = len(data) / self.nblocks
        rest = len(data) % self.nblocks
        i = 0
        while rest > 0:
            lengths[i] += 1
            rest -= 1
            i += 1
        assert sum(lengths) == len(data)
        # partition data
        thread_list = []
        for i in range(self.nblocks):
            lst = []
            lst = random.sample(data, lengths[i])
            for x in lst:
                print(x)
                data.remove(x)
            thread_list.append(lst)
        assert len(data) == 0
        data_list = np.zeros(sum(lengths), dtype=np.int32)
        init_index = np.zeros(self.nblocks, dtype=np.int32)
        k = 0
        for i in range(self.nblocks):
            init_index[i] = k
            for j in range(len(thread_list[i])):
                data_list[k] = thread_list[i][j]
                k += 1
        return data_list, init_index, lengths
    def create_index_list(self, shape):
        print(shape)
    def distribute_data_on_grid(self, data):
        list_data = list(data)
        # divide data in approx. equal parts
        lengths = np.zeros(self.gpu_size, dtype=int)
        lengths[:] = len(list_data) / self.gpu_size
        rest = len(list_data) % self.gpu_size
        i = 0
        while rest > 0:
            lengths[i] += 1
            rest -= 1
            i += 1
        assert sum(lengths) == len(list_data)
        ARR_SIZE = min(len(list_data), self.gpu_size)
        init_index = np.zeros(ARR_SIZE, dtype=int)
        size_list = np.zeros(ARR_SIZE, dtype=int)
        size_list[:] = lengths[:ARR_SIZE]
        print(size_list)
        exit()
        for i in range(1, self.gpu_size):
            init_index[i] = init_index[i-1] + lengths[i-1]
        return GPU_ARRAY(init_index, np.int32), GPU_ARRAY(lengths, np.int32)
    def reshape_array(self, index_list, gpu_array):
        pass
    def recover_data_from_grid(self, data):
        arr = np.zeros(self.nthr_block, dtype=type(data[0]))
        for th_x in range(self.BLOCK_SIZE[0]):
            for th_y in range(self.BLOCK_SIZE[1]):
                for th_z in range(self.BLOCK_SIZE[2]):
                    for bl_x in range(self.GRID_SIZE[0]):
                        for bl_y in range(self.GRID_SIZE[1]):
                            i = th_x + self.BLOCK_SIZE[0] * bl_x
                            j = th_y + self.BLOCK_SIZE[1] * bl_y
                            k = th_z
                            idx = i + j * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] + k * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] * self.BLOCK_SIZE[1] * self.GRID_SIZE[1]
                            xx = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            if idx < data.shape[0]:
                                arr[xx] += data[idx]
        return arr
    def recover_data_from_grid_apr(self, data, n, nx):
        arr = np.zeros((nx,n), dtype=type(data[0]))
        for th_x in range(self.BLOCK_SIZE[0]):
            for th_y in range(self.BLOCK_SIZE[1]):
                for th_z in range(self.BLOCK_SIZE[2]):
                    for bl_x in range(self.GRID_SIZE[0]):
                        for bl_y in range(self.GRID_SIZE[1]):
                            i = th_x + self.BLOCK_SIZE[0] * bl_x
                            j = th_y + self.BLOCK_SIZE[1] * bl_y
                            k = th_z
                            idx = i + j * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] + k * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] * self.BLOCK_SIZE[1] * self.GRID_SIZE[1]
                            xx  = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            mx  = bl_x + bl_y * self.GRID_SIZE[0]
                            if xx < nx and mx < n:
                                arr[xx,mx] = data[idx]
        return arr
    def recover_raman_force_from_grid(self, data, njby, nx):
        arr = np.zeros((njby,nx), dtype=type(data[0]))
        for th_x in range(self.BLOCK_SIZE[0]):
            for th_y in range(self.BLOCK_SIZE[1]):
                for th_z in range(self.BLOCK_SIZE[2]):
                    for bl_x in range(self.GRID_SIZE[0]):
                        for bl_y in range(self.GRID_SIZE[1]):
                            i = th_x + self.BLOCK_SIZE[0] * bl_x
                            j = th_y + self.BLOCK_SIZE[1] * bl_y
                            k = th_z
                            idx = i + j * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] + k * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] * self.BLOCK_SIZE[1] * self.GRID_SIZE[1]
                            mx = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            jx = bl_x + bl_y * self.GRID_SIZE[0]
                            if jx < njby and mx < nx:
                                arr[jx,mx] = data[idx]
        return arr
    def recover_eff_force_from_grid(self, Flqlqp, Flmqlqp, Flqlmqp, Flmqlmqp, nb, nthr):
        arr = np.zeros((4,nthr), dtype=np.complex128)
        for th_x in range(self.BLOCK_SIZE[0]):
            for th_y in range(self.BLOCK_SIZE[1]):
                for th_z in range(self.BLOCK_SIZE[2]):
                    for bl_x in range(self.GRID_SIZE[0]):
                        for bl_y in range(self.GRID_SIZE[1]):
                            i = th_x + self.BLOCK_SIZE[0] * bl_x
                            j = th_y + self.BLOCK_SIZE[1] * bl_y
                            k = th_z
                            idx = i + j * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] + k * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] * self.BLOCK_SIZE[1] * self.GRID_SIZE[1]
                            xx = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            bx = bl_x + bl_y * self.GRID_SIZE[0]
                            if xx < nthr and bx < nb:
                                arr[0,xx] += Flqlqp[idx]
                                arr[1,xx] += Flmqlqp[idx]
                                arr[2,xx] += Flqlmqp[idx]
                                arr[3,xx] += Flmqlmqp[idx]
        return arr
#