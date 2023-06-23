import numpy as np
import random
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
                            tx = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            if idx < data.shape[0]:
                                arr[tx] += data[idx]
        return arr
    def recover_data_from_grid_atr(self, data, na, nt):
        arr = np.zeros((nt,na), dtype=type(data[0]))
        for th_x in range(self.BLOCK_SIZE[0]):
            for th_y in range(self.BLOCK_SIZE[1]):
                for th_z in range(self.BLOCK_SIZE[2]):
                    for bl_x in range(self.GRID_SIZE[0]):
                        for bl_y in range(self.GRID_SIZE[1]):
                            i = th_x + self.BLOCK_SIZE[0] * bl_x
                            j = th_y + self.BLOCK_SIZE[1] * bl_y
                            k = th_z
                            idx = i + j * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] + k * self.BLOCK_SIZE[0] * self.GRID_SIZE[0] * self.BLOCK_SIZE[1] * self.GRID_SIZE[1]
                            tx  = th_x + th_y * self.BLOCK_SIZE[0] + th_z * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
                            ax  = bl_x + bl_y * self.GRID_SIZE[0]
                            if tx < nt and ax < na:
                                arr[tx,ax] = data[idx]
        return arr
#