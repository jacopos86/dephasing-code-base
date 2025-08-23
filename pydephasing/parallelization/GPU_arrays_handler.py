import numpy as np
from common.print_objects import print_2D_matrix, print_1D_array

#  handler to reshape and transform GPU arrays

#  set GPU array structure

class GPU_ARRAY:
    def __init__(self, array, type):
        self.shape = array.shape
        self.cpu_array = array
        self.type = type
    def to_gpu(self):
        return self.cpu_array.astype(self.type)
    def length(self):
        return np.int32(len(self.cpu_array))
    def print_array(self):
        print(self.shape)
        if len(self.shape) == 1:
            print_1D_array(self.cpu_array)
        elif len(self.shape) == 2:
            print_2D_matrix(self.cpu_array)