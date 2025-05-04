import numpy as np

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
    def reshape_gpu_array(self, gpu_array, index_list, shape):
        print('shape: ', shape)
        print('index shape: ', index_list.shape, len(index_list))
        print('gpu array shape', gpu_array.shape)
        array = np.zeros(shape)
        print(array.shape)
        return array