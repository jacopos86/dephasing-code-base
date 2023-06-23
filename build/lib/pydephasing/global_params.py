import yaml
from pydephasing.gpu import GPU_obj
#
MPI_ROOT = 0
# GPU section
try:
    f = open("./config.yml", 'r')
except:
    raise Exception("config.yml cannot be opened")
config = yaml.load(f, Loader=yaml.Loader)
#
GPU_ACTIVE = config['GPU']
if GPU_ACTIVE:
    import pycuda.autoinit
    import pycuda.driver as cuda
    print('Detected {} CUDA Capable device(s)'.format(cuda.Device.count()))
    ngpus = cuda.Device.count()
    if ngpus > 0:
        GPU_BLOCK_SIZE = config['GPU_BLOCK_SIZE']
        GPU_GRID_SIZE = config['GPU_GRID_SIZE']
        gpu = GPU_obj(GPU_BLOCK_SIZE, GPU_GRID_SIZE)
        gpu.set_grid_info()