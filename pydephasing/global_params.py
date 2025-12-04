import yaml
import site
#
MPI_ROOT = 0
ngpus = 0
# find code directory
site_packages = site.getsitepackages()[0].strip().split('/')
i=0
for d in site_packages:
    if d == 'lib':
        j = i - 2
    i += 1
PACKAGE_DIR = '/'.join(site_packages[:j+1])
# read config file
try:
    f = open(PACKAGE_DIR+"/config.yml", 'r')
except:
    raise Exception("config.yml cannot be opened")
config = yaml.load(f, Loader=yaml.Loader)
# GPU section
GPU_ACTIVE = config['GPU']
CUDA_SOURCE_DIR = None
if GPU_ACTIVE:
    import pycuda.driver as cuda
    from mpi4py import MPI
    from pydephasing.parallelization.gpu import GPU_obj
    # start GPU setting
    cuda.init()
    rank = MPI.COMM_WORLD.Get_rank()
    ngpus = cuda.Device.count()
    print('Detected {} CUDA Capable device(s)'.format(cuda.Device.count()))
    # Assign one GPU per rank (round-robin if more ranks than GPUs)
    device_id = rank % ngpus
    print(f"[Rank {rank}] Using CUDA device {device_id}/{ngpus}")
    if ngpus > 0:
        GPU_BLOCK_SIZE = config['GPU_BLOCK_SIZE']
        GPU_GRID_SIZE = config['GPU_GRID_SIZE']
        gpu = GPU_obj(GPU_BLOCK_SIZE, GPU_GRID_SIZE, device_id=device_id)
        gpu.set_grid_info()
    CUDA_SOURCE_DIR = PACKAGE_DIR + "/pydephasing/gpu_source/"