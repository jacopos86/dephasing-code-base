from pathlib import Path
import os
#
MPI_ROOT = 0
ngpus = 0
# === Repo root MUST be set via environment variable ===
root_env = os.environ.get("ROOT")
if root_env is None:
    raise EnvironmentError("ROOT environment variable is not set")
PACKAGE_DIR = Path(root_env).resolve()
if not PACKAGE_DIR.exists():
    raise FileNotFoundError(f"Specified PACKAGE_DIR does not exist: {PACKAGE_DIR}")
# === GPU Section ===
GPU_ACTIVE = os.environ.get("GPU_ACTIVE", "0") == "1"
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
    # CUDA SOURCE DIR
    CUDA_SOURCE_DIR = PACKAGE_DIR / "pydephasing" / "gpu_source"
    if not CUDA_SOURCE_DIR.exists():
        raise FileNotFoundError(f"CUDA source directory does not exist: {CUDA_SOURCE_DIR}")
    if ngpus > 0:
        GPU_BLOCK_SIZE = [int(x) for x in os.environ.get("BLOCK_SIZE").split()]
        GPU_GRID_SIZE  = [int(x) for x in os.environ.get("GRID_SIZE").split()]
        gpu = GPU_obj(GPU_BLOCK_SIZE, GPU_GRID_SIZE, CUDA_SOURCE_DIR, device_id=device_id)
        gpu.set_grid_info()
