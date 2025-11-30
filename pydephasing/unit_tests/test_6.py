import pytest
import numpy as np
from pydephasing.parallelization.mpi import MPI_obj

def test_random_split():
    mpi = MPI_obj()
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = mpi.random_split(array)
    # local size
    rest = len(array) % mpi.size
    if (mpi.rank < rest):
        size = len(array) // mpi.size + 1 
    else:
        size = len(array) // mpi.size
    assert len(result) == size