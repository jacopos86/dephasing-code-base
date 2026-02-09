import petsc4py
import sys
## Initialize petsc 
## By default uses MPI_COMM_WORLD as communicator
##     because MPI_INIT called previously
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np


def VecWrap(vec_in, n=None, PETSc=PETSc):
    ''' 
    Wrapper for constructing PETSc Vec 
    '''

    V = PETSc.Vec()
    V.create(PETSc.COMM_WORLD)
    if not n:
        n = vec_in.shape[0]
    V.setSizes(n) 
    V.setFromOptions()

    ## NOTE For later, add preallocate if we know the number of non-zero entries 
    rstart, rend = V.getOwnershipRange()
    for row in range(rstart, rend):
        V[row] = vec_in[row]
    
    V.assemblyBegin()
    V.assemblyEnd()
    return V

def MatWrap(mat_in, n=None, PETSc=PETSc):
    '''
    Wrapper for constructing PETSc Mat
    '''

    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    if not n:
        n = mat_in.shape[0]
    A.setSizes((n, n))

    ## NOTE For later, add preallocate if we know the number of non-zero entries 
    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        A[row,:] = mat_in[row]
    
    A.assemblyBegin()
    A.assemblyEnd()
    return A

def PETSc_StateVec(rho):
    """ Create Vec from mat, handles communicator set up as well """
    y = rho.createVecRight()
    
    bs = rho.getBlockSize()
    nbnd = int(np.sqrt(bs))
    rstart, rend = rho.getOwnershipRange()
    
    for i in range(rstart // bs, rend // bs):
        offset = i * bs
        indices = np.arange(offset, offset + bs, dtype=np.int32)
        block_data_flat = rho.getValues(indices, indices)
        block_2d = block_data_flat.reshape(bs, bs)

        y.setValues(indices, block_2d.diagonal())
    
    y.assemblyBegin()
    y.assemblyEnd()
    return y

def PETScMat_flatten(A):
    """ Equivelant to np.flatten() for PETSc.MAT objects """
    V = PETSc.Vec()
    V.create(PETSc.COMM_WORLD)
    n, m = A.size
    V.setSizes(n*m) 
    V.setFromOptions()

    v_rstart, v_rend = V.getOwnershipRange()
    for row in range(a_rstart, a_rend):
        values = A.getValues([row], range(n))
        v_start = row * n 
        v_end = (row+1) * n 
        V.setValues(range(v_start, v_end), values)


    V.assemblyBegin()
    V.assemblyEnd()
    return V

def PETScTrace(A):
    """ Get trace of a PETSc.MAT (handles MPI reduce)"""
    diag_vec = A.createVecLeft()
    A.getDiagonal(diag_vec)
    # Sum the entries (this automatically performs an MPI reduction)
    return diag_vec.sum()


