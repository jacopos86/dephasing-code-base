import petsc4py
import sys
## Initialize petsc 
## By default uses MPI_COMM_WORLD as communicator
##     because MPI_INIT called previously
petsc4py.init(sys.argv)
from petsc4py import PETSc


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

def PETScVec_from_Mat(A):

    V = PETSc.Vec()
    V.create(PETSc.COMM_WORLD)
    n, m = A.size
    V.setSizes(n*m) 
    V.setFromOptions()

    v_rstart, v_rend = V.getOwnershipRange()
    a_rstart, a_rend = A.getOwnershipRange()
    for row in range(a_rstart, a_rend):
        values = A.getValues([row], range(n))
        v_start = row * n 
        v_end = (row+1) * n 
        V.setValues(range(v_start, v_end), values)


    V.assemblyBegin()
    V.assemblyEnd()
    return V

