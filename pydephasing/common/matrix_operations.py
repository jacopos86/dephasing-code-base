import numpy as np

#
#  vector norm
#
def norm_realv(v):
    nrm = np.sqrt(sum(v[:]*v[:]))
    return nrm

# compute matrix elements
# <qs1|A|qs2>
def compute_matr_elements(A, s1, s2):
    r = np.einsum("ij,j->i", A, s2)
    mel = np.einsum("i,i", s1.conjugate(), r)
    return mel

#
#   function: set cross product matrix
#
def set_cross_prod_matrix(a):
    A = np.zeros((3,3))
    A[0,1] = -a[2]
    A[0,2] =  a[1]
    A[1,0] =  a[2]
    A[1,2] = -a[0]
    A[2,0] = -a[1]
    A[2,1] =  a[0]
    return A