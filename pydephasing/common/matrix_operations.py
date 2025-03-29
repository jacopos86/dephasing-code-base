import numpy as np

# compute matrix elements
# <qs1|A|qs2>
def compute_matr_elements(A, s1, s2):
    r = np.einsum("ij,j->i", A, s2)
    expv = np.einsum("i,i", s1.conjugate(), r)
    return expv