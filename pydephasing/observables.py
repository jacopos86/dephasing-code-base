import numpy as np

#  This modules computes different observables
#  from the system density matrix

#
#    compute spin magnetization
#    expect. value
#

def compute_spin_mag(H, dm_obj):
    # compute magnet. expect. value
    rho = dm_obj.matr
    nt = rho.shape[-1]
    M_oft = np.zeros((3, nt))
    # spin operators
    sx = H.Sx
    sy = H.Sy
    sz = H.Sz
    # run on time array
    for t in range(nt):
        M_oft[0,t] = np.matmul(rho[:,:,t], sx).trace().real
        M_oft[1,t] = np.matmul(rho[:,:,t], sy).trace().real
        M_oft[2,t] = np.matmul(rho[:,:,t], sz).trace().real
    return M_oft

#  
#   compute occupations probability
#

def compute_occup_prob(dm_obj):
    # compute occup. prob.
    rho = dm_obj.rho
    nt = rho.shape[-1]
    n = rho.shape[0]
    occup = np.zeros((n, nt))
    # run over time array
    for t in range(nt):
        for i in range(n):
            occup[i,t] = (rho[i,i,t] * rho[i,i,t].conj()).real
    return occup