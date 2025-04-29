import numpy as np
#
#  function 1)
#  RK4 ODE solver
#
def ODE_solver(y0, F, dt):
    # this routine solves
    # dy/dt = F(t) y -> y real 3d vector
    # F(t) is n x n matrix
    # using RK4 algorithm
    nt = F.shape[2]
    n  = F.shape[0]
    assert n == y0.shape[0]
    yt = np.zeros((n,int(nt/2)))
    yt[:,0] = y0[:]
    # iterate over t
    for i in range(int(nt/2)-1):
        y = np.zeros(n)
        y[:] = yt[:,i]
        # K1
        K1 = dt * np.matmul(F[:,:,2*i], y)
        y1 = y + K1 / 2.
        # K2
        K2 = dt * np.matmul(F[:,:,2*i+1], y1)
        y2 = y + K2 / 2.
        # K3
        K3 = dt * np.matmul(F[:,:,2*i+1], y2)
        y3 = y + K3
        # K4
        K4 = dt * np.matmul(F[:,:,2*i+2], y3)
        #
        yt[:,i+1] = y[:] + (K1[:] + 2.*K2[:] + 2.*K3[:] + K4[:]) / 6.
    return yt