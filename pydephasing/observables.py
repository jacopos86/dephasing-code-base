import numpy as np

#  This modules computes different observables
#  from the system density matrix

# compute spin magnetization
# expect. value
def compute_spin_mag(H, dm_obj):
    # compute magnet. expect. value
    rho = dm_obj.rho
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
    # plot spin magnetization
    if log.level <= logging.DEBUG:
        plt.ylim([-1.5, 1.5])
        plt.plot(self.time, self.Mt[0,:].real, label="X")
        plt.plot(self.time, self.Mt[1,:].real, label="Y")
        plt.plot(self.time, self.Mt[2,:].real, label="Z")
        plt.legend()
        plt.show()