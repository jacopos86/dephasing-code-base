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

# compute spin vector
def compute_spin_vector_evol(self, struct0, psi0, B):
		# initial state : psi0
		# magnetic field : B (gauss)
		# H = SDS + gamma_e B S
		# 1) compute SDS
		self.set_SDS(struct0)
		# n. time steps
		nt = len(self.time_dense)
		Ht = np.zeros((3,3,nt), dtype=np.complex128)
		# 2) set Ht in eV units
		# run over time steps
		for i in range(nt):
			Ht[:,:,i] = Ht[:,:,i] + hbar * self.SDS[:,:]
			# eV
			# add B field
			Ht[:,:,i] = Ht[:,:,i] + gamma_e * hbar * (B[0] * self.Sx[:,:] + B[1] * self.Sy[:,:] + B[2] * self.Sz[:,:])
		dt = self.time[1]-self.time[0]
		# ps units
		# triplet wave function evolution
		self.tripl_psit = triplet_evolution(Ht, psi0, dt)
		# set magnetization vector Mt
		self.set_magnetization()