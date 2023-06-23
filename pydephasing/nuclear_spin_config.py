#
#   This module defines
#   the nuclear spin configuration class
#   - number of non zero nuclear spins
#   - atomic sites with non zero spin
#   - nuclear spin value
#
import numpy as np
from scipy import integrate
import yaml
from pydephasing.utility_functions import set_cross_prod_matrix, norm_realv, ODE_solver
from pydephasing.phys_constants import gamma_n
import random
#
class nuclear_spins_config:
	#
	def __init__(self, nsp, B0):
		self.nsp = nsp
		self.B0 = np.array(B0)
		# applied mag. field (G) units
		self.nuclear_spins = []
		# I spins list
	def set_time(self, dt, T):
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		# finer array
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
		# micro sec units
	# set nuclear configuration method
	def set_nuclear_spins(self, nat, ic):
		# assume nuclear spin oriented along mag. field direction
		v = self.B0 / norm_realv(self.B0)
		# set spin list
		Ilist = []
		for isp in range(self.nsp):
			# set spin vector
			I = np.zeros(3)
			# compute components
			# in cart. coordinates
			I[:] = 0.5 * v[:]
			Ilist.append(I)
		# set atom's site
		random.seed(ic)
		sites = random.sample(range(1, nat+1), self.nsp)
		print(sites)
		# define dictionary
		keys = ['site', 'I']
		for isp in range(self.nsp):
			self.nuclear_spins.append(dict(zip(keys, [sites[isp], Ilist[isp]])))
	# set spin vector evolution
	# method to time evolve I
	def set_nuclear_spin_evol(self, Hss, unprt_struct):
		# dIa/dt = gamma_n B X Ia + (A_hf(a) M(t)) X Ia
		# B : applied magnetic field (Gauss)
		# spin_hamilt : spin Hamiltonian object
		# unprt_struct : unperturbed atomic structure
		Mt = Hss.Mt
		# ps units
		t = Hss.time
		T = t[-1]
		# compute <Mt> -> spin magnetization
		rx = integrate.simpson(Mt[0,:], t) / T
		ry = integrate.simpson(Mt[1,:], t) / T
		rz = integrate.simpson(Mt[2,:], t) / T
		M = np.array([rx, ry, rz])
		# n. time steps integ.
		nt = len(self.time_dense)
		# time interv. (micro sec)
		dt = self.time[1]-self.time[0]
		# set [B] matrix
		Btilde = set_cross_prod_matrix(self.B0)
		# run over the spins active
		# in the configuration
		for isp in range(self.nsp):
			site = self.nuclear_spins[isp]['site']
			# set HFI matrix (MHz)
			A = np.zeros((3,3))
			A[:,:] = 2.*np.pi*unprt_struct.Ahfi[site-1,:,:]
			# set F(t) = gamma_n B + Ahf(a) M
			Ft = np.zeros((3,3,nt))
			# A(a) M
			AM = np.matmul(A, M)
			AM_tilde = set_cross_prod_matrix(AM)
			for i in range(nt):
				Ft[:,:,i] = gamma_n * Btilde[:,:]
				Ft[:,:,i] = Ft[:,:,i] + AM_tilde[:,:]
				# MHz units
			I0 = self.nuclear_spins[isp]['I']
			It = ODE_solver(I0, Ft, dt)
			self.nuclear_spins[isp]['It'] = It
	# set nuclear spin time fluct.
	def set_nuclear_spin_time_fluct(self):
		# time steps
		nt = len(self.time)
		# run over different active spins
		# in the config.
		for isp in range(self.nsp):
			It = self.nuclear_spins[isp]['It']
			dIt = np.zeros((3,nt))
			dIt[0,:] = It[0,:] - It[0,0]
			dIt[1,:] = It[1,:] - It[1,0]
			dIt[2,:] = It[2,:] - It[2,0]
			self.nuclear_spins[isp]['dIt'] = dIt
	# write I(t) on ext. file
	def write_It_on_file(self, out_dir, ic):
		# write file name
		name_file = out_dir + "/config-sp" + str(self.nsp) + "-" + str(ic+1) + ".yml"
		# set dictionary
		dict = {'time' : 0, 'nuclear spins' : []}
		dict['time'] = self.time
		dict['nuclear spins'] = self.nuclear_spins
		# save data
		with open(name_file, 'w') as out_file:
			yaml.dump(dict, out_file)