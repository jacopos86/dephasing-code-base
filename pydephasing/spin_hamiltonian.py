#
#   This module defines
#   the spin Hamiltonian 
#
import numpy as np
from numpy import linalg as LA
import yaml
from common.special_functions import delta, triplet_evolution
from common.phys_constants import hbar, mu_B, eps, THz_to_ev
from pydephasing.log import log
from pydephasing.mpi import mpi
import logging
from abc import ABC
import matplotlib.pyplot as plt

#
#   function : set spin Hamiltonian
#

def set_spin_hamiltonian(struct0, B0, nuclear_config=None):
	# extract spin configuration
	struct0.extract_spin_state()
	# spin mult.
	if int(struct0.spin_multiplicity) == 3:
		if mpi.rank == mpi.root:
			log.info("\t SPIN TRIPLET CALCULATION")
		spin_hamilt = spin_triplet_hamiltonian()
	elif int(struct0.spin_multiplicity) == 2:
		if mpi.rank == mpi.root:
			log.info("\t SPIN DOUBLET CALCULATION")
		spin_hamilt = spin_doublet_hamiltonian()
	else:
		log.error("Wrong spin multiplicity : " + str(struct0.spin_multiplicity))
	spin_hamilt.set_zfs_levels(struct0, B0, nuclear_config)
	return spin_hamilt
#

class spin_hamiltonian(ABC):
	def __init__(self):
		pass
	def Splus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 - m2) * (j2 + m2 + 1)) * delta(j1, j2) * delta(m1, m2+1)
		return r
	def Sminus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 + m2) * (j2 - m2 + 1)) * delta(j1, j2) * delta(m1, m2-1)
		return r
	def set_Splus(self):
		#
		#    S+  ->
		#    <j1,m1|S+|j2,m2> = sqrt((j2-m2)(j2+m2+1)) delta(j1,j2) delta(m1,m2+1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Splus[r,c] = self.Splus_mtxel(j1, m1, j2, m2)
	def set_Sminus(self):
		#
		#    S-   ->
		#    <j1,m1|S-|j2,m2> = sqrt((j2+m2)(j2-m2+1)) delta(j1,j2) delta(m1,m2-1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Sminus[r,c] = self.Sminus_mtxel(j1, m1, j2, m2)
	def set_Sx(self):
		self.Sx[:,:] = (self.Splus[:,:] + self.Sminus[:,:]) / 2.
	def set_Sy(self):
		self.Sy[:,:] = (self.Splus[:,:] - self.Sminus[:,:]) / (2.*1j)
	def set_Ssq(self):
		self.Ssq = np.matmul(self.Sx, self.Sx) + np.matmul(self.Sy, self.Sy) + np.matmul(self.Sz, self.Sz)
		assert np.abs(self.Ssq[0,0] - self.s*(self.s+1.)) < eps
		assert np.abs(self.Ssq[1,1] - self.s*(self.s+1.)) < eps
		assert np.abs(self.Ssq[2,2] - self.s*(self.s+1.)) < eps
	def check_degeneracy(self):
		return False

#
#   spin triplet Hamiltonian class
#
#   for spin triplet ground states
#   
#   Hss = D [Sz^2 - S(S+1)/3] + E (Sx^2 - Sy^2) + \sum_i^N I_i Ahfi(i) S
#

class spin_triplet_hamiltonian(spin_hamiltonian):
	def __init__(self):
		super().__init__()
		self.s = 1.
		self.basis_vectors = [[self.s,1.], [self.s,0.], [self.s,-1.]]
		self.Splus = np.zeros((3,3), dtype=np.complex128)
		self.Sminus = np.zeros((3,3), dtype=np.complex128)
		self.Sx = np.zeros((3,3), dtype=np.complex128)
		self.Sy = np.zeros((3,3), dtype=np.complex128)
		self.Sz = np.zeros((3,3), dtype=np.complex128)
		self.qs = []
		# set spin matrices
		self.set_Sz()
		self.set_Splus()
		self.set_Sminus()
		self.set_Sx()
		self.set_Sy()
		self.set_Ssq()
	#   Sz
	def set_Sz(self):
		self.Sz[0,0] = 1.
		self.Sz[2,2] = -1.
	#   SDS
	def set_SDS(self, unprt_struct):
		Ddiag = unprt_struct.Ddiag*2.*np.pi*1.E-6
		#
		#  THz units
		#
		self.SDS = Ddiag[0]*np.matmul(self.Sx, self.Sx)
		self.SDS = self.SDS + Ddiag[1]*np.matmul(self.Sy, self.Sy)
		self.SDS = self.SDS + Ddiag[2]*np.matmul(self.Sz, self.Sz)
	# hyperfine interaction
	def hperfine_coupl(self, site, I, Ahfi):
		# hyperfine coupl. in eV from MHz
		Ahfi_ev = Ahfi * 1.E-6 * THz_to_ev
		I_Ahf = np.einsum("i,ij->j", I, Ahfi_ev[site,:,:])
		Hhf = I_Ahf[0] * self.Sx + I_Ahf[1] * self.Sy + I_Ahf[2] * self.Sz
		return Hhf
	# set ZFS energy levels
	def set_zfs_levels(self, unprt_struct, B, nuclear_config=None):
		Ddiag = unprt_struct.Ddiag*1.E-6  # THz
		Ddiag = Ddiag * THz_to_ev         # eV units
		# D = 3./2 Dz
		D = 3./2 * Ddiag[2]
		# E = (Dx - Dy)/2
		E = (Ddiag[0] - Ddiag[1]) / 2.
		# unperturbed H0 = D[Sz^2 - s(s+1)/3] + E(Sx^2 - Sy^2) - mu_B B Sz
		H0 = D * (np.matmul(self.Sz, self.Sz) - self.Ssq / 3.) 
		H0 +=E * (np.matmul(self.Sx, self.Sx) - np.matmul(self.Sy, self.Sy))
		H0 -=mu_B * 2.0 * (B[0] * self.Sx + B[1] * self.Sy + B[2] * self.Sz)
		if nuclear_config is not None:
			for isp in range(nuclear_config.nsp):
				I = nuclear_config.nuclear_spins[isp]['I']
				site = nuclear_config.nuclear_spins[isp]['site']
				H0 += self.hperfine_coupl(site, I, unprt_struct.Ahfi)
		# store eigenstates
		eig, eigv = LA.eig(H0)
		for s in range(3):
			self.qs.append({'eig':eig[s], 'eigv':eigv[:,s]})
	# set time array
	def set_time(self, dt, T):
		# set time in ps units
		# for spin vector evolution
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		#
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
	# compute magnetization array
	def set_magnetization(self):
		# compute magnet. expect. value
		nt = len(self.time)
		self.Mt = np.zeros((3,nt))
		# run on t
		for i in range(nt):
			vx = np.dot(self.Sx, self.tripl_psit[:,i])
			self.Mt[0,i] = np.dot(self.tripl_psit[:,i].conjugate(), vx).real
			#
			vy = np.dot(self.Sy, self.tripl_psit[:,i])
			self.Mt[1,i] = np.dot(self.tripl_psit[:,i].conjugate(), vy).real
			#
			vz = np.dot(self.Sz, self.tripl_psit[:,i])
			self.Mt[2,i] = np.dot(self.tripl_psit[:,i].conjugate(), vz).real
		# plot spin magnetization
		if log.level <= logging.DEBUG:
			plt.ylim([-1.5, 1.5])
			plt.plot(self.time, self.Mt[0,:].real, label="X")
			plt.plot(self.time, self.Mt[1,:].real, label="Y")
			plt.plot(self.time, self.Mt[2,:].real, label="Z")
			plt.legend()
			plt.show()
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
	# write spin vector on file
	def write_spin_vector_on_file(self, out_dir):
		# time steps
		nt = len(self.time)
		# write on file
		namef = out_dir + "/spin-vector.yml"
		# set dictionary
		dict = {'time' : 0, 'Mt' : 0}
		dict['time'] = self.time
		dict['Mt'] = self.Mt
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict, out_file)
		# mag. vector
		namef = out_dir + "/occup-prob.yml"
		# set dictionary
		dict2 = {'time' : 0, 'occup' : 0}
		occup = np.zeros((3,nt))
		# run over t
		for i in range(nt):
			occup[:,i] = np.dot(self.tripl_psit[:,i].conjugate(), self.tripl_psit[:,i]).real
		dict2['time'] = self.time
		dict2['occup']= occup
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict2, out_file)

#
#   spin doublet 
#   Hamiltonian class
#

class spin_doublet_hamiltonian(spin_hamiltonian):
	def __init__(self):
		super().__init__()
		self.s = 0.5
		self.basis_vectors = [[self.s, 0.5], [self.s, -0.5]]
		self.Splus = np.zeros((2,2), dtype=np.complex128)
		self.Sminus = np.zeros((2,2), dtype=np.complex128)
		self.Sx = np.zeros((2,2), dtype=np.complex128)
		self.Sy = np.zeros((2,2), dtype=np.complex128)
		self.Sz = np.zeros((2,2), dtype=np.complex128)
		self.qs = []
		# set spin matrices
		self.set_Sz()
		self.set_Splus()
		self.set_Sminus()
		self.set_Sx()
		self.set_Sy()
		self.set_Ssq()
	#   Sz
	def set_Sz(self):
		self.Sz[0,0] = 0.5
		self.Sz[1,1] =-0.5