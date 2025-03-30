import numpy as np
from math import exp
from pydephasing.log import log
from pydephasing.set_param_object import p
import yaml
#
# special functions module
#

# 1) delta function
#    input : x, y
# 2) lorentzian
#    input : x, eta
# 3) gaussian
#    input : x, eta
# 5) solve ODE dy/dt = Fy
#    input : y0, F, dt
# 6) set cross product matrix
#    input : a
#    output : [a] : axv = [a]v
# 7) triplet state evolution
#    input : Ht, psi0, dt
#    output : psit
# 8) print ZPL gradient data
#    input : gradZPL, hessZPL, outdir
#    output: None
# 9) print matrix as string
#    input : matrix
#    output : string

#
#  function 1)
#
def delta(x, y):
	if x == y:
		return 1.
	else:
		return 0.
#
#  function 2)
#
def lorentzian(x, eta):
	ltz = eta/2. / (x ** 2 + (eta/2.) ** 2) * 1./np.pi
	return ltz

#
#   function 3)
#
def gaussian():
	pass

#
#   function 7)
#
def triplet_evolution(Ht, psi0, dt):
	# this routine solves
	# dpsi/dt = -i/hbar Ht psi -> psi triplet wfc
	# H(t) is 3X3
	nt = Ht.shape[2]
	psit = np.zeros((3,int(nt/2)), dtype=np.complex128)
	psit[:,0] = psi0[:]
	# iterate over time
	for i in range(int(nt/2)-1):
		v = np.zeros(3, dtype=np.complex128)
		v[:] = psit[:,i]
		# K1
		F1 = -1j / hbar * np.matmul(Ht[:,:,2*i], v)
		K1 = dt * F1
		v1 = v + K1 / 2.
		# K2
		F2 = -1j / hbar * np.matmul(Ht[:,:,2*i+1], v1)
		K2 = dt * F2
		v2 = v + K2 / 2.
		# K3
		F3 = -1j / hbar * np.matmul(Ht[:,:,2*i+1], v2)
		K3 = dt * F3
		v3 = v + K3
		# K4
		F4 = -1j / hbar * np.matmul(Ht[:,:,2*i+2], v3)
		K4 = dt * F4
		psit[:,i+1] = v[:] + (K1[:] + 2.*K2[:] + 2.*K3[:] + K4[:]) / 6.
	return psit
#
# function 8) print ZPL gradient data
# on output file
#
def print_zpl_fluct(gradZPL, hessZPL, out_dir):
	# write tensor to file
	file_name = "ZPL_grad.yml"
	file_name = "{}".format(out_dir + '/' + file_name)
	data = {'gradZPL' : gradZPL, 'hessZPL' : hessZPL}
	# eV / ang - eV / ang^2 units
	with open(file_name, 'w') as out_file:
		yaml.dump(data, out_file)
#
# function 9) print matrix
#
def print_2D_matrix(A):
	size = A.shape
	log.info("\t Noise matrix :")
	for i in range(size[0]):
		line = ""
		for j in range(size[1]):
			line += "  {0:.3f}".format(A[i,j])
		log.info("\t " + line)
	log.info("\n")
	log.info("\t " + p.sep)
	return line