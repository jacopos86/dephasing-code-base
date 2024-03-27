#
#   This module defines the gradient
#   of the different interaction terms
#   1) class: gradient ZFS
#   2) class: gradient Hyperfine interaction tensor
#   3) class: gradient ZPL
#
import os
import numpy as np
import yaml
import logging
from pydephasing.neural_network_class import generate_NN_object
from pydephasing.utility_functions import print_2D_matrix
from pydephasing.phys_constants import eps
from pydephasing.input_parameters import p
from pydephasing.atomic_list_struct import atoms
import matplotlib.pyplot as plt
import math
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_structs import UnpertStruct
from tqdm import tqdm
from abc import ABC
#
class perturbation_ZFS(ABC):
	def __init__(self, out_dir):
		# out dir
		self.out_dir = out_dir	
	# read outcar file
	def read_outcar_diag(self, outcar):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		D = None
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) > 0 and l[0] == "D_diag":
				# Dxx
				j = i+2
				l2 = lines[j].split()
				Dxx = float(l2[0])
				# Dyy
				j = j+1
				l2 = lines[j].split()
				Dyy = float(l2[0])
				# Dzz
				j = j+1
				l2 = lines[j].split()
				Dzz = float(l2[0])
				#
				D = np.array([Dxx, Dyy, Dzz])
		f.close()
		if D is None:
			log.error("\t " + outcar + " -> ZFS: NOT FOUND")
		return D
	# read full tensor from outcar
	def read_outcar_full(self, outcar):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		D = None
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "D_xx" and l[5] == "D_yz":
				j = i+2
				l2 = lines[j].split()
				D = np.zeros((3,3))
				# D tensor components
				Dxx = float(l2[0])
				Dyy = float(l2[1])
				Dzz = float(l2[2])
				Dxy = float(l2[3])
				Dxz = float(l2[4])
				Dyz = float(l2[5])
				D[0,0] = Dxx
				D[1,1] = Dyy
				D[2,2] = Dzz
				D[0,1] = Dxy
				D[1,0] = Dxy
				D[0,2] = Dxz
				D[2,0] = Dxz
				D[1,2] = Dyz
				D[2,1] = Dyz
		f.close()
		if D is None:
			log.error("\t " + outcar + " ZFS -> NOT FOUND")
		return D
	# eliminate noise
	def compute_noise(self, displ_structs):
		nat = self.struct_0.nat
		D0 = self.struct_0.Dtensor
		# defect atom index
		defect_index = self.atom_info_dict['defect_atom_index'] - 1
		d = np.zeros(nat)
		# dr0
		out_dir_full = self.out_dir + '/' + self.default_dir
		for displ_struct in displ_structs:
			if displ_struct.outcars_dir == out_dir_full:
				dr0 = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
				# Ang units
			else:
				pass
		# run over atoms
		for ia in range(nat):
			dist = self.struct_0.struct.get_distance(ia, defect_index)
			d[ia] = dist
		# sorted array
		at_index = np.argsort(d)
		self.Dns = np.zeros((3,3))
		# run over atoms
		# start from half of list
		for ia in range(int(p.frac_kept_atoms*nat)+1, nat):
			iia = at_index[ia]
			for idx in range(3):
				pair = '(' + str(iia+1) + ',' + str(idx+1) + ')'
				if pair in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[pair]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# Ang units
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# read outcar
				file_name = str(iia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				D1 = self.read_outcar_full(outcar)
				dD1= np.zeros((3,3))
				dD1[:,:]= np.abs(D1[:,:] - D0[:,:]) * dr0[idx] / dr[idx]
				for i1 in range(3):
					for i2 in range(3):
						if dD1[i1,i2] > self.Dns[i1,i2]:
							self.Dns[i1,i2] = dD1[i1,i2]
				# read outcar
				file_name = str(iia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				D2 = self.read_outcar_full(outcar)
				dD2= np.zeros((3,3))
				dD2[:,:]= np.abs(D2[:,:] - D0[:,:]) * dr0[idx] / dr[idx]
				for i1 in range(3):
					for i2 in range(3):
						if dD2[i1,i2] > self.Dns[i1,i2]:
							self.Dns[i1,i2] = dD2[i1,i2]
	#
	# set local unperturbed structure
	def set_gs_zfs_tensor(self):
		self.struct_0 = UnpertStruct(self.gs_data_dir)
		self.struct_0.read_poscar()
		# get ZFS tensor
		self.struct_0.read_zfs_tensor()
#
# ZFS gradient
class gradient_ZFS(perturbation_ZFS):
	# initialization
	def __init__(self, out_dir, atoms_info):
		super().__init__(out_dir)
		# read atoms info data
		try:
			f = open(atoms_info)
		except:
			msg = "\t COULD NOT FIND: " + atoms_info
			log.error(msg)
		self.atom_info_dict = yaml.load(f, Loader=yaml.Loader)
		f.close()
		# set other variables
		self.gradDtensor = None
		self.U_gradD_U = None
		self.Dns = None
		self.gradD = None
		self.gradE = None
		# default dir
		self.default_dir = self.atom_info_dict['(0,0)'][0]
		# GS data dir
		self.gs_data_dir = self.out_dir + '/' + self.atom_info_dict['(0,0)'][1]
	#
	# set ZFS gradients
	#
	def set_tensor_gradient(self, displ_structs):
		# D0
		D0 = self.struct_0.Dtensor
		# initialize tensor gradient
		nat = self.struct_0.nat
		self.gradDtensor = np.zeros((3*nat,3,3))
		# dr0
		out_dir_full = self.out_dir + '/' + self.default_dir
		for displ_struct in displ_structs:
			if displ_struct.outcars_dir == out_dir_full:
				dr0 = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
				# Ang units
			else:
				pass
		# run over atoms
		jax = 0
		for ia in range(nat):
			for idx in range(3):
				pair = '(' + str(ia+1) + ',' + str(idx+1) + ')'
				if pair in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[pair]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# Ang units
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# write file name
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc1 = self.read_outcar_full(outcar)
				# subtract noise
				dD1 = np.zeros((3,3))
				dD1[:,:] = np.abs(Dc1[:,:]-D0[:,:])*dr0[idx]/dr[idx]
				for i1 in range(3):
					for i2 in range(3):
						if dD1[i1,i2] <= self.Dns[i1,i2]:
							Dc1[i1,i2] = D0[i1,i2]
				# read outcar
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc2 = self.read_outcar_full(outcar)
				# subtract noise
				dD2 = np.zeros((3,3))
				dD2[:,:] = np.abs(Dc2[:,:]-D0[:,:])*dr0[idx]/dr[idx]
				for i1 in range(3):
					for i2 in range(3):
						if dD2[i1,i2] <= self.Dns[i1,i2]:
							Dc2[i1,i2] = D0[i1,i2]
				#
				self.gradDtensor[jax,0,0] = (Dc1[0,0] - Dc2[0,0]) / (2.*dr[idx])
				self.gradDtensor[jax,1,1] = (Dc1[1,1] - Dc2[1,1]) / (2.*dr[idx])
				self.gradDtensor[jax,2,2] = (Dc1[2,2] - Dc2[2,2]) / (2.*dr[idx])
				self.gradDtensor[jax,0,1] = (Dc1[0,1] - Dc2[0,1]) / (2.*dr[idx])
				self.gradDtensor[jax,1,0] = (Dc1[1,0] - Dc2[1,0]) / (2.*dr[idx])
				self.gradDtensor[jax,0,2] = (Dc1[0,2] - Dc2[0,2]) / (2.*dr[idx])
				self.gradDtensor[jax,2,0] = (Dc1[2,0] - Dc2[2,0]) / (2.*dr[idx])
				self.gradDtensor[jax,1,2] = (Dc1[1,2] - Dc2[1,2]) / (2.*dr[idx])
				self.gradDtensor[jax,2,1] = (Dc1[2,1] - Dc2[2,1]) / (2.*dr[idx])
				#
				# MHz / Ang units
				#
				jax = jax + 1
	#
	# plot tensor gradient method
	#
	def plot_tensor_grad_component(self, displ_structs):
		nat = self.struct_0.nat
		Dc0= self.struct_0.Dtensor
		# run over atoms
		for ia in range(nat):
			for idx in range(3):
				pair = '(' + str(ia+1) + ',' + str(idx+1) + ')'
				if pair in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[pair]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						d = [-dr[idx], 0., dr[idx]]
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# write file name
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc1 = self.read_outcar_full(outcar)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc2 = self.read_outcar_full(outcar)
				#
				log.info('\t ' + str(ia+1)+'-'+str(idx+1)+' -> GRAD_D PLOTTED')
				Dxy = [Dc2[0,1], Dc0[0,1], Dc1[0,1]]
				Dxz = [Dc2[0,2], Dc0[0,2], Dc1[0,2]]
				Dyz = [Dc2[1,2], Dc0[1,2], Dc1[1,2]]
				modelxy = np.polyfit(d, Dxy, 2)
				modelxz = np.polyfit(d, Dxz, 2)
				modelyz = np.polyfit(d, Dyz, 2)
				ffitxy = np.poly1d(modelxy)
				ffitxz = np.poly1d(modelxz)
				ffityz = np.poly1d(modelyz)
				x_s = np.arange(d[0], d[2]+d[2], d[2])
				plt.scatter(d, Dxy, c='k')
				plt.plot(x_s, ffitxy(x_s), color="k")
				plt.savefig(p.write_dir+'/'+str(ia+1)+'-'+str(idx+1)+'-xy.png', dpi=600, format='png')
				plt.clf()
				#
				plt.scatter(d, Dyz, c='r')
				plt.plot(x_s, ffityz(x_s), color="r")
				plt.savefig(p.write_dir+'/'+str(ia+1)+'-'+str(idx+1)+'-yz.png', dpi=600, format='png')
				plt.clf()
				#
				plt.scatter(d, Dxz, c='b')
				plt.plot(x_s, ffitxz(x_s), color="b")
				plt.savefig(p.write_dir+'/'+str(ia+1)+'-'+str(idx+1)+'-xz.png', dpi=600, format='png')
				plt.clf()
	#
	# write grad D to file
	#
	def write_gradDtensor_to_file(self, write_dir):
		# write data on file
		file_name = "grad_Dtensor.yml"
		file_name = "{}".format(write_dir + '/' + file_name)
		data = {'gradD' : {'coeffs' : self.gradDtensor, 'units' : 'MHz/ang' }, 'UgradDU' : { 'coeffs' : self.U_gradD_U, 'units' : 'THz/ang' } }
		# THz / ang
		with open(file_name, 'w') as out_file:
			yaml.dump(data, out_file)
	#
	# method
	# set grad D
	def set_grad_D(self):
		nat = self.struct_0.nat
		self.gradD = np.zeros(3*nat)
		# D = 3/2 * Dzz
		self.gradD[:] = 3./2 * self.U_gradD_U[:,2,2]
		#
		#  (THz/Ang) units
		#
	# set grad E
	def set_grad_E(self):
		nat = self.struct_0.nat
		self.gradE = np.zeros(3*nat)
		# E = |Dxx - Dyy|/2
		# gradE = grad(Dxx - Dyy) * (Dxx - Dyy)/2/|Dxx - Dyy|
		D = self.struct_0.Ddiag
		if abs(D[0] - D[1]) < eps:
			sgn = math.copysign(1, D[0]-D[1])
			self.gradE[:] = sgn * (self.U_gradD_U[:,0,0] - self.U_gradD_U[:,1,1]) / 2.
		else:
			self.gradE[:] = (D[0]-D[1]) / (2.*abs(D[0]-D[1])) * (self.U_gradD_U[:,0,0] - self.U_gradD_U[:,1,1])
		#
		#  (THz/Ang) units
		#
	# set grad D tensor
	def set_UgradDU_tensor(self):
		nat = self.struct_0.nat
		# gradD = U^+ gD U
		self.U_gradD_U = np.zeros((3*nat, 3, 3))
		U = self.struct_0.Deigv
		gD = np.zeros((3,3))
		# iterate over jax
		for ia in range(nat):
			for idx in range(3):
				jax = ia*3+idx
				gD[:,:] = 0.
				gD[:,:] = self.gradDtensor[jax,:,:]
				gDU = np.matmul(gD, U)
				gDt = np.matmul(U.transpose(), gDU)
				self.U_gradD_U[jax,:,:] = gDt[:,:]
		self.U_gradD_U[:,:,:] = self.U_gradD_U[:,:,:] * 1.E-6
		#
		#   (THz/ang) units
		#
#  ------------------------------------------------------------------
#
#               2nd order gradient classes
#
# -------------------------------------------------------------------
def generate_2nd_order_grad_instance(out_dir, atoms_info):
	# read atoms info data
	try:
		f = open(atoms_info)
	except:
		msg = "\t COULD NOT FIND: " + atoms_info
		log.error(msg)
	atoms_info_dict = yaml.load(f, Loader=yaml.Loader)
	f.close()
	# select model
	if atoms_info_dict['NN_model'] == "MLP":
		return gradient_2nd_ZFS_MLP(out_dir, atoms_info_dict)
	elif atoms_info_dict['NN_model'] == "DNN":
		return gradient_2nd_ZFS_DNN(out_dir, atoms_info_dict)
	else:
		log.error("Wrong neural network model selection : MLP / DNN")
#
#   base 2nd order gradient
class gradient_2nd_ZFS(perturbation_ZFS):
	# initialization
	def __init__(self, out_dir, atoms_info_dict):
		super().__init__(out_dir)
		self.atom_info_dict = atoms_info_dict
		# set other variables
		self.grad2Dtensor = None
		self.U_grad2D_U = None
		# default dir
		self.default_dir = self.atom_info_dict['(0,0,0,0)'][0]
		# poscar dir
		self.default_poscar_dir = self.out_dir + '/' + self.atom_info_dict['(0,0,0,0)'][1]
		# GS data dir
		self.gs_data_dir = self.out_dir + '/' + self.atom_info_dict['(0,0,0,0)'][2]
		# cutoff distance between two atoms
		if self.atom_info_dict['cutoff_dist'] == "inf":
			self.d_cutoff = np.inf
		else:
			self.d_cutoff = self.atom_info_dict['cutoff_dist']
		# distance from defect
		self.defect_index = self.atom_info_dict['defect_atom_index'] - 1
		if self.atom_info_dict['max_dist_from_defect'] == "inf":
			self.dmax_defect = np.inf
		else:
			self.dmax_defect = self.atom_info_dict['max_dist_from_defect']
		if mpi.rank == mpi.root:
			log.info("\t Neural network model : " + self.atom_info_dict['NN_model'])
	#
	# main driver to compute U grad2D U
	def compute_2nd_order_gradients(self, displ_structs):
		# compute noise
		self.compute_noise(displ_structs)
		if mpi.rank == mpi.root:
			print_2D_matrix(self.Dns)
		# set NN model to find missing terms
		nat = self.struct_0.nat
		jax_list = mpi.random_split(range(3*nat))
		input_data, output_data = self.set_inp_out_NN_model(jax_list)
		# collect all data
		full_input = []
		for input_ij in input_data:
			input_ij = mpi.collect_list(input_ij)
			full_input.append(input_ij)
		full_output = []
		for output_ij in output_data:
			output_ij = mpi.collect_list(output_ij)
			full_output.append(output_ij)
		# learn model
		self.learn_network_model(full_input, full_output, p.write_dir)
		# compute grad ^ 2 tensor
		self.compute_2nd_derivative_tensor(jax_list, displ_structs)
		self.grad2Dtensor = mpi.collect_array(self.grad2Dtensor)
		mpi.comm.Barrier()
		# compute u Grad^2 D U
		self.set_Ugrad2DU_tensor()
	#
	# set ZFS 2nd order gradient
	def set_inp_out_NN_model(self, jax_list):
		# model input [dab, Dax, Dby, Daxby]
		input_00 = []
		input_01 = []
		input_02 = []
		input_11 = []
		input_12 = []
		input_22 = []
		# calculation [dab, Dax, Dby]
		output_00 = []
		output_01 = []
		output_02 = []
		output_11 = []
		output_12 = []
		output_22 = []
		# n. atoms
		nat = self.struct_0.nat
		D = self.struct_0.Dtensor
		# lattice parameters
		L1, L2, L3 = self.struct_0.struct.lattice.abc
		L = np.sqrt(L1**2 + L2**2 + L3**2)
		# poscar files list
		poscar_files = os.listdir(self.default_poscar_dir)
		poscar_files.remove("displ.yml")
		# set poscar files list
		for ia in range(nat):
			for idx in range(3):
				poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + "-1"
				if poscar in poscar_files:
					poscar_files.remove(poscar)
				else:
					log.warning("\t " + poscar + " FILE NOT FOUND IN " + self.default_poscar_dir)
				poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + "-2"
				if poscar in poscar_files:
					poscar_files.remove(poscar)
				else:
					log.warning("\t " + poscar + " FILE NOT FOUND IN " + self.default_poscar_dir)
		# eliminate poscars located on different processor
		tmp_list = []
		for poscar in poscar_files:
			index_list = poscar.split('-')
			ia = int(index_list[1])-1
			idx= int(index_list[2])-1
			jax= ia*3+idx
			if jax not in jax_list:
				pass
			else:
				tmp_list.append(poscar)
		mpi.comm.Barrier()
		poscar_files = []
		poscar_files = tmp_list
		del tmp_list
		mpi.comm.Barrier()
		# run over displ.
		# ia/idx
		for jax in tqdm(jax_list):
			ia = atoms.index_to_ia_map[jax]-1
			idx= atoms.index_to_idx_map[jax]
			da = self.struct_0.struct.get_distance(ia,self.defect_index)
			for ib in range(ia, nat):
				dab = self.struct_0.struct.get_distance(ia, ib)
				db = self.struct_0.struct.get_distance(ib,self.defect_index)
				for idy in range(3):
					# (ax,by)
					iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
					if iaxby in self.atom_info_dict.keys():
						outcars_dir = self.atom_info_dict[iaxby]
					else:
						outcars_dir = self.default_dir
					out_dir_full = ''
					out_dir_full = self.out_dir + '/' + outcars_dir
					out_dir_full += '/'
					# first order outcars
					file_name = str(ia+1) + '-' + str(idx+1) + '-1/OUTCAR'
					outcar = "{}".format(out_dir_full + file_name)
					isExist = os.path.exists(outcar)
					if not isExist:
						log.error("\t " + outcar + " FILE NOT FOUND")
					Dax1 = self.read_outcar_full(outcar)
					# first order outcars
					file_name = str(ia+1) + '-' + str(idx+1) + '-2/OUTCAR'
					outcar = "{}".format(out_dir_full + file_name)
					isExist = os.path.exists(outcar)
					if not isExist:
						log.error("\t " + outcar + " FILE NOT FOUND")
					Dax2 = self.read_outcar_full(outcar)
					# jby first order
					file_name = str(ib+1) + '-' + str(idy+1) + '-1/OUTCAR'
					outcar = "{}".format(out_dir_full + file_name)
					isExist = os.path.exists(outcar)
					if not isExist:
						log.error("\t " + outcar + " FILE NOT FOUND")
					Dby1 = self.read_outcar_full(outcar)
					# jby first order
					file_name = str(ib+1) + '-' + str(idy+1) + '-2/OUTCAR'
					outcar = "{}".format(out_dir_full + file_name)
					isExist = os.path.exists(outcar)
					if not isExist:
						log.error("\t " + outcar + " FILE NOT FOUND")
					Dby2 = self.read_outcar_full(outcar)
					# remove file from list
					poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + '-' + str(ib+1) + '-' + str(idy+1)
					if poscar in poscar_files:
						poscar_files.remove(poscar)
					# check atoms distance
					if dab <= self.d_cutoff and da <= self.dmax_defect and db <= self.dmax_defect:
						# extract data from files
						outcar = str(ia+1) + '-' + str(idx+1) + '-' + str(ib+1) + '-' + str(idy+1) + "/OUTCAR"
						outcar = "{}".format(out_dir_full + outcar)
						outExist = os.path.exists(outcar)
						poscar = "{}".format(self.default_poscar_dir + '/' + poscar)
						posExist = os.path.exists(poscar)
						if not outExist and posExist:
							log.error("\t " + outcar + " FILE NOT FOUND IN : " + out_dir_full)
						elif outExist and posExist:
							# 2nd order D
							Daxby = self.read_outcar_full(outcar)
							# inputs
							x1 = 1. - dab/L
							x2 = 1. - (dab/L) ** 2
							input_00.append([x1, x2, Dax1[0,0]-D[0,0], Dax2[0,0]-D[0,0], Dby1[0,0]-D[0,0], Dby2[0,0]-D[0,0]])
							input_01.append([x1, x2, Dax1[0,1]-D[0,1], Dax2[0,1]-D[0,1], Dby1[0,1]-D[0,1], Dby2[0,1]-D[0,1]])
							input_02.append([x1, x2, Dax1[0,2]-D[0,2], Dax2[0,2]-D[0,2], Dby1[0,2]-D[0,2], Dby2[0,2]-D[0,2]])
							input_11.append([x1, x2, Dax1[1,1]-D[1,1], Dax2[1,1]-D[1,1], Dby1[1,1]-D[1,1], Dby2[1,1]-D[1,1]])
							input_12.append([x1, x2, Dax1[1,2]-D[1,2], Dax2[1,2]-D[1,2], Dby1[1,2]-D[1,2], Dby2[1,2]-D[1,2]])
							input_22.append([x1, x2, Dax1[2,2]-D[2,2], Dax2[2,2]-D[2,2], Dby1[2,2]-D[2,2], Dby2[2,2]-D[2,2]])
							output_00.append(Daxby[0,0]-D[0,0])
							output_01.append(Daxby[0,1]-D[0,1])
							output_02.append(Daxby[0,2]-D[0,2])
							output_11.append(Daxby[1,1]-D[1,1])
							output_12.append(Daxby[1,2]-D[1,2])
							output_22.append(Daxby[2,2]-D[2,2])
						else:
							pass
		#
		if len(poscar_files) != 0:
			log.warning("\t POSCAR LIST NOT EMPTY: " + str(len(poscar_files)))
		mpi.comm.Barrier()
		input_data = [input_00, input_01, input_02, input_11, input_12, input_22]
		output_data = [output_00, output_01, output_02, output_11, output_12, output_22]
		return input_data, output_data
	#
	# learn model network
	def learn_network_model(self, input_data, output_data, write_dir):
		params = self.atom_info_dict['NN_parameters']
		# build neural network to learn missing Daxby
		# given input layer :
		# 1) Dax(h) 2) Dax(-h) 3) Dby(h) 4) Dby(-h) 4) 1-d/L 5) 1-(d/L)^2
		# first make training+test set
		# input data
		[input_00, input_01, input_02, input_11, input_12, input_22] = input_data
		params['input_shape'] = np.array(input_00).shape[1]
		# output data
		[output_00, output_01, output_02, output_11, output_12, output_22] = output_data
		# set neural network model - 00 component
		self.NN_obj_00 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_00.set_model(params)
		X_test_00, y_test_00 = self.NN_obj_00.fit(params, input_00, output_00)
		# set neural network - 01 component
		self.NN_obj_01 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_01.set_model(params)
		X_test_01, y_test_01 = self.NN_obj_01.fit(params, input_01, output_01)
		# set neural network - 02 component
		self.NN_obj_02 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_02.set_model(params)
		X_test_02, y_test_02 = self.NN_obj_02.fit(params, input_02, output_02)
		# set neural network - 11 component
		self.NN_obj_11 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_11.set_model(params)
		X_test_11, y_test_11 = self.NN_obj_11.fit(params, input_11, output_11)
		# set neural network - 12 component
		self.NN_obj_12 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_12.set_model(params)
		X_test_12, y_test_12 = self.NN_obj_12.fit(params, input_12, output_12)
		# set neural network - 22 component
		self.NN_obj_22 = generate_NN_object(self.atom_info_dict['NN_model'])
		self.NN_obj_22.set_model(params)
		X_test_22, y_test_22 = self.NN_obj_22.fit(params, input_22, output_22)
		#
		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t " + p.sep)
			log.info("\t REGR. NETWORK SCORE (00 COMPONENT): " + self.NN_obj_00.get_score(X_test_00, y_test_00))
			log.info("\t REGR. NETWORK SCORE (01 COMPONENT): " + self.NN_obj_01.get_score(X_test_01, y_test_01))
			log.info("\t REGR. NETWORK SCORE (02 COMPONENT): " + self.NN_obj_02.get_score(X_test_02, y_test_02))
			log.info("\t REGR. NETWORK SCORE (11 COMPONENT): " + self.NN_obj_11.get_score(X_test_11, y_test_11))
			log.info("\t REGR. NETWORK SCORE (12 COMPONENT): " + self.NN_obj_12.get_score(X_test_12, y_test_12))
			log.info("\t REGR. NETWORK SCORE (22 COMPONENT): " + self.NN_obj_22.get_score(X_test_22, y_test_22))
			log.info("\t " + p.sep)
			log.info("\n")
			if log.level <= logging.INFO:
				file_name = "grad2D_mlp_test.yml"
				file_name = "{}".format(write_dir + '/' + file_name)
				data = {'00': [None]*2, '01': [None]*2, '02': [None]*2, '11': [None]*2, '12': [None]*2, '22': [None]*2}
				data['00'][0] = y_test_00
				data['00'][1] = self.NN_obj_00.predict(X_test_00)
				data['01'][0] = y_test_01
				data['01'][1] = self.NN_obj_01.predict(X_test_01)
				data['02'][0] = y_test_02
				data['02'][1] = self.NN_obj_02.predict(X_test_02)
				data['11'][0] = y_test_11
				data['11'][1] = self.NN_obj_11.predict(X_test_11)
				data['12'][0] = y_test_12
				data['12'][1] = self.NN_obj_12.predict(X_test_12)
				data['22'][0] = y_test_22
				data['22'][1] = self.NN_obj_22.predict(X_test_22)
				with open(file_name, 'w') as out_file:
					yaml.dump(data, out_file)
		mpi.comm.Barrier()
	#
	# set grad_a grad_b D tensor -> quantization coordinate system
	#
	def set_Ugrad2DU_tensor(self):
    	# atoms number in simulation
		nat = self.struct_0.nat
		# grad grad D = U^+ ggD U
		self.U_grad2D_U = np.zeros((3*nat, 3*nat, 3, 3))
		# U operator
		U = self.struct_0.Deigv
		g2D = np.zeros((3,3))
		# iterate over (a,x)
		for jax in range(3*nat):
			for jby in range(3*nat):
				g2D[:,:] = 0.
				g2D[:,:] = self.grad2Dtensor[jax,jby,:,:]
				g2DU = np.matmul(g2D, U)
				g2Dt = np.matmul(U.transpose(), g2DU)
				self.U_grad2D_U[jax,jby,:,:] = g2Dt[:,:]
		self.U_grad2D_U[:,:,:,:] = self.U_grad2D_U[:,:,:,:] * 1.E-6
		#
		#    (THz/Ang^2)   units
		#
	#
	# write grad_ax,by D to file
	#
	def write_grad2Dtensor_to_file(self, write_dir):
		# write data on file
		file_name = "grad2_Dtensor.yml"
		file_name = "{}".format(write_dir + '/' + file_name)
		data = {'grad2D': {'coeffs' : self.grad2Dtensor, 'units' : 'MHz/ang^2'}, 'Ugrad2DU' : {'coeffs' : self.U_grad2D_U, 'units' : 'THz/ang^2'} }
		# write data
		with open(file_name, 'w') as out_file:
			yaml.dump(data, out_file)
	#
	# check size of tensor coefficients
	#
	def check_tensor_coefficients(self):
		nat = self.struct_0.nat
		# take average for each tensor component
		g2D00 = 0.
		g2D01 = 0.
		g2D02 = 0.
		g2D11 = 0.
		g2D12 = 0.
		g2D22 = 0.
		for jax in range(3*nat):
			for jby in range(3*nat):
				g2D00 += self.grad2Dtensor[jax,jby,0,0]
				g2D01 += self.grad2Dtensor[jax,jby,0,1]
				g2D02 += self.grad2Dtensor[jax,jby,0,2]
				g2D11 += self.grad2Dtensor[jax,jby,1,1]
				g2D12 += self.grad2Dtensor[jax,jby,1,2]
				g2D22 += self.grad2Dtensor[jax,jby,2,2]
		g2D00 = g2D00 / (9.*nat**2)
		g2D01 = g2D01 / (9.*nat**2)
		g2D02 = g2D02 / (9.*nat**2)
		g2D11 = g2D11 / (9.*nat**2)
		g2D12 = g2D12 / (9.*nat**2)
		g2D22 = g2D22 / (9.*nat**2)
		# loop over tensor components
		for ia in range(nat):
			for idx in range(3):
				jax = ia*3+idx
				for ib in range(nat):
					for idy in range(3):
						jby = ib*3+idy
						if np.abs(self.grad2Dtensor[jax,jby,0,0]-g2D00) > 1.E6:
							log.warning('\t 00 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
						if np.abs(self.grad2Dtensor[jax,jby,0,1]-g2D01) > 1.E6:
							log.warning('\t 01 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
						if np.abs(self.grad2Dtensor[jax,jby,0,2]-g2D02) > 1.E6:
							log.warning('\t 02 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
						if np.abs(self.grad2Dtensor[jax,jby,1,1]-g2D11) > 1.E6:
							log.warning('\t 11 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
						if np.abs(self.grad2Dtensor[jax,jby,1,2]-g2D12) > 1.E6:
							log.warning('\t 12 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
						if np.abs(self.grad2Dtensor[jax,jby,2,2]-g2D22) > 1.E6:
							log.warning('\t 22 COMPONENT :' + str(ia) + ' - ' + str(idx) + ' - ' + str(ib) + ' - ' + str(idy) + ' > AVERAGE BY 1.E+3' )
# 
#    MLP
#    2nd order ZFS gradient class
#    
class gradient_2nd_ZFS_MLP(gradient_2nd_ZFS):
	def __init__(self, out_dir, atoms_info_dict):
		super(gradient_2nd_ZFS_MLP, self).__init__(out_dir, atoms_info_dict)
	#
	# compute tensor 2nd derivative
	def compute_2nd_derivative_tensor(self, jax_list, displ_structs):
		# n. atoms
		nat = self.struct_0.nat
		# initialize grad ^ 2 D
		self.grad2Dtensor = np.zeros((3*nat, 3*nat, 3, 3))
		# set reference ZFS
		D = self.struct_0.Dtensor
		# lattice parameters
		L1, L2, L3 = self.struct_0.struct.lattice.abc
		L = np.sqrt(L1**2 + L2**2 + L3**2)
		# dr0
		out_dir_full = self.out_dir + '/' + self.default_dir
		for displ_struct in displ_structs:
			if displ_struct.outcars_dir == out_dir_full:
				dr0 = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
				# Ang units
			else:
				pass
		if dr0 is None:
			log.error("\t DEFAULT ATOM DISPLACEMENT NOT FOUND")
		# run over jax list
		for jax in tqdm(jax_list):
			ia = atoms.index_to_ia_map[jax]-1
			idx= atoms.index_to_idx_map[jax]
			# distance from defect center
			da = self.struct_0.struct.get_distance(ia, self.defect_index)
			for ib in range(ia, nat):
				# distance from defect center
				db = self.struct_0.struct.get_distance(ib, self.defect_index)
				# distance d(a,b)
				dab= self.struct_0.struct.get_distance(ia, ib)
				# check distance condition
				if dab <= self.d_cutoff and da <= self.dmax_defect and db <= self.dmax_defect:
					for idy in range(3):
						jby = ib*3+idy
						iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
						if iaxby in self.atom_info_dict.keys():
							outcars_dir = self.atom_info_dict[iaxby]
						else:
							outcars_dir = self.default_dir
						out_dir_full = ''
						out_dir_full = self.out_dir + '/' + outcars_dir
						# displ. structs
						for displ_struct in displ_structs:
							if displ_struct.outcars_dir == out_dir_full:
								dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
								# ang units
							else:
								pass
						out_dir_full += '/'
						# first order outcars
						file_name = str(ia+1) + '-' + str(idx+1) + '-1/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dax1 = self.read_outcar_full(outcar)
						# first order outcars
						file_name = str(ia+1) + '-' + str(idx+1) + '-2/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dax2 = self.read_outcar_full(outcar)
						# first order jby
						file_name = str(ib+1) + '-' + str(idy+1) + '-1/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dby1 = self.read_outcar_full(outcar)
						# read outcar
						file_name = str(ib+1) + '-' + str(idy+1) + '-2/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dby2 = self.read_outcar_full(outcar)
						# distances
						x1 = 1. - dab/L
						x2 = 1. - (dab/L) ** 2
						# Daxby calculation
						Daxby = np.zeros((3,3))
						input_00 = [x1, x2, Dax1[0,0]-D[0,0], Dax2[0,0]-D[0,0], Dby1[0,0]-D[0,0], Dby2[0,0]-D[0,0]]
						Daxby[0,0] = D[0,0] + self.NN_obj_00.predict([input_00])
						#
						input_01 = [x1, x2, Dax1[0,1]-D[0,1], Dax2[0,1]-D[0,1], Dby1[0,1]-D[0,1], Dby2[0,1]-D[0,1]]
						Daxby[0,1] = D[0,1] + self.NN_obj_01.predict([input_01])
						Daxby[1,0] = Daxby[0,1]
						#
						input_02 = [x1, x2, Dax1[0,2]-D[0,2], Dax2[0,2]-D[0,2], Dby1[0,2]-D[0,2], Dby2[0,2]-D[0,2]]
						Daxby[0,2] = D[0,2] + self.NN_obj_02.predict([input_02])
						Daxby[2,0] = Daxby[0,2]
						#
						input_11 = [x1, x2, Dax1[1,1]-D[1,1], Dax2[1,1]-D[1,1], Dby1[1,1]-D[1,1], Dby2[1,1]-D[1,1]]
						Daxby[1,1] = D[1,1] + self.NN_obj_11.predict([input_11])
						#
						input_12 = [x1, x2, Dax1[1,2]-D[1,2], Dax2[1,2]-D[1,2], Dby1[1,2]-D[1,2], Dby2[1,2]-D[1,2]]
						Daxby[1,2] = D[1,2] + self.NN_obj_12.predict([input_12])
						Daxby[2,1] = Daxby[1,2]
						#
						input_22 = [x1, x2, Dax1[2,2]-D[2,2], Dax2[2,2]-D[2,2], Dby1[2,2]-D[2,2], Dby2[2,2]-D[2,2]]
						Daxby[2,2] = D[2,2] + self.NN_obj_22.predict([input_22])
						# subtract noise
						dDax = np.zeros((3,3))
						dDax[:,:] = np.abs(Dax1[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDax[i1,i2] <= self.Dns[i1,i2]:
									Dax1[i1,i2] = D[i1,i2]
						#
						dDby = np.zeros((3,3))
						dDby[:,:] = np.abs(Dby1[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDby[i1,i2] <= self.Dns[i1,i2]:
									Dby1[i1,i2] = D[i1,i2]
						# Daxby
						dDaxby = np.zeros((3,3))
						dDaxby[:,:] = np.abs(Daxby[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDaxby[i1,i2] <= self.Dns[i1,i2]:
									Daxby[i1,i2] = D[i1,i2]
						# grad2 D
						self.grad2Dtensor[jax,jby,0,0] = (Daxby[0,0] - Dax1[0,0] - Dby1[0,0] + D[0,0]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,0,1] = (Daxby[0,1] - Dax1[0,1] - Dby1[0,1] + D[0,1]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,1,0] = self.grad2Dtensor[jax,jby,0,1]
						self.grad2Dtensor[jax,jby,0,2] = (Daxby[0,2] - Dax1[0,2] - Dby1[0,2] + D[0,2]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,2,0] = self.grad2Dtensor[jax,jby,0,2]
						self.grad2Dtensor[jax,jby,1,1] = (Daxby[1,1] - Dax1[1,1] - Dby1[1,1] + D[1,1]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,1,2] = (Daxby[1,2] - Dax1[1,2] - Dby1[1,2] + D[1,2]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,2,1] = self.grad2Dtensor[jax,jby,1,2]
						self.grad2Dtensor[jax,jby,2,2] = (Daxby[2,2] - Dax1[2,2] - Dby1[2,2] + D[2,2]) / (dr[idx] * dr[idy])
						#
						if jax != jby:
							self.grad2Dtensor[jby,jax,:,:] = self.grad2Dtensor[jax,jby,:,:]
				else:
					pass
		#
		# MHz / ang^2 units
		#
		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t " + p.sep)
			log.info("\t GRAD_ax;by CALCULATION COMPLETED")
			log.info("\t " + p.sep)
			log.info("\n")
#
#   DNN
#   2nd order ZFS gradient class
#
class gradient_2nd_ZFS_DNN(gradient_2nd_ZFS):
	def __init__(self, out_dir, atoms_info_dict):
		super(gradient_2nd_ZFS_DNN, self).__init__(out_dir, atoms_info_dict)
	#
	# compute tensor 2nd derivative
	def compute_2nd_derivative_tensor(self, jax_list, displ_structs):
		# n. atoms
		nat = self.struct_0.nat
		# initialize grad^2 D
		self.grad2Dtensor = np.zeros((3*nat, 3*nat, 3, 3))
		# reference ZFS
		D = self.struct_0.Dtensor
		# lattice parameters
		L1, L2, L3 = self.struct_0.struct.lattice.abc
		L = np.sqrt(L1**2 + L2**2 + L3**2)
		# dr0
		out_dir_full = self.out_dir + '/' + self.default_dir
		for displ_struct in displ_structs:
			if displ_struct.outcars_dir == out_dir_full:
				dr0 = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
				# ang units
			else:
				pass
		if dr0 is None:
			log.error("\t DEFAULT ATOM DISPLACEMENT NOT FOUND")
		# input lists for NN model
		input_00 = []
		input_01 = []
		input_02 = []
		input_11 = []
		input_12 = []
		input_22 = []
		Dax1_lst = []
		Dby1_lst = []
		# run jax list
		for jax in tqdm(jax_list):
			ia = atoms.index_to_ia_map[jax]-1
			idx= atoms.index_to_idx_map[jax]
			# distance from defect
			da = self.struct_0.struct.get_distance(ia, self.defect_index)
			for ib in range(ia, nat):
				# distance from defect
				db = self.struct_0.struct.get_distance(ib, self.defect_index)
				# distance d(a,b)
				dab = self.struct_0.struct.get_distance(ia, ib)
				# check distance
				if dab <= self.d_cutoff and da <= self.dmax_defect and db <= self.dmax_defect:
					for idy in range(3):
						iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
						if iaxby in self.atom_info_dict.keys():
							outcars_dir = self.atom_info_dict[iaxby]
						else:
							outcars_dir = self.default_dir
						out_dir_full = ''
						out_dir_full = self.out_dir + '/' + outcars_dir
						out_dir_full += '/'
						# first order outcars
						file_name = str(ia+1) + '-' + str(idx+1) + '-1/OUTCAR'			
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dax1 = self.read_outcar_full(outcar)
						Dax1_lst.append(Dax1)
						# first order outcars
						file_name = str(ia+1) + '-' + str(idx+1) + '-2/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dax2 = self.read_outcar_full(outcar)
						# first order jby
						file_name = str(ib+1) + '-' + str(idy+1) + '-1/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dby1 = self.read_outcar_full(outcar)
						Dby1_lst.append(Dby1)
						# read outcar
						file_name = str(ib+1) + '-' + str(idy+1) + '-2/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						Dby2 = self.read_outcar_full(outcar)
						# distances
						x1 = 1. - dab/L
						x2 = 1. - (dab/L) ** 2
						# append data to compute
						input_00.append(np.array([x1, x2, Dax1[0,0]-D[0,0], Dax2[0,0]-D[0,0], Dby1[0,0]-D[0,0], Dby2[0,0]-D[0,0]]))
						input_01.append(np.array([x1, x2, Dax1[0,1]-D[0,1], Dax2[0,1]-D[0,1], Dby1[0,1]-D[0,1], Dby2[0,1]-D[0,1]]))
						input_02.append(np.array([x1, x2, Dax1[0,2]-D[0,2], Dax2[0,2]-D[0,2], Dby1[0,2]-D[0,2], Dby2[0,2]-D[0,2]]))
						input_11.append(np.array([x1, x2, Dax1[1,1]-D[1,1], Dax2[1,1]-D[1,1], Dby1[1,1]-D[1,1], Dby2[1,1]-D[1,1]]))
						input_12.append(np.array([x1, x2, Dax1[1,2]-D[1,2], Dax2[1,2]-D[1,2], Dby1[1,2]-D[1,2], Dby2[1,2]-D[1,2]]))
						input_22.append(np.array([x1, x2, Dax1[2,2]-D[2,2], Dax2[2,2]-D[2,2], Dby1[2,2]-D[2,2], Dby2[2,2]-D[2,2]]))
		# predict Daxby
		Daxby_00 = self.NN_obj_00.predict(input_00)
		Daxby_01 = self.NN_obj_01.predict(input_01)
		Daxby_02 = self.NN_obj_02.predict(input_02)
		Daxby_11 = self.NN_obj_11.predict(input_11)
		Daxby_12 = self.NN_obj_12.predict(input_12)
		Daxby_22 = self.NN_obj_22.predict(input_22)
		# compute tensor gradient
		ii = 0
		for jax in jax_list:
			ia = atoms.index_to_ia_map[jax] - 1
			idx= atoms.index_to_idx_map[jax]
			# distance from defect
			da = self.struct_0.struct.get_distance(ia, self.defect_index)
			for ib in range(ia, nat):
				# distance from defect center
				db = self.struct_0.struct.get_distance(ib, self.defect_index)
				# distance d(a,b)
				dab= self.struct_0.struct.get_distance(ia, ib)
				# distance cut-off
				if dab <= self.d_cutoff and da <= self.dmax_defect and db <= self.dmax_defect:
					for idy in range(3):
						jby = ib*3+idy
						iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
						if iaxby in self.atom_info_dict.keys():
							outcars_dir = self.atom_info_dict[iaxby]
						else:
							outcars_dir = self.default_dir
						out_dir_full = ''
						out_dir_full = self.out_dir + '/' + outcars_dir
						# displ. structs
						for displ_struct in displ_structs:
							if displ_struct.outcars_dir == out_dir_full:
								dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
								# ang units
							else:
								pass
						# Dax1
						Dax1 = Dax1_lst[ii]
						# Dby1
						Dby1 = Dby1_lst[ii]
						#  Daxby calculation
						Daxby = np.zeros((3,3))
						Daxby[0,0] = Daxby_00[ii]
						Daxby[0,1] = Daxby_01[ii]
						Daxby[1,0] = Daxby[0,1]
						Daxby[0,2] = Daxby_02[ii]
						Daxby[2,0] = Daxby[0,2]
						Daxby[1,1] = Daxby_11[ii]
						Daxby[1,2] = Daxby_12[ii]
						Daxby[2,1] = Daxby[1,2]
						Daxby[2,2] = Daxby_22[ii]
						# subtract noise
						dDax = np.zeros((3,3))
						dDax[:,:] = np.abs(Dax1[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDax[i1,i2] <= self.Dns[i1,i2]:
									Dax1[i1,i2] = D[i1,i2]
						#
						dDby = np.zeros((3,3))
						dDby[:,:] = np.abs(Dby1[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDby[i1,i2] <= self.Dns[i1,i2]:
									Dby1[i1,i2] = D[i1,i2]
						# Daxby
						dDaxby = np.zeros((3,3))
						dDaxby[:,:] = np.abs(Daxby[:,:]-D[:,:])*dr0[idx]/dr[idx]
						for i1 in range(3):
							for i2 in range(3):
								if dDaxby[i1,i2] <= self.Dns[i1,i2]:
									Daxby[i1,i2] = D[i1,i2]
						# grad2 D
						self.grad2Dtensor[jax,jby,0,0] = (Daxby[0,0] - Dax1[0,0] - Dby1[0,0] + D[0,0]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,0,1] = (Daxby[0,1] - Dax1[0,1] - Dby1[0,1] + D[0,1]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,1,0] = self.grad2Dtensor[jax,jby,0,1]
						self.grad2Dtensor[jax,jby,0,2] = (Daxby[0,2] - Dax1[0,2] - Dby1[0,2] + D[0,2]) / (dr[idx] * dr[idy])
						self.grad2Dtensor[jax,jby,2,0] = self.grad2Dtensor[jax,jby,0,2]
						self.grad2Dtensor[jax,jby,1,1] = (Daxby[1,1] - Dax1[1,1] - Dby1[1,1] + D[1,1]) / (dr[idx] * dr[idy])

						ii += 1
#
#   class :
#   gradient hyperfine interaction
#
class perturbation_HFI(ABC):
	def __init__(self, out_dir, atoms_info, core):
		# out dir
		self.out_dir = out_dir
		# read atoms info data
		try:
			f = open(atoms_info)
		except:
			msg = "\t " + atoms_info + " FILE NOT FOUND"
			log.error(msg)
		self.atom_info_dict = yaml.load(f, Loader=yaml.Loader)
		f.close()
		# set core correction
		self.core = core
	# read outcar file
	def read_dipolar_int(self, outcar, nat):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 7:
				if l[0] == "ion" and l[6] == "A_yz":
					j = i+2
					A = np.zeros((nat,6))
					for k in range(j, j+nat):
						l2 = lines[k].split()
						for u in range(6):
							A[k-j,u] = float(l2[1+u])
		f.close()
		return A
	# read full HFI = dip + FC
	def read_full_hfi(self, outcar, nat):
		A_full = self.read_dipolar_int(outcar, nat)
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			# read fermi contact term
			if len(l) == 6:
				if l[0] == "ion" and l[5] == "A_tot":
					A_fc = 0.
					j = i+2
					for k in range(j, j+nat):
						l2 = lines[k].split()
						if self.core:
							A_fc = float(l2[4]) + float(l2[5])
						else:
							A_fc = float(l2[5])
						A_full[k-j,0] = A_full[k-j,0] + A_fc
						A_full[k-j,1] = A_full[k-j,1] + A_fc
						A_full[k-j,2] = A_full[k-j,2] + A_fc
		f.close()
		return A_full
	# compute noise
	def compute_noise(self, displ_structs):
		nat = self.struct_0.nat
		A0  = self.Ahfi_0
		# MHz
		# defect index
		defect_index = self.atom_info_dict['defect_atom_index'] - 1
		d = np.zeros(nat)
		# dr0
		out_dir_full = self.out_dir + '/' + self.default_dir
		for displ_struct in displ_structs:
			if displ_struct.outcars_dir == out_dir_full:
				dr0 = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
				# ang units
			else:
				pass
		# run over atoms
		for ia in range(nat):
			dist = self.struct_0.struct.get_distance(ia, defect_index)
			d[ia]= dist
		# sorted array
		at_index = np.argsort(d)
		self.Ahf_ns = np.zeros((nat,6))
		# run over atoms
		for ia in range(int(2*nat/3)+1, nat):
			iia = at_index[ia]
			for idx in range(3):
				pair = '(' + str(ia+1) + ',' + str(idx+1) + ')'
				if pair in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[pair]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# displ. struct.
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# ang units
					else:
						pass
				out_dir_full = out_dir_full + '/'
				# read outcars
				file_name = str(iia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				A1  = self.read_full_hfi(outcar, nat)
				dA1 = np.zeros((nat,6))
				dA1[:,:] = np.abs(A1[:,:] - A0[:,:]) * dr0[idx] / dr[idx]
				for aa in range(nat):
					for ix in range(6):
						if dA1[aa,ix] > self.Ahf_ns[aa,ix]:
							self.Ahf_ns[aa,ix] = dA1[aa,ix]
				# - 2
				file_name = str(iia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				A2 = self.read_full_hfi(outcar, nat)
				dA2 = np.zeros((nat,6))
				dA2[:,:] = np.abs(A2[:,:] - A0[:,:]) * dr0[idx] / dr[idx]
				for aa in range(nat):
					for ix in range(6):
						if dA2[aa,ix] > self.Ahf_ns[aa,ix]:
							self.Ahf_ns[aa,ix] = dA2[aa,ix]
	# set HFI GS tensor
	def set_gs_hfi_tensor(self):
		self.struct_0 = UnpertStruct(self.gs_data_dir)
		self.struct_0.read_poscar()
		# get zfs tensor
		self.struct_0.read_zfs_tensor()
		# set HFI tensor
		self.Ahfi_0 = self.struct_0.read_hfi_full(self.core)
#
# derived HFI classes
class gradient_HFI(perturbation_HFI):
	def __init__(self, out_dir, atoms_info, core):
		super().__init__(out_dir, atoms_info, core)
		self.gradAhfi = None
		# local basis array
		self.gAhfi_xx = []
		self.gAhfi_yy = []
		self.gAhfi_zz = []
		self.gAhfi_xy = []
		self.gAhfi_xz = []
		self.gAhfi_yz = []
		# default dir
		self.default_dir = self.atom_info_dict['(0,0)'][0]
		# GS data dir
		self.gs_data_dir = self.out_dir + '/' + self.atom_info_dict['(0,0)'][1]
	#
	# set Ahfi tensor gradient
	#
	def set_tensor_gradient(self, displ_structs):
		# nat
		nat = self.struct_0.nat
		# initialize array
		self.gradAhfi = np.zeros((3*nat,nat,3,3))
		# run over atoms
		for ia in range(nat):
			gradAxx = np.zeros((nat,3))
			gradAyy = np.zeros((nat,3))
			gradAzz = np.zeros((nat,3))
			gradAxy = np.zeros((nat,3))
			gradAxz = np.zeros((nat,3))
			gradAyz = np.zeros((nat,3))
			for idx in range(3):
				# check if (ia,idx) in pert_dirs
				pair = '(' + str(ia+1) + ',' + str(idx+1) + ')'
				if pair in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[pair]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# Ang units
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# write file names
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				A_p1 = self.read_full_hfi(outcar, nat)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				A_m1 = self.read_full_hfi(outcar, nat)
				#
				gradAxx[:,idx] = (A_p1[:,0] - A_m1[:,0]) / (2.*dr[idx])
				gradAyy[:,idx] = (A_p1[:,1] - A_m1[:,1]) / (2.*dr[idx])
				gradAzz[:,idx] = (A_p1[:,2] - A_m1[:,2]) / (2.*dr[idx])
				gradAxy[:,idx] = (A_p1[:,3] - A_m1[:,3]) / (2.*dr[idx])
				gradAxz[:,idx] = (A_p1[:,4] - A_m1[:,4]) / (2.*dr[idx])
				gradAyz[:,idx] = (A_p1[:,5] - A_m1[:,5]) / (2.*dr[idx])
				#
				# MHz/Ang units
				#
			self.gAhfi_xx.append(gradAxx)
			self.gAhfi_yy.append(gradAyy)
			self.gAhfi_zz.append(gradAzz)
			self.gAhfi_xy.append(gradAxy)
			self.gAhfi_xz.append(gradAxz)
			self.gAhfi_yz.append(gradAyz)
	# set grad Ahfi D diag basis set
	def set_U_gradAhfi_U_tensor(self):
		nat = self.struct_0.nat
		# set transf. matrix U
		U = self.struct_0.Deigv
		# start iterations over atoms
		jax = 0
		for ia in range(nat):
			gradAxx = self.gAhfi_xx[ia]
			gradAyy = self.gAhfi_yy[ia]
			gradAzz = self.gAhfi_zz[ia]
			gradAxy = self.gAhfi_xy[ia]
			gradAxz = self.gAhfi_xz[ia]
			gradAyz = self.gAhfi_yz[ia]
			#
			gradA = np.zeros((nat,3,3,3))
			gradA[:,:,0,0] = gradAxx[:,:]
			gradA[:,:,1,1] = gradAyy[:,:]
			gradA[:,:,2,2] = gradAzz[:,:]
			gradA[:,:,0,1] = gradAxy[:,:]
			gradA[:,:,1,0] = gradAxy[:,:]
			gradA[:,:,0,2] = gradAxz[:,:]
			gradA[:,:,2,0] = gradAxz[:,:]
			gradA[:,:,1,2] = gradAyz[:,:]
			gradA[:,:,2,1] = gradAyz[:,:]
			for idx in range(3):
				# run over every atoms
				for aa in range(nat):
					ga = np.zeros((3,3))
					ga[:,:] = gradA[aa,idx,:,:]
					# transform over new basis
					gaU = np.matmul(ga, U)
					ga = np.matmul(U.transpose(), gaU)
					self.gradAhfi[jax,aa,:,:] = ga[:,:]
					#
					#  MHz / Ang units
					#
				jax = jax + 1
		#
		#  THz / Ang units
		#
		self.gradAhfi[:,:,:,:] = self.gradAhfi[:,:,:,:] * 1.E-6
#
# 2nd order HFI
class gradient_2nd_HFI(perturbation_HFI):
	def __init__(self, out_dir, atoms_info, core):
		super().__init__(out_dir, atoms_info, core)
		self.grad2_Ahfi = None
		# local basis array
		self.g2_Ahfi_xx = []
		self.g2_Ahfi_yy = []
		self.g2_Ahfi_zz = []
		self.g2_Ahfi_xy = []
		self.g2_Ahfi_xz = []
		self.g2_Ahfi_yz = []
		# default dir
		self.default_dir = self.atom_info_dict['(0,0,0,0)'][0]
		# poscar dir
		self.default_poscar_dir = self.out_dir + '/' + self.atom_info_dict['(0,0,0,0)'][1]
		# GS data dir
		self.gs_data_dir = self.out_dir + '/' + self.atom_info_dict['(0,0,0,0)'][2]
		# cut off distance
		if self.atom_info_dict['cutoff_dist'] == 'inf':
			self.d_cutoff = np.inf
		else:
			self.d_cutoff = self.atom_info_dict['cutoff_dist']
		# distance from defect
		self.defect_index = self.atom_info_dict['defect_atom_index'] - 1
		if self.atom_info_dict['max_dist_from_defect'] == "inf":
			self.dmax_defect = np.inf
		else:
			self.dmax_defect = self.atom_info_dict['max_dist_from_defect']
	#
	# main driver to compute U grad2A U
	def extract_2nd_order_gradients(self):
		# set NN model
		nat = self.struct_0.nat
		jax_list = mpi.random_split(range(3*nat))
		# run over each atom aa= 1, nat
		inp_list, input_data, output_data = self.collect_ahf_data(jax_list)
		# collect all data on single proc.
		self.ahf_input_data = []
		self.ahf_input_data = mpi.collect_list(input_data)
		# output
		self.ahf_output_data = []
		self.ahf_output_data = mpi.collect_list(output_data)
		# inp list
		self.ahf_inp_list = []
		self.ahf_inp_list = mpi.collect_list(inp_list)
		if mpi.rank == mpi.root:
			log.info("\t 2ND ORDER GRADIENT ACQUISITION COMPLETED")
			log.info("\t " + p.sep)
			log.info("\n")
		mpi.comm.Barrier()
	#
	# set gradients
	def collect_ahf_data(self, jax_list):
		# set NN model : find missing terms
		nat = self.struct_0.nat
		inp_list = []
		# input data
		input_data = []
		# output data
		output_data= []
		# poscar files
		poscar_files = os.listdir(self.default_poscar_dir)
		poscar_files.remove("displ.yml")
		# poscar files list
		for ia in range(nat):
			for idx in range(3):
				poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + "-1"
				if poscar in poscar_files:
					poscar_files.remove(poscar)
				else:
					log.warning("\t " + poscar + " FILE NOT FOUND IN " + self.default_poscar_dir)
				poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + "-2"
				if poscar in poscar_files:
					poscar_files.remove(poscar)
				else:
					log.warning("\t " + poscar + " FILE NOT FOUND IN " + self.default_poscar_dir)
		# eliminate poscar accessed on different procs
		tmp_list = []
		for poscar in poscar_files:
			index_list = poscar.split('-')
			ia = int(index_list[1]) - 1
			idx= int(index_list[2]) - 1
			jax= 3*ia+idx
			if jax not in jax_list:
				pass
			else:
				tmp_list.append(poscar)
		mpi.comm.Barrier()
		poscar_files = []
		poscar_files = tmp_list
		del tmp_list
		mpi.comm.Barrier()
		# run over displ. ia/idx
		for jax in tqdm(jax_list):
			ia = atoms.index_to_ia_map[jax] - 1
			idx= atoms.index_to_idx_map[jax]
			da = self.struct_0.struct.get_distance(ia,self.defect_index)
			for ib in range(ia, nat):
				dab = self.struct_0.struct.get_distance(ia, ib)
				db  = self.struct_0.struct.get_distance(ib,self.defect_index)
				for idy in range(3):
					jby = 3*ib+idy
					# check atom distance
					if dab <= self.d_cutoff and da <= self.dmax_defect and db <= self.dmax_defect:
						# (ax,by)
						iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
						if iaxby in self.atom_info_dict.keys():
							outcars_dir = self.atom_info_dict[iaxby]
						else:
							outcars_dir = self.default_dir
						out_dir_full  = ''
						out_dir_full  = self.out_dir + '/' + outcars_dir
						out_dir_full += '/'
						# first order outcars
						file_name = str(ia+1) + '-' + str(idx+1) + '-1/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						A_ax1 = self.read_full_hfi(outcar, nat)
						# jby first order outcar
						file_name = str(ib+1) + '-' + str(idy+1) + '-1/OUTCAR'
						outcar = "{}".format(out_dir_full + file_name)
						isExist = os.path.exists(outcar)
						if not isExist:
							log.error("\t " + outcar + " FILE NOT FOUND")
						A_by1 = self.read_full_hfi(outcar, nat)
						# remove from list
						poscar = "POSCAR-" + str(ia+1) + '-' + str(idx+1) + '-' + str(ib+1) + '-' + str(idy+1)
						if poscar in poscar_files:
							poscar_files.remove(poscar)
						# data from files
						outcar = str(ia+1) + '-' + str(idx+1) + '-' + str(ib+1) + '-' + str(idy+1) + "/OUTCAR"
						outcar = "{}".format(out_dir_full + outcar)
						outExist = os.path.exists(outcar)
						poscar = "{}".format(self.default_poscar_dir + '/' + poscar)
						posExist = os.path.exists(poscar)
						if not outExist and posExist:
							log.error("\t " + outcar + " FILE NOT FOUND IN " + out_dir_full)
						elif outExist and posExist:
							# 2nd order A
							A_axby = self.read_full_hfi(outcar, nat)
							# inputs
							if ia == ib:
								natlst = range(nat)
							else:
								natlst = [ia, ib]
							# run over aa
							for aa in natlst:
								ax1 = np.zeros(6)
								ay1 = np.zeros(6)
								axy1= np.zeros(6)
								# data values
								ax1[:] = A_ax1[aa,:]
								#
								ay1[:] = A_by1[aa,:]
								#
								axy1[:]= A_axby[aa,:]
								#
								inp_list.append([aa, jax, jby])
								input_data.append([ax1, ay1])
								output_data.append(axy1)
						else:
							pass
		#
		if len(poscar_files) != 0:
			log.warning("\t POSCAR LIST NOT EMPTY : " + str(len(poscar_files)))
		mpi.comm.Barrier()
		#
		return inp_list, input_data, output_data
	#
	# compute at. resolved grad2 ahf
	def evaluate_at_resolved_grad2(self, aa, nat, displ_structs):
		# Ahf
		A0 =  self.Ahfi_0
		# compute missing Daxby
		# input data
		ind = []
		oud = []
		inax= []
		inby= []
		# look for aa elements
		for i in range(len(self.ahf_inp_list)):
			id = self.ahf_inp_list[i]
			id2= self.ahf_input_data[i]
			od = self.ahf_output_data[i]
			if id[0] == aa:
				inax.append(id[1])
				inby.append(id[2])
				ind.append(id2)
				oud.append(od)
		# grad2 Ahf
		grad2_ahf = np.zeros((3*nat, 3*nat, 6))
		assert len(ind) == len(oud)
		# run over d.o.f.
		for iiax in range(3*nat):
			ia = atoms.index_to_ia_map[iiax]-1
			idx= atoms.index_to_idx_map[iiax]
			# iiby index
			for iiby in [3*aa, 3*aa+1, 3*aa+2]:
				ib = atoms.index_to_ia_map[iiby]-1
				idy= atoms.index_to_idx_map[iiby]
				# extract atomic displ.
				iaxby = '(' + str(ia+1) + ',' + str(idx+1) + ',' + str(ib+1) + ',' + str(idy+1) + ')'
				ibyax = '(' + str(ib+1) + ',' + str(idy+1) + ',' + str(ia+1) + ',' + str(idx+1) + ')'
				if iaxby in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[iaxby]
				elif ibyax in self.atom_info_dict.keys():
					outcars_dir = self.atom_info_dict[ibyax]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = self.out_dir + '/' + outcars_dir
				# displ. structs
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# ang units
					else:
						pass
				# find (iiax,iiby) pair
				iil = [i for i, j in enumerate(inax) if j == iiax]
				ii = -1
				for x in iil:
					if inby[x] == iiby:
						ii = x
						break
				if ii == -1:
					iil = [i for i, j in enumerate(inby) if j == iiax]
					for x in iil:
						if inax[x] == iiby:
							ii = x
							break
				if ii == -1:
					# find closest available atoms
					d = np.inf
					km = -1
					for k in range(len(inax)):
						ia2 = int(inax[k]/3)
						ib2 = int(inby[k]/3)
						d_a = self.struct_0.struct.get_distance(ia2, ia)
						d_b = self.struct_0.struct.get_distance(ib2, ib)
						dist= np.sqrt(d_a ** 2 + d_b ** 2)
						if dist < d:
							km = k
							d = dist
						d_a = self.struct_0.struct.get_distance(ib2, ia)
						d_b = self.struct_0.struct.get_distance(ia2, ib)
						dist = np.sqrt(d_a ** 2 + d_b ** 2)
						if dist < d:
							km = k
							d = dist
					# extract data
					A_ax1 = ind[km][0]
					A_by1 = ind[km][1]
					A_axby= oud[km]
				else:
					# extract data
					A_ax1 = ind[ii][0]
					A_by1 = ind[ii][1]
					A_axby= oud[ii]
				# compute 2nd order gradient
				grad2_ahf[iiax,iiby,:] = (A_axby[:] - A_ax1[:] - A_by1[:] + A0[aa][:]) / (dr[idx] * dr[idy])
				if iiax != iiby:
					grad2_ahf[iiby,iiax,:] = grad2_ahf[iiax,iiby,:]
		return grad2_ahf
	# compute at. resolved : U grad2 ahf U
	def evaluate_at_resolved_U_grad2ahf_U(self, aa, nat, displ_structs):
		# compute grad2_ahf
		grad2_ahf = self.evaluate_at_resolved_grad2(aa, nat, displ_structs)
		# transf. matrix U
		U = self.struct_0.Deigv
		# U grad2 Ahf U
		U_g2Ahf_U = np.zeros((3*nat, 3*nat, 3, 3))
		# run over (iax,iby)
		for iax in range(3*nat):
			for iby in range(3*nat):
				gAhf = np.zeros((3,3))
				gAhf[0,0] = grad2_ahf[iax,iby,0]
				gAhf[1,1] = grad2_ahf[iax,iby,1]
				gAhf[2,2] = grad2_ahf[iax,iby,2]
				gAhf[0,1] = grad2_ahf[iax,iby,3]
				gAhf[1,0] = gAhf[0,1]
				gAhf[0,2] = grad2_ahf[iax,iby,4]
				gAhf[2,0] = gAhf[0,2]
				gAhf[1,2] = grad2_ahf[iax,iby,5]
				gAhf[2,1] = gAhf[1,2]
				# transf. new basis
				g2AhfU = np.matmul(gAhf, U)
				Ug2AhfU= np.matmul(U.transpose(), g2AhfU)
				# MHz / ang^2
				U_g2Ahf_U[iax,iby,:,:] = Ug2AhfU[:,:]
		#
		U_g2Ahf_U[:,:,:,:] = U_g2Ahf_U[:,:,:,:] * 1.E-6
		#
		# THz / ang^2 units
		#
		return U_g2Ahf_U
#
#  class 
#  gradient ZPL
#
class gradient_Eg:
	def __init__(self, atoms_info, dict_key):
		self.gradE = None
		self.forces = None
		self.force_const = None
		# read atoms info data
		try:
			f = open(atoms_info)
		except:
			msg = "\t COULD NOT FIND: " + atoms_info
			log.error(msg)
		self.atom_info_dict = yaml.load(f, Loader=yaml.Loader)
		f.close()
		# unpert dir
		self.unpert_dir = self.atom_info_dict[dict_key]['unpert_dir']
		self.unpert_dir = p.work_dir + '/' + self.unpert_dir
		self.hess_file  = self.atom_info_dict[dict_key]['hess_file']
		self.hess_file  = p.work_dir + '/' + self.hess_file
	# read outcar file
	def read_outcar(self, outcar):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "free" and l[1] == "energy" and l[2] == "TOTEN":
				E = float(l[4])   # eV
		return E
	# compute grad E
	def set_gradE(self, displ_structs, nat):
		dr = np.array([displ_structs.dx, displ_structs.dy, displ_structs.dz])
		# OUTCAR directory
		out_dir = displ_structs.outcars_dir
		# compute grad E
		gE = np.zeros(3*nat)
		jax = 0
		for ia in range(nat):
			for idx in range(3):
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir + '/' + file_name)
				E_1 = self.read_outcar(outcar)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir + '/' + file_name)
				E_2 = self.read_outcar(outcar)
				#
				gE[jax] = (E_1 - E_2) / (2.*dr[idx])
				#
				# eV / Ang units
				#
				jax = jax + 1
		self.gradE[:] = gE[:]
	# compute forces method
	def set_forces(self):
		# read forces
		F = self.struct_0.read_forces()
		# set up local force array
		nat = self.struct_0.nat
		self.forces = np.zeros(3*nat)
		# run over atoms
		for ia in range(nat):
			self.forces[3*ia]   = F[ia,0]
			self.forces[3*ia+1] = F[ia,1]
			self.forces[3*ia+2] = F[ia,2]
			# eV / ang 
			# units
	# extract force constants
	def set_force_constants(self):
		# read force constants
		Fc = self.struct_0.read_force_const(self.hess_file)
		# nat
		nat = self.struct_0.nat
		self.force_const = np.zeros((3*nat, 3*nat))
		# iterate over atomic index
		for jax in range(3*nat):
			ia = atoms.index_to_ia_map[jax]-1
			ix = atoms.index_to_idx_map[jax]
			for jby in range(3*nat):
				ib = atoms.index_to_ia_map[jby]-1
				iy = atoms.index_to_idx_map[jby]
				self.force_const[jax,jby] = Fc[ia,ib,ix,iy]
		# eV / ang^2
		# units
	# set HFI GS tensor
	def set_unpert_struct(self):
		self.struct_0 = UnpertStruct(self.unpert_dir)
		self.struct_0.read_poscar()
		# get energy
		self.struct_0.read_free_energy()