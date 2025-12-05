#
#   This module defines the displaced atomic
#   coordinates for the gradient calculation
#
import numpy as np
import os
from pathlib import Path
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Procar
import scipy.linalg as la
import h5py
import yaml
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.atomic_list_struct import atoms
#
#  atom dictionary class
#
class AtomListDict:
	def __init__(self):
		self.list_dict = None
		self.species_key = ''
		self.element_key = ''
		self.coord_xyz_key = ''
		self.coord_abc_key = ''
	def build_list_dict(self, atoms):
		self.list_dict = list(atoms)
		self.get_struct_keys()
	def get_struct_keys(self):
		log.debug("\t " + str(self.list_dict[0].keys()))
		keys = self.list_dict[0].keys()
		for k in keys:
			if k == "spec" or k == "specie" or k == "species":
				self.species_key = k
		if self.species_key == '':
			log.error("species key not found")
		spec_keys = self.list_dict[0][self.species_key][0].keys()
		for k in spec_keys:
			if k == "element" or k == "Element" or k == "symbol" or k == "Symbol":
				self.element_key = k
		if self.element_key == '':
			log.error("elements key not found")
		for k in keys:
			if k == "xyz" or k == "XYZ":
				self.coord_xyz_key = k
		if self.coord_xyz_key == '':
			log.error("xyz coord key not found")
		for k in keys:
			if k == "abc" or k == "ABC":
				self.coord_abc_key = k
		if self.coord_abc_key == '':
			log.error("abc coord key not found")
#
#  atomic struct dictionary class
#
class AtomicStructDict:
	def __init__(self):
		self.charge_key = ''
		self.atoms_key = ''
		self.lattice_key = ''
		self.unit_cell_key = ''
		self.dictionary = None
	def build_dict(self, struct):
		self.dictionary = struct.as_dict()
		self.get_struct_keys()
		self.get_lattice_keys()
		self.atoms_dictionary = AtomListDict()
		self.atoms_dictionary.build_list_dict(self.dictionary[self.atoms_key])
	def get_struct_keys(self):
		keys = list(self.dictionary.keys())
		for k in keys:
			if k == "atoms" or k == "Atoms" or k == "sites" or k == "Sites":
				self.atoms_key = k
		if self.atoms_key == '':
			log.error("atoms key not found in dict")
		for k in keys:
			if k == "lattice" or k == "Lattice" or k == "latt" or k == "Latt":
				self.lattice_key = k
		if self.lattice_key == '':
			log.error("lattice key not found in dict")
		for k in keys:
			if k == "charge" or k == "Charge":
				self.charge_key = k
		if self.charge_key == '':
			log.error("charge key not found in dict")
	def get_lattice_keys(self):
		keys = list(self.dictionary[self.lattice_key])
		log.debug("\t lattice keys :" + str(keys))
		for k in keys:
			if k == "matrix":
				self.unit_cell_key = k
		if self.unit_cell_key == '':
			log.error("unit cell key not found")
#
#  ground state structure
#  acquires ground state data
#
class UnpertStruct:
	def __init__(self, ipath):
		self.ipath = ipath + "/"
		# electronic parameters
		self.nkpt= None
		self.nbnd= None
		self.eigv = None
		self.occ = None
		self.nup = None
		self.ndw = None
		self.S = None
		self.spin_multiplicity = None
		self.has_soc = None
		self.spinor_wfc = None
		self.orbital_character = None
		self.orbitals = None
	### read POSCAR file
	def read_poscar(self):
		poscar = Poscar.from_file("{}".format(self.ipath+"POSCAR"), read_velocities=False)
		struct = poscar.structure
		self.struct = struct
		# set number of atoms
		struct_dict = AtomicStructDict()
		struct_dict.build_dict(struct)
		atoms_key = struct_dict.atoms_key
		self.nat = len(list(struct_dict.dictionary[atoms_key]))
		assert self.nat == atoms.nat
		log.debug("\t n. atoms: " + str(self.nat))
	### extract energy eigenvalues + occup.
	def extract_energy_eigv(self):
		file_name = "vasprun.xml"
		file_name = "{}".format(self.ipath+file_name)
		fil = Path(file_name)
		if fil.exists():
			self.extract_energy_eigv_from_vasprun()
			self.read_k_points_weights()
		else:
			file_name = "OUTCAR"
			file_name = "{}".format(self.ipath+file_name)
			fil = Path(file_name)
			if fil.exists():
				self.extract_energy_eigv_from_OUTCAR()
			else:
				log.error("data file not found in: " + str(self.ipath))
	def extract_energy_eigv_from_vasprun(self):
		file_name = "{}".format(self.ipath+"vasprun.xml")
		vasprun = Vasprun(file_name)
		bs = vasprun.get_band_structure()
		# num. bands / kpts
		self.nbnd = bs.nb_bands
		self.nkpt = len(bs.kpoints)
		if not bs.is_spin_polarized:
			nspin = 1
		else:
			nspin = 2
		if mpi.rank == mpi.root:
			log.info("\t n. kpts: " + str(self.nkpt))
			log.info("\t n. bands: " + str(self.nbnd))
			log.info("\t n. spin: " + str(nspin))
		# set eigv.
		bands = bs.bands
		self.eigv = np.zeros((self.nbnd, self.nkpt, nspin))
		self.occ  = np.zeros((self.nbnd, self.nkpt, nspin))
		for band_index, band_energies in enumerate(bands[Spin.up]):
			for k_index, energy in enumerate(band_energies):
				if mpi.rank == mpi.root:
					log.debug(f"Band {band_index}, k-point {k_index}, energy = {energy:.3f} eV")
				self.eigv[band_index, k_index, 0] = energy
		# occupations
		eig = vasprun.eigenvalues[Spin.up]
		for k_index in range(self.nkpt):
			for band_index in range(self.nbnd):
				self.occ[band_index, k_index, 0] = eig[k_index][band_index][1]
		if nspin == 2:
			for band_index, band_energies in enumerate(bands[Spin.down]):
				for k_index, energy in enumerate(band_energies):
					if mpi.rank == mpi.root:
						log.debug(f"Band {band_index}, k-point {k_index}, energy = {energy:.3f} eV")
					self.eigv[band_index, k_index, 1] = energy
			eig = vasprun.eigenvalues[Spin.down]
			for k_index in range(self.nkpt):
				for band_index in range(self.nbnd):
					self.occ[band_index, k_index, 1] = eig[k_index][band_index][1]
		nel = vasprun.parameters["NELECT"]
		if mpi.rank == mpi.root:
			log.info("\t number of electrons: " + str(nel))
		self.has_soc = vasprun.parameters["LSORBIT"]
		self.spinor_wfc = vasprun.parameters["LNONCOLLINEAR"]
		if mpi.rank == mpi.root:
			log.info("\t SOC: " + str(self.has_soc))
			log.info("\t NON COLLINEARITY: " + str(self.spinor_wfc))
	def extract_energy_eigv_from_OUTCAR(self):
		if mpi.rank == mpi.root:
			log.info("\n")
			log.info("\t " + p.sep)
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		nspin = 1
		for i in range(len(lines)):
			line = lines[i].strip().split()
			if len(line) > 2:
				if line[0] == "k-points":
					if line[1] == "NKPTS":
						self.nkpt = int(line[3])
					if line[-2] == "NBANDS=":
						self.nbnd = int(line[-1])
				if line[0] == "ISPIN":
					nspin = int(line[2])
				if line[0] == "LNONCOLLINEAR":
					if line[2] == ".TRUE.":
						self.spinor_wfc = True
					else:
						self.spinor_wfc = False
				if line[0] == "LSORBIT":
					if line[2] == ".TRUE.":
						self.has_soc = True
					else:
						self.has_soc = False
		if mpi.rank == mpi.root:
			log.info("\t n. kpts: " + str(self.nkpt))
			log.info("\t n. bands: " + str(self.nbnd))
			log.info("\t n. spin: " + str(nspin))
			log.info("\t SOC: " + str(self.has_soc))
			log.info("\t NON COLLINEARITY: " + str(self.spinor_wfc))
		self.eigv = np.zeros((self.nbnd, self.nkpt, nspin))
		self.occ  = np.zeros((self.nbnd, self.nkpt, nspin))
		for i in range(len(lines)):
			line = lines[i].strip().split()
			if len(line) > 2:
				if line[0] == "spin" and line[1] == "component":
					spi = int(line[2])-1
					ki = int(lines[i+2].strip().split()[1])-1
					i0 = i+4
					for j in range(i0, i0+self.nbnd):
						line2 = lines[j].strip().split()
						if len(line2) == 3:
							bi = int(line2[0])-1
							self.eigv[bi,ki,spi] = float(line2[1])
							self.occ[bi,ki,spi]  = float(line2[2])
	### read k points weights
	def read_k_points_weights(self):
		file_name = "{}".format(self.ipath+"vasprun.xml")
		vasprun = Vasprun(file_name)
		self.wk = vasprun.actual_kpoints_weights
		assert(len(self.wk) == self.nkpt)
	### extract orbital character
	### it tries to read PROCAR file
	### if file not found just return none
	def extract_orbital_character(self):
		file_name = "{}".format(self.ipath+"PROCAR")
		fil = Path(file_name)
		# ---------- CASE 1: file does not exist ----------------
		if not fil.exists():
			if mpi.rank == mpi.root:
				log.warning(f"PROCAR does not exist at {file_name}")
			return
		procar = None
		# ---------- CASE 2: Try parsing with pymatgen ----------
		try:
			if mpi.rank == mpi.root:
				log.info(f"\t Trying pymatgen Procar loader for {file_name}")
			procar = Procar(file_name)
			loading_method = "pymatgen"
		except Exception as exc:
			# ---------- CASE 3: fallback parsing ---------------
			if mpi.rank == mpi.root:
				log.warning(f"pymatgen Procar parsing failed: {exc}")
				log.warning("→ Trying fallback parser instead...")
			try:
				procar = self._fallback_parse_procar(file_name)
				loading_method = "fallback"
			except Exception as exc2:
				if mpi.rank == mpi.root:
					log.error(f"Fallback parser also failed: {exc2}")
				return
		# ---------- store parsed data ----------
		if loading_method == "pymatgen":
			# Total number of k-points, bands, and ions
			# spin indexed arrays
			if mpi.rank == mpi.root:
				log.info("\t " + p.sep)
				log.info(f"\t K-points: {procar.nkpoints}, Bands: {procar.nbands}, Ions: {procar.nions}")
				log.info("\t " + p.sep)
			self.orbital_character = procar.data
			self.orbitals = procar.orbitals
			assert(self.orbital_character[Spin.up].shape[0] == self.nkpt)
			assert(self.orbital_character[Spin.up].shape[1] == self.nbnd)
			assert(self.orbital_character[Spin.up].shape[3] == len(self.orbitals))
		else:
			# fallback parser delivers looser structure → no assertions
			self.orbital_character = procar["bands"]
			self.orbitals = procar["orbitals"]
			if mpi.rank == mpi.root:
				log.warning(
					"Fallback PROCAR interpretation loaded — shapes not validated!"
				)
	def _fallback_parse_procar(self, file_name):
		"""
		Minimal tolerant parse: returns dict-like data
		enough to inspect orbital contributions if pymatgen fails.
		"""
		bands = []
		orbitals = []
		# --- find orbital header line ---
		with open(file_name, "r") as f:
			for line in f:
				strip = line.strip()
				if strip.startswith("ion") and "tot" in strip:
					cols = strip.split()
					# expected form:
					# ion  s  py  pz  px  dxy  dyz  dz2  dxz  x2-y2  tot
					# remove ion and tot
					try:
						ion_index = cols.index("ion")
					except ValueError:
						ion_index = 0
					try:
						tot_index = cols.index("tot")
					except ValueError:
						tot_index = len(cols)
					orbitals = cols[ion_index + 1 : tot_index]
					break
		# dummy orbital labels — user can improve as needed
		# this ensures self.orbitals exists
		# so code that indexes orbitals doesn't break
		if not orbitals:
			orbitals = [f"orb{i}" for i in range(10)]
		# open file
		with open(file_name, "r") as f:
			block = []
			for line in f:
				line = line.strip()
				if line.startswith("band"):
					if block:
						bands.append(block)
					block = [line]
				elif block:
					block.append(line)
		if block:
			bands.append(block)
		return {"bands": bands, "orbitals": orbitals}
	### read OUTCAR
	### extract spin state
	def extract_spin_state(self):
		self.extract_energy_eigv()
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			line = lines[i].strip().split()
			if len(line) > 4:
				if line[0] == "Total" and line[1] == "magnetic" and line[2] == "moment":
					mag_mom = float(line[4])
					if mpi.rank == mpi.root:
						log.info("\t total magnetic moment: " + str(mag_mom))
		# compute occup. diff.
		self.nup = 0.
		self.ndw = 0.
		for ki in range(self.nkpt):
			for bi in range(self.nbnd):
				self.nup += 1./self.nkpt * self.occ[bi,ki,0]
				self.ndw += 1./self.nkpt * self.occ[bi,ki,1]
		if mpi.rank == mpi.root:
			log.info("\t n. up states = " + str(self.nup))
			log.info("\t n. dw states = " + str(self.ndw))
		self.S = 0.5 * np.abs(self.nup - self.ndw)
		assert int(2.*self.S) == int(round(mag_mom))
		self.spin_multiplicity = 2.*self.S + 1
		if mpi.rank == mpi.root:
			log.info("\t SPIN MULTIPLICITY : " + str(self.spin_multiplicity))
			log.info("\t " + p.sep)
			log.info("\n")
	### read ZFS from OUTCAR
	def read_zfs(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 3 and l[0] == "D_diag" and l[1] == "eigenvector":
				j = i+2
				l2 = lines[j].split()
				D1 = float(l2[0])     # MHz
				#
				l2 = lines[j+1].split()
				D2 = float(l2[0])     # MHz
				#
				l2 = lines[j+2].split()
				D3 = float(l2[0])     # MHz
		f.close()
		# set D tensor
		self.Ddiag = np.array([D1, D2, D3])
		# MHz
	### read full ZFS tensor
	def read_zfs_tensor(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "D_xx" and l[5] == "D_yz":
				j = i+2
				l2 = lines[j].split()
				# D tensor (MHz)
				Dxx = float(l2[0])
				Dyy = float(l2[1])
				Dzz = float(l2[2])
				Dxy = float(l2[3])
				Dxz = float(l2[4])
				Dyz = float(l2[5])
		f.close()
		self.Dtensor = np.zeros((3,3))
		self.Dtensor[0,0] = Dxx
		self.Dtensor[1,1] = Dyy
		self.Dtensor[2,2] = Dzz
		self.Dtensor[0,1] = Dxy
		self.Dtensor[1,0] = Dxy
		self.Dtensor[0,2] = Dxz
		self.Dtensor[2,0] = Dxz
		self.Dtensor[1,2] = Dyz
		self.Dtensor[2,1] = Dyz
		# set diagonalization
		# basis vectors
		[eig, eigv] = la.eig(self.Dtensor)
		eig = eig.real
		# set D spin quant. axis coordinates
		self.read_zfs()
		self.Deigv = np.zeros((3,3))
		for i in range(3):
			d = self.Ddiag[i]
			for j in range(3):
				if abs(d-eig[j]) < 1.E-2:
					self.Ddiag[i] = eig[j]
					self.Deigv[:,i] = eigv[:,j]
		# D units -> MHz
	### read HFI dipolar from OUTCAR
	def read_hfi_dipolar(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 7 and l[0] == "ion" and l[6] == "A_yz":
				j = i+2
				A = np.zeros((self.nat,6))
				for k in range(j, j+self.nat):
					l2 = lines[k].split()
					for u in range(6):
						A[k-j,u] = float(l2[1+u])
		f.close()
		return A
	# read full HFI tensor from OUTCAR
	def read_hfi_full(self, core=True):
		A_full = self.read_hfi_dipolar()
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			# read fermi contact term
			if len(l) == 6 and l[0] == "ion" and l[5] == "A_tot":
				A_fc = 0.
				j = i+2
				for k in range(j, j+self.nat):
					l2 = lines[k].split()
					if core:
						A_fc = float(l2[4]) + float(l2[5])
					else:
						A_fc = float(l2[5])
					A_full[k-j,0] = A_full[k-j,0] + A_fc
					A_full[k-j,1] = A_full[k-j,1] + A_fc
					A_full[k-j,2] = A_full[k-j,2] + A_fc
		f.close()
		return A_full
	### set HFI in D diag. basis
	def set_hfi_Dbasis(self, core=True):
		self.Ahfi = np.zeros((self.nat,3,3))
		# set transf. matrix U
		U = self.Deigv
		# read Ahfi from outcar
		ahf = self.read_hfi_full(core)
		# run over atoms
		for aa in range(self.nat):
			A = np.zeros((3,3))
			# set HF tensor
			A[0,0] = ahf[aa,0]
			A[1,1] = ahf[aa,1]
			A[2,2] = ahf[aa,2]
			A[0,1] = ahf[aa,3]
			A[1,0] = A[0,1]
			A[0,2] = ahf[aa,4]
			A[2,0] = A[0,2]
			A[1,2] = ahf[aa,5]
			A[2,1] = A[1,2]
			#
			AU = np.matmul(A, U)
			r = np.matmul(U.transpose(), AU)
			self.Ahfi[aa,:,:] = r[:,:]
			# MHz units
	### read free energy from OUTCAR
	def read_free_energy(self):
		# read file
		f = open(self.ipath+"OUTCAR")
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "free" and l[1] == "energy" and l[2] == "TOTEN":
				E = float(l[4])      # eV
		f.close()
		self.E = E
	### read forces from outcar
	def read_forces(self):
		FF = np.zeros((self.nat,3))
		# read file
		f = open(self.ipath+"OUTCAR")
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 3 and l[0] == "POSITION" and l[1] == "TOTAL-FORCE":
				j = i+2
				for k in range(j, j+self.nat):
					l2 = lines[k].split()
					F = np.zeros(3)
					# set force
					F[0] = float(l2[3])
					F[1] = float(l2[4])
					F[2] = float(l2[5])
					# eV / Ang
					FF[k-j,:] = F[:]
		f.close()
		return FF
	### read force constants from HDF5 file
	def read_force_const(self, inp_file):
		with h5py.File(inp_file, 'r') as f:
			# list all groups
			unit_key = list(f.keys())[2]
			units = str(list(f[unit_key])[0])
			if mpi.rank == mpi.root:
				log.info("\n")
				log.info("\t " + p.sep)
				log.info("\t FORCE CONSTANTS UNITS : " + units)
				log.info("\t " + p.sep)
				log.info("\n")
			if units != "b'eV/angstrom^2'":
				log.error("\t WRONG FORCE CONSTANTS UNITS")
				raise Exception("WRONG FORCE CONSTANTS UNITS")
			# extract force constants
			key = list(f.keys())[1]
			p2s_map = list(f[key])
			# force const.
			key_fc = list(f.keys())[0]
			Fc = np.array(f[key_fc])
		return Fc
#
#  displaced structures class
#  prepare the VASP displacement calculation
#
class DisplacedStructs:
	def __init__(self, out_dir, outcars_dir=''):
		self.out_dir = out_dir + '/'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		self.outcars_dir = outcars_dir
	# set atomic displacement (angstrom)
	def atom_displ(self, dr=np.array([0.1,0.1,0.1])):
		self.dx = dr[0]
		self.dy = dr[1]
		self.dz = dr[2]
		if mpi.rank == mpi.root:
			# write data on file
			file_name = self.out_dir + "displ.yml"
			isExist = os.path.exists(file_name)
			if not isExist:
				data = {'displ_ang' : list(dr)}
				with open(file_name, 'w') as outfile:
					yaml.dump(data, outfile)
	# generate new displaced structures
	def build_atom_displ_structs(self, struct_unprt, max_d_from_defect, defect_index):
		# struct -> unperturbed atomic structure
		struct_dict = AtomicStructDict()
		struct_dict.build_dict(struct_unprt.struct)
		# unit cell -> angstrom units
		lattice_key = struct_dict.lattice_key
		unit_cell_key = struct_dict.unit_cell_key
		self.unit_cell = np.array(struct_dict.dictionary[lattice_key][unit_cell_key])
		# charge
		charge_key = struct_dict.charge_key
		self.charge = struct_dict.dictionary[charge_key]
		# set species list
		self.species = []
		species_key = struct_dict.atoms_dictionary.species_key
		element_key = struct_dict.atoms_dictionary.element_key
		atoms = struct_dict.atoms_dictionary.list_dict
		for ia in range(struct_unprt.nat):
			self.species.append(atoms[ia][species_key][0][element_key])
		log.debug("\t " + str(self.species))
		# extract atomic cartesian coordinates
		coord_xyz_keys = struct_dict.atoms_dictionary.coord_xyz_key
		coord_abc_keys = struct_dict.atoms_dictionary.coord_abc_key
		atoms_xyz_coords = np.zeros((struct_unprt.nat, 3))
		atoms_abc_coords = np.zeros((struct_unprt.nat, 3))
		for ia in range(struct_unprt.nat):
			abc_coord_ia = atoms[ia][coord_abc_keys]
			atoms_abc_coords[ia,:] = abc_coord_ia[:]
			xyz_coord_ia = atoms[ia][coord_xyz_keys]
			atoms_xyz_coords[ia,:] = xyz_coord_ia[:]
		# build perturbed structures
		displ_struct_list = []
		for ia in range(struct_unprt.nat):
			# distance atom - defect
			dd = struct_unprt.struct.get_distance(ia, defect_index)
			if dd <= max_d_from_defect:
				# x - displ 1
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,0] = atoms_xyz_coords[ia,0] + self.dx
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '1', '1', struct])
				# x - displ 2
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,0] = atoms_xyz_coords[ia,0] - self.dx
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '1', '2', struct])
				# y - displ 1
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,1] = atoms_xyz_coords[ia,1] + self.dy
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '2', '1', struct])
				# y - displ 2
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,1] = atoms_xyz_coords[ia,1] - self.dy
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '2', '2', struct])
				# z - displ 1
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,2] = atoms_xyz_coords[ia,2] + self.dz
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '3', '1', struct])
				# z - displ 2
				atoms_xyz_displ = np.zeros((struct_unprt.nat,3))
				atoms_xyz_displ[:,:] = atoms_xyz_coords[:,:]
				atoms_xyz_displ[ia,2] = atoms_xyz_coords[ia,2] - self.dz
				struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_xyz_displ,
					charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
				displ_struct_list.append([str(ia+1), '3', '2', struct])
		# set up dictionary
		self.displ_structs = []
		keys = ['atom', 'x', 'sign', 'structure']
		for displ_struct in displ_struct_list:
			self.displ_structs.append(dict(zip(keys, displ_struct)))
	# write structures on file
	def write_structs_on_file(self, significant_figures=16, direct=True, vasp4_compatible=False):
		for displ_struct in self.displ_structs:
			struct = displ_struct['structure']
			at_index = displ_struct['atom']
			x = displ_struct['x']
			sgn = displ_struct['sign']
			# prepare file
			file_name = "POSCAR-" + at_index + "-" + x + "-" + sgn
			poscar = Poscar(struct)
			poscar.write_file(filename="{}".format(self.out_dir+file_name), direct=direct,
				vasp4_compatible=vasp4_compatible, significant_figures=significant_figures)
#
# 2nd order displaced structure
# class
#
class DisplacedStructures2ndOrder:
	def __init__(self, out_dir, outcars_dir=''):
		self.out_dir = out_dir + '/'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		self.outcars_dir = outcars_dir
		self.max_dab = np.inf
	def set_max_atoms_distance(self, max_d_ab):
		self.max_dab = max_d_ab
	# set atomic displacement (angstrom)
	def atom_displ(self, dr=np.array([0.1, 0.1, 0.1])):
		self.dx = dr[0]
		self.dy = dr[1]
		self.dz = dr[2]
		self.dr = [self.dx, self.dy, self.dz]
		# write data on file
		if mpi.rank == mpi.root:
			file_name = self.out_dir + "displ.yml"
			isExist = os.path.exists(file_name)
			if not isExist:
				data = {'displ_ang' : list(dr)}
				with open(file_name, 'w') as out_file:
					yaml.dump(data, out_file)
	# build atoms displ. structures
	def build_atom_displ_structs(self, struct_unprt, max_d_from_defect, defect_index):
		# unpert. atomic structure
		struct_dict = struct_unprt.struct.as_dict()
		atoms_key = list(struct_dict.keys())[4]
		lattice_key = list(struct_dict.keys())[3]
		charge_key = list(struct_dict.keys())[2]
		unit_cell_key = list(struct_dict[lattice_key].keys())[0]
		# unit cell -> angstrom units
		self.unit_cell = np.array(struct_dict[lattice_key][unit_cell_key])
		# charge
		self.charge = struct_dict[charge_key]
		# list of atoms dictionary
		atoms = list(struct_dict[atoms_key])
		# set species list
		self.species = []
		species_key = list(atoms[0].keys())[0]
		element_key = list(atoms[0][species_key][0].keys())[0]
		for ia in range(struct_unprt.nat):
			self.species.append(atoms[ia][species_key][0][element_key])
		# extract atomic cartesian coordinates
		coord_xyz_keys = list(atoms[0].keys())[2]
		atoms_cart_coords = np.zeros((struct_unprt.nat,3))
		coord_abc_keys = list(atoms[0].keys())[1]
		atoms_abc_coords = np.zeros((struct_unprt.nat,3))
		for ia in range(struct_unprt.nat):
			abc_coord_ia = atoms[ia][coord_abc_keys]
			atoms_abc_coords[ia,:] = abc_coord_ia[:]
			xyz_coord_ia = atoms[ia][coord_xyz_keys]
			atoms_cart_coords[ia,:] = xyz_coord_ia[:]
		# build perturbed structures
		displ_struct_list = []
		# distance to vacancy
		for ia in range(struct_unprt.nat):
			# distance atom - defect
			dda = struct_unprt.struct.get_distance(ia, defect_index)
			if dda <= max_d_from_defect:
				for ib in range(struct_unprt.nat):
					ddb = struct_unprt.struct.get_distance(ib, defect_index)
					if ddb <= max_d_from_defect:
						dab = struct_unprt.struct.get_distance(ia, ib)
						if dab <= self.max_dab:
							for ix in range(3):
								for iy in range(ix,3):
									# compute atomic structure
									atoms_cart_displ = np.zeros((struct_unprt.nat,3))
									atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
									if ia == ib:
										if ix == iy:
											atoms_cart_displ[ia,ix] = atoms_cart_coords[ia,ix] + 2.*self.dr[ix]
										else:
											atoms_cart_displ[ia,ix] = atoms_cart_coords[ia,ix] + self.dr[ix]
											atoms_cart_displ[ia,iy] = atoms_cart_coords[ia,iy] + self.dr[iy]
									else:
										atoms_cart_displ[ia,ix] = atoms_cart_coords[ia,ix] + self.dr[ix]
										atoms_cart_displ[ib,iy] = atoms_cart_coords[ib,iy] + self.dr[iy]
									# set new structure
									struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
										charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
									displ_struct_list.append([str(ia+1), str(ix+1), str(ib+1), str(iy+1), struct])
		# set up dictionary
		self.displ_structs = []
		keys = ['ia', 'ix', 'ib', 'iy', 'structure']
		for displ_struct in displ_struct_list:
			self.displ_structs.append(dict(zip(keys, displ_struct)))
	# write structures on file
	def write_structs_on_file(self, significant_figures=16, direct=True, vasp4_compatible=False):
		summary = []
		for displ_struct in self.displ_structs:
			struct = displ_struct['structure']
			ia = displ_struct['ia']
			ix = displ_struct['ix']
			ib = displ_struct['ib']
			iy = displ_struct['iy']
			# prepare file
			file_name = "POSCAR-" + str(ia) + "-" + str(ix) + "-" + str(ib) + "-" + str(iy)
			poscar = Poscar(struct)
			poscar.write_file(filename="{}".format(self.out_dir+file_name), direct=direct,
				vasp4_compatible=vasp4_compatible, significant_figures=significant_figures)
			# write summary on file
			summary.append(str(ia) + '-' + str(ix) + '-' + str(ib) + '-' + str(iy))
		# set summary file
		file_name = self.out_dir + "summary.yml"
		data = {'calc_list' : summary}
		with open(file_name, 'w') as outfile:
			yaml.dump(data, outfile)

#
#   JDFTx struct
#

class JDFTxStruct:
	def __init__(self, KPOINTS_FILE, EIGENV_FILE, OUTPUT_FILE):
		# electronic parameters
		self.nkpt = None
		self.Kpts = None
		self.nbnd = None
		self.eigv = None
		self.occ = None
		# variables as in QE : 2 for SOC
		self.npol = None
		# nsp_index : 2 for spin pol. 
		# 1 for SOC -> no spin index in energies / wfc
		self.nsp_index = None
		self.spintype = None
		# dft bands
		self.Eks = None
		self.mu = None
		self.nelec = None
		# external file
		self.OUT_FILE = OUTPUT_FILE
		self.KPTS_FILE = KPOINTS_FILE
		self.EIG_FILE = EIGENV_FILE
	def read_Kpts(self):
		self.Kpts = np.loadtxt(self.KPTS_FILE, skiprows=2, usecols=(1,2,3))
		self.nkpt = self.Kpts.shape[0]
		if mpi.rank == mpi.root:
			log.info("\t " + p.sep)
			log.info("\t n. K pts: " + str(self.nkpt))
	def read_nspin(self):
		if mpi.rank == mpi.root:
			log.info("\t EXTRACT N. SPIN FROM -> " + self.OUT_FILE)
		with open(self.OUT_FILE, "r") as fil:
			for line in fil:
				line = line.strip().split()
				if len(line) > 0:
					if line[0] == "spintype":
						self.spintype = line[1]
						if line[1] == "no-spin":
							''' spin unpolarized '''
							self.npol = 1
							self.nsp_index = 1
						elif line[1] == "spin-orbit":
							''' spin orbit with no magnetization '''
							self.npol = 2
							self.nsp_index = 1
						elif line[1] == "vector-spin":
							''' non collinear magnetism '''
							self.npol = 2
							self.nsp_index = 1
						elif line[1] == "z-spin":
							''' spin polarized calculation '''
							self.npol = 1
							self.nsp_index = 2
						else:
							log.error("spintype flag not recognized in " + self.OUT_FILE)
		if mpi.rank == mpi.root:
			log.info("\t n. spin indexes: " + str(self.nsp_index))
			log.info("\t spin polarization: " + str(self.npol))
			log.info("\t " + p.sep)
	def read_nbands(self):
		if mpi.rank == mpi.root:
			log.info("\t EXTRACT N. BANDS FROM -> " + self.OUT_FILE)
		with open(self.OUT_FILE, "r") as fil:
			for line in fil:
				line = line.strip().split()
				if len(line) > 0:
					if line[0] == "elec-n-bands":
						self.nbnd = int(line[1])
		if mpi.rank == mpi.root:
			log.info("\t " + p.sep)
			log.info("\t n. electronic bands: " + str(self.nbnd))
	def set_chem_pot(self):
		self.mu = np.nan
		initDone = False
		for line in open(self.OUT_FILE):
			if line.startswith('Initialization completed'):
				initDone = True
			if initDone and line.find('FillingsUpdate:')>=0:
				self.mu = float(line.split()[2])
			if (not initDone) and line.startswith('nElectrons:'):
				self.nelec = float(line.split()[1])
				if self.spintype in ("no-spin", "z-spin"):
					nval = int(self.nelec/2)               # valence bands
					self.mu = np.max(self.Eks[:,:,:nval])
				else:
					nval = int(self.nelec)
					self.mu = np.max(self.Eks[0,:,:nval])  # VBM
		if mpi.rank == mpi.root:
			log.info("\t " + p.sep)
			log.info("\t num. electrons: " + str(self.nelec))
			log.info("\t chemical potential: " + str(self.mu))
			log.info("\t " + p.sep)
	def read_band_struct(self):
		Edft = np.fromfile(self.EIG_FILE).reshape(self.nsp_index,self.nkpt,-1)
		assert Edft.shape[-1] == self.nbnd
		self.Eks = Edft
	def set_elec_parameters(self):
		# read K points
		self.read_Kpts()
		# set n. spins / bands
		self.read_nbands()
		self.read_nspin()
		# read electronic band structure
		if mpi.rank == mpi.root:
			log.info("\t EXTRACT BAND STRUCTURE from -> " + self.EIG_FILE)
		self.read_band_struct()
		# get mu/VBM from totalE.out
		self.set_chem_pot()