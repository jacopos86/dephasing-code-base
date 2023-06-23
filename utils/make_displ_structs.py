# this subroutine creates the input files
# to be run in the VASP calculation
from shutil import copyfile
import os
import yaml
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
import scipy.linalg as la
#
#  ground state structure
#  acquires ground state data
#
class UnpertStruct:
	def __init__(self, ipath):
		self.ipath = ipath + "/"
	### read POSCAR file
	def read_poscar(self):
		poscar = Poscar.from_file("{}".format(self.ipath+"POSCAR"), read_velocities=False)
		struct = poscar.structure
		self.struct = struct
		# set number of atoms
		struct_dict = struct.as_dict()
		atoms_key = list(struct_dict.keys())[4]
		self.nat = len(list(struct_dict[atoms_key]))
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
			log.info("force const. units: " + units)
			if units != "b'eV/angstrom^2'":
				log.error("wrong force constants units")
				raise Exception("wrong force constants units")
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
        coord_abc_keys = list(atoms[0].keys())[1]
        atoms_cart_coords = np.zeros((struct_unprt.nat, 3))
        for ia in range(struct_unprt.nat):
            cart_coord_ia = atoms[ia][coord_xyz_keys]
            atoms_cart_coords[ia,:] = cart_coord_ia[:]
        # build perturbed structures
        displ_struct_list = []
        for ia in range(struct_unprt.nat):
            # compute distance between atom and defect
            dd = struct_unprt.struct.get_distance(ia, defect_index)
            if dd <= max_d_from_defect:
                # x - displ 1
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,0] = atoms_cart_coords[ia,0] + self.dx
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
                    charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
                displ_struct_list.append([str(ia+1), '1', '1', struct])
                # x - displ 2
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,0] = atoms_cart_coords[ia,0] - self.dx
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
                    charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
                displ_struct_list.append([str(ia+1), '1', '2', struct])
                # y - displ 1
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,1] = atoms_cart_coords[ia,1] + self.dy
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
                    charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
                displ_struct_list.append([str(ia+1), '2', '1', struct])
                # y - displ 2
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,1] = atoms_cart_coords[ia,1] - self.dy
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
                    charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
                displ_struct_list.append([str(ia+1), '2', '2', struct])
                # z - displ 1
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,2] = atoms_cart_coords[ia,2] + self.dz
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
                    charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
                displ_struct_list.append([str(ia+1), '3', '1', struct])
                # z - displ 2
                atoms_cart_displ = np.zeros((struct_unprt.nat,3))
                atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
                atoms_cart_displ[ia,2] = atoms_cart_coords[ia,2] - self.dz
                struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
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
    def __init__(self, out_dir, max_d_ab, outcars_dir=''):
        self.out_dir = out_dir + '/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.outcars_dir = outcars_dir
        self.max_dab = max_d_ab
    # set atomic displacement (angstrom)
    def atom_displ(self, dr=np.array([0.1, 0.1, 0.1])):
        self.dx = dr[0]
        self.dy = dr[1]
        self.dz = dr[2]
        self.dr = [self.dx, self.dy, self.dz]
        # write data on file
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
        # first find distance to vacancy
        for ia in range(struct_unprt.nat):
            # compute distance between atom and defect
            dda = struct_unprt.struct.get_distance(ia, defect_index)
            if dda <= max_d_from_defect:
                for ib in range(struct_unprt.nat):
                    ddb = struct_unprt.struct.get_distance(ib, defect_index)
                    if ddb <= max_d_from_defect:
                        dab = struct_unprt.struct.get_distance(ia, ib)
                        if dab <= self.max_dab:
                            for ix in range(3):
                                for iy in range(ix,3):
                                    #
                                    # compute atomic structure
                                    #
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
def gen_poscars(unpert_dir, displ_poscar_dir, atoms_displ, displ_outcar_dir, copy_files_dir, max_dist_from_defect, defect_index):
    # struct 0
    struct0 = UnpertStruct(unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define all the displced atoms structures
    for i in range(len(displ_poscar_dir)):
        displ_struct = DisplacedStructs(displ_poscar_dir[i])
        # set atoms displacements
        displ_struct.atom_displ(atoms_displ[i])      # Ang
        # build displaced atomic structures
        displ_struct.build_atom_displ_structs(struct0, max_dist_from_defect, defect_index)
        # write data on file
        displ_struct.write_structs_on_file()
        # delete structure object
        del displ_struct
    # create calculation directory
    nat = struct0.nat
    for i in range(len(displ_outcar_dir)):
        if not os.path.exists(displ_outcar_dir[i]):
            os.mkdir(displ_outcar_dir[i])
        # run over all calculations
        for ia in range(struct0.nat):
            # compute distance between atom and defect
            dd = struct0.struct.get_distance(ia, defect_index)
            if dd <= max_dist_from_defect:
                for idx in range(3):      # x-y-z index
                    for s in range(2):    # +/- index
                        namef = displ_poscar_dir[i] + "/POSCAR-" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                        named = displ_outcar_dir[i] + "/" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                        if not os.path.exists(named):
                            os.mkdir(named)
                        # copy files in new directory
                        files = os.listdir(copy_files_dir)
                        for fil in files:
                            copyfile(copy_files_dir + "/" + fil, named + "/" + fil)
                        copyfile(namef, named + "/POSCAR")
                        print(str(ia+1) + " - " + str(idx+1) + " - " + str(s+1) + " -> completed   ")
#
# generate 2nd order displ
# POSCAR
#
def gen_2ndorder_poscar(unpert_dir, displ_poscar_dir, atoms_displ, displ_outcar_dir, copy_files_dir, max_dist_from_defect, defect_index, max_d_ab):
    struct0 = UnpertStruct(unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define displ. atomic structures
    for i in range(len(displ_poscar_dir)):
        # init structures
        displ_struct = DisplacedStructures2ndOrder(displ_poscar_dir[i], max_d_ab)
        # set atoms displacement
        displ_struct.atom_displ(atoms_displ[i])   # ang
        # build displ. atoms structures
        displ_struct.build_atom_displ_structs(struct0, max_dist_from_defect, defect_index)
        # write data on file
        displ_struct.write_structs_on_file()
        # check if directory exists
        if not os.path.exists(displ_outcar_dir[i]):
            os.mkdir(displ_outcar_dir[i])
        # read summary files
        input_file = displ_poscar_dir[i] + "/summary.yml"
        try:
            f = open(input_file)
        except:
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        dirlist = data['calc_list']
        # run over all calculations
        for named in dirlist:
            # check named exists
            out_dir = displ_outcar_dir[i]
            if not os.path.exists(out_dir + "/" + named):
                os.mkdir(out_dir + "/" + named)
            # copy files in new directory
            files = os.listdir(copy_files_dir)
            for fil in files:
                copyfile(copy_files_dir + "/" + fil, out_dir + "/" + named + "/" + fil)
            namef = displ_poscar_dir[i] + "/POSCAR-" + named
            copyfile(namef, out_dir + "/" + named + "/POSCAR")
            print("   " + named + " -> completed   ")
#
unpert_dir = "/home/jacopo/Documents/dephasing-code-project/examples/CV-defect"
displ_poscar_dir = ["/home/jacopo/Documents/dephasing-code-project/examples/CV-defect/POSCAR-FILES"]
atoms_displ = [np.array([0.001, 0.001, 0.001])]
displ_outcar_dir = ["/home/jacopo/Documents/dephasing-code-project/examples/CV-defect"]
copy_files_dir = "/home/jacopo/Documents/dephasing-code-project/examples/CV-defect/COPY-FOLDER"
max_dist_from_defect = 4.0  # in ang
defect_index = 1            # start from 0
max_d_ab = 2.0
#
gen_poscars(unpert_dir, displ_poscar_dir, atoms_displ, displ_outcar_dir, copy_files_dir, max_dist_from_defect, defect_index)
gen_2ndorder_poscar(unpert_dir, displ_poscar_dir, atoms_displ, displ_outcar_dir, copy_files_dir, max_dist_from_defect, defect_index, max_d_ab)