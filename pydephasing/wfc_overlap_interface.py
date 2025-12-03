import re
import os
import numpy as np
import xml.etree.ElementTree as ET
from parallelization.mpi import mpi

def read_file(path_to_calc, filename):
    if os.path.isfile(os.path.join(path_to_calc, filename)):
        path2filename = os.path.join(path_to_calc,filename)
    else:
        raise ValueError(f"Unable to read {filename}")
    return path2filename

# 1st class: class to read 1 VASP calculation. Inside the directory, there is 1 vasprun.xml file, CONTCAR or POSCAR, and optionally WSWQ file
# With this class we can read an arbitrary VASP calculation.

class read_VASP_files:
    def __init__(self, path_to_calc):
        ''' Initialize class by reading vasprun.xml file:
        Inputs:
        - path_to_calc -> path/to/vasp/calculation
        The directory of calculation should contain vasprun.xml
        '''
        
        self.path_to_calc = path_to_calc
        vasprun = read_file(path_to_calc, "vasprun.xml")
        self.vasprun =  ET.parse(vasprun).getroot() # use xml library to read vasprun.xml

    def get_kpoints(self):
        '''
        Function to get number of K points and number of K points
        Return: kpoints (array), number of kpoints (int)
        '''
        nk_root = self.vasprun.find("kpoints//varray[@name='kpointlist']")
        kpoints = []
        for v in nk_root.findall("v"):
            kpoints.append([float(x) for x in v.text.split()])
        
        num_kpoints = len(kpoints)
        kpoints = np.array(kpoints)
        return kpoints, num_kpoints
    
    def get_nbands(self):
        '''
        Function to get number of KS-states or number of bands
        Return: number of bands (int)
        '''
        nbands_root = self.vasprun.find("parameters//separator[@name='electronic']/i[@name='NBANDS']")
        nbands = int(nbands_root.text.strip())
        return nbands
    
    def get_nspin(self):
        '''
        Function to get number of spin components
        Return: 1 or 2 (int)
        '''
        spin_root = self.vasprun.find("parameters//separator[@name='electronic spin']/i[@name='ISPIN']")
        nspin = int(spin_root.text.strip())
        return nspin

    def read_eigenvals(self):
        '''
        Read eigenvalues from vasprun.xml file of VASP

        Input:
        - vasprun_file: path to vasprun.xml file

       Output:
        - ar: array with eigenvalues with shape nkpoints, nspin, nbands, occupations
        '''
        vasprun = self.vasprun
        kpoints, num_kpoints = self.get_kpoints()
        nbands = self.get_nbands()
        nspin = self.get_nspin()
        list_spin = [1, 2] if nspin == 2 else [1]

        eigenvals = None
        for ik in range(1, num_kpoints+1):
            for ispin, stspin in enumerate(list_spin):
                eigenvals_root = vasprun.find("calculation/eigenvalues/array/set/set[@comment='spin %s']/set[@comment='kpoint %i']" % (stspin, ik))
                data = []
                occ = []
                for pair in eigenvals_root.findall("r"):
                    data.append(float(pair.text.split()[0]))
                    occ.append(float(pair.text.split()[1]))
                data = np.asarray(data)
                occ = np.asarray(occ)

                if (eigenvals is None):
                    eigenvals = np.zeros((num_kpoints, len(list_spin), nbands, 2), dtype=np.float64)
                eigenvals[ik-1, ispin, :, 0] = data
                eigenvals[ik-1, ispin, :, 1] = occ

        return eigenvals # shape: (kpoint, spin, bands, occupation)

    def read_poscar(self, filename):
        '''
		Read the cell and ionic positions from POSCAR or CONTCAR
		
        Input:
        - filename: "POSCAR" or "CONTCAR" file

        Output:
        - vecR: Lattice vector (in angstrom) in columns format
        - positions: List of atomic positions in crystal coordinates
        '''

        path2filename = read_file(self.path_to_calc, filename)

        with open(path2filename, 'r') as f:
            lines = f.readlines()
        vecR = np.asarray([[float(x) for x in line.split()] for line in lines[2:5]]).T
        atoms = lines[5].split()
        natoms = [int(x) for x in lines[6].split()]
        positions = []
        i = 8
        for atom, n in zip(atoms, natoms):
            for j in range(n):
                positions.append({"species": atom, "pos": np.asarray([float(x) for x in lines[i].split()])})
                i += 1

        return vecR, positions

    def read_wswq(self):
        '''
        Read WSWQ file from VASP calculation (overlap of wavefunctions) 

        Output:
        - wswq: numpy array containing wavefunction overlaps with shape (nspins, nkpoints, nbands, nbands)
        '''

        if not os.path.isfile(os.path.join(self.path_to_calc, "WSWQ")):
            raise ValueError(f"Unable to read WSWQ file in path {self.path_to_calc}")
        else:
            wswq_file = os.path.join(self.path_to_calc, "WSWQ")
            kpoints, num_kpoints = self.get_kpoints()
            nbands = self.get_nbands()
            nspin = self.get_nspin()

            wswq = np.zeros((nspin, num_kpoints, nbands, nbands), dtype=np.complex_)
            with open(wswq_file) as f:
                lines = f.readlines()
                for line in lines:
                    spin_match = re.search(r'\s*spin=(\d+),', str(line))
                    kpoint_match = re.search(r' kpoint=\s*(\d+)', str(line))
                    data_match = re.search(r'i=\s*(\d+), 'r'j=\s*(\d+)\s*:\s*([0-9\-.]+)\s+([0-9\-.]+)',str(line))
                    if spin_match and kpoint_match:
                        spin = int(spin_match.group(1))
                        kpoint = int(kpoint_match.group(1))
                    elif data_match:
                        band_i = int(data_match.group(1))
                        band_j = int(data_match.group(2))
                        overlap = complex(float(data_match.group(3)), float(data_match.group(4)))
                        wswq[spin-1, kpoint-1, band_i-1, band_j-1] = overlap
            return wswq    


# 2nd class: read a directory of multiple perturbations, each perturbation directory contains 1 WSWQ file and 1 vasprun.xml file

class VASP_wfc_overlap_:
    def __init__(self, perturbations_dir, header):
        self.perturbations_dir = perturbations_dir     # root directory. It stores all perturbation directories which contains each WSWQ file and vasprun.xml file
        self.header = header

    def read_displacements(self, f_disp):
        """
        Input:
        - f_disp: The phononpy_disp.yaml file
        
        Output:
        - result_displacement: [[atom, direction, dR],...]; 
            result_dispacement[imode] = [atom, direction, dR] 
            provide the information of mode=imode correspopnding to the 
            `atom`th atom's displacement at `direction` direction with displacement = `dR` angstrom
        """
        import yaml
        # Load a YAML file
        with open(f_disp, "r") as f:
            data = yaml.safe_load(f)  # safer than yaml.load

        print(f"Loading displacement from file {f_disp} ...")
        # Print all top-level keys
        print("All keys in file:")
        for key in data.keys():
            print(" -", key)
        print("reading displacement....")

        N_displacement = len(data["displacements"])
        print(f"found {N_displacement} modes...")
        result_displacement = []
        data["displacements"][0]
        for imode in range(N_displacement):
            disp_info = [0, 0, 0]
            atom = int(data["displacements"][imode]['atom']) #atom number
            vec_dR = data["displacements"][imode]['displacement']
            direction = ''
            if vec_dR[0] !=0:
                direction +='x'
                dR = vec_dR[0]
            if vec_dR[1] != 0:
                direction += 'y'
                dR = vec_dR[1]
            if vec_dR[2] != 0:
                direction += 'z'
                dR = vec_dR[2]
            if len(direction) != 1:
                print(f"WARNING, the direction '{direction}' for mode {imode} is not orthonormal, ")
            disp_info = [atom, direction, dR] #displacement info, include number of atom, direction, displacement
            result_displacement.append(disp_info)
        print("Loading done.")
        return result_displacement

    def read_all_wswq(self, result_displacement):
        """
        Get the list of WSWQ file paths for positive and negative displacements

        Input:
        - result_displacement: the displacement read from file

        Output:
        - N_mode: number of phonon modes
        - list_wswq_file_Rp: list of path for R=+dR (positive displacements)
        - list_wswq_file_Rm: list of path for R=-dR (negative displacements)
        """

        N_mode = int(len(result_displacement)/2)
        #Rp_wswq_list = [read_VASP_files(os.path.join(self.perturbations_dir,f"{self.header}{2*n+1:03d}")).read_wswq() for n in range(N_mode)]
        #Rm_wswq_list = [read_VASP_files(os.path.join(self.perturbations_dir,f"{self.header}{2*n+1+1:03d}")).read_wswq() for n in range(N_mode)]
        
        Rp_wswq_list = [os.path.join(self.perturbations_dir,f"{self.header}{2*n+1:03d}") for n in range(N_mode)]
        Rm_wswq_list = [os.path.join(self.perturbations_dir,f"{self.header}{2*n+1+1:03d}") for n in range(N_mode)]

        # -----  TO-DO: implement MPI parallelization here -----

        # # MPI parallelization to read WSWQ files
        # pathRp = [os.path.join(self.perturbations_dir,f"{self.header}{2*n+1:03d}") for n in range(N_mode)]
        # split_pathRp = mpi.random_split(pathRp) 
        # pathRp_local = split_pathRp[mpi.rank]

        # Rp_wswq_list = [read_VASP_files(path).read_wswq() for path in pathRp_local]
        # Rp_wswq_list = mpi.collect_list(Rp_wswq_list)

        # -------------------------------------------------

        return N_mode, Rp_wswq_list, Rm_wswq_list
