import re
import numpy as np
import xml.etree.ElementTree as ET

class VASP_wfc_overlap_:
    def __init__(self, calc_dir):
        self.calc_dir = calc_dir     # calculation directory

    
    def read_wswq(self, wswq_file, vasprun_file):

        '''
        Read WSWQ file from VASP (overlap of wavefunctions) 
        -wswq_file: path/to/WSWQ file
        -vasprun_file: path/to/vasprun.xml file
        '''
        
        # We read the vasprun.xml file to get the number of spins, number of kpoints and number of bands
        t1 = ET.parse(vasprun_file).getroot() # use xml library to read files

        nk_root = t1.find("kpoints//varray[@name='kpointlist']")
        kpoints = []
        for v in nk_root.findall("v"):
            kpoints.append([float(x) for x in v.text.split()])
        nkpoints = len(kpoints)

        nbands_element = t1.find("parameters//separator[@name='electronic']/i[@name='NBANDS']")
        nbands = int(nbands_element.text.strip())

        spin_root = t1.find("parameters//separator[@name='electronic spin']/i[@name='ISPIN']")
        nspins = int(spin_root.text.strip())
        wswq = np.zeros((nspins, nkpoints, nbands, nbands), dtype=np.complex_)

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
    
    def read_eig_vasp(self, vasprun_file):
        '''
        Function for VASP
        Read eigenvalues from save folder
        :return: Array in ik, ispin, ib, 1/2 (for eigenvalues and occupations numbers)
        '''
        t1 = ET.parse(vasprun_file).getroot()
        nk_root = t1.find("kpoints//varray[@name='kpointlist']")
        kpoints = []
        for v in nk_root.findall("v"):
            kpoints.append([float(x) for x in v.text.split()])
        nk = len(kpoints)

        spin_root = t1.find("parameters//separator[@name='electronic spin']/i[@name='ISPIN']")
        list_spin = [1, 2] if int(spin_root.text.strip()) == 2 else [1]

        nbands_element = t1.find("parameters//separator[@name='electronic']/i[@name='NBANDS']")
        nbands = int(nbands_element.text.strip())

        ar = None
        for ik in range(1, nk+1):
            for ispin, stspin in enumerate(list_spin):
                eigenvals_root = t1.find("calculation/eigenvalues/array/set/set[@comment='spin %s']/set[@comment='kpoint %i']" % (stspin, ik))
                data = []
                occ = []
                for pair in eigenvals_root.findall("r"):
                    data.append(float(pair.text.split()[0]))
                    occ.append(float(pair.text.split()[1]))
                data = np.asarray(data)
                occ = np.asarray(occ)

                if (ar is None):
                    ar = np.zeros((nk, len(list_spin), nbands, 2), dtype=np.float64)
                ar[ik-1, ispin, :, 0] = data
                ar[ik-1, ispin, :, 1] = occ

        return ar # shape: (kpoint, spin, bands, occupation)

    def read_wswq_dir(self, root, result_displacement):
        """
        Root: Root with all the WSWQ calculations
        result_displacement: the displacement read from file
        OUTPUT: 
        list of path for R=+dR and R=-dR 
        """
        N_mode = int(len(result_displacement)/2)
        List_wswq_file_Rp = [os.path.join(root,f"disp-{2*n+1:03d}/WSWQ") for n in range(N_mode)]
        List_wswq_file_Rm = [os.path.join(root,f"disp-{2*n+1+1:03d}/WSWQ") for n in range(N_mode)]
        return N_mode, List_wswq_file_Rp, List_wswq_file_Rm

