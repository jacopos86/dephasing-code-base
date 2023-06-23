import yaml
import numpy as np
from pydephasing.log import log
from pydephasing.set_structs import UnpertStruct
from pydephasing.input_parameters import p
#
#  ground state structure for static HFI
#
class perturbation_HFI_stat:
    def __init__(self, out_dir, atoms_info, core):
        # out dir
        self.out_dir = out_dir
        # read atoms info data
        try:
            f = open(atoms_info)
        except:
            msg = "could not find: " + atoms_info
            log.error(msg)
        self.atom_info_dict = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # core correction
        self.core = core
        # GS data dir
        self.gs_data_dir = self.out_dir + '/' + self.atom_info_dict['(0,0)'][1]
    # set GS structure
    def set_gs_struct(self):
        self.struct_0 = UnpertStruct(self.gs_data_dir)
        self.struct_0.read_poscar()
        # ZFS tensor
        self.struct_0.read_zfs_tensor()
        # HFI tensor
        self.struct_0.set_hfi_Dbasis(self.core)
    # compute spin fluct. forces
    def compute_stat_force_HFS(self, Hss, config):
        # compute spin coeffs.
        DS  = Hss.set_DeltaS(p.qs1, p.qs2)
        # run over active spins
        for isp in range(config.nsp):
            site = config.nuclear_spins[isp]['site']
            # set HFI matrix (THz)
            Ahf = np.zeros((3,3))
            Ahf[:,:] = 2.*np.pi*self.struct_0.Ahfi[site-1,:,:]*1.E-6
            # force vector (THz)
            F = np.zeros(3)
            F = np.matmul(Ahf, DS)
            config.nuclear_spins[isp]['F'] = F