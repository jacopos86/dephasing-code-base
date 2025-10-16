import numpy as np

#
#    compute electron density
#

class ElectronDensity:
    def __init__(self):
        self.rho_r = None
    def compute_elec_density_ofG(self, wfc):
        isp = 1
        ikpt = 1
        ibnd = 1
        gvec = wfc.set_gvectors(ikpt)
        cnk_G = wfc.read_cnk_ofG(isp, ikpt, ibnd, norm=True)
        print(cnk_G, np.linalg.norm(cnk_G))