

#
#   define electron density matrix 

class elec_dmatr(object):
    # initialization
    def __init__(self, smearing, Te):
        self.nbnd = None
        self.nkpt = None
        self.nspin = None
        # set smearing
        self.smearing = smearing
        # elec. temperature
        self.Te = Te
        # temperature must be in eV
        print(self.smearing, self.Te)