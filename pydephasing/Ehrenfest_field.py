
#
#   define Ehrenfest field class
#

class ehr_field(object):
    # initialization
    def __init__(self):
        self.nqpt = None
        self.nmd = None
    def initialize_Bfield(self, nqpt, nmd):
        self.nqpt = nqpt
        self.nmd = nmd
        # define displ. field