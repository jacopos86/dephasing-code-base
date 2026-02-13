import numpy as np
from pydephasing.common.phys_constants import hbar

#
#   define Ehrenfest field class
#

class ehr_field(object):
    # initialization
    def __init__(self):
        self.nqpt = None
        self.nmd = None
        self.Bq = None
        self.map_qtomq = None
        # time arrays
        self.nsnap = None
        self.Bq_t = None
        self.sum_Bqt = None
        # ph. drive local
        self._ph_drive = None
    def initialize_Bfield(self, nqpt, nmd):
        self.nqpt = nqpt
        self.nmd = nmd
        # define displ. field
        self.Bq = np.zeros((self.nqpt, self.nmd), 
            dtype=np.complex128
        )
    def get_PETSc_Bq(self):
        ''' Get Bq array '''
        B = PETSc.Mat().createDense(
            size=(self.nqpt, self.nmd),
            array=self.Bq,
            comm=PETSc.COMM_SELF
        )
        B.assemble()
        return B
    # phonon drive
    def set_drive(self, ph_drive):
        self._ph_drive = ph_drive
    def compute_dBq_dt(self, Bq, omega_q, gql, rho, t):
        dBq = np.zeros_like(Bq)
        if omega_q is not None:
            dBq = -1j / hbar * omega_q * Bq
        # --- add external drive if available ---
        if self._ph_drive is not None:
            Fq = self._ph_drive.set_force(t)   # shape (nq, nmd)
            dBq += -1j / hbar * Fq
        wk = np.ones(len(rho)) / len(rho)
        # iterate over (iq,il)
        iql = 0
        for iq in range(Bq.shape[0]):
            for im in range(Bq.shape[1]):
                gqp = gql[:,:,iql].conj()
                # iterate over k
                for ik in range(len(rho)):
                    dBq[iq,im] += -1j / hbar * wk[ik] * np.einsum(
                        'ab,ab->',
                        gqp,
                        rho[ik],
                        optimize=True
                    )
                iql += 1
        return dBq
    def init_td_arrays(self, nsteps, save_every):
        # number of snapshots to store
        self.nsnap = (nsteps + save_every -1) // save_every + 1
        # storage array
        self.Bq_t = np.zeros(
            (self.nqpt, self.nmd, self.nsnap),
            dtype=np.complex128
        )
        self.sum_Bqt = np.zeros(self.nsnap, dtype=np.complex128)
    def store_Bq_time(self, istep, Bq):
        self.Bq_t[:,:,istep] = Bq[:,:]
        self.sum_Bqt[istep] = np.sum(Bq)