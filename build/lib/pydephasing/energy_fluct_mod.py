#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
from pydephasing.phys_constants import hbar
from pydephasing.log import log
from pydephasing.input_parameters import p
#
# energy levels fluctuations
class energy_level_fluctuations_ofq:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2 = input_params.nt2
        self.nph = 3*nat
        # atom res. calc.
        if atom_res:
            self.deltaEq_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # ph. resolved
        if ph_res:
            self.deltaEq_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        self.deltaEq_oft = np.zeros((self.nt,input_params.ntmp))
    # compute energy fluctuations
    def compute_deltaEq_oft(self, atom_res, ph_res, qv, wuq, nat, Fax, Fc_axby, input_params, index_to_ia_map, atoms_dict, ph_ampl):
        #
        # compute deltaE(t)
        #
        eiqR = np.zeros(3*nat, dtype=np.complex128)
        for iax in range(3*nat):
            ia = index_to_ia_map[iax] - 1
            Ra = atoms_dict[ia]['coordinates']
            eiqR[iax] = np.exp(-1j*2.*np.pi*np.dot(qv, Ra))
        # iterate over ph. modes
        for im in range(self.nph):
            log.debug(str(im) + ' -> ' + str(self.nph))
            if wuq[im] > p.min_freq:
                # temporal part
                if atom_res or ph_res:
                    eiwt2 = np.zeros(self.nt2, dtype=np.complex128)
                    eiwt2[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time2[:])
                eiwt = np.zeros(self.nt, dtype=np.complex128)
                eiwt[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time[:])
                # run over atom index ia/ix
                for iax in range(3*nat):
                    # run over the temperature list
                    for iT in range(input_params.ntmp):
                        # compute energy fluct.
                        if atom_res:
                            self.deltaEq_atr[:,ia,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * Fax[iax]
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * Fax[iax]
                        self.deltaEq_oft[:,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt[:] * eiqR[iax]).real * Fax[iax]
                    # run over atom index ja/iy
                    for iby in range(3*nat):
                        # run over T list
                        for iT in range(input_params.ntmp):
                            # compute 2nd order fluct.
                            if atom_res:
                                self.deltaEq_atr[:,ia,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * Fc_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                            if ph_res:
                                self.deltaEq_phm[:,im,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * Fc_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                            self.deltaEq_oft[:,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt[:] * eiqR[iax]).real * Fc_axby[iax,iby] * (eiwt[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
        # eV units
        #
# total energy fluct. class
class energy_level_fluctuations:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        # atom res. calc.
        if atom_res:
            self.deltaE_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # ph. resolved
        if ph_res:
            self.deltaE_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        self.deltaE_oft = np.zeros((self.nt,input_params.ntmp))
    # compute final energy fluctuations
    def compute_deltaE_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        #
        if atom_res:
            self.deltaE_atr[:,:,:] += wq * deltaEq.deltaEq_atr[:,:,:]
        if ph_res:
            self.deltaE_phm[:,:,:] += wq * deltaEq.deltaEq_phm[:,:,:]
        # energy fluct.
        self.deltaE_oft[:,:] += wq * deltaEq.deltaEq_oft[:,:]
#
# static spin fluctuations class
class ZFS_ph_fluctuations:
    pass