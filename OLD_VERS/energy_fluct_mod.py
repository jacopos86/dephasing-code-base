#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
from tqdm import tqdm
from pydephasing.phys_constants import hbar
from pydephasing.log import log
from pydephasing.input_parameters import p
from pydephasing.mpi import mpi
from pydephasing.atomic_list_struct import atoms
#
class spin_level_fluctuations_ofq:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params, homo=True, nconf=0):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph = 3*nat
        if homo:
            # if atom resolved
            if atom_res:
                self.deltaEq_atr = np.zeros((self.nt2,nat,input_params.ntmp))
            # if phonon resolved
            if ph_res:
                self.deltaEq_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
            # full auto correlation
            self.deltaEq_oft = np.zeros((self.nt,input_params.ntmp))
        else:
            # HFI calculation
            if atom_res:
                self.deltaEq_hfi_atr = np.zeros((self.nt2,nat))
            # phonon res.
            if ph_res:
                self.deltaEq_hfi_phm = np.zeros((self.nt2,self.nph))
            # full fluct.
            self.deltaEq_hfi_oft = np.zeros((self.nt,nconf))
    # compute energy fluctuations
    def compute_deltaEq_oft(self, atom_res, ph_res, qv, wuq, nat, D_ax, input_params, index_to_ia_map, atoms_dict, ph_ampl):
        #
        # compute deltaEq_oft
        #
        # iterate over ph. modes
        for im in range(self.nph):
            if wuq[im] > min_freq:
                for jax in range(3*nat):
                    ia = index_to_ia_map[jax] - 1
                    # direct atom coordinate
                    R0 = atoms_dict[ia]['coordinates']
                    # temporal part
                    # cos(wq t - q R0)
                    if ph_res or atom_res:
                        cos_wt2 = np.zeros(self.nt2)
                        cos_wt2[:] = np.cos(2*np.pi*wuq[im]*input_params.time2[:]-2*np.pi*np.dot(qv,R0))
                    cos_wt = np.zeros(self.nt)
                    cos_wt[:] = np.cos(2*np.pi*wuq[im]*input_params.time[:]-2*np.pi*np.dot(qv,R0))
                    # run over temperature list
                    for iT in range(input_params.ntmp):
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] += ph_ampl.u_ql_ja[im,jax,iT] * D_ax[jax] * cos_wt2[:]
                        if atom_res:
                            self.deltaEq_atr[:,ia,iT] += ph_ampl.u_ql_ja[im,jax,iT] * D_ax[jax] * cos_wt2[:]
                        self.deltaEq_oft[:,iT] += ph_ampl.u_ql_ja[im,jax,iT] * D_ax[jax] * cos_wt[:]
    # compute 2nd order fluctuations
    def compute_2ndorder_deltaEq_oft(self, atom_res, ph_res, qv, wuq, nat, D_ax, D_axby, input_params, index_to_ia_map, atoms_dict, ph_ampl):
        #
        # compute deltaEq_oft
        #
        eiqR = np.zeros(3*nat, dtype=np.complex128)
        for iax in range(3*nat):
            ia = index_to_ia_map[iax] - 1
            Ra = atoms_dict[ia]['coordinates']
            eiqR[iax] = np.exp(-1j*2.*np.pi*np.dot(qv, Ra))
        # iterate over ph. modes
        for im in range(self.nph):
            log.debug(str(im) + ' -> ' + str(self.nph))
            if wuq[im] > min_freq:
                # time part
                if atom_res or ph_res:
                    eiwt2 = np.zeros(self.nt2, dtype=np.complex128)
                    eiwt2[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time2[:])
                eiwt = np.zeros(self.nt, dtype=np.complex128)
                eiwt[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time[:])
                # run over iax
                for iax in range(3*nat):
                    # run over T
                    for iT in range(input_params.ntmp):
                        # compute energy fluct.
                        if atom_res:
                            ia = index_to_ia_map[iax]
                            self.deltaEq_atr[:,ia,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_ax[iax]
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_ax[iax]
                        self.deltaEq_oft[:,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt[:] * eiqR[iax]).real * D_ax[iax]
                    # run over jb/iy
                    for iby in range(iax, 3*nat):
                        for iT in range(input_params.ntmp):
                            # compute 2nd order fluct.
                            if atom_res:
                                if iax == iby:
                                    self.deltaEq_atr[:,ia,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                                else:
                                    self.deltaEq_atr[:,ia,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                            if ph_res:
                                if iax == iby:
                                    self.deltaEq_phm[:,im,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                                else:
                                    self.deltaEq_phm[:,im,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt2[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt2[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                            if iax == iby:
                                self.deltaEq_oft[:,iT] += 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * (eiwt[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
                            else:
                                self.deltaEq_oft[:,iT] += ph_ampl.u_ql_ja[im,iax,iT] * (eiwt[:] * eiqR[iax]).real * D_axby[iax,iby] * (eiwt[:] * eiqR[iby]).real * ph_ampl.u_ql_ja[im,iby,iT]
    # compute HFI energy fluctuations
    def compute_deltaEq_hfi_oft(self, atom_res, ph_res, qv, wuq, nat, spin_config, input_params, index_to_ia_map, atoms_dict, ph_ampl, icl):
        #
        # compute deltaEq_hfi(t)
        #
        # applied force (THz/Ang)
        F_ax = spin_config.Fax
        # iterate over ph. modes
        for im in range(self.nph):
            if wuq[im] > min_freq:
                for jax in range(3*nat):
                    ia = index_to_ia_map[jax] - 1
                    # direct atom coord.
                    R0 = atoms_dict[ia]['coordinates']
                    # temporal part
                    # cos(wq t - q R0)
                    if ph_res or atom_res:
                        cos_wt2 = np.zeros(self.nt2)
                        cos_wt2[:] = np.cos(2.*np.pi*wuq[im]*input_params.time2[:]-2.*np.pi*np.dot(qv,R0))
                    cos_wt = np.zeros(self.nt)
                    cos_wt[:] = np.cos(2.*np.pi*wuq[im]*input_params.time[:]-2.*np.pi*np.dot(qv,R0))
                    # ph res / at res
                    if ph_res:
                        self.deltaEq_hfi_phm[:,im] = self.deltaEq_hfi_phm[:,im] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt2[:]
                    if atom_res:
                        self.deltaEq_hfi_atr[:,ia] = self.deltaEq_hfi_atr[:,ia] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt2[:]
                    self.deltaEq_hfi_oft[:,icl] = self.deltaEq_hfi_oft[:,icl] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt[:]
                    # THz units
# compute final deltaE(t)
class spin_level_fluctuations:
    # initialization
    def __init__(self, nat, homo=True, nconf=0):
        if homo:
            # atom resolved calc.
            if p.at_resolved:
                self.deltaE_atr = np.zeros((p.nt2,nat,p.ntmp))
            # ph. resolved calc.
            if p.ph_resolved:
                if p.nphr == 0:
                    log.error("ph. resolved calculation must have > 0 ph. list")
                    sys.exit(1)
                self.deltaE_phm = np.zeros((p.nt2,p.nphr,p.ntmp))
            self.deltaE_oft = np.zeros((p.nt,p.ntmp))
        else:
            # HFI calculation
            if p.at_resolved:
                self.deltaE_hfi_atr = np.zeros((self.nt2,nat))
            if p.ph_resolved:
                self.nphr = len(p.phm_list)
                if self.nphr == 0:
                    log.error("ph. resolved calculation must have > 0 ph. list")
                    sys.exit(1)
                self.deltaE_hfi_phm = np.zeros((self.nt2,self.nphr))
            # time fluct.
            self.deltaE_hfi_oft = np.zeros((self.nt,nconf))
    # compute energy fluctuations
    def compute_deltaE_oft(self, nat, ph_ampl, D_ax):
        #
        # first : divide atoms in groups
        jax_list = mpi.split_data(3*nat)
        for jax in jax_list:
            for iT in range(p.ntmp):
                if p.at_resolved:
                    ia = atoms.index_to_ia_map[jax] - 1
                    self.deltaE_atr[:,ia,iT] += ph_ampl.u_ja_t[iT,jax,:p.nt2] * D_ax[jax]
                self.deltaE_oft[:,iT] += ph_ampl.u_ja_t[iT,jax,:] * D_ax[jax]
        # eV units
    def compute_deltaE_2nd_oft(self, nat, ph_ampl, D_ax, D_axby):
        # compute first + 2nd order
        # energy fluctuations
        #
        jax_list = mpi.split_data(3*nat)
        for jax in jax_list:
            # run over temperatures
            for iT in range(p.ntmp):
                if p.at_resolved:
                    ia = atoms.index_to_ia_map[jax] - 1
                    self.deltaE_atr[:,ia,iT] += ph_ampl.u_ja_t[iT,jax,:p.nt2] * D_ax[jax]
                self.deltaE_oft[:,iT] += ph_ampl.u_ja_t[iT,jax,:] * D_ax[jax]
            # second jby index
            for jby in range(jax, 3*nat):
                for iT in range(p.ntmp):
                    if p.at_resolved:
                        if jax == jby:
                            self.deltaE_atr[:,ia,iT] += 0.5 * ph_ampl.u_ja_t[iT,jax,:p.nt2] * D_axby[jax,jby] * ph_ampl.u_ja_t[iT,jby,:p.nt2]
                        else:
                            self.deltaE_atr[:,ia,iT] += ph_ampl.u_ja_t[iT,jax,:p.nt2] * D_axby[jax,jby] * ph_ampl.u_ja_t[iT,jby,:p.nt2]
                    if jax == jby:
                        self.deltaE_oft[:,iT] += 0.5 * ph_ampl.u_ja_t[iT,jax,:] * D_axby[jax,jby] * ph_ampl.u_ja_t[iT,jby,:]
                    else:
                        self.deltaE_oft[:,iT] += ph_ampl.u_ja_t[iT,jax,:] * D_axby[jax,jby] * ph_ampl.u_ja_t[iT,jby,:]
        # eV units
    def compute_deltaEphr_oft(self, nat, ph_ampl, D_ax, u, wu, nq, qpts, wq):
        # make ph modes list
        phr_list = mpi.split_list(p.phm_list)
        # run over list
        for iph in tqdm(range(len(phr_list))):
            # compute phonon amplitude
            u_jal_t = ph_ampl.ph_res_ampl(nat, phr_list[iph], u, wu, nq, qpts, wq)
            iphr = p.phm_list.index(phr_list[iph])
            # run over temperatures
            for iT in range(p.ntmp):
                for jax in range(3*nat):
                    self.deltaE_phm[:,iphr,iT] += u_jal_t[iT,jax,:] * D_ax[jax]
        # eV units
    def compute_deltaEphr_2nd_oft(self, nat, ph_ampl, D_ax, D_axby, u, wu, nq, qpts, wq):
        # make ph modes list
        phr_list = mpi.split_list(p.phm_list)
        # run over iph
        log.info("start phonon resolved calculation")
        for iph in tqdm(range(len(phr_list))):
            # compute ph. amplitude
            u_jal_t = ph_ampl.ph_res_ampl(nat, phr_list[iph], u, wu, nq, qpts, wq)
            iphr = p.phm_list.index(phr_list[iph])
            # run over temperatures
            for iT in range(p.ntmp):
                for jax in range(3*nat):
                    self.deltaE_phm[:,iphr,iT] += u_jal_t[iT,jax,:] * D_ax[jax]
                    for jby in range(3*nat):
                        self.deltaE_phm[:,iphr,iT] += u_jal_t[iT,jax,:] * D_axby[jax,jby] * u_jal_t[iT,jby,:]
        # eV units
    # HFI energy fluct.
    def compute_deltaE_hfi_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        if atom_res:
            self.deltaE_hfi_atr[:,:] = self.deltaE_hfi_atr[:,:] + wq * deltaEq.deltaEq_hfi_atr[:,:] * hbar
        if ph_res:
            self.deltaE_hfi_phm[:,:] = self.deltaE_hfi_phm[:,:] + wq * deltaEq.deltaEq_hfi_phm[:,:] * hbar
        # energy fluct.
        self.deltaE_hfi_oft[:,:] = self.deltaE_hfi_oft[:,:] + wq * deltaEq.deltaEq_hfi_oft[:,:] * hbar
        # eV units
    # collect dE(t) between processes
    def collect_deltaen_oft_between_proc(self, nat):
        if p.ph_resolved:
            for iT in range(p.ntmp):
                for iph in range(len(p.phm_list)):
                    f_oft = np.zeros(p.nt2)
                    f_oft[:] = self.deltaE_phm[:,iph,iT]
                    ff_oft = mpi.collect_time_array(f_oft)
                    self.deltaE_phm[:,iph,iT] = 0.
                    self.deltaE_phm[:,iph,iT] = ff_oft[:]
        if p.at_resolved:
            for iT in range(p.ntmp):
                for ia in range(nat):
                    f_oft = np.zeros(p.nt2)
                    f_oft[:] = self.deltaE_atr[:,ia,iT]
                    ff_oft = mpi.collect_time_array(f_oft)
                    self.deltaE_atr[:,ia,iT] = 0.
                    self.deltaE_atr[:,ia,iT] = ff_oft[:]
        for iT in range(p.ntmp):
            f_oft = np.zeros(p.nt)
            f_oft[:] = self.deltaE_oft[:,iT]
            ff_oft = mpi.collect_time_array(f_oft)
            self.deltaE_oft[:,iT] = 0.
            self.deltaE_oft[:,iT] = ff_oft[:]
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
            if wuq[im] > min_freq:
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
class spin_level_static_fluctuations:
    # initialization
    def __init__(self, nt):
        self.nt = nt
        # deltaE(t)
        self.deltaE_oft = np.zeros(self.nt)
    # compute energy fluctuations
    def compute_deltaE_oft(self, spins_config):
        # run over nuclear spins
        for isp in range(spins_config.nsp):
            # spin fluctuations
            dIt = spins_config.nuclear_spins[isp]['dIt']
            # forces (THz)
            F = spins_config.nuclear_spins[isp]['F']
            # run over time steps
            for t in range(self.nt):
                self.deltaE_oft[t] = self.deltaE_oft[t] + np.dot(F[:], dIt[:,t])
        # eV units
        self.deltaE_oft[:] = self.deltaE_oft[:] * hbar