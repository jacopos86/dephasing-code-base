import numpy as np
from pydephasing.mpi import mpi
from pydephasing.phys_constants import hbar
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.input_parameters import p
from pydephasing.log import log
from pydephasing.input_parser import parser
import yaml
# T2 inverse class
class T2i_class(object):
    # T2i is in ps^-1
    def __init__(self):
        self.T2_sec = None
        self.T2s_avg = None
        self.T2s_atr = None
        self.T2s_atr_avg = None
        self.T2s_phr = None
        self.T2s_phr_avg = None
        self.T2s_wql = None
        self.T2s_wql_avg = None
    def generate_instance(self, nat, nconf=None):
        if not p.deph and not p.relax:
            if p.dyndec:
                return T2i_inhom_dd()
            else:
                return T2i_inhom()
        else:
            return T2i_ofT().generate_instance(nat, nconf)
# T2i abstract
class T2i_ofT(T2i_class):
    def __init__(self):
        super(T2i_ofT, self).__init__()
    def generate_instance(self, nat, nconf=None):
        calc_type2 = parser.parse_args().ct2
        if calc_type2 == "homo":
            return T2i_homo_ofT (nat)
        elif calc_type2 == "inhomo" or calc_type2 == "full":
            return T2i_inhom_ofT (nat, nconf)
        else:
            if mpi.rank == mpi.root:
                log.error("\t ct2 PARAMETER NOT RECOGNIZED ...")
    def get_T2_sec(self):
        return self.T2_sec
    def get_T2_atr_sec(self):
        return self.T2s_atr
    def get_T2_phr_sec(self):
        return self.T2s_phr
    def get_T2_wql_sec(self):
        return self.T2s_wql
# concrete T2i -> time resolved
# calculation
class T2i_homo_ofT(T2i_ofT):
    def __init__(self, nat):
        super(T2i_homo_ofT, self).__init__()
        self.T2_sec = np.zeros(p.ntmp)
        if p.at_resolved:
            self.T2s_atr = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.T2s_phr = np.zeros((p.nphr,p.ntmp))
            self.T2s_wql = np.zeros((p.nwbn,p.ntmp))
    # compute deph/relax time
    def set_T2_sec(self, iT, T2i):
        self.T2_sec[iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_atr(self, ia, iT, T2i):
        self.T2s_atr[ia,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_phr(self, iph, iT, T2i):
        self.T2s_phr[iph,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_wql(self, iwb, iT, T2i):
        self.T2s_wql[iwb,iT] = 1./T2i * 1.E-12
        # sec
    def collect_atr_from_other_proc(self, iT):
        T2s_atr_full = mpi.collect_array(self.T2s_atr[:,iT])
        self.T2s_atr[:,iT] = 0.
        self.T2s_atr[:,iT] = T2s_atr_full[:]
    def collect_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            T2s_phr_full = mpi.collect_array(self.T2s_phr[:,iT])
            self.T2s_phr[:,iT] = 0.
            self.T2s_phr[:,iT] = T2s_phr_full[:]
        # wql
        T2s_wql_full = mpi.collect_array(self.T2s_wql[:,iT])
        self.T2s_wql[:,iT] = 0.
        self.T2s_wql[:,iT] = T2s_wql_full[:]
# concrete T2i -> time resolved 
# calculation
class T2i_inhom_ofT(T2i_ofT):
    def __init__(self, nat, nconf):
        super(T2i_inhom_ofT, self).__init__()
        self.T2_sec = np.zeros((nconf,p.ntmp))
        self.T2s_avg= np.zeros(p.ntmp)
        if p.at_resolved:
            self.T2s_atr = np.zeros((nat,nconf,p.ntmp))
            self.T2s_atr_avg = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.T2s_phr = np.zeros((p.nphr,nconf,p.ntmp))
                self.T2s_phr_avg = np.zeros((p.nphr,p.ntmp))
            self.T2s_wql = np.zeros((p.nwbn,nconf,p.ntmp))
            self.T2s_wql_avg = np.zeros((p.nwbn,p.ntmp))
    # compute deph/relax time
    def set_T2_sec(self, ic, iT, T2i):
        self.T2_sec[ic,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_avg(self, iT, T2i):
        self.T2s_avg[iT] = 1./T2i * 1.E-12
    def set_T2_atr(self, ia, ic, iT, T2i):
        self.T2s_atr[ia,ic,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_atr_avg(self, ia, iT, T2i):
        self.T2s_atr_avg[ia,iT] = 1./T2i * 1.E-12
    def set_T2_phr(self, iph, ic, iT, T2i):
        self.T2s_phr[iph,ic,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_phr_avg(self, iph, iT, T2i):
        self.T2s_phr_avg[iph,iT] = 1./T2i * 1.E-12
    def set_T2_wql(self, iwb, ic, iT, T2i):
        self.T2s_wql[iwb,ic,iT] = 1./T2i * 1.E-12
        # sec
    def set_T2_wql_avg(self, iwb, iT, T2i):
        self.T2s_wql_avg[iwb,iT] = 1./T2i * 1.E-12
    def collect_atr_from_other_proc(self, ic, iT):
        T2s_atr_full = mpi.collect_array(self.T2s_atr[:,ic,iT])
        self.T2s_atr[:,ic,iT] = 0.
        self.T2s_atr[:,ic,iT] = T2s_atr_full[:]
    def collect_avg_atr_from_other_proc(self, iT):
        T2s_atr_full = mpi.collect_array(self.T2s_atr_avg[:,iT])
        self.T2s_atr_avg[:,iT] = 0.
        self.T2s_atr_avg[:,iT] = T2s_atr_full[:]
    def collect_phr_from_other_proc(self, ic, iT):
        if p.nphr > 0:
            T2s_phr_full = mpi.collect_array(self.T2s_phr[:,ic,iT])
            self.T2s_phr[:,ic,iT] = 0.
            self.T2s_phr[:,ic,iT] = T2s_phr_full[:]
        # wql
        T2s_wql_full = mpi.collect_array(self.T2s_wql[:,ic,iT])
        self.T2s_wql[:,ic,iT] = 0.
        self.T2s_wql[:,ic,iT] = T2s_wql_full[:]
    def collect_avg_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            T2s_phr_full = mpi.collect_array(self.T2s_phr_avg[:,iT])
            self.T2s_phr_avg[:,iT] = 0.
            self.T2s_phr_avg[:,iT] = T2s_phr_full[:]
        # wql
        T2s_wql_full = mpi.collect_array(self.T2s_wql_avg[:,iT])
        self.T2s_wql_avg[:,iT] = 0.
        self.T2s_wql_avg[:,iT] = T2s_wql_full[:]
    def get_T2_avg(self):
        return self.T2s_avg
    def get_T2_atr_avg(self):
        return self.T2s_atr_avg
    def get_T2_phr_avg(self):
        return self.T2s_phr_avg
    def get_T2_wql_avg(self):
        return self.T2s_wql_avg
#
# -> T2i dd class
class T2i_inhom_dd(T2i_class):
    # T2i is in ps^-1
    def __init__(self, nconf):
        super(T2i_inhom_dd, self).__init__()
        npl = len(p.n_pulses)
        self.T2_psec = np.zeros((npl,nconf+1))
    def get_T2_psec(self):
        return self.T2_psec
    def get_T2_musec(self):
        return self.T2_psec * 1.E-6
    def set_T2_psec(self, ipl, ic, T2i):
        self.T2_psec[ipl,ic] = 1./T2i
        # psec units
    def collect_from_other_proc(self, ipl):
        T2s_full = mpi.collect_array(self.T2_sec[ipl,:])
        self.T2_sec[ipl,:] = 0.
        self.T2_sec[ipl,:] = T2s_full[:]
# T2 inverse class
class T2i_inhom(T2i_class):
    # T2i is in ps^-1
    def __init__(self, nconf):
        super(T2i_inhom, self).__init__()
        self.T2_psec = np.zeros(nconf+1)
    def get_T2_psec(self):
        return self.T2_psec
    def get_T2_musec(self):
        return self.T2_psec * 1.E-6
    def set_T2_psec(self, ic, T2i):
        self.T2_psec[ic] = 1./T2i
        # psec units
    def collect_from_other_proc(self):
        T2ps_full = mpi.collect_array(self.T2_psec)
        self.T2_psec[:] = 0.
        self.T2_psec[:] = T2ps_full[:]
# --------------------------------------------------
#
#       Delta class
# --------------------------------------------------
class Delta_class(object):
    # Delta is in eV
    # acf(t=0) -> only time resolved calcuations
    def __init__(self):
        self.Delt = None
        self.Delt_avg = None
        self.Delt_atr = None
        self.Delt_atr_avg = None
        self.Delt_phr = None
        self.Delt_phr_avg = None
        self.Delt_wql = None
        self.Delt_wql_avg = None
    def generate_instance(self, nat, nconf=None):
        if not p.deph and not p.relax:
            if p.dyndec:
                return Delta_inhom_dd()
            else:
                return Delta_inhom()
        else:
            if p.time_resolved:
                return Delta_ofT().generate_instance(nat, nconf)
            else:
                return None
    def get_Delt(self):
        return self.Delt
        # eV
class Delta_ofT(Delta_class):
    def __init__(self):
        super(Delta_ofT, self).__init__()
    def generate_instance(self, nat, nconf=None):
        calc_type2 = parser.parse_args().ct2
        if calc_type2 == "homo":
            return Delta_homo_ofT(nat)
        elif calc_type2 == "inhomo" or calc_type2 == "full":
            return Delta_inhom_ofT(nat, nconf)
        else:
            if mpi.rank == mpi.root:
                log.error("\t WRONG calc_type2 VALUE")
    def get_Delt_atr(self):
        return self.Delt_atr
    def get_Delt_phr(self):
        return self.Delt_phr
    def get_Delt_wql(self):
        return self.Delt_wql
class Delta_homo_ofT(Delta_ofT):
    def __init__(self, nat):
        super(Delta_homo_ofT, self).__init__()
        self.Delt = np.zeros(p.ntmp)
        if p.at_resolved:
            self.Delt_atr = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.Delt_phr = np.zeros((p.nphr,p.ntmp))
            self.Delt_wql = np.zeros((p.nwbn,p.ntmp))
    def set_Delt(self, iT, D2):
        self.Delt[iT] = np.sqrt(D2)
        # eV units
    def set_Delt_atr(self, ia, iT, D2):
        self.Delt_atr[ia,iT] = np.sqrt(D2)
    def set_Delt_phr(self, iph, iT, D2):
        self.Delt_phr[iph,iT] = np.sqrt(D2)
    def set_Delt_wql(self, iwb, iT, D2):
        self.Delt_wql[iwb,iT] = np.sqrt(D2)
    def collect_atr_from_other_proc(self, iT):
        Delt_atr_full = mpi.collect_array(self.Delt_atr[:,iT])
        self.Delt_atr[:,iT] = 0.
        self.Delt_atr[:,iT] = Delt_atr_full[:]
    def collect_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            Delt_phr_full = mpi.collect_array(self.Delt_phr[:,iT])
            self.Delt_phr[:,iT] = 0.
            self.Delt_phr[:,iT] = Delt_phr_full[:]
        # wql
        Delt_wql_full = mpi.collect_array(self.Delt_wql[:,iT])
        self.Delt_wql[:,iT] = 0.
        self.Delt_wql[:,iT] = Delt_wql_full[:]
class Delta_inhom_ofT(Delta_ofT):
    def __init__(self, nat, nconf):
        super(Delta_inhom_ofT, self).__init__()
        self.Delt = np.zeros((nconf,p.ntmp))
        self.Delt_avg = np.zeros(p.ntmp)
        if p.at_resolved:
            self.Delt_atr = np.zeros((nat,nconf,p.ntmp))
            self.Delt_atr_avg = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.Delt_phr = np.zeros((p.nphr,nconf,p.ntmp))
                self.Delt_phr_avg = np.zeros((p.nphr,p.ntmp))
            self.Delt_wql = np.zeros((p.nwbn,nconf,p.ntmp))
            self.Delt_wql_avg = np.zeros((p.nwbn,p.ntmp))
    def set_Delt(self, ic, iT, D2):
        self.Delt[ic,iT] = np.sqrt(D2)
        # eV units
    def set_Delt_avg(self, iT, D2):
        self.Delt_avg[iT] = np.sqrt(D2)
    def set_Delt_atr(self, ia, ic, iT, D2):
        self.Delt_atr[ia,ic,iT] = np.sqrt(D2)
    def set_Delt_atr_avg(self, ia, iT, D2):
        self.Delt_atr_avg[ia,iT] = np.sqrt(D2)
    def set_Delt_phr(self, iph, ic, iT, D2):
        self.Delt_phr[iph,ic,iT] = np.sqrt(D2)
    def set_Delt_phr_avg(self, iph, iT, D2):
        self.Delt_phr_avg[iph,iT] = np.sqrt(D2)
    def set_Delt_wql(self, iwb, ic, iT, D2):
        self.Delt_wql[iwb,ic,iT] = np.sqrt(D2)
    def set_Delt_wql_avg(self, iwb, iT, D2):
        self.Delt_wql_avg[iwb,iT] = np.sqrt(D2)
    def collect_atr_from_other_proc(self, ic, iT):
        Delt_atr_full = mpi.collect_array(self.Delt_atr[:,ic,iT])
        self.Delt_atr[:,ic,iT] = 0.
        self.Delt_atr[:,ic,iT] = Delt_atr_full[:]
    def collect_avg_atr_from_other_proc(self, iT):
        Delt_atr_full = mpi.collect_array(self.Delt_atr_avg[:,iT])
        self.Delt_atr_avg[:,iT] = 0.
        self.Delt_atr_avg[:,iT] = Delt_atr_full[:]
    def collect_phr_from_other_proc(self, ic, iT):
        if p.nphr > 0:
            Delt_phr_full = mpi.collect_array(self.Delt_phr[:,ic,iT])
            self.Delt_phr[:,ic,iT] = 0.
            self.Delt_phr[:,ic,iT] = Delt_phr_full[:]
        # wql
        Delt_wql_full = mpi.collect_array(self.Delt_wql[:,ic,iT])
        self.Delt_wql[:,ic,iT] = 0.
        self.Delt_wql[:,ic,iT] = Delt_wql_full[:]
    def collect_avg_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            Delt_phr_full = mpi.collect_array(self.Delt_phr_avg[:,iT])
            self.Delt_phr_avg[:,iT] = 0.
            self.Delt_phr_avg[:,iT] = Delt_phr_full[:]
        # wql
        Delt_wql_full = mpi.collect_array(self.Delt_wql_avg[:,iT])
        self.Delt_wql_avg[:,iT] = 0.
        self.Delt_wql_avg[:,iT] = Delt_wql_full[:]
    def get_Delt_avg(self):
        return self.Delt_avg
    def get_Delt_atr_avg(self):
        return self.Delt_atr_avg
    def get_Delt_phr_avg(self):
        return self.Delt_phr_avg
    def get_Delt_wql_avg(self):
        return self.Delt_wql_avg
class Delta_inhom_dd(Delta_class):
    # Delta is in eV
    def __init__(self, nconf):
        super(Delta_inhom_dd, self).__init__()
        npl = len(p.n_pulses)
        self.Delt = np.zeros((npl,nconf+1))
    def set_Delt(self, ipl, ic, D2):
        self.Delt[ipl,ic] = np.sqrt(D2)
        # eV units
    def collect_from_other_proc(self, ic):
        Delt_full = mpi.collect_array(self.Delt[:,ic])
        self.Delt[:,ic] = 0.
        self.Delt[:,ic] = Delt_full[:]
# Delta inhom class
class Delta_inhom(Delta_class):
    # Delta is in eV
    def __init__(self, nconf):
        super(Delta_inhom, self).__init__()
        self.Delt = np.zeros(nconf+1)
    def set_Delt(self, ic, D2):
        self.Delt[ic] = np.sqrt(D2)
        # eV units
    def collect_from_other_proc(self):
        Delt_full = mpi.collect_array(self.Delt)
        self.Delt[:] = 0.
        self.Delt[:] = Delt_full[:]
        # eV
# ---------------------------------------------------
# 
#         tauc class
# ---------------------------------------------------
class tauc_class(object):
    # -> only time resolved calculation 
    # tauc is in ps
    def __init__(self):
        self.tauc_ps = None
        self.tauc_avg= None
        self.tauc_atr = None
        self.tauc_atr_avg = None
        self.tauc_phr = None
        self.tauc_phr_avg = None
        self.tauc_wql = None
        self.tauc_wql_avg = None
    def generate_instance(self, nat, nconf=None):
        if not p.deph and not p.relax:
            if p.dyndec:
                return tauc_inhom_dd()
            else:
                return tauc_inhom()
        else:
            if p.time_resolved:
                return tauc_ofT().generate_instance(nat, nconf)
            else:
                return None
    def get_tauc(self):
        return self.tauc_ps
class tauc_ofT(tauc_class):
    def __init__(self):
        super(tauc_ofT, self).__init__()
    def generate_instance(self, nat, nconf=None):
        calc_type2 = parser.parse_args().ct2
        if calc_type2 == "homo":
            return tauc_homo_ofT(nat)
        elif calc_type2 == "inhomo" or calc_type2 == "full":
            return tauc_inhom_ofT(nat, nconf)
        else:
            if mpi.rank == mpi.root:
                log.error("\t WRONG calc_type2 VALUE")
    def get_tauc_atr(self):
        return self.tauc_atr
    def get_tauc_phr(self):
        return self.tauc_phr
    def get_tauc_wql(self):
        return self.tauc_wql
class tauc_homo_ofT(tauc_ofT):
    def __init__(self, nat):
        super(tauc_homo_ofT, self).__init__()
        self.tauc_ps = np.zeros(p.ntmp)
        if p.at_resolved:
            self.tauc_atr = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.tauc_phr = np.zeros((p.nphr,p.ntmp))
            self.tauc_wql = np.zeros((p.nwbn,p.ntmp))
    def set_tauc(self, iT, tau_c):
        self.tauc_ps[iT] = tau_c
        # ps units
    def set_tauc_atr(self, ia, iT, tau_c):
        self.tauc_atr[ia,iT] = tau_c
        # ps units
    def set_tauc_phr(self, iph, iT, tau_c):
        self.tauc_phr[iph,iT] = tau_c
        # ps units
    def set_tauc_wql(self, iwb, iT, tau_c):
        self.tauc_wql[iwb,iT] = tau_c
        # ps units
    def collect_atr_from_other_proc(self, iT):
        tauc_atr_full = mpi.collect_array(self.tauc_atr[:,iT])
        self.tauc_atr[:,iT] = 0.
        self.tauc_atr[:,iT] = tauc_atr_full[:]
    def collect_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            tauc_phr_full = mpi.collect_array(self.tauc_phr[:,iT])
            self.tauc_phr[:,iT] = 0.
            self.tauc_phr[:,iT] = tauc_phr_full[:]
        # wql
        tauc_wql_full = mpi.collect_array(self.tauc_wql[:,iT])
        self.tauc_wql[:,iT] = 0.
        self.tauc_wql[:,iT] = tauc_wql_full[:]
# tauc class inhom (T)
class tauc_inhom_ofT(tauc_ofT):
    # tauc in ps
    def __init__(self, nat, nconf):
        self.tauc_ps = np.zeros((nconf,p.ntmp))
        self.tauc_avg= np.zeros(p.ntmp)
        if p.at_resolved:
            self.tauc_atr = np.zeros((nat,nconf,p.ntmp))
            self.tauc_atr_avg = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.tauc_phr = np.zeros((p.nphr,nconf,p.ntmp))
                self.tauc_phr_avg = np.zeros((p.nphr,p.ntmp))
            self.tauc_wql = np.zeros((p.nwbn,nconf+1,p.ntmp))
            self.tauc_wql_avg = np.zeros((p.nwbn,p.ntmp))
    def set_tauc(self, ic, iT, tau_c):
        self.tauc_ps[ic,iT] = tau_c
        # ps units
    def set_tauc_avg(self, iT, tau_c):
        self.tauc_avg[iT] = tau_c
    def set_tauc_atr(self, ia, ic, iT, tau_c):
        self.tauc_atr[ia,ic,iT] = tau_c
        # ps units
    def set_tauc_atr_avg(self, ia, iT, tau_c):
        self.tauc_atr_avg[ia,iT] = tau_c
    def set_tauc_phr(self, iph, ic, iT, tau_c):
        self.tauc_phr[iph,ic,iT] = tau_c
        # ps units
    def set_tauc_phr_avg(self, iph, iT, tau_c):
        self.tauc_phr_avg[iph,iT] = tau_c
    def set_tauc_wql(self, iwb, ic, iT, tau_c):
        self.tauc_wql[iwb,ic,iT] = tau_c
        # ps units
    def set_tauc_wql_avg(self, iwb, iT, tau_c):
        self.tauc_wql_avg[iwb,iT] = tau_c
    def collect_atr_from_other_proc(self, ic, iT):
        tauc_atr_full = mpi.collect_array(self.tauc_atr[:,ic,iT])
        self.tauc_atr[:,ic,iT] = 0.
        self.tauc_atr[:,ic,iT] = tauc_atr_full[:]
    def collect_avg_atr_from_other_proc(self, iT):
        tauc_atr_full = mpi.collect_array(self.tauc_atr_avg[:,iT])
        self.tauc_atr_avg[:,iT] = 0.
        self.tauc_atr_avg[:,iT] = tauc_atr_full[:]
    def collect_phr_from_other_proc(self, ic, iT):
        if p.nphr > 0:
            tauc_phr_full = mpi.collect_array(self.tauc_phr[:,ic,iT])
            self.tauc_phr[:,ic,iT] = 0.
            self.tauc_phr[:,ic,iT] = tauc_phr_full[:]
        # wql
        tauc_wql_full = mpi.collect_array(self.tauc_wql[:,ic,iT])
        self.tauc_wql[:,ic,iT] = 0.
        self.tauc_wql[:,ic,iT] = tauc_wql_full[:]
    def collect_avg_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            tauc_phr_full = mpi.collect_array(self.tauc_phr_avg[:,iT])
            self.tauc_phr_avg[:,iT] = 0.
            self.tauc_phr_avg[:,iT] = tauc_phr_full[:]
        # wql
        tauc_wql_full = mpi.collect_array(self.tauc_wql_avg[:,iT])
        self.tauc_wql_avg[:,iT] = 0.
        self.tauc_wql_avg[:,iT] = tauc_wql_full[:]
    def get_tauc_avg(self):
        return self.tauc_avg
    def get_tauc_atr_avg(self):
        return self.tauc_atr_avg
    def get_tauc_phr_avg(self):
        return self.tauc_phr_avg
    def get_tauc_wql_avg(self):
        return self.tauc_wql_avg
class tauc_inhom_dd(tauc_class):
    # tauc is in ps
    def __init__(self, nconf):
        npl = len(p.n_pulses)
        self.tauc_ps = np.zeros((npl,nconf+1))
    def set_tauc(self, ipl, ic, tauc_mus):
        self.tauc_ps[ipl,ic] = tauc_mus * 1.E6
        # ps units
    def collect_from_other_proc(self, ic):
        tauc_full = mpi.collect_array(self.tauc_ps[:,ic])
        self.tauc_ps[:,ic] = 0.
        self.tauc_ps[:,ic] = tauc_full[:]
# tauc class inhom
class tauc_inhom(tauc_class):
    # tauc is in ps
    def __init__(self, nconf):
        self.tauc_ps = np.zeros(nconf+1)
    def set_tauc(self, ic, tauc_mus):
        self.tauc_ps[ic] = tauc_mus * 1.E6
        # ps units
    def collect_from_other_proc(self):
        tauc_full = mpi.collect_array(self.tauc_ps)
        self.tauc_ps[:] = 0.
        self.tauc_ps[:] = tauc_full[:]
# -----------------------------------------------
#
#         lw class
# -----------------------------------------------
class lw_class(object): 
    # lw in eV in all classes
    def __init__(self):
        self.lw_eV = None
        self.lw_avg= None
        self.lw_atr= None
        self.lw_atr_avg = None
        self.lw_phr= None
        self.lw_phr_avg = None
        self.lw_wql= None
        self.lw_wql_avg = None
    def generate_instance(self, nat, nconf=None):
        if not p.deph and not p.relax:
            if p.dyndec:
                return lw_inhom_dd()
            else:
                return lw_inhom()
        else:
            return lw_ofT().generate_instance(nat, nconf)
    def get_lw(self):
        return self.lw_eV
class lw_ofT(lw_class):
    # lw in eV units
    def __init__(self):
        super(lw_ofT, self).__init__()
    def generate_instance(self, nat, nconf=None):
        calc_type2 = parser.parse_args().ct2
        if calc_type2 == "homo":
            return lw_homo_ofT(nat)
        elif calc_type2 == "inhomo" or calc_type2 == "full":
            return lw_inhom_ofT(nat, nconf)
        else:
            log.error("\t WRONG calc_type2 VALUE")
    def get_lw_atr(self):
        return self.lw_atr
    def get_lw_phr(self):
        return self.lw_phr
    def get_lw_wql(self):
        return self.lw_wql
class lw_homo_ofT(lw_ofT):
    def __init__(self, nat):
        super(lw_homo_ofT, self).__init__()
        self.lw_eV = np.zeros(p.ntmp)
        if p.at_resolved:
            self.lw_atr = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.lw_phr = np.zeros((p.nphr,p.ntmp))
            self.lw_wql = np.zeros((p.nwbn,p.ntmp))
    def set_lw(self, iT, T2i):
        self.lw_eV[iT] = 2.*np.pi*hbar*T2i
    def set_lw_atr(self, ia, iT, T2i):
        self.lw_atr[ia,iT] = 2.*np.pi*hbar*T2i
    def set_lw_phr(self, iph, iT, T2i):
        self.lw_phr[iph,iT] = 2.*np.pi*hbar*T2i
    def set_lw_wql(self, iwb, iT, T2i):
        self.lw_wql[iwb,iT] = 2.*np.pi*hbar*T2i
    def collect_atr_from_other_proc(self, iT):
        lw_atr_full = mpi.collect_array(self.lw_atr[:,iT])
        self.lw_atr[:,iT] = 0.
        self.lw_atr[:,iT] = lw_atr_full[:]
    def collect_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            lw_phr_full = mpi.collect_array(self.lw_phr[:,iT])
            self.lw_phr[:,iT] = 0.
            self.lw_phr[:,iT] = lw_phr_full[:]
        # wql
        lw_wql_full = mpi.collect_array(self.lw_wql[:,iT])
        self.lw_wql[:,iT] = 0.
        self.lw_wql[:,iT] = lw_wql_full[:]
class lw_inhom_ofT(lw_ofT):
    def __init__(self, nat, nconf):
        super(lw_inhom_ofT, self).__init__()
        self.lw_eV = np.zeros((nconf,p.ntmp))
        self.lw_avg= np.zeros(p.ntmp)
        if p.at_resolved:
            self.lw_atr = np.zeros((nat,nconf,p.ntmp))
            self.lw_atr_avg = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            if p.nphr > 0:
                self.lw_phr = np.zeros((p.nphr,nconf,p.ntmp))
                self.lw_phr_avg = np.zeros((p.nphr,p.ntmp))
            self.lw_wql = np.zeros((p.nwbn,nconf,p.ntmp))
            self.lw_wql_avg = np.zeros((p.nwbn,p.ntmp))
    def set_lw(self, ic, iT, T2i):
        self.lw_eV[ic,iT] = 2.*np.pi*hbar*T2i
    def set_lw_avg(self, iT, T2i):
        self.lw_avg[iT] = 2.*np.pi*hbar*T2i
    def set_lw_atr(self, ia, ic, iT, T2i):
        self.lw_atr[ia,ic,iT] = 2.*np.pi*hbar*T2i
    def set_lw_atr_avg(self, ia, iT, T2i):
        self.lw_atr_avg[ia,iT] = 2.*np.pi*hbar*T2i
    def set_lw_phr(self, iph, ic, iT, T2i):
        self.lw_phr[iph,ic,iT] = 2.*np.pi*hbar*T2i
    def set_lw_phr_avg(self, iph, iT, T2i):
        self.lw_phr_avg[iph,iT] = 2.*np.pi*hbar*T2i
    def set_lw_wql(self, iwb, ic, iT, T2i):
        self.lw_wql[iwb,ic,iT] = 2.*np.pi*hbar*T2i
    def set_lw_wql_avg(self, iwb, iT, T2i):
        self.lw_wql_avg[iwb,iT] = 2.*np.pi*hbar*T2i
    def collect_atr_from_other_proc(self, ic, iT):
        lw_atr_full = mpi.collect_array(self.lw_atr[:,ic,iT])
        self.lw_atr[:,ic,iT] = 0.
        self.lw_atr[:,ic,iT] = lw_atr_full[:]
    def collect_avg_atr_from_other_proc(self, iT):
        lw_atr_full = mpi.collect_array(self.lw_atr_avg[:,iT])
        self.lw_atr_avg[:,iT] = 0.
        self.lw_atr_avg[:,iT] = lw_atr_full[:]
    def collect_phr_from_other_proc(self, ic, iT):
        if p.nphr > 0:
            lw_phr_full = mpi.collect_array(self.lw_phr[:,ic,iT])
            self.lw_phr[:,ic,iT] = 0.
            self.lw_phr[:,ic,iT] = lw_phr_full[:]
        # wql
        lw_wql_full = mpi.collect_array(self.lw_wql[:,ic,iT])
        self.lw_wql[:,ic,iT] = 0.
        self.lw_wql[:,ic,iT] = lw_wql_full[:]
    def collect_avg_phr_from_other_proc(self, iT):
        if p.nphr > 0:
            lw_phr_full = mpi.collect_array(self.lw_phr_avg[:,iT])
            self.lw_phr_avg[:,iT] = 0.
            self.lw_phr_avg[:,iT] = lw_phr_full[:]
        # wql
        lw_wql_full = mpi.collect_array(self.lw_wql_avg[:,iT])
        self.lw_wql_avg[:,iT] = 0.
        self.lw_wql_avg[:,iT] = lw_wql_full[:]
    def get_lw_avg(self):
        return self.lw_avg
    def get_lw_atr_avg(self):
        return self.lw_atr_avg
    def get_lw_phr_avg(self):
        return self.lw_phr_avg
    def get_lw_wql_avg(self):
        return self.lw_wql_avg
class lw_inhom_dd(lw_class):
    # lw is in eV
    def __init__(self, nconf):
        npl = len(p.n_pulses)
        self.lw_eV = np.zeros((npl,nconf+1))
    def set_lw(self, ipl, ic, T2i):
        self.lw_eV[ipl,ic] = 2.*np.pi*hbar*T2i
        # eV units
    def collect_from_other_proc(self, ic):
        lw_full = mpi.collect_array(self.lw_eV[:,ic])
        self.lw_eV[:,ic] = 0.
        self.lw_eV[:,ic] = lw_full[:]
# tauc class inhom
class lw_inhom(lw_class):
    # lw is in eV
    def __init__(self, nconf):
        self.lw_eV = np.zeros(nconf+1)
    def set_lw(self, ic, T2i):
        self.lw_eV[ic] = 2.*np.pi*hbar*T2i
        # eV units
    def collect_from_other_proc(self):
        lw_full = mpi.collect_array(self.lw_eV)
        self.lw_eV[:] = 0.
        self.lw_eV[:] = lw_full[:]
#
# ext. function : print dephasing data
#
def print_decoher_data(data):
	# first print data on dict
    if not p.deph and not p.relax:
        if p.dyndec:
            print_decoher_data_dyndec(data)
        else:
            print_decoher_data_stat(data)
#
#  dynamical decoupling
#
def print_decoher_data_dyndec(data):
    T2_obj   = data['T2']
    Delt_obj = data['Delt']
    tauc_obj = data['tau_c']
    lw_obj   = data['lw']
	# first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'lw_eV' : None, 'pulses' : None}
    deph_dict['T2']   = T2_obj.get_T2_musec()
    deph_dict['Delt'] = Delt_obj.get_Delt()
    deph_dict['tau_c']= tauc_obj.get_tauc()
    deph_dict['lw_eV']= lw_obj.get_lw()
    deph_dict['pulses'] = p.n_pulses
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
# ext. function : print dephasing data (static)
#
def print_decoher_data_stat(data):
    T2_obj   = data['T2']
    Delt_obj = data['Delt']
    tauc_obj = data['tau_c']
    lw_obj   = data['lw']
	# first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'lw_eV' : None}
    deph_dict['T2']   = T2_obj.get_T2_musec()
    deph_dict['Delt'] = Delt_obj.get_Delt()
    deph_dict['tau_c']= tauc_obj.get_tauc()
    deph_dict['lw_eV']= lw_obj.get_lw()
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
