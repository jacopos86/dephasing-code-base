import numpy as np
from pydephasing.mpi import mpi
from pydephasing.phys_constants import hbar
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.input_parameters import p
from pydephasing.log import log
import yaml
# T2 inverse class
class T2i_ofT(object):
    # T2i is in ps^-1
    def __init__(self):
        self.T2_sec = None
        self.T2s_atr = None
        self.T2s_phr = None
        self.T2s_wql = None
    def generate_instance(self):
        if not p.deph and not p.relax:
            if p.dyndec:
                return T2i_inhom_dd ()
            else:
                return T2i_inhom()
        else:
            return T2i_homo_ofT().generate_instance()
    def get_T2_sec(self):
        return self.T2_sec
    def set_T2(self, iT, T2i):
        for ic in range(2):
            if T2i[ic] == 0.:
                self.T2_sec[ic,iT] = np.inf
            else:
                self.T2_sec[ic,iT] = 1./T2i[ic] * 1.E-12
        # sec units
    def get_T2_atr_sec(self):
        return self.T2s_atr
    def set_T2_atr(self, ia, iT, T2ai):
        for ic in range(2):
            if T2ai[ic] == 0.:
                self.T2s_atr[ic,ia,iT] = np.inf
            else:
                self.T2s_atr[ic,ia,iT] = 1./T2ai[ic] * 1.E-12
        # sec
    def get_T2_phr_sec(self):
        return self.T2s_phr
    def get_T2_wql_sec(self):
        return self.T2s_wql
    def set_T2_phr(self, iph, iT, T2pi):
        for ic in range(2):
            if T2pi[ic] == 0.:
                self.T2s_phr[ic,iph,iT] = np.inf
            else:
                self.T2s_phr[:,iph,iT] = 1./T2pi[:] * 1.E-12
        # sec
    def set_T2_wql(self, iwb, iT, T2pi):
        for ic in range(2):
            if T2pi[ic] == 0.:
                self.T2s_wql[ic,iwb,iT] = np.inf
            else:
                self.T2s_wql[:,iwb,iT] = 1./T2pi[:] * 1.E-12
        # sec
    def collect_atr_from_other_proc(self, iT):
        T2s_atr_full = mpi.collect_array(self.T2s_atr[:,:,iT])
        self.T2s_atr[:,:,iT] = 0.
        self.T2s_atr[:,:,iT] = T2s_atr_full[:,:]
    def collect_phr_from_other_proc(self, iT):
        T2s_phr_full = mpi.collect_array(self.T2s_phr[:,:,iT])
        self.T2s_phr[:,:,iT] = 0.
        self.T2s_phr[:,:,iT] = T2s_phr_full[:,:]
        # wql
        T2s_wql_full = mpi.collect_array(self.T2s_wql[:,:,iT])
        self.T2s_wql[:,:,iT] = 0.
        self.T2s_wql[:,:,iT] = T2s_wql_full[:,:]
# T2i abstract
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
    def generate_instance(self):
        if p.time_resolved:
            return T2i_time_resolved ()
        elif p.w_resolved:
            return T2i_w_resolved ()
        else:
            log.error("neither time nor freq. resolved...")
# concrete T2i -> time resolved 
# calculation
class T2i_time_resolved(T2i_homo_ofT):
    def __init__(self):
        super(T2i_time_resolved, self).__init__()
# concrete T2i -> time resolved 
# calculation
class T2i_w_resolved(T2i_homo_ofT):
    def __init__(self):
        super(T2i_w_resolved, self).__init__()
#
# -> T2i dd class
class T2i_inhom_dd(T2i_ofT):
    # T2i is in ps^-1
    def __init__(self):
        super(T2i_inhom_dd, self).__init__()
        npl = len(p.n_pulses)
        self.T2_psec = np.zeros((npl,p.ntmp))
    def get_T2_sec(self):
        return self.T2_sec
    def set_T2(self, ipl, iT, T2i):
        for ic in range(2):
            if T2i[ic] == 0.:
                self.T2_sec[ic,ipl,iT] = np.inf
            else:
                self.T2_sec[ic,ipl,iT] = 1./T2i[ic] * 1.E-12
        # sec units
    def collect_from_other_proc(self, iT):
        T2s_full = mpi.collect_array(self.T2_sec[:,:,iT])
        self.T2_sec[:,:,iT] = 0.
        self.T2_sec[:,:,iT] = T2s_full[:,:]
# T2 inverse class
class T2i_inhom(T2i_ofT):
    # T2i is in ps^-1
    def __init__(self, nconf):
        super(T2i_inhom, self).__init__()
        self.T2_psec = np.zeros(nconf+1)
    def get_T2_psec(self):
        return self.T2_psec
    def get_T2_musec(self):
        return self.T2_psec*1.E-6
    def set_T2_psec(self, ic, T2i):
        if T2i == 0.:
            self.T2_psec[ic] = np.inf
        else:
            self.T2_psec[ic] = 1./T2i
        # psec units
    def collect_from_other_proc(self):
        T2ps_full = mpi.collect_array(self.T2_psec)
        self.T2_psec[:] = 0.
        self.T2_psec[:] = T2ps_full[:]
# Delta class
class Delta_ofT:
    # Delta is in eV
    def __init__(self, nat):
        self.Delt = np.zeros(p.ntmp)
        if p.at_resolved:
            self.Delt_atr = np.zeros((nat,p.ntmp))
        if p.ph_resolved:
            self.Delt_phr = np.zeros((len(p.phm_list),p.ntmp))
            self.Delt_wql = np.zeros((p.nwbn,p.ntmp))
    def get_Delt(self):
        return self.Delt
    def set_Delt(self, iT, D2):
        self.Delt[iT] = np.sqrt(D2)
        # eV units
    def get_Delt_atr(self):
        return self.Delt_atr
    def set_Delt_atr(self, ia, iT, D2):
        self.Delt_atr[ia,iT] = np.sqrt(D2)
    def get_Delt_phr(self):
        return self.Delt_phr
    def get_Delt_wql(self):
        return self.Delt_wql
    def set_Delt_phr(self, iph, iT, D2):
        self.Delt_phr[iph,iT] = np.sqrt(D2)
    def set_Delt_wql(self, iwb, iT, D2):
        self.Delt_wql[iwb,iT] = np.sqrt(D2)
    def collect_atr_from_other_proc(self, iT):
        Delt_atr_full = mpi.collect_array(self.Delt_atr[:,iT])
        self.Delt_atr[:,iT] = 0.
        self.Delt_atr[:,iT] = Delt_atr_full[:]
    def collect_phr_from_other_proc(self, iT):
        Delt_phr_full = mpi.collect_array(self.Delt_phr[:,iT])
        self.Delt_phr[:,iT] = 0.
        self.Delt_phr[:,iT] = Delt_phr_full[:]
        # wql
        Delt_wql_full = mpi.collect_array(self.Delt_wql[:,iT])
        self.Delt_wql[:,iT] = 0.
        self.Delt_wql[:,iT] = Delt_wql_full[:]
# Delta class
class Delta_dd_ofT:
    # Delta is in eV
    def __init__(self):
        npl = len(p.n_pulses)
        self.Delt = np.zeros((npl,p.ntmp))
    def get_Delt(self):
        return self.Delt
    def set_Delt(self, ipl, iT, D2):
        self.Delt[ipl,iT] = np.sqrt(D2)
        # eV units
    def collect_from_other_proc(self, iT):
        Delt_full = mpi.collect_array(self.Delt[:,iT])
        self.Delt[:,iT] = 0.
        self.Delt[:,iT] = Delt_full[:]
# Delta inhom class
class Delta_inhom:
    # Delta is in eV
    def __init__(self, nconf):
        self.Delt = np.zeros(nconf+1)
    def get_Delt(self):
        return self.Delt
    def set_Delt(self, ic, D2):
        self.Delt[ic] = np.sqrt(D2)
        # eV units
    def collect_from_other_proc(self):
        Delt_full = mpi.collect_array(self.Delt)
        self.Delt[:] = 0.
        self.Delt[:] = Delt_full[:]
        # eV
# tauc class
class tauc_ofT:
    # tauc is in ps
    def __init__(self, nat):
        self.tauc_ps = np.zeros((2,p.ntmp))
        if p.at_resolved:
            self.tauc_atr = np.zeros((2,nat,p.ntmp))
        if p.ph_resolved:
            self.tauc_phr = np.zeros((2,len(p.phm_list),p.ntmp))
            self.tauc_wql = np.zeros((2,p.nwbn,p.ntmp))
    def set_tauc(self, iT, tau_c):
        self.tauc_ps[:,iT] = tau_c[:]
        # ps units
    def get_tauc(self):
        return self.tauc_ps
    def set_tauc_atr(self, ia, iT, tau_ca):
        self.tauc_atr[:,ia,iT] = tau_ca[:]
        # ps units
    def get_tauc_atr(self):
        return self.tauc_atr
    def set_tauc_phr(self, iph, iT, tau_cp):
        self.tauc_phr[:,iph,iT] = tau_cp[:]
        # ps units
    def set_tauc_wql(self, iwb, iT, tau_cp):
        self.tauc_wql[:,iwb,iT] = tau_cp[:]
        # ps units
    def get_tauc_phr(self):
        return self.tauc_phr
    def get_tauc_wql(self):
        return self.tauc_wql
    def collect_atr_from_other_proc(self, iT):
        tauc_atr_full = mpi.collect_array(self.tauc_atr[:,:,iT])
        self.tauc_atr[:,:,iT] = 0.
        self.tauc_atr[:,:,iT] = tauc_atr_full[:,:]
    def collect_phr_from_other_proc(self, iT):
        tauc_phr_full = mpi.collect_array(self.tauc_phr[:,:,iT])
        self.tauc_phr[:,:,iT] = 0.
        self.tauc_phr[:,:,iT] = tauc_phr_full[:,:]
        # wql
        tauc_wql_full = mpi.collect_array(self.tauc_wql[:,:,iT])
        self.tauc_wql[:,:,iT] = 0.
        self.tauc_wql[:,:,iT] = tauc_wql_full[:,:]
# tauc dd class
class tauc_dd_ofT:
    # tauc is in ps
    def __init__(self):
        npl = len(p.n_pulses)
        self.tauc_ps = np.zeros((2,npl,p.ntmp))
    def set_tauc(self, ipl, iT, tau_c):
        self.tauc_ps[:,ipl,iT] = tau_c[:]
        # ps units
    def get_tauc(self):
        return self.tauc_ps
    def collect_from_other_proc(self, iT):
        tauc_full = mpi.collect_array(self.tauc_ps[:,:,iT])
        self.tauc_ps[:,:,iT] = 0.
        self.tauc_ps[:,:,iT] = tauc_full[:,:]
# tauc class inhom
class tauc_inhom:
    # tauc is in ps
    def __init__(self, nconf):
        self.tauc_ps = np.zeros(nconf+1)
    def set_tauc(self, ic, tau_c):
        self.tauc_ps[ic] = tau_c
        # ps units
    def get_tauc(self):
        return self.tauc_ps
    def collect_from_other_proc(self):
        tauc_full = mpi.collect_array(self.tauc_ps)
        self.tauc_ps[:] = 0.
        self.tauc_ps[:] = tauc_full[:]
# lw class
class lw_ofT:
    # lw in eV units
    def __init__(self, nat):
        self.lw_eV = np.zeros((2,p.ntmp))
        if p.at_resolved:
            self.lw_atr = np.zeros((2,nat,p.ntmp))
        if p.ph_resolved:
            self.lw_phr = np.zeros((2,len(p.phm_list),p.ntmp))
            self.lw_wql = np.zeros((2,p.nwbn,p.ntmp))
    def set_lw(self, iT, T2i):
        self.lw_eV[:,iT] = 2.*np.pi*hbar*T2i[:]
    def get_lw(self):
        return self.lw_eV
    def set_lw_atr(self, ia, iT, T2ai):
        self.lw_atr[:,ia,iT] = 2.*np.pi*hbar*T2ai[:]
    def get_lw_atr(self):
        return self.lw_atr
    def set_lw_phr(self, iph, iT, T2pi):
        self.lw_phr[:,iph,iT] = 2.*np.pi*hbar*T2pi[:]
    def set_lw_wql(self, iwb, iT, T2pi):
        self.lw_wql[:,iwb,iT] = 2.*np.pi*hbar*T2pi[:]
    def get_lw_phr(self):
        return self.lw_phr
    def get_lw_wql(self):
        return self.lw_wql
    def collect_atr_from_other_proc(self, iT):
        lw_atr_full = mpi.collect_array(self.lw_atr[:,:,iT])
        self.lw_atr[:,:,iT] = 0.
        self.lw_atr[:,:,iT] = lw_atr_full[:,:]
    def collect_phr_from_other_proc(self, iT):
        lw_phr_full = mpi.collect_array(self.lw_phr[:,:,iT])
        self.lw_phr[:,:,iT] = 0.
        self.lw_phr[:,:,iT] = lw_phr_full[:,:]
        # wql
        lw_wql_full = mpi.collect_array(self.lw_wql[:,:,iT])
        self.lw_wql[:,:,iT] = 0.
        self.lw_wql[:,:,iT] = lw_wql_full[:,:]
#
# ext. function : print dephasing data
#
def print_dephas_data(T2_obj, tauc_obj, Delt_obj, lw_obj=None):
	# first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'lw_eV' : None, 'temperature' : None}
    deph_dict['T2']   = T2_obj.get_T2_sec()
    deph_dict['Delt'] = Delt_obj.get_Delt()
    deph_dict['tau_c']= tauc_obj.get_tauc()
    deph_dict['temperature'] = p.temperatures
    if lw_obj != None:
        deph_dict['lw_eV'] = lw_obj.get_lw()
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
#  dynamical decoupling
#
def print_dephas_data_dyndec(T2_obj, tauc_obj, Delt_obj):
	# first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'temperature' : None, 'pulses' : None}
    deph_dict['T2']   = T2_obj.get_T2_sec()
    deph_dict['Delt'] = Delt_obj.get_Delt()
    deph_dict['tau_c']= tauc_obj.get_tauc()
    deph_dict['temperature'] = p.temperatures
    deph_dict['pulses'] = p.n_pulses
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
# ph. res. data
#
def print_dephas_data_phr(T2_obj, tauc_obj, Delt_obj, lw_obj=None):
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'lw_eV' : None, 'temperature' : None, 'w_lambda' : None}
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    assert mesh[0]*mesh[1]*mesh[2] == nq
    assert len(qpts) == nq
    assert len(u) == nq
    wl = np.zeros(len(wu[0]))
    for iq in range(nq):
        wuq = wu[iq]
        wl[:] = wl[:] + wq[iq] * wuq[:]
    wiph = np.zeros(p.nphr)
    for iph in range(p.nphr):
        im = p.phm_list[iph]
        wiph[iph] = wl[im]
    deph_dict['T2']   = T2_obj.get_T2_phr_sec()
    deph_dict['Delt'] = Delt_obj.get_Delt_phr()
    deph_dict['tau_c']= tauc_obj.get_tauc_phr()
    deph_dict['temperature'] = p.temperatures
    deph_dict['w_ql'] = wiph
    if lw_obj != None:
        deph_dict['lw_eV'] = lw_obj.get_lw_phr()
    # wql data
    deph_dict['wql_bins'] = p.wql_grid
    deph_dict['T2_bins'] = T2_obj.get_T2_wql_sec()
    deph_dict['Delt_bins'] = Delt_obj.get_Delt_wql()
    deph_dict['tauc_bins'] = tauc_obj.get_tauc_wql()
    if lw_obj != None:
        deph_dict['lw_eV_bins'] = lw_obj.get_lw_wql()
    # write yaml file
    namef = "T2-phr-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
# at res. data
#
def print_dephas_data_atr(T2_obj, tauc_obj, Delt_obj, lw_obj=None):
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'lw_eV' : None, 'temperature' : None}
    deph_dict['T2']   = T2_obj.get_T2_atr_sec()
    deph_dict['Delt'] = Delt_obj.get_Delt_atr()
    deph_dict['tau_c']= tauc_obj.get_tauc_atr()
    if lw_obj != None:
        deph_dict['lw_eV'] = lw_obj.get_lw_atr()
    deph_dict['temperature'] = p.temperatures
    # write yaml file
    namef = "T2-atr_data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
# HFI functions
#
def print_dephas_data_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst):
    # first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'temperature' : None}
    deph_dict['T2']  = []
    for ic in range(len(T2_obj_lst)):
        deph_dict['T2'].append(T2_obj_lst[ic].get_T2_sec())
    deph_dict['Delt'] = []
    for ic in range(len(Delt_obj_lst)):
        deph_dict['Delt'].append(Delt_obj_lst[ic].get_Delt())
    deph_dict['tau_c'] = []
    for ic in range(len(tauc_obj_lst)):
        deph_dict['tau_c'].append(tauc_obj_lst[ic].get_tauc())
    deph_dict['temperature'] = p.temperatures
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
def print_dephas_data_atr_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst):
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'temperature' : None}
    deph_dict['T2']   = []
    for ic in range(len(T2_obj_lst)):
        deph_dict['T2'].append(T2_obj_lst[ic].get_T2_atr_sec())
    deph_dict['Delt'] = []
    for ic in range(len(Delt_obj_lst)):
        deph_dict['Delt'].append(Delt_obj_lst[ic].get_Delt_atr())
    deph_dict['tau_c'] = []
    for ic in range(len(tauc_obj_lst)):
        deph_dict['tau_c'].append(tauc_obj_lst[ic].get_tauc_atr())
    deph_dict['temperature'] = p.temperatures
    # write yaml file
    namef = "T2-atr-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
def print_dephas_data_phr_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst):
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None, 'temperature' : None, 'w_ql' : None}
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    assert mesh[0]*mesh[1]*mesh[2] == nq
    assert len(qpts) == nq
    assert len(u) == nq
    wl = np.zeros(len(wu[0]))
    for iq in range(nq):
        wuq = wu[iq]
        wl[:] = wl[:] + wq[iq] * wuq[:]
    wiph = np.zeros(p.nphr)
    for iph in range(p.nphr):
        im = p.phm_list[iph]
        wiph[iph] = wl[im]
    deph_dict['T2']   = []
    for ic in range(len(T2_obj_lst)):
        deph_dict['T2'].append(T2_obj_lst[ic].get_T2_phr_sec())
    deph_dict['Delt'] = []
    for ic in range(len(Delt_obj_lst)):
        deph_dict['Delt'].append(Delt_obj_lst[ic].get_Delt_phr())
    deph_dict['tau_c'] = []
    for ic in range(len(tauc_obj_lst)):
        deph_dict['tau_c'].append(tauc_obj_lst[ic].get_tauc_phr())
    deph_dict['temperature'] = p.temperatures
    deph_dict['w_ql'] = wiph
    # write yaml file
    namef = "T2-phr-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)
#
# ext. function : print dephasing data (static)
#
def print_dephas_data_stat(T2_obj, tauc_obj, Delt_obj):
	# first print data on dict
    deph_dict = {'T2' : None, 'Delt' : None, 'tau_c' : None}
    deph_dict['T2']   = T2_obj.get_T2_musec()
    deph_dict['Delt'] = Delt_obj.get_Delt()
    deph_dict['tau_c']= tauc_obj.get_tauc()
    # write yaml file
    namef = "T2-data.yml"
    with open(p.write_dir+'/'+namef, 'w') as out_file:
        yaml.dump(deph_dict, out_file)