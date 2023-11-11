#
# This module sets up the method
# needed to compute T2 / T2*
# given the energy fluct. auto correlation function
#
import numpy as np
import scipy
from pydephasing.T2_classes import T2i_class, Delta_class, tauc_class, lw_class
from pydephasing.phys_constants import hbar
from pydephasing.log import log
from pydephasing.input_parameters import p
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")
#
# functions : Exp ExpSin exp_gt gt
#
def Exp(x, c):
    return np.exp(-c * x)
# fit gaussian+lorentzian decay
def Explg(x, a, b, c, sig):
    return a * np.exp(-c * x) + b * np.exp(-x**2 / 2 / sig**2)
# fitting function int_0^t acf(t)
def fitting_function(x, offset, a, b, c, d):
    '''use y(x) = offset + a * cos(x) / (b + c * exp(d*x))'''
    y = offset + a * np.cos(x) / (b + c * np.exp(d * x))
    return y
#
#   abstract T2_eval_class
class T2_eval_class_time_res(ABC):
    def __init__(self):
        self.T2_obj = None
        self.lw_obj = None
        self.tauc_obj = None
        self.Delt_obj = None
    @classmethod
    def set_up_param_objects(self):
        self.T2_obj   = T2i_class().generate_instance()
        self.Delt_obj = Delta_class().generate_instance()
        self.tauc_obj = tauc_class().generate_instance()
        self.lw_obj   = lw_class().generate_instance()
    @classmethod
    def parametrize_acf(self, t, acf_oft):
        # units of tau_c depends on units of t
        # if it is called by static calculation -> mu sec
        # otherwise p sec
        D2 = acf_oft[0]
        Ct = acf_oft / D2
        # check non Nan
        if not np.isfinite(Ct).all():
            return None, None, None, None
        # set parametrization
        # e^-t/tau parametrization
        # fit over exp. function
        p0 = 1    # start with values near those we expect
        res = scipy.optimize.curve_fit(Exp, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (ps^-1/musec^-1)
        tau_c = 1./p[0]
        return D2, tau_c, Ct, Exp(t, p)
    @classmethod
    def get_T2_data(self):
        decoher_dict = {'T2' : None, 'lw' : None, 'Delt' : None, 'tau_c' : None}
        decoher_dict['T2']   = self.T2_obj
        decoher_dict['Delt'] = self.Delt_obj
        decoher_dict['tau_c']= self.tauc_obj
        decoher_dict['lw']   = self.lw_obj
'''
# --------------------------------------------------------------
#  time resolved calculation -> concrete class implementation
#  fit the autocorrelation over 
#  (1) e^-t or sin(wt) (2) e^-t model
#  -> depending on the model -> different g(t)
#  depending on Delta tau_c value determine T2 / linwidth
# --------------------------------------------------------------
class T2_eval_fit_model_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
        # fft sample points
        self.N = p.N_df
        # sample spacing -> ps
        self.T = p.T_df
        # max iteration curve fitting
        self.maxiter = p.maxiter
    def get_T2_data(self):
        super().get_T2_data()
    def generate_instance(self):
        if not p.deph and not p.relax:
            return T2_eval_fit_model_stat_class()
        else:
            return T2_eval_fit_model_dyn_class()
    #
    # e^-g(t) -> g(t)=D2*tau_c^2[e^(-t/tau_c)+t/tau_c-1]
    # D2 -> eV^2
    # tau_c -> ps
    def exp_gt(self, x, D2, tau_c):
        r = np.exp(-self.gt(x, D2, tau_c))
        return r
    #
    def gt(self, x, D2, tau_c):
        r = np.zeros(len(x))
        r[:] = D2 / hbar ** 2 * tau_c ** 2 * (np.exp(-x[:]/tau_c) + x[:]/tau_c - 1)
        return r
    #
    # compute T2*
    # input : t, Ct, D2
    # output: tauc, T2_inv, [expsin, fit]
    def evaluate_T2(self, D2, tau_c):
        # check non Nan
        if not np.isfinite(Ct).all():
            return [None, None, None]
        # perform the fit
        p0 = [1., 1., 1.]
        res = scipy.optimize.curve_fit(ExpSin, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (mus^-1)
        tau_c = 1./p[2]
        # tau_c (mu sec)
        r = np.sqrt(D2) / hbar * tau_c * 1.E+6
        #
        # check limit r conditions
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c * 1.E+6
            # ps^-1
        elif r > 1.E+4:
            T2_inv = np.sqrt(D2) / hbar * 1.E+6
            # ps^-1
        else:
            # -> implement here
            tauc_ps = tau_c * 1.E+6
            T2_inv = self.T2inv_interp_eval(D2, tauc_ps)
            
        return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])
'''
# -------------------------------------------------------------
# this class is unique for dynamical calculations
# relax / dephas calculations
# extract T2 from integrated auto correlation function directly
# -------------------------------------------------------------
class T2_eval_from_integ_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
    def get_T2_data(self):
        super().get_T2_data()
    @classmethod
    def evaluate_T2(self, t, acf_int_oft):
        '''use y(x) = offset + a * cos(x) / (b + c * exp(d*x))'''
        p0 = [1., 1., 1., 1., 1.]              # start with values close to those expected
        p = scipy.optimize.curve_fit(fitting_function, t, acf_int_oft, p0, maxfev=self.maxiter)
        T2i = p[0]
        return T2i, fitting_function(t, p[0], p[1], p[2], p[3], p[4])
# -------------------------------------------------------------
# subclass of the integral model
# to be used for dynamical homogeneous calculations
# -------------------------------------------------------------
class T2_eval_from_integ_homo_class(T2_eval_from_integ_class):
    def __init__(self):
        super(T2_eval_from_integ_homo_class, self).__init__()
    def parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        acf_integ_oft = np.zeros(p.nt)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,0,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(iT, D2)
        self.tauc_obj.set_tauc(iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf[:,1,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_sec(iT, T2_inv)
        self.lw_obj.set_lw(iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,0,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_atr[:,1,ia,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_atr(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # ph. res. version
    def phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,0,iph,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_phr[:,1,iph,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_phr(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # wql resolved
    def wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,0,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_wql[:,1,iwql,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_wql(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
# -------------------------------------------------------------
# subclass of the integral model
# to be used for dynamical inhomogeneous calculations
# -------------------------------------------------------------
class T2_eval_from_integ_inhom_class(T2_eval_from_integ_class):
    def __init__(self):
        super(T2_eval_from_integ_inhom_class, self).__init__()
    def parameter_eval_driver(self, acf_obj, ic, iT):
        acf_oft = np.zeros(p.nt)
        acf_integ_oft = np.zeros(p.nt)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,0,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(ic, iT, D2)
        self.tauc_obj.set_tauc(ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf[:,1,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_sec(ic, iT, T2_inv)
        self.lw_obj.set_lw(ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,0,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, ic, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_atr[:,1,ia,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_atr(ia, ic, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # ph. resolved version
    def phr_parameter_eval_driver(self, acf_obj, iph, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,0,iph,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, ic, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_phr[:,1,iph,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_phr(iph, ic, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
    # wql resolved version
    def wql_parameter_eval_driver(self, acf_obj, iwql, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,0,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, ic, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_wql[:,1,iwql,iT])
        T2_inv, f_it = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_wql(iwql, ic, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft, f_it
# -------------------------------------------------------------
# subclass -> template for pure fitting calculation
# this is also abstract -> real class we must specifiy
# if static / dyn. homogeneous / dyn. inhomogeneous 
# -------------------------------------------------------------
class T2_eval_fit_model_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def evaluate_T2(self, D2, tauc_ps):
        '''method to implement'''
        return
    def T2inv_interp_eval(self, D2, tauc_ps):
        # fft sample points
        N = self.N
        T = self.T
        x = np.linspace(0.0, N*T, N, endpoint=False)
        x_ps = x * 1.E+6
        y = self.exp_gt(x_ps, D2, tauc_ps)
        try:
            c0 = D2 / hbar ** 2 * tauc_ps * 1.E+6   # mu s^-1
            s0 = hbar / np.sqrt(D2) * 1.E-6         # mu sec
            p0 = [0.5, 0.5, c0, s0]                 # start with values close to those expected
            res = scipy.optimize.curve_fit(Explg, x, y, p0, maxfev=self.maxiter)
            p1 = res[0]
            # gauss vs lorentzian
            if p1[0] > p1[1]:
                # T2 -> lorentzian (mu sec)
                T2_inv = p1[2]
                # ps^-1
                T2_inv = T2_inv * 1.E-6
            else:
                T2_inv = 1./p1[3]
                # mu s^-1 units
                T2_inv = T2_inv * 1.E-6
                # ps^-1 units
        except RuntimeError:
            T2_inv = None
        return T2_inv
# -------------------------------------------------------------
# abstract subclass of the fitting model
# template for homo/inhomo dynamical calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_class(T2_eval_fit_model_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_class, self).__init__()
    def evaluate_T2(self, D2, tauc_ps):
        # compute r = Delta tau_c
        # dynamical calculations -> r << 1
        r = np.sqrt(D2) / hbar * tauc_ps
        if r < 1.E-4:
            # lorentzian decay
            T2_inv = D2 / hbar ** 2 * tauc_ps
            # ps^-1
        else:
            log.warning("dynamica calc: r >~ 1 : " + str(r))
            T2_inv = None
        return T2_inv
# -------------------------------------------------------------
# subclass of the fitting model
# to be used for dynamical calculation -> different fitting
# time domain wrt to static calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_homo_class(T2_eval_fit_model_dyn_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_homo_class, self).__init__()
    def parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(iT, D2)
        self.tauc_obj.set_tauc(iT, tauc_ps)
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        self.T2_obj.set_T2_sec(iT, T2_inv)
        # lw obj
        self.lw_obj.set_lw(iT, T2_inv)
        return Ct, ft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, iT, tauc_ps)
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        self.T2_obj.set_T2_atr(ia, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_atr(ia, iT, T2_inv)
        return Ct, ft
    # ph. resolved version
    def phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,iph,iT])
        # parametrize acf
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, iT, tauc_ps)
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        self.T2_obj.set_T2_phr(iph, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_phr(iph, iT, T2_inv)
        return Ct, ft
# -------------------------------------------------------------
# subclass of the fitting model
# to be used for dynamical inhomo calculation -> different fitting
# time domain wrt to static calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_inhom_class(T2_eval_fit_model_dyn_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_inhom_class, self).__init__()
    def parameter_eval_driver(self, acf_obj, ic, iT):
        acf_oft = np.zeros(p.nt)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(ic, iT, D2)
        self.tauc_obj.set_tauc(ic, iT, tauc_ps)
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        # store results in objects
        self.T2_obj.set_T2_sec(ic, iT, T2_inv)
        # lw object
        self.lw_obj.set_lw(ic, iT, T2_inv)
        return Ct, ft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, ic, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, ic, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, ic, iT, tauc_ps)
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        self.T2_obj.set_T2_atr(ia, ic, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_atr(ia, ic, iT, T2_inv)
        return Ct, ft
# -------------------------------------------------------------
# subclass of the fitting model
# to be used for static calculation -> different fitting
# -------------------------------------------------------------
class T2_eval_fit_model_stat_class(T2_eval_fit_model_class):
    def __init__(self):
        super(T2_eval_fit_model_stat_class, self).__init__()
    def parameter_eval_driver(self, acf_obj):
        # store acf_oft
        acf_oft = acf_obj.acf
        # parametrize acf_oft
        D2, tauc_mus, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(ic, iT, D2)
        self.tauc_obj.set_tauc(ic, tauc_mus)
        # convert to psec
        tauc_ps = tauc_mus * 1.E+6
        # compute T2_inv
        T2_inv = self.evaluate_T2(D2, tauc_ps)
        self.T2_obj.set_T2_psec(ic, T2_inv)
        # lw obj.
        self.lw_obj.set_lw(ic, T2_inv)
        return Ct, ft
    def evaluate_T2(self, D2, tauc_ps):
        pass
#
# generate initial parameters function
def generate_initial_params(r, D2, tau_c):
    #
    p0 = []
    if r < 1.:
        a = 1.
        b = 0.
    else:
        a = 0.
        b = 1.
    p0.append(a)
    p0.append(b)
    c0 = D2 / hbar**2 * tau_c    # ps^-1
    p0.append(c0)
    s0 = hbar / np.sqrt(D2)      # ps
    p0.append(s0)
    #
    return p0
#
# class T2_eval definition
#
class T2_eval:
    # initialization calculation
    # parameters
    def __init__(self):
        # fft sample points
        self.N = p.N_df
        # sample spacing -> ps
        self.T = p.T_df
        # max iteration curve fitting
        self.maxiter = p.maxiter
    # extract T2 fulle
    def extract_T2(self, t, Ct, D2):
        # check Ct finite
        if not np.isfinite(Ct).all():
            return [None, None, None]
        # compute exp
        tau_c1, T2_inv1, ft1 = self.extract_T2_Exp(t, Ct, D2)
        # compute exp sin
        tau_c2, T2_inv2, ft2 = self.extract_T2_ExpSin(t, Ct, D2)
        # final object
        tau_c = np.array([tau_c1, tau_c2], dtype=object)
        T2_inv = np.array([T2_inv1, T2_inv2], dtype=object)
        ft = [ft1, ft2]
        return tau_c, T2_inv, ft
    #
    # T2 calculation methods
    #
    # 1)  compute T2
    # input : t, Ct, D2
    # output: tauc, T2_inv, [exp. fit]
    def extract_T2_Exp(self, t, Ct, D2):
        # t : time array
        # Ct : acf
        # D2 : Delta^2
        #
        # fit over exp. function
        p0 = 1    # start with values near those we expect
        res = scipy.optimize.curve_fit(Exp, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (ps^-1)
        tau_c = 1./p[0]
        # ps units
        r = np.sqrt(D2) / hbar * tau_c
        # check r size
        # see Mukamel book
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c
            # ps^-1
        elif r > 1.E4:
            T2_inv = np.sqrt(D2) / hbar
            # ps^-1
        else:
            N = self.N
            T = self.T
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y = exp_gt(x, D2, tau_c)
            try:
                init_param = generate_initial_params(r, D2, tau_c)
                res = scipy.optimize.curve_fit(Explg, x, y, init_param, maxfev=self.maxiter, xtol=1.E-3, ftol=1.E-3)
                p3 = res[0]
                # gauss vs lorentzian
                if p3[0] > p3[1]:
                    # T2 -> lorentzian (psec)
                    T2_inv = p3[2]
                else:
                    T2_inv = 1./p3[3]
                    # ps^-1 units
            except RuntimeError:
                # set T2_inv1 to None
                log.warning("T2_inv is None")
                T2_inv = None
        #
        return tau_c, T2_inv, Exp(t, p)
    #
    # function 2 -> compute T2
    #
    # input : t, Ct, D2
    # output: tauc, T2_inv, [expsin, fit]
    def extract_T2_ExpSin(self, t, Ct, D2):
        # t : time array
        # Ct : acf
        # D2 : Delta^2
        #
        # fit over exp. function
        p0 = [1., 1., 1.]      # start with values near those we expect
        res = scipy.optimize.curve_fit(ExpSin, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (ps^-1)
        tau_c = 1. / p[2]      # ps
        # r -> Mukamel book
        r = np.sqrt(D2) / hbar * tau_c
        #
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c
            # ps^-1
        elif r > 1.E4:
            T2_inv = np.sqrt(D2) / hbar
        else:
            N = self.N
            T = self.T
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y = exp_gt(x, D2, tau_c)
            try:
                # perform fit
                init_param = generate_initial_params(r, D2, tau_c)
                res = scipy.optimize.curve_fit(Explg, x, y, init_param, maxfev=self.maxiter, xtol=1.E-3, ftol=1.E-3)
                # gauss vs lorentzian
                p3 = res[0]
                if p3[0] > p3[1]:
                    # T2 -> lorentzian (psec)
                    T2_inv = p3[2]
                else:
                    T2_inv = 1./p3[3]
                    # ps^-1 units
            except RuntimeError:
                # set T2_inv to None
                log.warning("T2_inv is None")
                T2_inv = None
        #
        return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])