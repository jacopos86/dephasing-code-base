import numpy as np
from abc import ABC, abstractmethod
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.utilities.plot_functions import plot_ph_pulse

#
#    Initialize phonon drive object
#

def set_up_phonon_drive(ph_drive_params, ph, qgr):
    if not ph_drive_params.get("enabled", False):
        return
    # envelope
    env_data = ph_drive_params["envelope"]
    # amplitude
    ampl_data = ph_drive_params["amplitude"]
    amplitude = build_amplitude(ampl_data, ph, qgr)
    # envelope
    env_type = env_data["type"].lower()
    if env_type == "cw":
        phdr = MonochromaticPhononDrive(
            omega_d=env_data["omega_d"],
            amplitude=amplitude,
            phase=env_data.get("phase", 0.0)
        )
    elif env_type == "gaussian":
        phdr = GaussianPhononPulse(
            t0=env_data["t0"],
            sigma_t=env_data["sigma_t"],
            amplitude=amplitude,
            omega_d=env_data.get("omega_d", 0.0),
            phase=env_data.get("phase", 0.0)
        )
    else:
        raise ValueError(f"Unknown envelope type: {env_type}")
    phdr.plot(p.evol_params)
    return phdr

#
#    build amplitude -> function
#

def build_amplitude(ampl_data, ph, qgr):
    """Return amplitude array (nq, nmd) from YAML config"""
    ampl_type = ampl_data["modes_profile"].lower()
    if ampl_type not in ("constant", "selective", "gaussian"):
        log.error("ampl_type must be: constant / selective / gaussian")
    # set amplitude
    max_val = ampl_data["max_value"]
    nq = len(qgr)
    A = np.zeros((nq, ph.nmodes))
    if ampl_type == "constant":
        A[:,:] = max_val
    # selective mode
    elif ampl_type == "selective":
        for q in ampl_data["q_indices"]:
            A[q, :] = max_val
    # gaussian amplitude
    elif ampl_type == "gaussian":
        q0 = ampl_data["q0"]
        sigma_q = ampl_data["sigma_q"]
        # q grid
        qs = np.arange(nq)
        env = np.exp(-(qs - q0)**2 / (2*sigma_q**2))
        A[:,:] = max_val * env[:, None]
    else:
        log.error(f"Unknown amplitude mode: {ampl_type}")
    return A

#
#    Phonon external pulse module
#    THz pulse phonon field
#

class PhononExternalField(ABC):
    @abstractmethod
    def set_force(self, t):
        """ Return F_q(t) with shape (nq, nmd), complex """
        raise NotImplementedError
    def plot(self, evol_params):
        dt = evol_params.get("time_step")                # ps
        nt = evol_params.get("num_steps")
        # time grid
        tgrid = np.arange(nt) * dt
        Fq = self.set_force(tgrid)                       # expect shape (nq, nmd) or broadcastable
        # plot ph pulse
        plot_ph_pulse(tgrid, Fq)

#
#     External CW perturbation
#

class MonochromaticPhononDrive(PhononExternalField):
    def __init__(self, omega_d, amplitude, phase):
        self.omega_d = omega_d
        self.amplitude = amplitude
        self.phase = phase
    # set force field
    def set_force(self, t):
        if np.isscalar(t):
            return self.amplitude * np.sin(self.omega_d * t + self.phase)
        else:
            return self.amplitude[:,:,None] * np.sin(self.omega_d * t + self.phase)[None,None,:]

#
#     External Gaussian pulse
#

class GaussianPhononPulse(PhononExternalField):
    def __init__(self, t0, sigma_t, amplitude, omega_d, phase):
        self.t0 = t0
        self.sigma = sigma_t
        self.amplitude = amplitude
        self.omega_d = omega_d
        self.phase = phase
    # set external force
    def set_force(self, t):
        env = np.exp(-(t - self.t0)**2 / (2*self.sigma**2)) * np.sin(self.omega_d * t + self.phase)
        if np.isscalar(t):
            return self.amplitude * env
        else:
            return self.amplitude[:,:,None] * env[None,None,:]