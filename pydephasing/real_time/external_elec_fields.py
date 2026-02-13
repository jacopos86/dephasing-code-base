
import numpy as np
from abc import ABC, abstractmethod
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.utilities.plot_functions import plot_A_pulse

#
#    Initialize phonon drive object
#

def set_up_vector_potential(ext_Apot_params):
    if not ext_Apot_params.get("enabled", False):
        return
    # envelope
    env_data = ext_Apot_params["envelope"]
    # amplitude
    amplitude = ext_Apot_params["amplitude"]
    # envelope
    env_type = env_data["type"].lower()
    if env_type == "cw":
        Apulse = MonochromaticPulse(
            omega_d=env_data["omega_d"],
            amplitude=amplitude,
            phase=env_data.get("phase", 0.0)
        )
    elif env_type == "gaussian":
        Apulse = GaussianPulse(
            t0=env_data["t0"],
            sigma_t=env_data["sigma_t"],
            amplitude=amplitude,
            omega_d=env_data.get("omega_d", 0.0),
            phase=env_data.get("phase", 0.0)
        )
    else:
        raise ValueError(f"Unknown envelope type: {env_type}")
    Apulse.plot(p.evol_params)
    return Apulse

#
#    Phonon external pulse module
#    THz pulse phonon field
#

class ExternalVectorField(ABC):
    @abstractmethod
    def set_A(self, t):
        """ Return F_q(t) with shape (nq, nmd), complex """
        raise NotImplementedError
    def plot(self, evol_params):
        dt = evol_params.get("time_step")                # ps
        nt = evol_params.get("num_steps")
        # time grid
        tgrid = np.arange(nt) * dt
        A_t = self.set_A(tgrid)                       # expect shape (nq, nmd) or broadcastable
        # plot ph pulse
        plot_A_pulse(tgrid, A_t)

#
#     External CW perturbation
#

class MonochromaticPulse(ExternalVectorField):
    def __init__(self, omega_d, amplitude, phase):
        self.omega_d = omega_d
        self.amplitude = amplitude
        self.phase = phase
    # set force field
    def set_A(self, t):
        if np.isscalar(t):
            return self.amplitude * np.sin(self.omega_d * t + self.phase)
        else:
            return self.amplitude * np.sin(self.omega_d * t + self.phase)[:]

#
#     External Gaussian pulse
#

class GaussianPulse(ExternalVectorField):
    def __init__(self, t0, sigma_t, amplitude, omega_d, phase):
        self.t0 = t0
        self.sigma = sigma_t
        self.amplitude = amplitude
        self.omega_d = omega_d
        self.phase = phase
    # set external force
    def set_A(self, t):
        env = np.exp(-(t - self.t0)**2 / (2*self.sigma**2)) * np.sin(self.omega_d * t + self.phase)
        if np.isscalar(t):
            return self.amplitude * env
        else:
            return self.amplitude * env[:]