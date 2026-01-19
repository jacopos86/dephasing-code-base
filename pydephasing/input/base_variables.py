import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from pydephasing.common.matrix_operations import norm_cmplxv
from pydephasing.common.units import ps

@dataclass
class TimeGrid:
    # internal unit of time : ps
    T: object
    dt: object
    nt: int
    def __post_init__(self):
        self._validate_dimensions()
        self._normalize_units()
        self._validate_values()
        self._validate_size_consistency()
    def _validate_dimensions(self):
        if not self.T.check(ps):
            raise ValueError("T must have dimensions of time")
        if not self.dt.check(ps):
            raise ValueError("dt must have dimensions of time")
        if not isinstance(self.nt, int):
            raise TypeError("nt must be integer")
    def _normalize_units(self):
        self.T = self.T.to(ps)
        self.dt = self.dt.to(ps)
    def _validate_values(self):
        if self.T.magnitude <= 0:
            raise ValueError("T must be positive")
        if self.dt.magnitude <= 0:
            raise ValueError("dt must be positive")
        if self.nt <= 0:
            raise ValueError("nt must be positive")
    def _validate_size_consistency(self):
        exp_T = self.dt.magnitude * self.nt
        if not (0.999 * exp_T <= self.T.magnitude <= 1.001 * exp_T):
            raise ValueError(
                f"Inconsistent TimeGrid: T={self.T.magnitude} ps, "
                f"dt={self.dt.magnitude} ps, nt={self.nt}, "
                f"expected total T ≈ {exp_T} ps"
            )

@dataclass
class TemperatureGrid:
    # internally everything in K
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    def __post_init__(self):
        self._validate_and_normalize()
    def _validate_and_normalize(self):
        temps = []
        for t in self.temperatures:
            if not isinstance(t, Q_):
                t = Q_(t, K)
            if t.check(ureg.degC):
                t = t.to(K)
            if not t.check(K):
                raise ValueError(f"temperature {t} has wrong dimension")
            temps.append(t.magnitude)
        self.temperatures = np.array(temps, dtype=float)
    @property
    def ntmp(self):
        return len(self.temperatures)

@dataclass
class MagneticField:
    B0: np.ndarray          # static field (Tesla)
    Bt: Optional[dict] = None

@dataclass
class InitialState:
    psi0: np.ndarray
    def normalize(self):
        self.psi0 = self.psi0 / norm_cmplxv(self.psi0)

