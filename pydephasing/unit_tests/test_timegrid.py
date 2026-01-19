from pydephasing.input.base_variables import TimeGrid
from pydephasing.common.units import ns, fs, ps, eV
import pytest

def test_timegrid_1():
    tgr = TimeGrid(
        T  = 1.0 * ns,
        dt = 10.0 * fs,
        nt = 100000
    )
    assert tgr.T.units == ps
    assert tgr.dt.units == ps
    assert tgr.T.magnitude == pytest.approx(1000., abs=1.e-8)
    assert tgr.dt.magnitude == pytest.approx(0.01, abs=1.e-8)

def test_timegrid_2():
    with pytest.raises(ValueError):
        TimeGrid(
            T  = 1.0 * eV,
            dt = 1.0 * ps,
            nt = 10
        )

def test_timegrid_3():
    with pytest.raises(ValueError):
        TimeGrid(
            T  =-1.0 * ps,
            dt = 1.0 * ps,
            nt = 10
        )

def test_timegrid_4():
    with pytest.raises(ValueError):
        TimeGrid(
            T  = 1.0 * ps,
            dt = 0.2 * ps,
            nt = 10
        )