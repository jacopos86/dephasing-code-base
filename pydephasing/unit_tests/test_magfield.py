import numpy as np
import pytest
from pydephasing.magnetic_field import MagneticField
from pydephasing.common.units import T

#
#   magnetic field unit test
#

def test_magnetic_field_1():
    # B0 as list in Tesla
    B0_list = [0.1, 0.2, 0.3]
    mf = MagneticField(B0=B0_list)
    assert isinstance(mf.B0, np.ndarray)
    assert np.allclose(mf.B0, np.array(B0_list))
    assert mf.unit == T

def test_magnetic_field_2():
    # B0 in milliTesla, should convert to Tesla
    B0_mT = [100, 200, 300]  # mT
    mf = MagneticField(B0=B0_mT, unit="millitesla")
    assert np.allclose(mf.B0, np.array([0.1, 0.2, 0.3]))

def test_magnetic_field_3():
    # B0 must be length 3
    with pytest.raises(ValueError):
        MagneticField(B0=[0.1, 0.2])  # only 2 elements

def test_magnetic_field_4():
    Bt = {"expr_x": "sin(x)", "expr_z": "cos(x)"}
    mf = MagneticField(B0=[0,0,0], Bt_expr=Bt, var="x")
    assert mf.Bt_expr["expr_x"] == "sin(x)"
    assert mf.Bt_expr["expr_y"] == "0"
    assert mf.Bt_expr["expr_z"] == "cos(x)"
    assert mf.Bt_expr["var"] == "x"

def test_magnetic_field_5():
    # Bt_expr is None -> stays None
    mf = MagneticField(B0=[0,0,0], Bt_expr=None)
    assert mf.Bt_expr is None