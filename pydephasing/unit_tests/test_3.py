from pydephasing.phonons_module import PhononsClass
import pytest
from pydephasing.common.phys_constants import kb

#
#  test phonons_module
#

def test_ph_occupations():
    ph = PhononsClass()
    n = ph.ph_occup(0., 0.)
    assert n == pytest.approx(0., abs=1.e-7)
    n = ph.ph_occup(1., 0.)
    assert n == pytest.approx(0., abs=1.e-7)
    n = ph.ph_occup(0.01, 10.)
    assert n == pytest.approx(9.12485E-6, abs=1.e-7)
    n = ph.ph_occup(0.01, 100.)
    assert n == pytest.approx(0.4563345, abs=1.e-7)
    n = ph.ph_occup(0.01, 500.)
    assert n == pytest.approx(3.82799015, abs=1.e-7)