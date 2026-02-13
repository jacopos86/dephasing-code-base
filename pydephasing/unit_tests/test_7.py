import os
import numpy as np
from types import SimpleNamespace
from pydephasing.q_grid import phonopy_qgridClass
from pydephasing.utilities.input_parser import parser

#
#   test Phonopy q grid class
#

TESTS_DIR = os.environ.get("TESTS_DIR")
if TESTS_DIR is None:
    raise EnvironmentError("TESTS_DIR environment variable is not set")

def test_irred_qqplist(monkeypatch):
    # set parameters
    yml_inp = TESTS_DIR+"/2/inputE.yml"
    # Simulate CLI arguments
    # Create a dummy namespace with all arguments _init() expects
    dummy_args = SimpleNamespace(
        co=["spin-qubit"],
        ct1=["LR"],
        ct2="homo",
        yml_inp=yml_inp
    )
    # Monkeypatch parser.parse_args to return our dummy args
    parser.parse_args = lambda *a, **kw: args
    # Monkeypatch parser.parse_args globally
    monkeypatch.setattr(parser, "parse_args", lambda *a, **kw: dummy_args)
    from pydephasing.set_param_object import p
    p.read_yml_data(yml_inp)
    qgr = phonopy_qgridClass()
    qgr.set_qgrid()
    assert qgr.nq == 27
    qqplst = qgr.build_qqp_pairs()
    assert len(qqplst) == 729
    qlst = qgr.set_q2mq_list()
    assert len(qlst) == 14
    for (iq1,iq2) in qlst:
        np.testing.assert_allclose(qgr.qpts[iq1], -qgr.qpts[iq2])
    irr_qlst = qgr.build_irred_qqp_pairs()
    assert len(irr_qlst) == (1 + (qgr.nq*qgr.nq - 1)/2)