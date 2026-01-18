from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from pydephasing.quantum.qubitization_module import PauliTerm
from pydephasing.quantum.pauli_polynomial_class import fermion_plus_operator, fermion_minus_operator
import pytest

#
#  QISKIT test unit
#

def test_hadamard():
    # create quantum circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    # simulate circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=2000)
    result = job.result()
    counts = result.get_counts()
    # assert outcomes
    assert '0' in counts
    assert '1' in counts
    # assert probability ~ 0.5 for both
    p0 = counts['0'] / 2000
    p1 = counts['1'] / 2000
    assert p0 == pytest.approx(0.5, abs=0.05)
    assert p1 == pytest.approx(0.5, abs=0.05)

def test_bell_circuit():
    # Bell circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    assert qc.width() == 4
    assert qc.size() == 4
    # run circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=2000)
    result = job.result()
    counts = result.get_counts()
    # assert results
    assert '00' in counts
    assert '11' in counts
    assert counts['00'] + counts['11'] > 900  # > 90% fidelity

def test_pauli_words():
    pt1 = PauliTerm(5, ps='eexyz', pc=1.)
    assert pt1.p_coeff == pytest.approx(1., abs=1.e-7)
    assert pt1.pw[0].symbol == 'e'
    assert pt1.pw[0].phase == pytest.approx(1., abs=1.e-8)
    assert pt1.pw[2].symbol == 'x'
    assert pt1.pw[2].phase == pytest.approx(1., abs=1.e-8)
    assert pt1.pw[4].symbol == 'z'
    assert pt1.pw[4].phase == pytest.approx(1., abs=1.e-8)

def test_pauli_term_product():
    pt1 = PauliTerm(5, ps='eexyz', pc=1.)
    pt2 = PauliTerm(5, ps='xyezy', pc=1j)
    r = pt1 * pt2
    assert r.p_coeff.real == pytest.approx(0., abs=1.e-7)
    assert r.p_coeff.imag == pytest.approx(1., abs=1.e-7)
    assert r.pw[0].symbol == 'x'
    assert r.pw[1].symbol == 'y'
    assert r.pw[2].symbol == 'x'
    assert r.pw[3].symbol == 'x'
    assert r.pw[4].symbol == 'x'
    # replace pt1
    pt3 = PauliTerm(5, ps='zzzzz', pc=1.)
    pt1 *= pt3
    assert pt1.p_coeff.real == pytest.approx(1., abs=1.e-7)
    assert pt1.p_coeff.imag == pytest.approx(0., abs=1.e-7)
    assert pt1.pw[0].symbol == 'z'
    assert pt1.pw[1].symbol == 'z'
    assert pt1.pw[2].symbol == 'y'
    assert pt1.pw[3].symbol == 'x'
    assert pt1.pw[4].symbol == 'e'

def test_pauli_pol_reduction():
    f2q_mode = "JW"
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    pp = c_jdagg.return_polynomial()
    assert len(pp) == 2
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(-0.5, 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'y'
    assert pp[1].p_coeff.real == pytest.approx(0.5, 1.e-7)
    assert pp[1].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp[1].pw[0].symbol == 'e'
    assert pp[1].pw[1].symbol == 'e'
    assert pp[1].pw[2].symbol == 'x'
    c_jdagg2= fermion_plus_operator(f2q_mode, 3, 0)
    c_jdagg += c_jdagg2
    pp2 = c_jdagg.return_polynomial()
    assert len(pp2) == 2
    assert pp2[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp2[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp2[0].pw[0].symbol == 'e'
    assert pp2[0].pw[1].symbol == 'e'
    assert pp2[0].pw[2].symbol == 'x'
    assert pp2[1].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp2[1].p_coeff.imag == pytest.approx(-1., 1.e-7)
    assert pp2[1].pw[0].symbol == 'e'
    assert pp2[1].pw[1].symbol == 'e'
    assert pp2[1].pw[2].symbol == 'y'
    c_j = fermion_minus_operator(f2q_mode, 3, 0)
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    cc_j = c_j + c_jdagg
    pp3 = cc_j.return_polynomial()
    assert len(pp3) == 1
    assert pp3[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp3[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp3[0].pw[0].symbol == 'e'
    assert pp3[0].pw[1].symbol == 'e'
    assert pp3[0].pw[2].symbol == 'x'

def test_pauli_pol_product():
    f2q_mode = "JW"
    c_j = fermion_minus_operator(f2q_mode, 3, 0)
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    cc_j = c_j + c_jdagg
    cc_j2 = cc_j * cc_j
    pp = cc_j2.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'
    # test product complex number
    cc_j3 = 1j * cc_j2
    pp = cc_j3.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(1., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'
    cc_j4 = 2. * cc_j3
    pp = cc_j4.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(2., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'