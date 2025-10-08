from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
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