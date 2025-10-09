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