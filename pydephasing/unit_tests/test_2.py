from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from quantum.qubitization_module import PauliTerm
from quantum.pauli_polynomial_class import fermion_plus_operator, fermion_minus_operator, PauliPolynomial
import pytest
import numpy as np
import matplotlib.pyplot as plt

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

def test_Ising_model():
    # Define the Ising Hamiltonian H = -J * Z_i Z_(i+1)
    # Let's use J=1 and apply it to a 2-qubit system for simplicity.
    J = 1
    nq = 2
    # H = - Z0 Z1
    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -J)
    ])
    # exact solution
    exact_ener = -J
    # -----------------------
    # Ansatz
    # -----------------------
    ansatz = TwoLocal(
        num_qubits=nq,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="full",
        reps=1
    )
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    assert exact_ener == pytest.approx(energy, 1.e-4)
    # exact simulation 3 qubits
    J1 = 1
    J2 = 0.5
    nq = 3
    # H = -J1 Z0 Z1 -J2 Z1 Z2
    hamiltonian = SparsePauliOp.from_list([
        ("ZZI", -J1),
        ("IZZ", -J2)
    ])
    # exact solution
    exact_ener2 = -J1 - J2
    # -----------------------
    # Ansatz
    # -----------------------
    ansatz = TwoLocal(
        num_qubits=nq,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="full",
        reps=1
    )
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result2 = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy2 = result2.eigenvalue.real
    assert exact_ener2 == pytest.approx(energy2, 1.e-4)

def test_reduced_Hubbard_dimer():
    pass

def test_Hubbard_dimer():
    f2q_mode = "JW"
    nq = 4
    t = 1.   # hopping coeff.
    U = 2.   # Hubbard parameter
    H = PauliPolynomial(f2q_mode)
    # 4 qubits |1u>, |1d>, |2u>, |2d>
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 2)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 0)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 3)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 1)
    Hpol = H.return_polynomial()
    pauli_list = []
    for it in range(len(Hpol)):
        pt = Hpol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    Hop = SparsePauliOp.from_list(pauli_list)
    # add penalty term to impose half filling
    lmbd = 10.0
    N = PauliPolynomial(f2q_mode)
    N += fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 0)
    N += fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 1)
    N += fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 2)
    N += fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 3)
    H += lmbd * (N - 2.0) ** 2
    Sz = PauliPolynomial(f2q_mode)
    Sz += 0.5 * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 0)
    Sz -= 0.5 * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 1)
    Sz += 0.5 * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 2)
    Sz -= 0.5 * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 3)
    H += lmbd * Sz ** 2
    # pauli list
    Hpol = H.return_polynomial()
    pauli_list = []
    for it in range(len(Hpol)):
        pt = Hpol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    # set hamiltonian
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    Npol = N.return_polynomial()
    pauli_list = []
    for it in range(len(Npol)):
        pt = Npol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    Nop = SparsePauliOp.from_list(pauli_list)
    Spol = Sz.return_polynomial()
    pauli_list = []
    for it in range(len(Spol)):
        pt = Spol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    Sop = SparsePauliOp.from_list(pauli_list)
    # exact solution
    exact_ener = -2*t
    # -----------------------
    # Ansatz
    # -----------------------
    ansatz = TwoLocal(
        num_qubits=nq,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="full",
        reps=1
    )
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    print(energy, exact_ener)
    N_expect = estimator.run(
        circuits=[result.optimal_circuit],
        observables=[Nop],
        parameter_values=[list(result.optimal_parameters.values())],
    ).result().values[0]
    print("⟨N⟩ =", N_expect)
    S_expect = estimator.run(
        circuits=[result.optimal_circuit],
        observables=[Sop],
        parameter_values=[list(result.optimal_parameters.values())],
    ).result().values[0]
    print("⟨Sz⟩ =", S_expect)
    H_expect = estimator.run(
        circuits=[result.optimal_circuit],
        observables=[Hop],
        parameter_values=[list(result.optimal_parameters.values())],
    ).result().values[0]
    print("⟨H⟩ =", H_expect)
    H = PauliPolynomial(f2q_mode)
    # 4 qubits |1u>, |1d>, |2u>, |2d>
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 2)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 0)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 3)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 1)
    Hpol = H.return_polynomial()
    for it in range(len(Hpol)):
        pt = Hpol[it].pw2sparsePauliOp()
        print(pt)
    Hmat = Hop.to_matrix()
    basis = [
        0b1010,  # |1↑ 2↓>
        0b0101,  # |1↓ 2↑>
        0b1100,  # |1↑1↓>
        0b0011,  # |2↑2↓>
    ]
    Hred = Hmat[np.ix_(basis, basis)]
    eigvals = np.linalg.eigvalsh(Hred)
    print(eigvals)
    #assert exact_ener == pytest.approx(energy, 1.e-4)