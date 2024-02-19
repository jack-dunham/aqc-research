# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for manipulating the state vectors in MPS format.
"""
import logging
import re
from random import choice
from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import aqc_research.checking as chk
import aqc_research.utils as helper
from aqc_research.circuit_structures import create_ansatz_structure
from aqc_research.circuit_transform import ansatz_to_qcircuit
from aqc_research.parametric_circuit import ParametricCircuit

# Method had been moved in AirSimulator in the latest Qiskit. FIXME
# Instance of 'QuantumCircuit' has no 'set_matrix_product_state' memberPylint(E1101:no-member)


logger = logging.getLogger(__name__)

_NO_TRUNCATION_THR = 1e-16

# Type of MPS data as it outputted by Qiskit.
QiskitMPS = Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]


# class MPSdata:
#     """Class keep *preprocessed* MPS data, see _preprocess_mps() for details."""
#
#     def __init__(
#         self,
#         mps_matrices: List[np.ndarray],
#         mps_phase: Optional[float] = 0.0,
#         state: Optional[np.ndarray] = None,
#     ):
#         assert chk.is_list(mps_matrices)
#         assert len(mps_matrices) == 0 or chk.complex_3d(mps_matrices[0])
#         assert chk.is_float(mps_phase)
#         assert state is None or chk.complex_1d(state)
#
#         self._matrices = mps_matrices
#         self._phase = mps_phase
#         self._state = state
#         assert self._is_consistent()
#
#     def _is_consistent(self) -> bool:
#         if self._state is None or len(self._matrices) == 0:
#             return True
#         return self._state.size == 2 ** len(self._matrices)
#
#     @property
#     def num_qubits(self) -> int:
#         if len(self.matrices) > 0:
#             return len(self._matrices)
#         elif self._state is not None and self._state.size > 0:
#             return int(round(np.log2(float(self._state.size))))
#         else:
#             return int(0)
#
#     @property
#     def matrices(self) -> List[np.ndarray]:
#         return self._matrices
#
#     @property
#     def phase(self) -> float:
#         return self._phase
#
#     @property
#     def state(self) -> Union[np.ndarray, None]:
#         return self._state


def no_truncation_threshold() -> float:
    """Returns "no truncation" threshold value for MPS computation."""
    return _NO_TRUNCATION_THR


def check_mps(qiskit_mps: QiskitMPS) -> bool:
    """
    Checks that the input argument is really an MPS decomposition produced
    by Qiskit framework.

    Args:
        qiskit_mps: state in MPS representation.

    Returns:
        True in case of valid input structure.
    """
    if not (isinstance(qiskit_mps, tuple) and len(qiskit_mps) == 2):
        return False  # expects MPS as a tuple of size 2

    num_qubits = len(qiskit_mps[0])
    gam, lam = qiskit_mps[0], qiskit_mps[1]  # Gamma and diagonal Lambda matrices
    if len(gam) != num_qubits:
        return False  # wrong number of Gamma matrices
    if len(lam) != num_qubits - 1:
        return False  # wrong number of Lambda matrices

    for n in range(num_qubits):
        g_n = gam[n]
        if g_n[0].ndim != 2:
            return False  # Gamma is not a matrix
        if len(g_n) != 2:
            return False  # expects 2 Gammas per qubit
        if g_n[0].shape != g_n[1].shape:
            return False  # unequal shapes of Gammas
        if n < num_qubits - 1:
            l_n = lam[n]
            if not (l_n.ndim == 1 or (l_n.ndim == 2 and min(l_n.shape) == 1)):
                return False  # expects vector input
            l_n = l_n.ravel()
            if not np.all(l_n[:-1] >= l_n[1:]):
                return False  # expects vector sorted in descending order
    return True


def _preprocess_mps(qiskit_mps: QiskitMPS, conjugate: bool = False) -> List[np.ndarray]:
    """
    Converts Qiskit MPS representation is more compact, custom one, which is
    better suitable for further computation. Namely, we combine Gamma matrices
    for both bit states 0/1 into a single 3D tensor and multiply it by the
    subsequent diagonal Lambda matrix (except the very last Gamma).

    Args:
        qiskit_mps: MPS decomposition of a quantum state as it comes out from
                    Qiskit framework.
        conjugate: optional conjugation of the output MPS tensors.

    Returns:
        list of size ``num_qubits`` of MPS tensors.
    """
    assert check_mps(qiskit_mps) and isinstance(conjugate, bool)
    num_qubits = len(qiskit_mps[0])
    gam, lam = qiskit_mps[0], qiskit_mps[1]  # Gamma and diagonal Lambda matrices
    my_mps = list([])
    for n in range(num_qubits):
        g_n = gam[n]
        g_n = np.stack((g_n[0], g_n[1]))  # combine into a single tensor
        if n < num_qubits - 1:
            g_n *= np.expand_dims(lam[n], (0, 1))  # gamma <-- gamma * lambda

        if conjugate:
            my_mps.append(np.conjugate(g_n, out=g_n))
        else:
            my_mps.append(g_n)

    return my_mps


def mps_to_vector(
    qiskit_mps: QiskitMPS, already_preprocessed: Optional[bool] = False
) -> np.ndarray:
    """
    Computes coefficients ``coef`` of individual quantum basis states:
    ``coef_{i1,i2,...,in} * |i1> kron |i2> kron ... kron |in>``, and
    builds a state vector from MPS representation.

    **Note**, the operation is slow and time-consuming for a large number
    of qubits. It was designed primarily for testing.

    Args:
        qiskit_mps: MPS decomposition of the state produced by Qiskit framework.
        already_preprocessed: Set to True if qiskit_mps has been preprocessed already

    Returns:
        quantum state as a vector of size ``2^n``.
    """
    if already_preprocessed:
        mps = qiskit_mps
    else:
        mps = _preprocess_mps(qiskit_mps)

    num_qubits = len(mps)
    state = np.zeros(2**num_qubits, dtype=np.cfloat)
    for k in range(state.size):  # for all combinations of individual bits ...
        coef = None
        for n in range(num_qubits):
            b = (k >> n) & 1  # k-th bit state (0/1) at qubit 'n'
            g_n = mps[n][b, ...]  # Gamma at qubit 'n'
            if coef is None:
                coef = g_n.copy()
            else:
                coef = np.tensordot(coef, g_n, axes=([1], [0]))

        state[k] = coef.item()  # must be a scalar at the end

    return state


def mps_dot(
    qiskit_mps1: QiskitMPS, qiskit_mps2: QiskitMPS, already_preprocessed: Optional[bool] = False
) -> np.cfloat:
    """
    Computes dot product between MPS decompositions of two quantum states:
    ``< mps1 | mps2 >``.

    Args:
        qiskit_mps1: MPS decomposition of the left state.
        qiskit_mps2: MPS decomposition of the right state.
        already_preprocessed: Set to True if mps1 and mps2 have been preprocessed already

    Returns:
        complex dot product value.
    """
    if already_preprocessed == False:
        mat1, mat2 = _preprocess_mps(qiskit_mps1), _preprocess_mps(qiskit_mps2)
    else:
        mat1, mat2 = qiskit_mps1, qiskit_mps2

    a, b = np.squeeze(mat1[0], axis=1), np.squeeze(mat2[0], axis=1)
    a_b = np.tensordot(np.conj(a), b, axes=([0], [0]))

    assert len(mat1) == len(mat2)  # same number of qubits
    for n in range(1, len(mat1)):
        a_b = np.tensordot(a_b, np.conj(mat1[n]), axes=([0], [1]))
        a_b = np.tensordot(a_b, mat2[n], axes=([0, 1], [1, 0]))

    return np.cfloat(a_b.item())


def mps_expectation(
    qiskit_mps: QiskitMPS,
    operator: str,
    qubit_index: int,
    already_preprocessed: Optional[bool] = False,
):
    """
    Computes expectation of a Pauli operator (P) for a given qubit:
    ``< mps | P | mps >``.

    Args:
        qiskit_mps: MPS decomposition of the state.
        operator: 'X', 'Y' or 'Z': the Pauli operator P
        qubit_index: the qubit that P acts on
        already_preprocessed: Set to True if qiskit_mps has been preprocessed already

    Returns:
        real expectation value.
    """

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.0j], [1.0j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    if operator == "X":
        op = X
    elif operator == "Y":
        op = Y
    elif operator == "Z":
        op = Z

    if already_preprocessed:
        mat1, mat2 = qiskit_mps.copy(), qiskit_mps.copy()
    else:
        mat1, mat2 = _preprocess_mps(qiskit_mps), _preprocess_mps(qiskit_mps)

    # Apply op (Pauli X, Y, or Z) to qubit 'qubit_index'
    mat2[qubit_index] = np.tensordot(op, mat2[qubit_index], axes=([1], [0]))

    expectation = mps_dot(mat1, mat2, already_preprocessed=True)

    assert np.abs(np.imag(expectation)) <= 1e-10

    return np.real(expectation)


def partial_trace(
    qiskit_mps: QiskitMPS, qubits_to_keep: List, already_preprocessed: Optional[bool] = False
):
    """
    Given a system of qubits described by a density matrix ⍴_AB, take the partial trace with respect
    to B to obtain the reduced density matrix ⍴_A = tr_B(⍴_AB) = ∑_i\inb ⟨i|⍴_AB|i⟩
    :param qiskit_mps: MPS in the form outputted from Qiskit
    :param qubits_to_keep: List of qubits in A such that all other qubits are traced over
    :param already_preprocessed: Set to True if qiskit_mps has been preprocessed already
    :return: Reduced density matrix ⍴_A
    """
    if already_preprocessed:
        mat1 = qiskit_mps
    else:
        mat1 = _preprocess_mps(qiskit_mps)

    n_qubits = len(mat1)
    qubits_to_contract = [q for q in range(n_qubits) if q not in qubits_to_keep]

    # n=0 step
    # start with first block from top and bottom MPS.
    # remove redundant left legs.
    a, b = np.squeeze(mat1[0], axis=1), np.squeeze(mat1[0], axis=1)
    if 0 in qubits_to_contract:
        # contract physical leg from first top and bottom block
        a_b = np.tensordot(a, np.conj(b), axes=([0], [0]))
    else:
        # do not contract physical leg from blocks (tensor product).
        a_b = np.tensordot(a, np.conj(b), axes=([], []))
        # reorder to place uncontracted physical leg indices to left.
        a_b = np.moveaxis(a_b, [2], [1])

    for n in range(1, len(mat1)):
        a_b = np.tensordot(
            a_b, mat1[n], axes=([-2], [1])
        )  # contract inner legs of top MPS to inner leg of next block.
        if n in qubits_to_contract:
            # contract inner leg of bottom MPS to inner leg of new bottom block.
            # contract physical leg of top MPS with physical leg of new bottom block.
            a_b = np.tensordot(a_b, np.conj(mat1[n]), axes=([-3, -2], [1, 0]))
        else:
            # contract inner leg of bottom MPS to inner leg of new bottom block.
            a_b = np.tensordot(a_b, np.conj(mat1[n]), axes=([-3], [1]))
            # reorder to place uncontracted physical leg indices to left of uncontracted inner leg indices.
            a_b = np.swapaxes(a_b, -2, -3)

    # remove redundant open right legs
    a_b = np.squeeze(a_b, axis=-1)
    a_b = np.squeeze(a_b, axis=-1)

    n_qubit_kept = len(qubits_to_keep)

    # group all top physical legs to be left indices of array. bottom legs to right indices.
    # within groups of legs, reorder to be in reverse qubit notation to match qiskit.
    new_shape = list(reversed(range(n_qubit_kept * 2)[::2])) + list(
        reversed(range(n_qubit_kept * 2)[1::2])
    )
    a_b = np.transpose(a_b, new_shape)

    # reshape to be 2^n by 2^n density matrix.
    a_b = np.reshape(a_b, (2 ** len(qubits_to_keep), 2 ** len(qubits_to_keep)))
    return a_b


def mps_from_circuit(
    qc: QuantumCircuit,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
    out_state: Optional[np.ndarray] = None,
    print_log_data: Optional[bool] = False,
    return_preprocessed: Optional[bool] = False,
) -> QiskitMPS:
    """
    Computes MPS representation of output state (in Qiskit format) after quantum
    circuit acting on zero state: ``output = circuit @ |0>``.

    **Note**, this function modifies the input circuit in a way that one cannot
    apply it twice. Qiskit limitation: ``save_statevector()`` is applicable
    only once.

    Args:
        qc: quantum circuit that acts on state ``|0>``.
        trunc_thr: truncation threshold in MPS representation.
        out_state: output array for storing state as a normal vector; *note*,
                   state generation can be a slow and even intractable operation
                   for the large number of qubits; useful for testing only.
        print_log_data: flag enables printing of MPS internal information;
                        useful for debugging and testing.
        return_preprocessed: set to True to return preprocessed MPS

    Returns:
        MPS state representation as outputted by Qiskit framework.
    """
    assert isinstance(qc, QuantumCircuit)
    assert chk.is_float(trunc_thr, 0 <= trunc_thr <= 0.1)
    assert isinstance(print_log_data, bool)

    if isinstance(out_state, np.ndarray):
        qc.save_statevector(label="my_sv")
        assert chk.complex_1d(out_state, out_state.size == 2**qc.num_qubits)

    qc.save_matrix_product_state(label="my_mps")
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=trunc_thr,
        mps_log_data=print_log_data,
    )
    result = sim.run(qc, shots=1).result()
    data = result.data(0)

    if print_log_data:
        mps_log_string = result.results[0].metadata["MPS_log_data"]
        bond_dimensions = re.findall(r"\[(.*?)\]", mps_log_string)
        max_chi = max(map(int, " ".join(bond_dimensions).split()))
        print(mps_log_string)
        logger.debug(f"MPS created with max bond dimension {max_chi}")
    if isinstance(out_state, np.ndarray):
        np.copyto(out_state, np.asarray(data["my_sv"]))

    if return_preprocessed:
        return _preprocess_mps(data["my_mps"])
    else:
        return data["my_mps"]


def max_chi_from_circuit(
    qc: QuantumCircuit,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
) -> int:
    """
    Finds the maximum bond dimension at any stage of the computation when computing the MPS representation of output
    state (in Qiskit format) after quantum circuit acting on zero state: ``output = circuit @ |0>``.

    Args:
        qc: quantum circuit that acts on state ``|0>``.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        Maximum bond dimension
    """
    assert isinstance(qc, QuantumCircuit)
    assert chk.is_float(trunc_thr, 0 <= trunc_thr <= 0.1)

    qc_copy = qc.copy()

    N = qc_copy.num_qubits

    # Linear coupling map for transpilation
    linear_coupling = [0] * (N - 1)
    for i in range(N - 1):
        linear_coupling[i] = [i, i + 1]

    # Transpile to linear coupling, this means the simulator won't have to perform any internal swaps. As a result, the
    # number of of times the bond dimensions are logged is equal to the number of CNOTs in the transpiled circuit
    qc_copy = transpile(
        qc_copy,
        basis_gates=["cx", "rx", "ry", "rz"],
        coupling_map=linear_coupling,
        optimization_level=0,
    )
    num_2_qubit_gates = qc_copy.count_ops()["cx"]

    qc_copy.save_matrix_product_state(label="my_mps")
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=trunc_thr,
        mps_log_data=True,
    )
    result = sim.run(qc_copy, shots=1).result()

    mps_log_string = result.results[0].metadata["MPS_log_data"]
    bond_dimensions = re.findall(r"\[(.*?)\]", mps_log_string)

    # Remove all entries from previous circuits
    del bond_dimensions[:-num_2_qubit_gates]
    max_chi = max(map(int, " ".join(bond_dimensions).split()))

    return max_chi


def qcircuit_mul_mps(
    qc: QuantumCircuit,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
    out_state: Optional[np.ndarray] = None,
) -> QiskitMPS:
    """
    Applies quantum circuit to right-hand side state vector given in MPS
    format and returns augmented MPS, i.e. circuit times vector product.

    Args:
        qc: quantum circuit to be applied to a vector in MPS format.
        mps_vec: right-hand side vector to be multiplied by the circuit.
        trunc_thr: truncation threshold in MPS representation.
        out_state: output array for storing state as a normal vector;
                   *note*, this can be very slow and even intractable
                   for a large number of qubits; useful for testing only.

    Returns:
        product of circuit and right-hand vector in MPS format.
    """
    assert isinstance(qc, QuantumCircuit)
    assert chk.is_tuple(mps_vec) and len(mps_vec[0]) == qc.num_qubits

    # Note, the order of operations is crucial. We fill in the circuit after (!)
    # invocation of set_matrix_product_state().
    qc2 = QuantumCircuit(qc.num_qubits)
    qc2.set_matrix_product_state(mps_vec)
    qc2.compose(qc, inplace=True)
    return mps_from_circuit(qc2, trunc_thr=trunc_thr, out_state=out_state)


def rand_mps_vec(
    num_qubits: int,
    out_state: Optional[np.ndarray] = None,
    num_layers: int = 3,
) -> QiskitMPS:
    """
    Generates a random vector in MPS format.

    Args:
        num_qubits: number of qubits.
        out_state: output array for storing state as a normal vector (slow!).
        num_layers: number of layers with (n-1) blocks each.

    Returns:
        (1) Qiskit MPS vector, (2) global phase, (3) state vector or None.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers, num_layers > 0)
    blocks = create_ansatz_structure(num_qubits, "spin", "full", num_layers * (num_qubits - 1))
    circ = ParametricCircuit(num_qubits, choice(["cx", "cz", "cp"]), blocks)
    thetas = helper.rand_thetas(circ.num_thetas)
    qc = ansatz_to_qcircuit(circ, thetas)
    return mps_from_circuit(qc, out_state=out_state)


def v_mul_mps(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
) -> QiskitMPS:
    """
    Multiplies circuit matrix by the right-hand side vector: ``out = V @ vec``.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mps_vec: right-hand side vector in Qiskit MPS format.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        out = V @ vec.
    """
    qc = ansatz_to_qcircuit(circ, thetas)
    return qcircuit_mul_mps(qc, mps_vec, trunc_thr=trunc_thr)


def v_dagger_mul_mps(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
) -> QiskitMPS:
    """
    Multiplies conjugate-transposed circuit matrix by right-hand side vector:
    ``out = V.H @ vec``. The gates are applied to the vector in inverse order
    because conjugate-transposed matrix is flipped over.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mps_vec: right-hand side vector in Qiskit MPS format.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        out = V.H @ vec.
    """
    qc = ansatz_to_qcircuit(circ, thetas).inverse()  # V.H
    return qcircuit_mul_mps(qc, mps_vec, trunc_thr=trunc_thr)
