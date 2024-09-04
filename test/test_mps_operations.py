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
Tests correctness of our interpretation of Qiskit MPS implementation.
"""
import test.utils_for_testing as tut
import unittest
from time import perf_counter
from typing import Dict, List
from unittest import TestCase

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace, Statevector, SparsePauliOp
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from qiskit_aer import AerSimulator

import aqc_research.circuit_transform as ctr
import aqc_research.mps_operations as mpsop
import aqc_research.utils as helper
from aqc_research.circuit_structures import create_ansatz_structure
from aqc_research.circuit_transform import ansatz_to_qcircuit
from aqc_research.job_executor import run_jobs
from aqc_research.model_sp_lhs.trotter.trotter import trotter_circuit
from aqc_research.parametric_circuit import ParametricCircuit


class TestMPS(TestCase):
    """
    Tests correctness of our interpretation of Qiskit MPS implementation.
    **Note**, MPS is defined up to global phase factor (after transpiler?).
    Often the phase factor can be safely dropped, but not always.
    """

    _max_num_qubits = 7  # max. number of qubits
    _num_repeats = 4  # number of test repetitions per qubit number
    _seed = 0x696969  # seed for random generator
    _tol = float(np.sqrt(np.finfo(float).eps))  # tolerance

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def _make_configs(self) -> List[Dict]:
        """Generates configurations for all the simulations."""
        return [
            {"num_qubits": n, "entangler": e}
            for n in range(2, self._max_num_qubits + 1)
            for e in ["cx", "cz", "cp"]
            for _ in range(self._num_repeats)
        ]

    def _job_mps_to_vector(self, _: int, conf: dict) -> dict:
        """Job function for the test_mps_to_vector()."""
        num_qubits = int(conf["num_qubits"])
        tol = (2 ** max(num_qubits - 10, 0)) * self._tol

        # Generates a random state in MPS format.
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        mps = mpsop.rand_mps_vec(num_qubits, out_state=state1)
        self.assertTrue(mpsop.check_mps(mps))

        # Reconstructed MPS vector must coincide with the one returned by Qiskit.
        tic = perf_counter()
        state2 = mpsop.mps_to_vector(mps)
        clock = perf_counter() - tic
        residual = tut.relative_diff(state1, state2)  # note, phase == 0
        self.assertTrue(residual < tol, f"too large residual: {residual}")
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_mps_to_vector(self):
        """Tests the function mpsop.mps_to_vector()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_mps_to_vector)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_mps_dot(self, _: int, conf: dict) -> dict:
        """Job function for the test_mps_dot()."""
        num_qubits = int(conf["num_qubits"])
        tol = (2 ** max(num_qubits - 10, 0)) * self._tol
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        state2 = np.zeros(2**num_qubits, dtype=np.cfloat)

        mps1 = mpsop.rand_mps_vec(num_qubits, out_state=state1)
        mps2 = mpsop.rand_mps_vec(num_qubits, out_state=state2)

        tic = perf_counter()
        dot11 = mpsop.mps_dot(mps1, mps1)
        dot12 = mpsop.mps_dot(mps1, mps2)
        dot22 = mpsop.mps_dot(mps2, mps2)
        clock = float(perf_counter() - tic) / 3

        err11 = abs(dot11 - np.cfloat(np.vdot(state1, state1)))
        err12 = abs(dot12 - np.cfloat(np.vdot(state1, state2)))
        err22 = abs(dot22 - np.cfloat(np.vdot(state2, state2)))

        err11 = max(abs(dot11 - 1), err11)
        err22 = max(abs(dot22 - 1), err22)

        for err, kind in zip([err11, err12, err22], ["11", "12", "22"]):
            self.assertTrue(err < tol, f"too large residual{kind}: {err}")

        residual = max(err11, err12, err22)
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_mps_dot(self):
        """Tests the function mpsop.mps_dot()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_mps_dot)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_qcircuit_mul_mps(self, _: int, conf: dict) -> dict:
        """Job function for the test_qcircuit_mul_mps()."""
        tol = 2 * self._tol
        num_qubits = int(conf["num_qubits"])

        # Create 2 circuits.
        qcirc = list([])
        for _ in range(2):
            blocks = create_ansatz_structure(num_qubits, "spin", "full", 3 * (num_qubits - 1))
            circ = ParametricCircuit(num_qubits, conf["entangler"], blocks)
            thetas = helper.rand_thetas(circ.num_thetas)
            qcirc.append(ansatz_to_qcircuit(circ, thetas))

        # **Recall**, one can get state vector from circuit only once!
        # This is Qiskit limitation, so we create a copy of the circuit.

        # state1 <-- MPS(qc1) * MPS(qc0).
        mps0 = mpsop.mps_from_circuit(qcirc[0].copy())  # copy!
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        tic = perf_counter()
        mpsop.qcircuit_mul_mps(qcirc[1].copy(), mps0, out_state=state1)  # copy!
        clock = perf_counter() - tic

        # state2 <-- qc1 * qc0, state after concatenated circuits.
        qc2 = qcirc[0].compose(qcirc[1])
        state2 = ctr.qcircuit_to_state(qc2)

        residual = np.linalg.norm(state1 - state2)
        self.assertTrue(np.isclose(residual, 0, atol=tol, rtol=tol), f"residual: {residual}")
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_qcircuit_mul_mps(self):
        """Tests the function mpsop.qcircuit_mul_mps()."""
        configs = self._make_configs()
        results = run_jobs(configs, self._seed, self._job_qcircuit_mul_mps)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_v_mul_vec(self, _: int, conf: dict) -> dict:
        """Job function for the test_v_mul_vec()."""
        tol = self._tol
        residual = 0.0
        num_qubits = int(conf["num_qubits"])
        entangler = conf["entangler"]

        # Generate a random circuit and full-size vector.
        blocks = tut.rand_circuit(num_qubits, np.random.randint(20, 50))
        circ = ParametricCircuit(num_qubits, entangler, blocks)
        thetas = helper.rand_thetas(circ.num_thetas)
        vec = np.zeros(2**num_qubits, dtype=np.cfloat)
        mps_vec = mpsop.rand_mps_vec(num_qubits, out_state=vec)

        # vec2 = V @ V.H @ vec == vec.
        vec1 = mpsop.v_dagger_mul_mps(circ, thetas, mps_vec)
        vec2 = mpsop.v_mul_mps(circ, thetas, vec1)
        dot = mpsop.mps_dot(vec2, mps_vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # MPS(V.H @ vec) == V.H @ vec, by converting from MPS to normal vector.
        qcirc = ctr.ansatz_to_qcircuit(circ, thetas)
        vh_mat = ctr.qcircuit_to_matrix(qcirc.inverse())
        dot = np.vdot(mpsop.mps_to_vector(vec1), vh_mat @ vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # vec2 = V.H @ V @ vec == vec.
        vec1 = mpsop.v_mul_mps(circ, thetas, mps_vec)
        vec2 = mpsop.v_dagger_mul_mps(circ, thetas, vec1)
        dot = mpsop.mps_dot(vec2, mps_vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # MPS(V @ vec) == V @ vec, by converting from MPS to normal vector.
        qcirc = ctr.ansatz_to_qcircuit(circ, thetas)
        v_mat = ctr.qcircuit_to_matrix(qcirc)
        dot = np.vdot(mpsop.mps_to_vector(vec1), v_mat @ vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        result = dict({})
        return result

    def test_v_mul_vec(self):
        """Tests the functions mpsop.v_mul_vec() and mpsop.v_dagger_mul_vec()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_v_mul_vec)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def test_given_input_sim_same_as_default_when_mps_from_circuit_then_same_output(self):
        qc = random_circuit(4, 4)
        qc = transpile(qc, basis_gates=["cx", "rx", "ry", "rz"])
        qc2 = qc.copy()

        mps1 = mpsop.mps_from_circuit(qc)

        SIM = AerSimulator(method="matrix_product_state")
        mps2 = mpsop.mps_from_circuit(qc2, sim=SIM)

        assert mps1 == mps2


class TestRDMFromMPS:
    @pytest.mark.parametrize("num_qubits", list(range(2, 6)))
    def test_given_random_state_when_partial_trace_on_random_subset_then_mps_method_matches_qiskit(
        self, num_qubits
    ):
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        mps1 = mpsop.rand_mps_vec(num_qubits, out_state=state1)
        qubits_to_keep = np.random.choice(
            range(num_qubits), size=np.random.randint(1, num_qubits), replace=False
        )
        rdm_mps = mpsop.partial_trace(mps1, qubits_to_keep)
        qubits_to_contract = [q for q in range(num_qubits) if q not in qubits_to_keep]
        rdm_qiskit = partial_trace(Statevector(state1), qubits_to_contract)
        np.testing.assert_array_almost_equal(
            rdm_mps,
            rdm_qiskit.data,
            err_msg="RDM as calculated by our MPS method should equal Qiskit RDM",
        )

    @pytest.mark.parametrize("num_qubits", list(range(2, 6)))
    def test_given_random_mps_when_partial_trace_on_all_qubits_then_equals_one(self, num_qubits):
        mps1 = mpsop.rand_mps_vec(num_qubits)
        rdm_mps = mpsop.partial_trace(mps1, qubits_to_keep=[])
        np.testing.assert_almost_equal(rdm_mps.item(), 1 + 0j)

    @pytest.mark.parametrize("num_qubits", list(range(2, 6)))
    def test_given_random_mps_when_partial_trace_random_subset_then_rdm_hermitian(self, num_qubits):
        mps1 = mpsop.rand_mps_vec(num_qubits)
        qubits_to_keep = np.random.choice(
            range(num_qubits), size=np.random.randint(1, num_qubits), replace=False
        )
        rdm_mps = mpsop.partial_trace(mps1, qubits_to_keep=qubits_to_keep)
        np.testing.assert_array_almost_equal(rdm_mps, rdm_mps.conj().T)

    @pytest.mark.parametrize("num_qubits", list(range(2, 6)))
    def test_given_random_mps_when_partial_trace_random_subset_then_rdm_has_non_neg_eigenvalues(
        self, num_qubits
    ):
        mps1 = mpsop.rand_mps_vec(num_qubits)
        qubits_to_keep = np.random.choice(
            range(num_qubits), size=np.random.randint(1, num_qubits), replace=False
        )
        rdm_mps = mpsop.partial_trace(mps1, qubits_to_keep=qubits_to_keep)
        eigenvalues = np.linalg.eig(rdm_mps)[0].real
        assert np.all(eigenvalues >= -1e-10)


class TestExpectationFromMPS:
    @pytest.mark.parametrize("op, expected", [("Z", 1), ("Y", 0), ("X", 0)])
    def test_given_zero_state_mps_when_pauli_expectation_then_is_correct(self, op, expected):
        qc = QuantumCircuit(4)
        mps = mpsop.mps_from_circuit(qc)
        expectation = [mpsop.mps_expectation(mps, op, i) for i in range(4)]
        np.testing.assert_allclose(expectation, expected)

    @pytest.mark.parametrize("op", ["Z", "X", "Y"])
    def test_given_random_mps_when_pauli_expectation_then_matches_analytic(self, op):
        n = 4
        state = np.zeros(2**n, dtype=np.cfloat)
        mps = mpsop.rand_mps_vec(n, out_state=state)
        mps_qubit_evals = [mpsop.mps_expectation(mps, op, i) for i in range(n)]

        # Construct single-qubit operator for all qubits
        ops = list(reversed([SparsePauliOp("I" * i + op + "I" * (n - i - 1)) for i in range(n)]))

        analytic_qubit_evals = [
            np.real(np.dot(state.conj(), np.dot(op.to_matrix(), state))) for op in ops
        ]

        np.testing.assert_allclose(mps_qubit_evals, analytic_qubit_evals)


class TestMaxChiFromCircuit:
    def test_given_product_state_when_max_chi_from_circuit_then_1(self):
        for _ in range(10):
            qc = QuantumCircuit(5)
            # max_chi_from_circuit() will not work for circuits with no 2-qubit gates because Aer simulator will not log
            # any bond dimension data unless it sees a 2-qubit gate. This CNOT does not change the state.
            qc.cx(0, 1)
            for i in range(5):
                qc.u(
                    4 * np.pi * np.random.random(),
                    2 * np.pi * np.random.random(),
                    2 * np.pi * np.random.random(),
                    i,
                )
            assert mpsop.max_chi_from_circuit(qc) == 1

    def test_given_bell_state_when_max_chi_from_circuit_then_2(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert mpsop.max_chi_from_circuit(qc) == 2

    def test_given_N_qubit_rand_state_when_max_chi_from_circuit_then_bounded(self):
        for N in range(2, 10):
            qc = random_circuit(N, 2 * N)
            assert mpsop.max_chi_from_circuit(qc) <= 2 ** (N / 2)

    def test_given_1_and_2_trotter_steps_when_max_chi_from_circuit_then_larger_for_2_steps(self):
        # This tests two things: a) entanglement grows with time and b) max_chi_from_circuit() is not returning the
        # cumulative maximum bond dimension
        n = 20
        dt = 0.4
        delta = 1.0

        qc1 = QuantumCircuit(n)
        qc2 = QuantumCircuit(n)

        # Initialise Neel state
        qc1.x(range(0, n, 2))
        qc2.x(range(0, n, 2))

        trotter_circuit(qc1, dt=dt, delta=delta, num_trotter_steps=1, second_order=True)
        trotter_circuit(qc2, dt=dt, delta=delta, num_trotter_steps=2, second_order=True)

        # Call max_chi_from_circuit(qc2) first to break test if returning cumulative max bond dimension
        max_chi_2 = mpsop.max_chi_from_circuit(qc2)
        max_chi_1 = mpsop.max_chi_from_circuit(qc1)

        assert max_chi_2 > max_chi_1

    def test_given_input_sim_same_as_default_when_max_chi_from_circuit_then_same_output(self):
        qc = random_circuit(4, 4)

        chi1 = mpsop.max_chi_from_circuit(qc)

        SIM = AerSimulator(method="matrix_product_state")
        chi2 = mpsop.max_chi_from_circuit(qc, sim=SIM)

        assert chi1 == chi2


class TestAlreadyPreprocessedFunctionality(TestCase):
    basis_gates = ["cx", "rx", "ry", "rz"]

    def test_when_mps_from_circuit_with_preprocessed_flag_then_output_not_QiskitMPS(self):

        qc = random_circuit(2, 2)
        qc = transpile(qc, basis_gates=self.basis_gates)
        preprocessed_mps = mpsop.mps_from_circuit(qc, return_preprocessed=True)
        self.assertIsInstance(preprocessed_mps, list)
        self.assertIsInstance(preprocessed_mps[0], np.ndarray)

    def test_given_already_preprocessed_mps_when_mps_to_vector_then_output_as_expected(self):
        qc = random_circuit(5, 5)
        qc = transpile(qc, basis_gates=self.basis_gates)

        vec1 = mpsop.mps_to_vector(mpsop.mps_from_circuit(qc.copy()))
        vec2 = mpsop.mps_to_vector(
            mpsop.mps_from_circuit(qc.copy(), return_preprocessed=True),
            already_preprocessed=True
        )

        np.testing.assert_array_equal(vec1, vec2)

    def test_given_already_preprocessed_mps_when_mps_dot_then_output_as_expected(self):
        qc1 = random_circuit(5, 5)
        qc2 = random_circuit(5, 5)
        qc1 = transpile(qc1, basis_gates=self.basis_gates)
        qc2 = transpile(qc2, basis_gates=self.basis_gates)

        dot = mpsop.mps_dot(mpsop.mps_from_circuit(qc1.copy()), mpsop.mps_from_circuit(qc2.copy()))
        dot_preprocessed = mpsop.mps_dot(
            mpsop.mps_from_circuit(qc1.copy(), return_preprocessed=True),
            mpsop.mps_from_circuit(qc2.copy(), return_preprocessed=True),
            already_preprocessed=True)

        self.assertEqual(dot, dot_preprocessed)


class TestAlreadyPreprocessedFunctionalityParameterized:

    @pytest.mark.parametrize("op", ['Z', 'Y', 'X'])
    @pytest.mark.parametrize("index", list(range(5)))
    def test_given_already_preprocessed_mps_when_mps_expectation_then_output_as_expected(self, op, index):
        qc = random_circuit(5, 5)
        qc = transpile(qc, basis_gates=['cx', 'rx', 'ry', 'rz'])

        expectation = mpsop.mps_expectation(mpsop.mps_from_circuit(qc.copy()), op, index)
        expectation_preprocessed = mpsop.mps_expectation(
            mpsop.mps_from_circuit(qc.copy(), return_preprocessed=True),
            op,
            index,
            already_preprocessed=True
        )

        np.testing.assert_equal(expectation, expectation_preprocessed)

    @pytest.mark.parametrize("index_1", list(range(5)))
    @pytest.mark.parametrize("index_2", list(range(5)))
    def test_given_already_preprocessed_mps_when_partial_trace_then_output_as_expected(self, index_1, index_2):
        qc = random_circuit(5, 5)
        qc = transpile(qc, basis_gates=['cx', 'rx', 'ry', 'rz'])

        if index_2 > index_1:
            rdm = mpsop.partial_trace(mpsop.mps_from_circuit(qc.copy()), [index_1, index_2])
            rdm_preprocessed = mpsop.partial_trace(
                mpsop.mps_from_circuit(qc.copy(), return_preprocessed=True),
                [index_1, index_2],
                already_preprocessed=True
            )

            np.testing.assert_array_equal(rdm, rdm_preprocessed)


class TestExtractAmplitude(TestCase):

    def test_given_random_state_extract_amplitude_works_for_all_basis_states(self):
        n = 5
        qc = random_circuit(n, n)
        qc = transpile(qc, basis_gates=['cx', 'rx', 'ry', 'rz'])
        sv = Statevector(qc)
        mps = mpsop.mps_from_circuit(qc, return_preprocessed=True)

        for i in range(2**n):
            amplitude = mpsop.extract_amplitude(mps, i, already_preprocessed=True)
            self.assertAlmostEqual(amplitude, sv[i], places=10)

    def test_given_known_state_then_extract_amplitude_works(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        mps = mpsop.mps_from_circuit(qc, return_preprocessed=True)

        a_00 = mpsop.extract_amplitude(mps, 0, already_preprocessed=True)
        a_01 = mpsop.extract_amplitude(mps, 1, already_preprocessed=True)
        a_10 = mpsop.extract_amplitude(mps, 2, already_preprocessed=True)
        a_11 = mpsop.extract_amplitude(mps, 3, already_preprocessed=True)

        self.assertEqual(a_00, 0)
        self.assertEqual(a_01, 1)
        self.assertEqual(a_10, 0)
        self.assertEqual(a_11, 0)

    def test_given_invalid_basis_state_then_extract_amplitude_throws_error(self):
        qc = QuantumCircuit(3)
        mps = mpsop.mps_from_circuit(qc, return_preprocessed=True)

        # 0 (000) and 7 (111) should be valid
        mpsop.extract_amplitude(mps, 0, True)
        mpsop.extract_amplitude(mps, 7, True)
        # -1 and 8 (1000) shouldn't be valid
        with self.assertRaises(ValueError):
            mpsop.extract_amplitude(mps, -1, True)
        with self.assertRaises(ValueError):
            mpsop.extract_amplitude(mps, 8, True)


if __name__ == "__main__":
    unittest.main()
