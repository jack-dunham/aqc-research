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
Tests correctness of Trotter framework implementation.
"""

import unittest
from unittest import TestCase

import numpy as np
import pytest
import qiskit.quantum_info as qinfo
import scipy as sp
from qiskit import QuantumCircuit

import aqc_research.model_sp_lhs.trotter.trotter as trotop
import aqc_research.utils as helper
from aqc_research.circuit_transform import qcircuit_to_state
from aqc_research.mps_operations import mps_to_vector


class TestTrotterFramework(TestCase):
    """
    Tests correctness of Trotter framework implementation.
    """

    _max_num_qubits = 5  # max. number of qubits
    _seed = 0x696969  # seed for random generator
    _mps_trunc_thr = 1e-6  # MPS truncation threshold

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def test_trotter_vs_exact(self):
        """Tests time evolution routines Trotter vs exact."""
        num_steps, delta = 30, 1.0
        for num_qubits in range(2, self._max_num_qubits + 1):
            hamiltonian = trotop.make_hamiltonian(num_qubits, delta)
            for second_order in [False, True]:
                for evol_tm in [0.5, 0.7, 1.0, 1.5, 2.0]:
                    ini_state_func = trotop.neel_init_state
                    timer = helper.MyTimer()

                    # Exact evolution for the full time interval.
                    # By default, we set Trotter global phase to 0 (because it
                    # is difficult to keep track of phase factor everywhere);
                    # instead, we compensate the global phase of the exact
                    # state by the inverse phase factor.
                    with timer("exact"):
                        exact_state = trotop.exact_evolution(
                            hamiltonian, ini_state_func(num_qubits), evol_tm
                        )
                        exact_state *= np.exp(
                            -1j * trotop.trotter_global_phase(num_qubits, num_steps, second_order)
                        )

                    # Apply Trotter twice over the half-time intervals.
                    with timer("trotter"):
                        half_trot1 = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm * 0.5,
                            num_steps=num_steps // 2,
                            delta=delta,
                            second_order=second_order,
                        )
                        half_trot2 = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm * 0.5,
                            num_steps=num_steps - num_steps // 2,
                            delta=delta,
                            second_order=second_order,
                        )
                        trot_state = half_trot2.as_qcircuit(
                            half_trot1.as_qcircuit(ini_state_func(num_qubits))
                        )
                        trot_state = qcircuit_to_state(trot_state)

                    # MPS evolution for the full time interval.
                    with timer("mps"):
                        full_trot = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm,
                            num_steps=num_steps,
                            delta=delta,
                            second_order=second_order,
                        )
                        mps = full_trot.as_mps(
                            ini_state_func(num_qubits), trunc_thr=self._mps_trunc_thr
                        )

                    fid = trotop.fidelity(trot_state, exact_state)
                    mps_fid = trotop.fidelity(trot_state, mps_to_vector(mps))
                    self.assertTrue(fid > 0.9)
                    self.assertTrue(mps_fid > 0.9)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("delta", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("field", [0.0, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("dt", [0.1])
@pytest.mark.parametrize("num_steps", [1, 10])
def test_given_trotter_circuit_when_converted_to_operator_then_matches_analytic(
    num_qubits, delta, field, second_order, dt, num_steps
):
    # Exact unitary
    evol_time = dt * num_steps
    hamiltonian = trotop.make_hamiltonian(num_qubits, delta, field)
    desired = sp.linalg.expm(-1j * hamiltonian * evol_time)

    # Unitary from a trotter circuit
    qc = trotop.trotter_circuit(
        QuantumCircuit(num_qubits),
        dt=dt,
        delta=delta,
        field=field,
        num_trotter_steps=num_steps,
        second_order=second_order,
    )
    actual = qinfo.Operator(qc).data

    # Compute Frobenius product
    frob = np.abs(np.trace(actual.conj().T @ desired)) / 2**num_qubits
    np.testing.assert_allclose(1.0, frob, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("Jx", [-0.1, 1.0])
@pytest.mark.parametrize("Jy", [-0.1, 1.0])
@pytest.mark.parametrize("Jz", [-0.1, 1.0])
@pytest.mark.parametrize("field", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("dt", [0.1])
@pytest.mark.parametrize("num_steps", [1, 10])
def test_given_xyz_trotter_circuit_when_converted_to_operator_then_matches_analytic(
    num_qubits, Jx, Jy, Jz, field, second_order, dt, num_steps
):
    # Exact unitary
    evol_time = dt * num_steps
    hamiltonian = trotop.make_xyz_hamiltonian(num_qubits, Jx, Jy, Jz, field)
    desired = sp.linalg.expm(-1j * hamiltonian * evol_time)

    # Unitary from a trotter circuit
    qc = trotop.xyz_trotter_circuit(
        QuantumCircuit(num_qubits),
        dt=dt,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        field=field,
        num_trotter_steps=num_steps,
        second_order=second_order,
    )
    actual = qinfo.Operator(qc).data

    # Compute Frobenius product
    frob = np.abs(np.trace(actual.conj().T @ desired)) / 2**num_qubits
    np.testing.assert_allclose(1.0, frob, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
