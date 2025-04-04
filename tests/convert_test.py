# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import numpy as np
from qiskit_aqt_provider.api_client import models

from pytket.circuit import Circuit, OpType
from pytket.extensions.aqt.backends.aqt import (
    AQTBackend,
    _aqt_rebase,
    _pytket_to_aqt_circuit,
)


def tk_to_aqt(circ: Circuit) -> tuple[models.Circuit, str]:
    """Convert a circuit to AQT list representation"""
    c = circ.copy()
    AQTBackend().default_compilation_pass().apply(c)
    return _pytket_to_aqt_circuit(c)


def test_convert() -> None:
    circ = Circuit(4, 4)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.noop, [1])
    circ.CRz(0.5, 1, 2)
    circ.add_barrier([2])
    circ.ZZPhase(0.3, 2, 3).CX(3, 0).Tdg(1)
    circ.Measure(0, 0)
    circ.Measure(1, 2)
    circ.Measure(2, 3)
    circ.Measure(3, 1)

    circ_aqt = tk_to_aqt(circ)
    assert json.loads(circ_aqt[1]) == [0, 3, 1, 2]


def test_rebase_CX() -> None:
    circ = Circuit(2)
    circ.CX(0, 1)
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_singleq() -> None:
    circ = Circuit(1)
    # some arbitrary unitary
    circ.add_gate(OpType.U3, [0.01231, 0.848, 38.200], [0])
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)


def test_rebase_large() -> None:
    circ = Circuit(3)
    # some arbitrary unitary
    circ.Rx(0.21, 0).Rz(0.12, 1).Rz(8.2, 2).X(2).CX(0, 1).CX(1, 2).Rz(0.44, 1).Rx(
        0.43, 0
    )
    orig_circ = circ.copy()

    _aqt_rebase().apply(circ)

    u1 = orig_circ.get_unitary()
    u2 = circ.get_unitary()

    assert np.allclose(u1, u2)
