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
from typing import TypeAlias

from pytket import Circuit, Qubit
from pytket._tket.circuit import OpType

DepthList: TypeAlias = list[list[tuple[int, int]]]  # noqa: UP040


def get_2q_gate_pairs_from_circuit(circuit: Circuit) -> list[tuple[int, int]]:
    pair_list: list[tuple[int, int]] = []
    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1 or cmd.op.type == OpType.Measure:
            continue
        if (
            n_args == 2  # noqa: PLR2004
            and isinstance(cmd.args[0], Qubit)
            and isinstance(cmd.args[1], Qubit)
        ):
            qubit0 = cmd.args[0].index[0]
            qubit1 = cmd.args[1].index[0]
            # always smaller index first
            if qubit0 < qubit1:
                pair_list.append((qubit0, qubit1))
            else:
                pair_list.append((qubit1, qubit0))
    return pair_list


def get_depth_list(n_qubits: int, gate_pairs: list[tuple[int, int]]) -> DepthList:
    depth_list: list[list[tuple[int, int]]] = []
    current_depth_per_qubit: list[int] = [0] * n_qubits
    for pair in gate_pairs:
        qubit0 = pair[0]
        qubit1 = pair[1]
        depth = max(current_depth_per_qubit[qubit0], current_depth_per_qubit[qubit1])
        assert len(depth_list) >= depth
        if depth > 0 and (
            (qubit0, qubit1) in depth_list[depth - 1]
            or (
                qubit1,
                qubit0,
            )
            in depth_list[depth - 1]
        ):
            depth_list[depth - 1].append((qubit0, qubit1))
            continue
        if len(depth_list) > depth:
            depth_list[depth].append((qubit0, qubit1))
        else:
            depth_list.append([(qubit0, qubit1)])
        current_depth_per_qubit[qubit0] = depth + 1
        current_depth_per_qubit[qubit1] = depth + 1
    return depth_list


def get_initial_depth_list(circuit: Circuit) -> DepthList:
    """From a given Circuit get the Depth list used to determine gate priority.

    Used for the initial placement of ions based on partitioning algorithms
    """
    n_qubits = circuit.n_qubits
    gate_pairs = get_2q_gate_pairs_from_circuit(circuit)
    return get_depth_list(n_qubits, gate_pairs)


def get_updated_depth_list(
    n_qubits: int,
    qubit_to_zone: list[int],
    gate_zones: list[int],
    depth_list: DepthList,
) -> DepthList:
    """From a given placement of qubits in zones
     update the DepthList used to determine gate priority.

    Used for graph partitioning routing.
    """
    pruned_depth_list = [depth.copy() for depth in depth_list]
    # prune current depth list
    prune_stage = False
    prune_touched = set()
    for i, depth in enumerate(depth_list):
        for qubit_pair in depth:
            zone_0 = qubit_to_zone[qubit_pair[0]]
            zone_1 = qubit_to_zone[qubit_pair[1]]
            if zone_0 == zone_1 and zone_0 in gate_zones:
                if not prune_stage or (
                    qubit_pair[0] not in prune_touched
                    and qubit_pair[1] not in prune_touched
                ):
                    pruned_depth_list[i].remove(qubit_pair)
                else:
                    prune_touched.update({qubit_pair[0], qubit_pair[1]})
            else:
                prune_touched.update({qubit_pair[0], qubit_pair[1]})
        if pruned_depth_list[i]:
            prune_stage = True
        if len(prune_touched) >= n_qubits - 1:
            break
    # flatten depth list
    flattened_depth_list = [pair for depth in pruned_depth_list for pair in depth]
    # new depth list
    new_depth_list = get_depth_list(n_qubits, flattened_depth_list)
    return new_depth_list  # noqa: RET504
