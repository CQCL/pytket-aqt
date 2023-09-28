# Copyright 2020-2023 Cambridge Quantum Computing
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

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any, Optional, Iterator

from sympy import symbols, Expr

from pytket.circuit import UnitID
from pytket.circuit import Circuit
from pytket.circuit import CustomGateDef
from pytket.circuit import OpType

from ..architecture import EdgeType
from ..architecture import MultiZoneArchitecture
from ..architecture import source_edge_type
from ..architecture import target_edge_type
from ..macro_architecture_graph import empty_macro_arch_from_architecture
from ..macro_architecture_graph import MultiZoneMacroArch
from ..macro_architecture_graph import ZoneId

ParamType = Expr | float


class QubitPlacementError(Exception):
    pass


class MoveError(Exception):
    pass


class AcrossZoneOperationError(Exception):
    pass


class VirtualZonePosition(Enum):
    VirtualLeft = 0
    VirtualRight = 1


dz, se, te = symbols("destination_zone source_edge target_edge")

move_def_circ = Circuit(1)
move_def_circ.add_barrier([0])
move_gate = CustomGateDef("MOVE", move_def_circ, [dz])

shuttle_def_circ = Circuit(1)
shuttle_gate = CustomGateDef("SHUTTLE", shuttle_def_circ, [dz, se, te])

swap_def_circ = Circuit(2)
swap_gate = CustomGateDef("PSWAP", swap_def_circ, [])


@dataclass
class SwapWithinZone:
    """This class holds all information for defining a PSWAP"""

    qubit_0: int
    qubit_1: int

    def __str__(self) -> str:
        return f"{self.qubit_0}: {self.qubit_1}"

    def append_to_circuit(self, circuit: "MultiZoneCircuit") -> None:
        circuit.pytket_circuit.add_custom_gate(
            swap_gate, [], [self.qubit_0, self.qubit_1]
        )


@dataclass
class Shuttle:
    """This class holds all information for defining a SHUTTLE operation"""

    qubit: int
    zone: int

    source_edge: EdgeType
    target_edge: EdgeType

    source_edge_int_encoding: int = field(init=False)
    target_edge_int_encoding: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Encode left and right edges as negative or non-negative
         number for use as tket custom op parameters
        Yes, this is a hack
        """
        self.source_edge_int_encoding = -1 if self.source_edge == EdgeType.Left else 1
        self.target_edge_int_encoding = -1 if self.target_edge == EdgeType.Left else 1

    def __str__(self) -> str:
        return f"{self.qubit}: {self.zone}"

    def append_to_circuit(self, circuit: "MultiZoneCircuit") -> None:
        circuit.pytket_circuit.add_custom_gate(
            shuttle_gate,
            [self.zone, self.source_edge_int_encoding, self.target_edge_int_encoding],
            [self.qubit],
        )


@dataclass
class Init:
    qubit: int
    zone: int


MZAOperation = SwapWithinZone | Shuttle


def _swap_left_to_right_through_list(
    qubit: int, qubit_list: list[int]
) -> list[MZAOperation]:
    """Generate a list of swap operations moving an ion from left to right
    through a zone"""
    return [SwapWithinZone(qubit, swap_qubit) for swap_qubit in qubit_list]


def _swap_right_to_left_through_list(
    qubit: int, qubit_list: list[int]
) -> list[MZAOperation]:
    """Generate a list of swap operations moving an ion from right to left
    through a zone"""
    return [SwapWithinZone(swap_qubit, qubit) for swap_qubit in reversed(qubit_list)]


def _move_from_zone_position_to_connected_zone_edge(
    qubit: int,
    zone_qubit_list: list[int],
    position_in_zone: int | VirtualZonePosition,
    move_source_edge_type: EdgeType,
    move_target_edge_type: EdgeType,
    target_zone: int,
) -> list[MZAOperation]:
    """Generate a list of swap and shuttle operations moving an ion from a
    given position within a zone to the edge of a target zone"""
    move_operations = []
    match (move_source_edge_type, position_in_zone):
        case (EdgeType.Right, VirtualZonePosition.VirtualLeft):
            move_operations.extend(
                _swap_left_to_right_through_list(qubit, zone_qubit_list)
            )
        case (EdgeType.Left, VirtualZonePosition.VirtualRight):
            move_operations.extend(
                _swap_right_to_left_through_list(qubit, zone_qubit_list)
            )
        case (EdgeType.Right, VirtualZonePosition.VirtualRight):
            pass
        case (EdgeType.Left, VirtualZonePosition.VirtualLeft):
            pass
        case (EdgeType.Right, position) if isinstance(position, int):
            move_operations.extend(
                _swap_left_to_right_through_list(qubit, zone_qubit_list[position + 1 :])
            )
        case (EdgeType.Left, position) if isinstance(position, int):
            move_operations.extend(
                _swap_right_to_left_through_list(qubit, zone_qubit_list[:position])
            )
    move_operations.append(
        Shuttle(qubit, target_zone, move_source_edge_type, move_target_edge_type)
    )
    return move_operations


class MultiZoneCircuit:
    """Circuit for AQT Multi-Zone architectures

    Adds operations for initialisation of ions within zones and
    movement of ions between zones.

    Also validates correctness of circuit with respect to the
    architecture constraints

    """

    architecture: MultiZoneArchitecture
    macro_arch: MultiZoneMacroArch
    qubit_to_zones: dict[int, list[int]]
    zone_to_qubits: dict[int, list[int]]
    initial_zone_to_qubits: dict[int, list[int]]
    multi_zone_operations: dict[int, list[list[MZAOperation]]]
    pytket_circuit: Circuit
    _is_compiled: bool = False

    def __init__(
        self,
        multi_zone_arch: MultiZoneArchitecture,
        initial_zone_to_qubits: dict[int, list[int]],
        *args: int,
        **kwargs: str,
    ):
        self.architecture = multi_zone_arch
        self.macro_arch = empty_macro_arch_from_architecture(multi_zone_arch)
        self.pytket_circuit = Circuit(*args, **kwargs)
        self.qubit_to_zones = {}
        self.initial_zone_to_qubits = initial_zone_to_qubits
        self.zone_to_qubits = {
            zone_id: [] for zone_id, _ in enumerate(multi_zone_arch.zones)
        }
        self.multi_zone_operations = {
            qubit: [] for qubit in range(multi_zone_arch.n_qubits_max)
        }
        for zone, qubits in initial_zone_to_qubits.items():
            self._place_qubits(zone, qubits)

        self.all_qubit_list = list(range(len(self.pytket_circuit.qubits)))
        move_barrier_def_circ = Circuit(len(self.all_qubit_list))
        move_barrier_def_circ.add_barrier(self.all_qubit_list)
        self.move_barrier_gate = CustomGateDef(
            "MOVE_BARRIER", move_barrier_def_circ, []
        )
        for zone, qubit_list in initial_zone_to_qubits.items():
            init_def_circ = Circuit(len(qubit_list))
            custom_init = CustomGateDef("INIT", init_def_circ, [dz])
            self.pytket_circuit.add_custom_gate(custom_init, [zone], qubit_list)

    def __iter__(self) -> Iterator:
        return self.pytket_circuit.__iter__()

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    @is_compiled.setter
    def is_compiled(self, new_value: bool) -> None:
        self._is_compiled = new_value

    def add_move_barrier(self) -> None:
        """Add custom gate MOVE_BARRIER

        This is internally a barrier over all qubits
        It is necessary to prevent reordering of shuttling
        during compilation
        """
        self.pytket_circuit.add_custom_gate(
            self.move_barrier_gate, [], self.all_qubit_list
        )

    def _place_qubit(self, zone: int, qubit: int) -> None:
        if qubit in self.qubit_to_zones:
            raise QubitPlacementError(f"Qubit {qubit} was already placed")
        if (
            self.architecture.get_zone_max_ions(zone)
            < len(self.zone_to_qubits[zone]) + 1
        ):
            raise QubitPlacementError(
                f"Cannot add ion to zone {zone}, maximum ion capacity already reached"
            )
        self.qubit_to_zones[qubit] = [zone]
        self.zone_to_qubits[zone].append(qubit)

    def _place_qubits(self, zone: int, qubits: list[int]) -> None:
        for qubit in qubits:
            self._place_qubit(zone, qubit)

    def move_qubit(self, qubit: int, new_zone: int, precompiled: bool = False) -> None:
        """Move a qubit from its current zone to new_zone

        Calculates the needs "PSWAP" and "SHUTTLE" operations to implement move.
        Adds custom gates to underlying Circuit to signify move and prevent optimisation
        through the move.
        Raises error is move is not possible

        If precompiled=False, the needed "PSWAP" and "SHUTTLE" operations are added to
        lists for each qubit and "MoveBarriers" are added to underlying pytket circuit.
        The "MoveBarriers" serve as markers to add the physical operations to the
        circuit after compilation

        If precompiled=True (should not be used for manual routing), the underlying
        circuit is assumed to already be compiled (but not yet routed). the "PSWAP"
        and "SHUTTLE" operations will be added to the circuit directly.

        :param qubit: qubit to move
        :param new_zone: zone to move it too
        :param precompiled: whether the underlying pytket circuit has already been
         compiled (but not yet routed)
        """
        if qubit not in self.qubit_to_zones:
            raise QubitPlacementError("Cannot move qubit that was never placed")
        old_zone = self.qubit_to_zones[qubit][-1]
        if old_zone == new_zone:
            raise MoveError(
                f"Requested move has no effect,"
                f" qubit {qubit} is already in zone {new_zone}"
            )
        move_operations = []
        shortest_path = self.macro_arch.shortest_path(
            ZoneId(old_zone), ZoneId(new_zone)
        )
        if not shortest_path:
            raise MoveError(
                f"Cannot move ion to zone {new_zone},"
                f" no path found from current zone {old_zone}"
            )

        old_zone_qubits = self.zone_to_qubits[old_zone]
        position_in_zone: int | VirtualZonePosition = old_zone_qubits.index(qubit)

        for source_zone, target_zone in itertools.pairwise(shortest_path):
            n_qubits_in_target_zone = len(self.zone_to_qubits[target_zone])
            if (
                self.architecture.get_zone_max_ions(target_zone)
                < n_qubits_in_target_zone + 1
            ):
                if target_zone == new_zone:
                    raise MoveError(
                        f"Cannot move ion to zone {target_zone},"
                        f" maximum ion capacity already reached"
                    )
                raise MoveError(
                    f"Move requires shuttling ion through zone {target_zone},"
                    f" but this zone is at maximum capacity"
                )

            connection_type = self.architecture.get_connection_type(
                source_zone, target_zone
            )
            source_zone_qubits = self.zone_to_qubits[source_zone]
            move_operations.extend(
                _move_from_zone_position_to_connected_zone_edge(
                    qubit,
                    source_zone_qubits,
                    position_in_zone,
                    source_edge_type(connection_type),
                    target_edge_type(connection_type),
                    target_zone,
                )
            )
            if target_edge_type(connection_type) == EdgeType.Right:
                position_in_zone = VirtualZonePosition.VirtualRight
            else:
                position_in_zone = VirtualZonePosition.VirtualLeft

        if not precompiled:
            self.pytket_circuit.add_custom_gate(move_gate, [new_zone], [qubit])
            self.add_move_barrier()
        old_zone_qubits.remove(qubit)
        if position_in_zone is VirtualZonePosition.VirtualLeft:
            self.zone_to_qubits[new_zone].insert(0, qubit)
        else:
            self.zone_to_qubits[new_zone].append(qubit)
        self.qubit_to_zones[qubit].append(new_zone)
        self.multi_zone_operations[qubit].append(move_operations)
        if precompiled:
            for multi_op in move_operations:
                multi_op.append_to_circuit(self)

    def add_gate(
        self,
        op_type: OpType,
        args: list[UnitID] | list[int],
        params: Optional[list[ParamType]] = None,
    ) -> "MultiZoneCircuit":
        if op_type == OpType.Barrier:
            raise ValueError(
                "The manual addition of barriers is not currently"
                " allowed within Multi Zone Circuits"
            )
        if params is None:
            self.pytket_circuit.add_gate(op_type, args)
        else:
            self.pytket_circuit.add_gate(op_type, params, args)
        return self

    def CX(self, control: int, target: int, **kwargs: Any) -> "MultiZoneCircuit":
        self.pytket_circuit.CX(control, target, **kwargs)
        return self

    def measure_all(self) -> "MultiZoneCircuit":
        self.pytket_circuit.measure_all()
        return self

    def copy(self) -> "MultiZoneCircuit":
        new_circuit = MultiZoneCircuit(self.architecture, self.initial_zone_to_qubits)
        new_circuit.pytket_circuit = self.pytket_circuit.copy()
        new_circuit.qubit_to_zones = deepcopy(self.qubit_to_zones)
        new_circuit.zone_to_qubits = deepcopy(self.zone_to_qubits)
        new_circuit.multi_zone_operations = deepcopy(self.multi_zone_operations)
        return new_circuit

    def get_n_shuttles(self) -> int:
        """
        Get the number of shuttles used to route the circuit to the architecture
        """
        count = 0
        for cmd in self.pytket_circuit.get_commands():
            op = cmd.op
            optype = op.type
            op_string = f"{op}"
            if optype == OpType.CustomGate and "SHUTTLE" in op_string:
                count += 1
        return count

    def get_n_pswaps(self) -> int:
        """
        Get the number of pswaps used to route the circuit to the architecture
        """
        count = 0
        for cmd in self.pytket_circuit.get_commands():
            op = cmd.op
            optype = op.type
            op_string = f"{op}"
            if optype == OpType.CustomGate and "PSWAP" in op_string:
                count += 1
        return count

    def validate(self) -> None:
        if self._is_compiled:
            logging.warning(
                "Skipping requested validation because circuit is already compiled"
                "MultiZoneCircuit.validate() only validates circuits pre-compilation"
            )
            return
        current_multiop_index_per_qubit: dict[int, int] = {
            k: 0 for k in self.multi_zone_operations.keys()
        }
        for i, cmd in enumerate(self.pytket_circuit):
            op = cmd.op
            if "MOVE_BARRIER" in f"{op}":
                pass
            elif "MOVE" in f"{op}":
                qubit = cmd.args[0].index[0]
                current_multiop_index = current_multiop_index_per_qubit[qubit]
                current_multiop_index_per_qubit[qubit] = current_multiop_index + 1
            else:
                qubits: list[int] = [q.index[0] for q in cmd.args]
                cmd_qubit_zones = [
                    self.qubit_to_zones[q][current_multiop_index_per_qubit[q]]
                    for q in qubits
                ]
                if not all(zone == cmd_qubit_zones[0] for zone in cmd_qubit_zones):
                    qubit_to_zone_message = " ".join(
                        [
                            f"q[{qz[0]}] in zone {qz[1]},"
                            for qz in zip(qubits, cmd_qubit_zones)
                        ]
                    )
                    raise AcrossZoneOperationError(
                        f"Operation {i} = {cmd} involved qubits across multiple"
                        f"zones. {qubit_to_zone_message}"
                    )
