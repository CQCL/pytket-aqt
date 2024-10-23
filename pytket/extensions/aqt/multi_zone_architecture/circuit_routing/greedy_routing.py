# Copyright 2020-2024 Quantinuum
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
from copy import deepcopy

from pytket import Circuit, Qubit

from ..architecture import MultiZoneArchitecture
from ..circuit.helpers import ZonePlacement, ZoneRoutingError
from ..circuit.multizone_circuit import MultiZoneCircuit
from .settings import RoutingSettings


class GreedyCircuitRouter:
    """Uses a simple greedy algorithm to add shuttles and swaps to a circuit

    The routed circuit can be directly run on the given Architecture

    :param circuit: The circuit to be routed
    :param arch: The architecture to route to
    :param initial_placement: The initial placement of ions in the ion trap zones
    :param settings: The settings used for routing
    """

    def __init__(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitecture,
        initial_placement: ZonePlacement,
        settings: RoutingSettings,
    ):
        self._circuit = circuit
        self._arch = arch
        self._initial_placement = initial_placement
        self._settings = settings

    def get_routed_circuit(self) -> MultiZoneCircuit:
        """Returns the routed MultiZoneCircuit"""
        n_qubits = self._circuit.n_qubits
        mz_circuit = MultiZoneCircuit(
            self._arch, self._initial_placement, n_qubits, self._circuit.n_bits
        )
        current_qubit_to_zone = {}
        for zone, qubit_list in self._initial_placement.items():
            for qubit in qubit_list:
                current_qubit_to_zone[qubit] = zone
        current_zone_to_qubits = deepcopy(self._initial_placement)

        for cmd in self._circuit.get_commands():
            n_args = len(cmd.args)
            if n_args == 1:
                mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
            elif n_args == 2:
                if isinstance(cmd.args[0], Qubit) and isinstance(cmd.args[1], Qubit):
                    _make_necessary_moves(
                        (cmd.args[0].index[0], cmd.args[1].index[0]),
                        mz_circuit,
                        current_qubit_to_zone,
                        current_zone_to_qubits,
                    )
                mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
            else:
                raise ZoneRoutingError("Circuit must be rebased to the AQT gate set")
        return mz_circuit


def _make_necessary_moves(
    qubits: tuple[int, int],
    mz_circ: MultiZoneCircuit,
    current_qubit_to_zone: dict[int, int],
    current_placement: ZonePlacement,
) -> None:
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move two qubits into the same zone

    :param qubits: tuple of two qubits
    :param mz_circ: the MultiZoneCircuit
    :param current_qubit_to_zone: dictionary containing the current
     mapping of qubits to zones (may be altered)
    :param current_placement: dictionary the current mapping of zones
     to lists of qubits contained within them (may be altered)
    """

    def _move_qubit(qubit_to_move: int, starting_zone: int, target_zone: int) -> None:
        mz_circ.move_qubit(qubit_to_move, target_zone, precompiled=True)
        current_placement[starting_zone].remove(qubit_to_move)
        current_placement[target_zone].append(qubit_to_move)
        current_qubit_to_zone[qubit_to_move] = target_zone

    qubit0 = qubits[0]
    qubit1 = qubits[1]

    zone0 = current_qubit_to_zone[qubit0]
    zone1 = current_qubit_to_zone[qubit1]
    if zone0 == zone1:
        return
    free_space_zone_0 = mz_circ.architecture.get_zone_max_ions(zone0) - len(
        current_placement[zone0]
    )
    free_space_zone_1 = mz_circ.architecture.get_zone_max_ions(zone1) - len(
        current_placement[zone1]
    )
    match (free_space_zone_0, free_space_zone_1):
        case (0, 0):
            raise ValueError("Should not allow two full registers")
        case (1, 1):
            # find first qubit in zone1 that isn't qubit1
            uninvolved_qubit = [
                qubit for qubit in current_placement[zone1] if qubit != qubits[1]
            ][0]
            # send it to zone0
            _move_qubit(uninvolved_qubit, zone1, zone0)
            # send qubit0 to zone1
            _move_qubit(qubits[0], zone0, zone1)
        case (a, b) if a < 0 or b < 0:
            raise ValueError("Should never be negative")
        case (free0, free1) if free0 >= free1:
            _move_qubit(qubits[1], zone1, zone0)
        case (_, _):
            _move_qubit(qubits[0], zone0, zone1)
