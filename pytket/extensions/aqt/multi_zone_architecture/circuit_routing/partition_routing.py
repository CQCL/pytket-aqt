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
import math
from collections.abc import Generator
from copy import deepcopy

from pytket import Circuit

from ..architecture import MultiZoneArchitectureSpec
from ..circuit.helpers import ZonePlacement, ZoneRoutingError
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..depth_list.depth_list import (
    DepthList,
    get_initial_depth_list,
    get_updated_depth_list,
)
from ..graph_algs.graph import GraphData
from ..graph_algs.mt_kahypar import MtKahyparPartitioner
from ..macro_architecture_graph import empty_macro_arch_from_architecture
from .settings import RoutingSettings


class PartitionCircuitRouter:
    """Uses graph partitioning to add shuttles and swaps to a circuit

    The routed circuit can be directly run on the given Architecture

    :param circuit: The circuit to be routed
    :param arch: The architecture to route to
    :param initial_placement: The initial placement of ions in the ion trap zones
    :param settings: The settings used for routing
    """

    def __init__(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitectureSpec,
        initial_placement: ZonePlacement,
        settings: RoutingSettings,
    ):
        self._circuit = circuit
        self._arch = arch
        self._macro_arch = empty_macro_arch_from_architecture(arch)
        self._initial_placement = initial_placement
        self._settings = settings

    def get_routed_circuit(self) -> MultiZoneCircuit:  # noqa: PLR0912
        """Returns the routed MultiZoneCircuit"""
        n_qubits = self._circuit.n_qubits
        depth_list = get_initial_depth_list(self._circuit)
        commands = self._circuit.get_commands().copy()
        mz_circuit = MultiZoneCircuit(
            self._arch, self._initial_placement, n_qubits, self._circuit.n_bits
        )
        for old_place, new_place in self.placement_generator(depth_list):
            if self._settings.debug_level > 0:
                print("-------")  # noqa: T201
                for zone in range(self._arch.n_zones):
                    changes_str = ", ".join(
                        [
                            f"+{i}"
                            for i in set(new_place[zone]).difference(old_place[zone])
                        ]
                        + [
                            f"-{i}"
                            for i in set(old_place[zone]).difference(new_place[zone])
                        ]
                    )
                    print(  # noqa: T201
                        f"Z{zone}: {old_place[zone]} ->"
                        f" {new_place[zone]} -- ({changes_str})"
                    )
            leftovers = []
            # stragglers are qubits with pending 2 qubit gates that cannot
            # be performed in the old placement
            # they have to wait for the next iteration
            stragglers: set[int] = set()
            qubit_to_zone_old = _get_qubit_to_zone(n_qubits, old_place)
            last_cmd_index = 0
            for i, cmd in enumerate(commands):
                last_cmd_index = i
                n_args = len(cmd.args)
                qubit0 = cmd.args[0].index[0]
                zone0 = qubit_to_zone_old[qubit0]
                if n_args == 1:
                    if qubit0 in stragglers or zone0 not in self._macro_arch.gate_zones:
                        leftovers.append(cmd)
                    else:
                        mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
                elif n_args == 2:  # noqa: PLR2004
                    qubit1 = cmd.args[1].index[0]
                    if qubit0 in stragglers:
                        stragglers.add(qubit1)
                        leftovers.append(cmd)
                        continue
                    if qubit1 in stragglers:
                        stragglers.add(qubit0)
                        leftovers.append(cmd)
                        continue
                    if (
                        zone0 == qubit_to_zone_old[qubit1]
                        and zone0 in self._macro_arch.gate_zones
                    ):
                        mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
                    else:
                        leftovers.append(cmd)
                        stragglers.update([qubit0, qubit1])
                if len(stragglers) >= n_qubits - 1:
                    break
            if last_cmd_index == len(commands) - 1:
                commands = leftovers
            else:
                commands = leftovers + commands[last_cmd_index + 1 :]
            _make_necessary_config_moves((old_place, new_place), mz_circuit)
        for cmd in commands:
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
        return mz_circuit

    def placement_generator(
        self, depth_list: DepthList
    ) -> Generator[tuple[ZonePlacement, ZonePlacement], None, None]:
        """Generates pairs of ZonePlacements representing one shuttling step

        The first placement represents the current state of the trap, the second
        represents the "optimal" next state to implement the remaining gates in
        the depth list.

        :param depth_list: The list of gates used to determine the next ion placement.
        """
        current_placement = deepcopy(self._initial_placement)
        n_qubits = self._circuit.n_qubits
        qubit_to_zone = _get_qubit_to_zone(n_qubits, current_placement)
        depth_list = get_updated_depth_list(
            n_qubits, qubit_to_zone, self._macro_arch.gate_zones, depth_list
        )
        max_iter = len(depth_list) * 2
        iteration = 0
        while depth_list:
            new_placement = self.new_placement_graph_partition_alg(
                depth_list, current_placement
            )
            yield current_placement, new_placement
            qubit_to_zone = _get_qubit_to_zone(n_qubits, new_placement)
            depth_list = get_updated_depth_list(
                n_qubits, qubit_to_zone, self._macro_arch.gate_zones, depth_list
            )
            current_placement = new_placement
            if iteration > max_iter:
                raise ZoneRoutingError("placement alg is not converging")
            iteration += 1

    def new_placement_graph_partition_alg(
        self,
        depth_list: DepthList,
        starting_placement: ZonePlacement,
    ) -> ZonePlacement:
        """Generates a new ZonePlacement to implement the next gates

        The returned ZonePlacement
        represents the "optimal" next state to implement the remaining gates in
        the depth list.

        :param depth_list: The list of gates used to determine the next ion placement.
        :param starting_placement: The starting configuration of ions in ion trap zones
        """
        n_qubits = self._circuit.n_qubits
        n_qubits_max = self._arch.n_qubits_max
        if n_qubits > n_qubits_max:
            raise ZoneRoutingError(
                f"Attempting to route circuit with {n_qubits}"
                f" qubits, but architecture only supports up to {n_qubits_max}"
            )

        num_zones = self._arch.n_zones
        shuttle_graph_data = self.get_circuit_shuttle_graph_data(
            starting_placement, depth_list
        )
        partitioner = MtKahyparPartitioner(log_level=self._settings.debug_level)
        if self._settings.debug_level > 0:
            print("Depth List:")  # noqa: T201
            for i in range(min(4, len(depth_list))):
                print(depth_list[i])  # noqa: T201
        vertex_to_part = partitioner.partition_graph(shuttle_graph_data, num_zones)
        new_placement: ZonePlacement = {i: [] for i in range(num_zones)}
        part_to_zone = [-1] * num_zones
        for vertex in range(n_qubits, n_qubits + num_zones):
            part_to_zone[vertex_to_part[vertex]] = vertex - n_qubits
        for vertex in range(n_qubits):
            new_placement[part_to_zone[vertex_to_part[vertex]]].append(vertex)
        return new_placement

    def get_circuit_shuttle_graph_data(
        self, starting_placement: ZonePlacement, depth_list: DepthList
    ) -> GraphData:
        """Calculate graph data for qubit-zone graph to be partitioned"""
        n_qubits = self._circuit.n_qubits
        num_zones = self._arch.n_zones
        places_per_zone = [
            self._arch.get_zone_max_ions(i) for i, _ in enumerate(self._arch.zones)
        ]
        num_spots = sum(places_per_zone)
        edges: list[tuple[int, int]] = []
        edge_weights: list[int] = []

        # add gate edges
        max_considered_depth = min(self._settings.max_depth, len(depth_list))
        max_weight = math.ceil(math.pow(2, 18))
        for depth, pairs in enumerate(depth_list):
            if depth > max_considered_depth:
                break
            weight = math.ceil(math.exp(-2 * depth) * max_weight)
            edges.extend(pairs)
            edge_weights.extend([weight] * len(pairs))

        # "assign" depth 0 qubits to gate zones
        if self._macro_arch.has_memory_zones:
            edge_pair_pairs = [
                (pair[i], zone + n_qubits)
                for i in [0, 1]
                for pair in depth_list[0]
                for zone in self._macro_arch.gate_zones
            ]
            edge_pair_weights = (
                [max_weight] * len(depth_list[0]) * 2 * len(self._macro_arch.gate_zones)
            )
            edges.extend(edge_pair_pairs)
            edge_weights.extend(edge_pair_weights)

        # add shuttling penalty (just distance between zones for now,
        # should later be dependent on shuttling cost)
        max_shuttle_weight = math.ceil(max_weight / 2)
        for zone, qubits in starting_placement.items():
            for other_zone in range(num_zones):
                weight = math.ceil(
                    math.exp(-0.8 * self.shuttling_penalty(zone, other_zone))
                    * max_shuttle_weight
                )
                if weight < 1:
                    continue
                edges.extend([(other_zone + n_qubits, qubit) for qubit in qubits])
                edge_weights.extend([weight for _ in qubits])

        num_vertices = num_spots
        vertex_weights = [1 for _ in range(num_vertices)]

        fixed_list = (
            [-1] * n_qubits
            + [zone for zone in range(num_zones)]  # noqa: C416
            + [-1] * (num_vertices - n_qubits - num_zones)
        )

        return GraphData(
            num_vertices,
            vertex_weights,
            edges,
            edge_weights,
            fixed_list,
            places_per_zone,
        )

    def shuttling_penalty(self, zone1: int, other_zone1: int) -> int:
        """Calculate penalty for shuttling from one zone to another"""
        shortest_path = self._macro_arch.shortest_path(int(zone1), int(other_zone1))
        if shortest_path:
            return len(shortest_path) - 1
        raise ZoneRoutingError(
            f"Shortest path could not be calculated"
            f" between zones {zone1} and {other_zone1}"
        )


def _get_qubit_to_zone(n_qubits: int, placement: ZonePlacement) -> list[int]:
    qubit_to_zone: list[int] = [-1] * n_qubits
    for zone, qubits in placement.items():
        for qubit in qubits:
            qubit_to_zone[qubit] = zone
    return qubit_to_zone


def _make_necessary_config_moves(
    configs: tuple[ZonePlacement, ZonePlacement],
    mz_circ: MultiZoneCircuit,
) -> None:
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move from one zone placement to another

    :param configs: tuple of two ZonePlacements [Old, New]
    :param mz_circ: the MultiZoneCircuit
     mapping of qubits to zones (may be altered)
    """
    n_qubits = mz_circ.pytket_circuit.n_qubits
    old_place = configs[0]
    new_place = configs[1]
    qubit_to_zone_old = _get_qubit_to_zone(n_qubits, old_place)
    qubit_to_zone_new = _get_qubit_to_zone(n_qubits, new_place)
    qubits_to_move: list[tuple[int, int, int]] = []
    current_placement = deepcopy(old_place)
    for qubit in range(n_qubits):
        old_zone = qubit_to_zone_old[qubit]
        new_zone = qubit_to_zone_new[qubit]
        if old_zone != new_zone:
            qubits_to_move.append(
                (qubit, qubit_to_zone_old[qubit], qubit_to_zone_new[qubit])
            )
    # sort based on ascending number of free places in the target zone (at beginning)
    qubits_to_move.sort(
        key=lambda x: mz_circ.architecture.get_zone_max_ions(x[2])
        - len(current_placement[x[2]])
    )

    def _move_qubit(qubit_to_move: int, starting_zone: int, target_zone: int) -> None:
        mz_circ.move_qubit(qubit_to_move, target_zone, precompiled=True)
        current_placement[starting_zone].remove(qubit_to_move)
        current_placement[target_zone].append(qubit_to_move)

    while qubits_to_move:
        qubit, start, targ = qubits_to_move[-1]
        free_space_target_zone = mz_circ.architecture.get_zone_max_ions(targ) - len(
            current_placement[targ]
        )
        match free_space_target_zone:
            case 0:
                raise ValueError("Should not allow full register here")
            case 1:
                _move_qubit(qubit, start, targ)
                # remove this move from list
                qubits_to_move.pop()
                # find a qubit in target zone that needs to move and put it at end
                # of qubits_to_move, so it comes next
                moves_with_start_equals_current_targ = [
                    i
                    for i, move_tup in enumerate(qubits_to_move)
                    if move_tup[1] == targ
                ]
                if not moves_with_start_equals_current_targ:
                    raise ValueError("This shouldn't happen")
                next_move_index = moves_with_start_equals_current_targ[0]
                next_move = qubits_to_move.pop(next_move_index)
                qubits_to_move.append(next_move)
            case a if a < 0:
                raise ValueError("Should never be negative")
            case _:
                _move_qubit(qubit, start, targ)
                # remove this move from list
                qubits_to_move.pop()
