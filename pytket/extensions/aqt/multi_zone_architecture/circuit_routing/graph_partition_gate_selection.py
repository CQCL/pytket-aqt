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

from pytket.circuit import Command

from ..architecture import MultiZoneArchitectureSpec
from ..architecture_portgraph import MultiZonePortGraph
from ..circuit.helpers import TrapConfiguration, ZonePlacement, ZoneRoutingError
from ..depth_list.depth_list import (
    DepthList,
    depth_list_from_command_list,
)
from ..graph_algs.graph import GraphData
from ..graph_algs.mt_kahypar import MtKahyparPartitioner
from ..macro_architecture_graph import empty_macro_arch_from_architecture
from .greedy_gate_selection import (
    handle_only_single_qubits_remaining,
    handle_unused_qubits,
)
from .qubit_tracker import QubitTracker
from .settings import RoutingSettings


class PartitionGateSelector:
    """Uses graph partitioning to add shuttles and swaps to a circuit

    The routed circuit can be directly run on the given Architecture

    :param arch: The architecture to route to
    :param settings: The settings used for routing
    """

    def __init__(
        self,
        arch: MultiZoneArchitectureSpec,
        settings: RoutingSettings,
    ):
        self._arch = arch
        self._macro_arch = empty_macro_arch_from_architecture(arch)
        self._port_graph = MultiZonePortGraph(self._arch)
        self._settings = settings

    def next_config(
        self,
        current_configuration: TrapConfiguration,
        remaining_commands: list[Command],
    ) -> TrapConfiguration:
        """Generates a new TrapConfiguration to implement the next gates

        The returned TrapConfiguration
        represents the "optimal" next state to implement the remaining gates in
        the depth list.

        :param current_configuration: The starting configuration of ions in ion trap zones
        :param remaining_commands: The list of gate commands used to determine the next ion placement.
        """
        n_qubits = current_configuration.n_qubits
        depth_list = depth_list_from_command_list(n_qubits, remaining_commands)
        if not self._settings.ignore_swap_costs:
            # Update occupancy of port graph to current configuration
            for zone, occupants in current_configuration.zone_placement.items():
                self._port_graph.update_zone_occupancy_weight(zone, len(occupants))
        if depth_list:
            return self.handle_depth_list(current_configuration, depth_list)
        return self.handle_only_single_qubit_gates_remaining(
            current_configuration, remaining_commands
        )

    def handle_depth_list(
        self,
        current_configuration: TrapConfiguration,
        depth_list: list[list[tuple[int, int]]],
    ) -> TrapConfiguration:
        num_zones = self._arch.n_zones
        n_qubits = current_configuration.n_qubits
        shuttle_graph_data = self.get_circuit_shuttle_graph_data(
            current_configuration, depth_list
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
        return TrapConfiguration(n_qubits, new_placement)

    def handle_only_single_qubit_gates_remaining(
        self,
        current_configuration: TrapConfiguration,
        remaining_commands: list[Command],
    ) -> TrapConfiguration:
        qubit_tracker = QubitTracker(current_configuration.zone_placement)
        handle_only_single_qubits_remaining(
            remaining_commands,
            qubit_tracker,
            self._arch,
            self._macro_arch,
            self._port_graph,
            self._settings.ignore_swap_costs,
        )
        # Now move any unused qubits to vacant spots in new config
        handle_unused_qubits(self._arch, self._macro_arch, qubit_tracker)
        return TrapConfiguration(
            current_configuration.n_qubits, qubit_tracker.new_placement()
        )

    def get_circuit_shuttle_graph_data(
        self, starting_config: TrapConfiguration, depth_list: DepthList
    ) -> GraphData:
        """Calculate graph data for qubit-zone graph to be partitioned"""
        n_qubits = starting_config.n_qubits
        num_zones = self._arch.n_zones
        places_per_zone = [
            self._arch.get_zone_max_ions_gates(i) + 1
            for i, _ in enumerate(
                self._arch.zones
            )  # +1 is for the fixed vertex for each zone itself
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
        for zone, qubits in starting_config.zone_placement.items():
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
        if not self._settings.ignore_swap_costs:
            shortest_path_port0, path_length0, target_port_0 = (
                self._port_graph.shortest_port_path_length(zone1, 0, other_zone1)
            )
            shortest_path_port1, path_length1, target_port_1 = (
                self._port_graph.shortest_port_path_length(zone1, 1, other_zone1)
            )
            return min(path_length0, path_length1)
        shortest_path = self._macro_arch.shortest_path(int(zone1), int(other_zone1))
        if shortest_path:
            return len(shortest_path) - 1
        raise ZoneRoutingError(
            f"Shortest path could not be calculated"
            f" between zones {zone1} and {other_zone1}"
        )
