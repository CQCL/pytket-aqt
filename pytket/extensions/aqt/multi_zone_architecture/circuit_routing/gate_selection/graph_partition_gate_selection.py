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
import logging
import math

from pytket.circuit import Command

from ...circuit.helpers import ZonePlacement
from ...cost_model import DynamicArch, RoutingCostModel
from ...depth_list.depth_list import (
    DepthList,
    depth_list_from_command_list,
)
from ...graph_algs.graph import GraphData
from ...graph_algs.mt_kahypar import MtKahyparPartitioner
from ..settings import RoutingSettings
from .config_selector_protocol import ConfigSelector
from .greedy_gate_selection import (
    handle_only_single_qubits_remaining,
    handle_unused_qubits,
)
from .qubit_tracker import QubitTracker

logger = logging.getLogger(__name__)


def log_depth_list(depth_list):
    logger.debug("--- Depth List ---")
    for i in range(min(4, len(depth_list))):
        msg = f"{depth_list[i]}"
        logger.debug(msg)


class PartitionGateSelector(ConfigSelector):
    """Uses graph partitioning to add shuttles and swaps to a circuit

    The routed circuit can be directly run on the given Architecture

    :param cost_model: Cost model for determining movement costs
    :param settings: The settings used for routing
    """

    def __init__(
        self,
        cost_model: RoutingCostModel,
        settings: RoutingSettings,
    ):
        self._settings = settings
        self._cost_model = cost_model

    def next_config(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        """Generates a new qubit placement in zones to implement the next gates

        The returned ZonePlacement
        represents the "optimal" next state to implement the remaining gates in
        the depth list. The ordering of the qubits within the zones is arbitrary. The correct
        ordering will be determined at the qubit routing stage.

        :param dyn_arch: The dynamic architecture containing current configuration of ions in ion trap zones
        :param remaining_commands: The list of gate commands used to determine the next ion placement.
        """
        current_configuration = dyn_arch.trap_configuration
        n_qubits = current_configuration.n_qubits
        depth_list = depth_list_from_command_list(n_qubits, remaining_commands)
        if depth_list:
            return self.handle_depth_list(dyn_arch, depth_list)
        return self.handle_only_single_qubit_gates_remaining(
            dyn_arch, remaining_commands
        )

    def handle_depth_list(
        self,
        dyn_arch: DynamicArch,
        depth_list: list[list[tuple[int, int]]],
    ) -> ZonePlacement:
        num_zones = dyn_arch.n_zones
        n_qubits = dyn_arch.n_qubits
        shuttle_graph_data = self.get_circuit_shuttle_graph_data(dyn_arch, depth_list)
        partitioner = MtKahyparPartitioner()
        log_depth_list(depth_list)
        vertex_to_part = partitioner.partition_graph(shuttle_graph_data, num_zones)
        new_placement: ZonePlacement = [[] for _ in range(num_zones)]
        part_to_zone = [-1] * num_zones
        for vertex in range(n_qubits, n_qubits + num_zones):
            part_to_zone[vertex_to_part[vertex]] = vertex - n_qubits
        for vertex in range(n_qubits):
            new_placement[part_to_zone[vertex_to_part[vertex]]].append(vertex)
        return new_placement

    def handle_only_single_qubit_gates_remaining(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        qubit_tracker = QubitTracker(
            dyn_arch.n_qubits, dyn_arch.trap_configuration.zone_placement
        )
        handle_only_single_qubits_remaining(
            dyn_arch, self._cost_model, remaining_commands, qubit_tracker
        )
        # Now move any unused qubits to vacant spots in new config
        handle_unused_qubits(dyn_arch, self._cost_model, qubit_tracker)
        return qubit_tracker.new_placement()

    def get_circuit_shuttle_graph_data(
        self, dyn_arch: DynamicArch, depth_list: DepthList
    ) -> GraphData:
        """Calculate graph data for qubit-zone graph to be partitioned"""
        n_qubits = dyn_arch.n_qubits
        num_zones = dyn_arch.n_zones
        places_per_zone = [
            dyn_arch.zone_max_gate_cap[i]
            + 1  # +1 is for the fixed vertex for each zone itself
            for i in range(num_zones)
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
        if dyn_arch.has_memory_zones:
            edge_pair_pairs = [
                (pair[i], zone + n_qubits)
                for i in [0, 1]
                for pair in depth_list[0]
                for zone in dyn_arch.gate_zones
            ]
            edge_pair_weights = (
                [max_weight] * len(depth_list[0]) * 2 * len(dyn_arch.gate_zones)
            )
            edges.extend(edge_pair_pairs)
            edge_weights.extend(edge_pair_weights)

        # add shuttling penalty (just distance between zones for now,
        # should later be dependent on shuttling cost)
        max_shuttle_weight = math.ceil(max_weight / 2)
        for zone, qubits in enumerate(dyn_arch.trap_configuration.zone_placement):
            for qubit in qubits:
                for other_zone in range(num_zones):
                    # disregard initial swap costs for now
                    weight = math.ceil(
                        math.exp(
                            -0.8
                            * self.shuttling_penalty(dyn_arch, qubit, zone, other_zone)
                        )
                        * max_shuttle_weight
                    )
                    if weight < 1:
                        continue
                    edges.append((other_zone + n_qubits, qubit))
                    edge_weights.append(weight)

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

    def shuttling_penalty(
        self,
        dyn_arch: DynamicArch,
        qubit: int,
        src_zone: int,
        trg_zone: int,
    ) -> int:
        """Calculate penalty for shuttling from one zone to another"""
        move_result_0 = self._cost_model.move_cost_src_port_0(
            dyn_arch, [qubit], src_zone, trg_zone
        )
        move_result_1 = self._cost_model.move_cost_src_port_1(
            dyn_arch, [qubit], src_zone, trg_zone
        )
        match (move_result_0 is not None, move_result_1 is not None):
            case (True, True):
                return min(
                    move_result_0.path_cost,
                    move_result_1.path_cost,
                )
            case (True, False):
                return move_result_0.path_cost
            case (False, True):
                return move_result_1.path_cost
        raise ValueError("Could note determine path")
