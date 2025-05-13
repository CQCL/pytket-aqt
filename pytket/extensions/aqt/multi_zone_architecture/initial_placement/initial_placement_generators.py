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
from dataclasses import dataclass
from logging import getLogger
from typing import Protocol

from pytket import Circuit

from ..architecture import MultiZoneArchitectureSpec
from ..circuit.helpers import ZonePlacement
from ..depth_list.depth_list import DepthList, get_initial_depth_list
from ..graph_algs.graph import GraphData
from ..graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
    MissingMtKahyparInstallError,
)
from .settings import InitialPlacementAlg, InitialPlacementSettings

logger = getLogger("initial_placement_logger")


class InitialPlacementError(Exception):
    pass


class InitialPlacementGenerator(Protocol):
    """Protocol for classes implementing an initial placement of ions in ion traps"""

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitectureSpec
    ) -> ZonePlacement: ...


@dataclass
class ManualInitialPlacement(InitialPlacementGenerator):
    """Used to generate an initial placement from manual input"""

    placement: ZonePlacement

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitectureSpec
    ) -> ZonePlacement:
        _check_n_qubits(circuit, arch)
        placed_qubits = []
        for zone, qubits in self.placement.items():
            placed_qubits.extend(qubits)
            if len(qubits) > arch.get_zone_max_ions(zone):
                raise InitialPlacementError(
                    f"Specified manual initial placement is faulty, zone {zone},"
                    f" can hold {arch.get_zone_max_ions(zone)} qubits, but"
                    f" {len(qubits)} were placed"
                )
        counts = [placed_qubits.count(i) for i in range(circuit.n_qubits)]
        duplicates = ".".join(
            [
                f"Qubit {i} placed {count} times"
                for i, count in enumerate(counts)
                if count > 1
            ]
        )
        if duplicates:
            raise InitialPlacementError(
                f"Duplicate placements detected in manual"
                f" initial placement. {duplicates}"
            )
        unplaced_qubits = {i for i in range(circuit.n_qubits)}.difference(placed_qubits)  # noqa: C416
        if unplaced_qubits:
            raise InitialPlacementError(
                f"Some qubits missing in manual initial placement."
                f" Missing qubits: {unplaced_qubits}"
            )
        for zone in range(arch.n_zones):
            if zone not in self.placement:
                self.placement[zone] = []
        return self.placement


@dataclass
class QubitOrderInitialPlacement(InitialPlacementGenerator):
    """Used to generate an initial placement based on qubit order.

    Zones are filled in increasing number order with qubits in increasing
    number order
    """

    zone_free_space: int

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitectureSpec
    ) -> ZonePlacement:
        _check_n_qubits(circuit, arch)
        placement: ZonePlacement = {}
        i_start = 0
        for zone in range(arch.n_zones):
            places_avail = arch.get_zone_max_ions(zone) - self.zone_free_space
            i_end = min(i_start + places_avail, circuit.n_qubits)
            placement[zone] = [i for i in range(i_start, i_end)]  # noqa: C416
            i_start = i_end
        return placement


@dataclass
class GraphMapInitialPlacement(InitialPlacementGenerator):
    """Used to generate an initial placement based on graph algorithms.

    Graph partitioning and graph mapping are used to assign qubits to zones
    """

    zone_free_space: int
    max_depth: int

    def initial_placement(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitectureSpec,
    ) -> ZonePlacement:
        try:
            from ..graph_algs.mt_kahypar import MtKahyparPartitioner
        except ImportError as e:
            logger.error(
                "Using graph partitioning algorithms requires"
                " the installation of mt-kahypar python bindings, "
                "see installation instructions for details"
            )
            raise e
        _check_n_qubits(circuit, arch)
        n_parts = arch.n_zones
        n_qubits = circuit.n_qubits
        initial_depth_list = get_initial_depth_list(circuit)
        circuit_graph_data = self.get_circuit_graph_data(initial_depth_list, arch)
        arch_graph_data = self.get_arch_graph_data(arch)
        partitioner = MtKahyparPartitioner()
        vertex_to_part = partitioner.map_graph_to_target_graph(
            circuit_graph_data, arch_graph_data
        )
        qubit_to_part = vertex_to_part[:n_qubits]
        placement: ZonePlacement = {i: [] for i in range(n_parts)}
        for qubit, part in enumerate(qubit_to_part):
            placement[part].append(qubit)
        return placement

    def get_circuit_graph_data(
        self, depth_list: DepthList, arch: MultiZoneArchitectureSpec
    ) -> GraphData:
        # Vertices up to n_qubit represent qubits,
        # the rest available spaces for qubits in the arch
        places_per_zone = [arch.get_zone_max_ions(i) for i, _ in enumerate(arch.zones)]
        free_places_per_zone = [
            self.zone_free_space if places_avail > 3 else 1  # noqa: PLR2004
            for places_avail in places_per_zone
        ]
        block_weights = [
            max(0, places_per_zone[i] - free_places_per_zone[i])
            for i, _ in enumerate(arch.zones)
        ]
        num_vertices = sum(block_weights)
        vertex_weights = [1 for _ in range(num_vertices)]

        # Edges
        edges: list[tuple[int, int]] = []
        edge_weights: list[int] = []
        max_weight = math.pow(2, 20)
        for i, pairs in enumerate(depth_list):
            if i > self.max_depth:
                break
            weight = math.ceil(math.exp(-1 * i) * max_weight)
            for pair in pairs:
                if pair in edges:
                    index = edges.index(pair)
                    edge_weights[index] = edge_weights[index] + weight
                else:
                    edges.append(pair)
                    edge_weights.append(weight)

        return GraphData(
            num_vertices,
            vertex_weights,
            edges,
            edge_weights,
            part_max_sizes=block_weights,
        )

    @staticmethod
    def get_arch_graph_data(arch: MultiZoneArchitectureSpec) -> GraphData:
        # Turn MultiZoneArchitecture into GraphData
        n_vertices = arch.n_zones
        arch_vertex_weights = [1] * n_vertices
        arch_edges = []
        arch_edge_weights = []
        for connection in arch.connections:
            arch_edges.append(
                (connection.zone_port_spec0.zone_id, connection.zone_port_spec1.zone_id)
            )
            arch_edge_weights.append(1)
        return GraphData(n_vertices, arch_vertex_weights, arch_edges, arch_edge_weights)


def _check_n_qubits(circuit: Circuit, arch: MultiZoneArchitectureSpec) -> None:
    n_qubits = circuit.n_qubits
    n_qubits_max = arch.n_qubits_max
    if n_qubits > n_qubits_max:
        raise InitialPlacementError(
            f"Attempting to place circuit with {n_qubits}"
            f" qubits, but architecture only supports up to {n_qubits_max}"
        )


def get_initial_placement_generator(
    settings: InitialPlacementSettings,
) -> InitialPlacementGenerator:
    """Return an initial placement generator from the initial placement settings"""
    match settings.algorithm:
        case InitialPlacementAlg.graph_partition:
            if MT_KAHYPAR_INSTALLED:
                return GraphMapInitialPlacement(
                    zone_free_space=settings.zone_free_space,
                    max_depth=settings.max_depth,
                )
            raise MissingMtKahyparInstallError()  # noqa: RSE102
        case InitialPlacementAlg.qubit_order:
            return QubitOrderInitialPlacement(zone_free_space=settings.zone_free_space)
        case InitialPlacementAlg.manual:
            assert settings.manual_placement is not None
            return ManualInitialPlacement(placement=settings.manual_placement)


def get_initial_placement(
    settings: InitialPlacementSettings,
    circuit: Circuit,
    arch: MultiZoneArchitectureSpec,
) -> ZonePlacement:
    initial_placement_generator = get_initial_placement_generator(settings)
    return initial_placement_generator.initial_placement(circuit, arch)
