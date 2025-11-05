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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .dynamic_architecture import DynamicArch


@dataclass
class MoveCostResult:
    optimal_path: list[int]
    path_cost: int
    src_port: int
    trg_port: int


def unwrap_move_cost_result(move_cost_result: MoveCostResult | None) -> MoveCostResult:
    if move_cost_result is None:
        raise ValueError("Error extracting move cost result")
    return move_cost_result


class RoutingCostModel(Protocol):
    def move_cost(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        pass

    def move_cost_src_port_0(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        pass

    def move_cost_src_port_1(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        pass

    def closest_zones(
        self, dyn_arch: DynamicArch, qubit: int, src_zone: int
    ) -> Iterable[int]:
        pass


class ShuttleOnlyCostModel:
    def move_cost(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        if not all(z[0] == src_zone for z in dyn_arch.qubit_to_zone_pos[qubits]):
            raise ValueError("All qubits must be src_zone")
        return self._move_cost_any_src_port(dyn_arch, qubits, src_zone, trg_zone)

    def move_cost_src_port_0(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        if not all(z[0] == src_zone for z in qubit_zone_pos):
            raise ValueError("All qubits must be src_zone")
        return self._move_cost_any_src_port(dyn_arch, qubits, src_zone, trg_zone)

    def move_cost_src_port_1(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        if not all(z[0] == src_zone for z in qubit_zone_pos):
            raise ValueError("All qubits must be src_zone")
        return self._move_cost_any_src_port(dyn_arch, qubits, src_zone, trg_zone)

    def closest_zones(
        self, dyn_arch: DynamicArch, qubit: int, src_zone: int
    ) -> Iterable[int]:
        return _closest_zones(dyn_arch, self, qubit, src_zone)

    @staticmethod
    def _move_cost_any_src_port(
        dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        # This model will not distinguish between ports because swap costs are neglected
        # Thus starting from port 0 has same cost as starting from port 1
        length, path, capacity = dyn_arch.shuttle_only_shortest_path_and_path_capacity(
            src_zone, trg_zone
        )
        if capacity < len(qubits):
            return None
        src_port, _ = dyn_arch.connection_ports(path[0], path[1])
        _, trg_port = dyn_arch.connection_ports(path[-2], path[-1])
        return MoveCostResult(path, length, src_port, trg_port)


class ShuttlePSwapCostModel:
    def move_cost(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        if not all(z[0] == src_zone for z in qubit_zone_pos):
            raise ValueError("All qubits must be src_zone")
        move_result_0 = self._move_cost_src_port_0(dyn_arch, qubits, src_zone, trg_zone)
        move_result_1 = self._move_cost_src_port_1(dyn_arch, qubits, src_zone, trg_zone)
        match (move_result_0 is not None, move_result_1 is not None):
            case (True, True):
                if move_result_0.path_cost <= move_result_1.path_cost:
                    return move_result_0
                return move_result_1
            case (True, False):
                return move_result_0
            case (False, True):
                return move_result_1
        return None

    def move_cost_src_port_0(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        if not all(z[0] == src_zone for z in qubit_zone_pos):
            raise ValueError("All qubits must be src_zone")
        return self._move_cost_src_port_0(dyn_arch, qubits, src_zone, trg_zone)

    def move_cost_src_port_1(
        self, dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        if not all(z[0] == src_zone for z in qubit_zone_pos):
            raise ValueError("All qubits must be src_zone")
        return self._move_cost_src_port_1(dyn_arch, qubits, src_zone, trg_zone)

    def closest_zones(
        self, dyn_arch: DynamicArch, qubit: int, src_zone: int
    ) -> Iterable[int]:
        return _closest_zones(dyn_arch, self, qubit, src_zone)

    @staticmethod
    def _move_cost_src_port_0(
        dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        n_move = len(qubits)
        shortest_path_port0, path_length0, targ_port0 = (
            dyn_arch.shortest_port_path_length(src_zone, 0, trg_zone, n_move)
        )
        if shortest_path_port0 is None:
            return None

        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        swap_cost_src_zone = dyn_arch.zone_swap_costs[src_zone]
        size_discount = np.arange(n_move).sum()
        # Cost of moving all qubits to port 0
        # Subtract size_discount because the qubits don't need to swap past the other
        # qubits that moved before it (1 less for second qubit, 2 less for third, ... = sum(0, 1, ..., n_move)
        swap_costs_0 = (qubit_zone_pos[:, 1].sum() - size_discount) * swap_cost_src_zone
        total_cost_0 = path_length0 + swap_costs_0
        return MoveCostResult(shortest_path_port0, total_cost_0, 0, targ_port0)

    @staticmethod
    def _move_cost_src_port_1(
        dyn_arch: DynamicArch, qubits: list[int], src_zone: int, trg_zone: int
    ) -> MoveCostResult | None:
        n_move = len(qubits)
        shortest_path_port1, path_length1, targ_port1 = (
            dyn_arch.shortest_port_path_length(src_zone, 1, trg_zone, n_move)
        )
        if shortest_path_port1 is None:
            return None

        edge_pos_src = dyn_arch.zone_occupancy[src_zone] - 1
        qubit_zone_pos = dyn_arch.qubit_to_zone_pos[qubits]
        swap_cost_src_zone = dyn_arch.zone_swap_costs[src_zone]
        size_discount = np.arange(n_move).sum()
        # Cost of moving all qubits to port 1
        # Subtract size_discount because the qubits don't need to swap past the other
        # qubits that moved before it (1 less for second qubit, 2 less for third, ...)
        swap_costs_1 = (
            (edge_pos_src - qubit_zone_pos[:, 1]).sum() - size_discount
        ) * swap_cost_src_zone
        total_cost_1 = path_length1 + swap_costs_1
        return MoveCostResult(shortest_path_port1, total_cost_1, 1, targ_port1)


def _closest_zones(
    dyn_arch: DynamicArch, cost_model: RoutingCostModel, qubit: int, src_zone: int
) -> Iterable[int]:
    # this is not 100% accurate but should be good enough (path cost can be lower for zones in later bfs layers
    layers = dyn_arch.macro_graph_bfs_layers(src_zone)
    # skip first layer which is just src_zone itself
    next(layers)
    for layer in layers:
        ordered_trg_move_costs = sorted(
            [
                (
                    zone,
                    unwrap_move_cost_result(
                        cost_model.move_cost(dyn_arch, [qubit], src_zone, zone)
                    ).path_cost,
                )
                for zone in layer
            ],
            key=lambda x: x[1],
        )
        for result in ordered_trg_move_costs:
            yield result[0]
