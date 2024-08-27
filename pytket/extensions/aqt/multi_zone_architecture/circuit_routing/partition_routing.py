import math
from copy import deepcopy

from pytket import Circuit

from ..circuit.helpers import ZonePlacement, ZoneRoutingError
from .settings import RoutingSettings
from ..architecture import MultiZoneArchitecture
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..depth_list.depth_list import (
    get_initial_depth_list,
    get_updated_depth_list,
    DepthList,
)
from ..graph_algs.graph import GraphData
from ..graph_algs.mt_kahypar import MtKahyparPartitioner
from ..macro_architecture_graph import empty_macro_arch_from_architecture, ZoneId


class PartitionCircuitRouter:
    def __init__(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitecture,
        initial_placement: ZonePlacement,
        settings: RoutingSettings,
    ):
        self._circuit = circuit
        self._arch = arch
        self._macro_arch = empty_macro_arch_from_architecture(arch)
        self._initial_placement = initial_placement
        self._settings = settings

    def get_routed_circuit(self) -> MultiZoneCircuit:
        n_qubits = self._circuit.n_qubits
        depth_list = get_initial_depth_list(self._circuit)
        commands = self._circuit.get_commands().copy()
        mz_circuit = MultiZoneCircuit(
            self._arch, self._initial_placement, n_qubits, self._circuit.n_bits
        )
        for old_place, new_place in self.placement_generator(depth_list):
            if self._settings.debug_level > 0:
                print("-------")
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
                    print(
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
                if n_args == 1:
                    qubit0 = cmd.args[0].index[0]
                    if qubit0 in stragglers:
                        leftovers.append(cmd)
                    else:
                        mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
                elif n_args == 2:
                    qubit0 = cmd.args[0].index[0]
                    qubit1 = cmd.args[1].index[0]
                    if qubit0 in stragglers:
                        stragglers.add(qubit1)
                        leftovers.append(cmd)
                        continue
                    if qubit1 in stragglers:
                        stragglers.add(qubit0)
                        leftovers.append(cmd)
                        continue
                    if qubit_to_zone_old[qubit0] == qubit_to_zone_old[qubit1]:
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
            # old_n_shuttles = mz_circuit.get_n_shuttles()
            _make_necessary_config_moves((old_place, new_place), mz_circuit)
            # print("Added shuttles: ", mz_circuit.get_n_shuttles() - old_n_shuttles)
        for cmd in commands:
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
        return mz_circuit

    def placement_generator(
        self, depth_list: DepthList
    ) -> tuple[ZonePlacement, ZonePlacement]:
        # generates pairs of configs representing one shuttling step
        current_placement = deepcopy(self._initial_placement)
        n_qubits = self._circuit.n_qubits
        qubit_to_zone = _get_qubit_to_zone(n_qubits, current_placement)
        depth_list = get_updated_depth_list(n_qubits, qubit_to_zone, depth_list)
        max_iter = len(depth_list)
        iteration = 0
        while depth_list:
            new_placement = self.new_placement_graph_partition_alg(
                depth_list, current_placement
            )
            yield current_placement, new_placement
            qubit_to_zone = _get_qubit_to_zone(n_qubits, new_placement)
            depth_list = get_updated_depth_list(n_qubits, qubit_to_zone, depth_list)
            current_placement = new_placement
            if iteration > max_iter:
                raise Exception("placement alg is not converging")
            iteration += 1

    def new_placement_graph_partition_alg(
        self,
        depth_list: DepthList,
        starting_placement: ZonePlacement,
    ) -> ZonePlacement:
        n_qubits = self._circuit.n_qubits
        n_qubits_max = self._arch.n_qubits_max
        if n_qubits > n_qubits_max:
            raise ZoneRoutingError(
                f"Attempting to route circuit with {n_qubits}"
                f" qubits, but architecture only supports up to {n_qubits_max}"
            )

        num_zones = self._arch.n_zones
        shuttle_graph_data, fixed_list = self.get_circuit_shuttle_graph_data(
            starting_placement, depth_list
        )
        partitioner = MtKahyparPartitioner(
            self._settings.n_threads, log_level=self._settings.debug_level
        )
        vertex_to_part = partitioner.partition_graph(
            shuttle_graph_data, num_zones, fixed_list
        )
        new_placement = {i: [] for i in range(num_zones)}
        part_to_zone = [-1] * num_zones
        for vertex in range(n_qubits, n_qubits + num_zones):
            part_to_zone[vertex_to_part[vertex]] = vertex - n_qubits
        for vertex in range(n_qubits):
            new_placement[part_to_zone[vertex_to_part[vertex]]].append(vertex)
        return new_placement

    def get_circuit_shuttle_graph_data(
        self, starting_placement: ZonePlacement, depth_list: DepthList
    ) -> tuple[GraphData, list[int]]:
        n_qubits = self._circuit.n_qubits
        num_zones = self._arch.n_zones
        num_spots = sum(
            [self._arch.get_zone_max_ions(i) for i, _ in enumerate(self._arch.zones)]
        )
        edges: list[tuple[int, int]] = []
        edge_weights: list[int] = []

        # add gate edges
        max_considered_depth = min(200, len(depth_list))
        max_weight = math.ceil(math.pow(2, 18))
        for i, pairs in enumerate(depth_list):
            if i > max_considered_depth:
                break
            # weight = math.ceil(math.exp(-3/avg_block_weight * i) * max_weight)
            weight = math.ceil(math.exp(-2 * i) * max_weight)
            for pair in pairs:
                if pair in edges:
                    index = edges.index(pair)
                    edge_weights[index] = edge_weights[index] + weight
                else:
                    edges.append(pair)
                    edge_weights.append(weight)

        # add shuttling penalty (just distance between zones for now,
        # should later be dependent on shuttling cost)

        # max_shuttle_weight = math.ceil(math.exp(-3/avg_block_weight * 5) * max_weight)
        max_shuttle_weight = math.ceil(max_weight - 10000)
        for zone, qubits in starting_placement.items():
            for other_zone in range(num_zones):
                weight = math.ceil(
                    math.exp(-0.8 * (self.shuttling_penalty(zone, other_zone) + 4))
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
            + [zone for zone in range(num_zones)]
            + [-1] * (num_vertices - n_qubits - num_zones)
        )

        return GraphData(num_vertices, vertex_weights, edges, edge_weights), fixed_list

    def shuttling_penalty(self, zone1: int, other_zone1: int):
        shortest_path = self._macro_arch.shortest_path(
            ZoneId(zone1), ZoneId(other_zone1)
        )
        return len(shortest_path)


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
