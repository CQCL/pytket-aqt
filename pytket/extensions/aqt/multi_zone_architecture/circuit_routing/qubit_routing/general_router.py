import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from ...architecture import PortId
from ...architecture_portgraph import MultiZonePortGraph
from ...circuit.helpers import TrapConfiguration, ZonePlacement, get_qubit_to_zone
from ...circuit.multizone_circuit import MultiZoneCircuit
from ...circuit_routing.settings import RoutingSettings
from ..routing_ops import PSwap, RoutingBarrier, RoutingOp, Shuttle
from .router import Router


@dataclass
class MoveGroup:
    # tuples where first int is qubit and second is its place in source zone
    qubit_src_index: list[tuple[int, int]]
    source: int
    target: int
    target_free_space: int

    def __post_init__(self):
        # sort qubits_srcspots based on their srcspots
        self.qubit_src_index.sort(key=lambda x: x[1])


class GeneralRouter(Router):
    def __init__(
        self,
        circuit: MultiZoneCircuit,
        settings: RoutingSettings,
    ):
        self._mz_circ = circuit
        self._arch = circuit.architecture
        self._macro_arch = circuit.macro_arch
        self._port_graph = MultiZonePortGraph(self._arch)
        self._settings = settings

    def route_source_to_target_config(  # noqa: PLR0912, PLR0915
        self,
        source: TrapConfiguration,
        target: ZonePlacement,
    ) -> tuple[list[RoutingOp], TrapConfiguration]:
        n_qubits = source.n_qubits
        new_place = target
        old_place = source.zone_placement
        if self._settings.debug_level > 0:
            print("-------")  # noqa: T201
            for zone in range(self._arch.n_zones):
                changes_str = ", ".join(
                    [f"+{i}" for i in set(new_place[zone]).difference(old_place[zone])]
                    + [
                        f"-{i}"
                        for i in set(old_place[zone]).difference(new_place[zone])
                    ]
                )
                print(  # noqa: T201
                    f"Z{zone}: {old_place[zone]} ->"
                    f" {new_place[zone]} -- ({changes_str})"
                )
        qubit_to_zone_old = get_qubit_to_zone(n_qubits, old_place)
        qubit_to_zone_new = get_qubit_to_zone(n_qubits, new_place)
        qubits_to_move: list[tuple[int, int, int]] = []
        current_placement = deepcopy(old_place)
        if not self._settings.ignore_swap_costs:
            for zone, occupants in enumerate(current_placement):
                self._port_graph.update_zone_occupancy_weight(zone, len(occupants))
        for qubit in range(n_qubits):
            old_zone = qubit_to_zone_old[qubit]
            new_zone = qubit_to_zone_new[qubit]
            if old_zone != new_zone:
                qubits_to_move.append(
                    (qubit, qubit_to_zone_old[qubit], qubit_to_zone_new[qubit])
                )

        move_ops: list[RoutingOp] = [RoutingBarrier()]
        qubits_to_move2 = deepcopy(qubits_to_move)
        current_placement_copy = deepcopy(current_placement)

        def free_space_in_zone(zone_loc: int):
            # use the transport limit - 1 >= gate limit as the base capacity,
            # otherwise the implementation of a move group may leave the target zone
            # in a blocked state
            return (
                self._mz_circ.architecture.get_zone_max_ions_transport(zone_loc)
                - 1
                - len(current_placement_copy[zone_loc])
            )

        soft_locked = False  # soft locked means only
        transport_blocked_zone = None
        while qubits_to_move2:
            grouped = defaultdict(list)
            for qubit, src, trg in qubits_to_move2:
                grouped[(src, trg)].append(
                    (qubit, current_placement_copy[src].index(qubit))
                )
            move_groups = (
                [
                    MoveGroup(grouped_qbts, src, trg, free_space_in_zone(trg))
                    for (src, trg), grouped_qbts in grouped.items()
                ]
                if transport_blocked_zone is None
                else [
                    # If a zone is transport blocked, only consider moves out of it, in order to unblock it
                    # There must be one since the end state is not allowed to be transport blocked
                    MoveGroup(grouped_qbts, src, trg, free_space_in_zone(trg))
                    for (src, trg), grouped_qbts in grouped.items()
                    if src == transport_blocked_zone
                ]
            )
            if (
                soft_locked
                or sum(group.target_free_space for group in move_groups) == 0
            ):
                soft_locked = True
                # This can happen if qubits need to swap between full zones or
                # there is a cycle between full zones.
                # To solve, use the full transport capacity for all following rounds
                # The sum will remain zero once this point is reached since any movement
                # from one zone to another will cause free_space calculation to give +1 and -1 for
                # those zones.
                for group in move_groups:
                    group.target_free_space += 1
            min_cost = 1000000
            chosen_path = []
            chosen_move_group = None
            chosen_qubit_indices = []
            for move_group in move_groups:
                edge_pos_src = len(current_placement_copy[move_group.source]) - 1
                for n_move in range(
                    min(len(move_group.qubit_src_index), move_group.target_free_space),
                    0,
                    -1,
                ):
                    qubit_indices, path, cost, _, _ = self._calc_move_path_cost(
                        move_group, n_move, edge_pos_src
                    )
                    if cost:
                        cost_per_qubit = cost / n_move
                        if cost_per_qubit < min_cost:
                            min_cost = cost_per_qubit
                            chosen_path = path
                            chosen_move_group = move_group
                            chosen_qubit_indices = qubit_indices

            # Do move/ update placement
            move_ops.extend(
                self.move_qubits(
                    chosen_qubit_indices,
                    chosen_move_group.source,
                    chosen_move_group.target,
                    current_placement_copy,
                    chosen_path,
                )
            )
            move_ops.append(RoutingBarrier())

            # When soft locked a move may cause a zone to become transport blocked
            # resulting in its calculated free space taking the value -1
            # This makes sure the next move will be out of this zone, unblocking it
            if soft_locked and free_space_in_zone(chosen_move_group.target) == -1:
                transport_blocked_zone = chosen_move_group.target
            else:
                transport_blocked_zone = None
            # update port graph weights
            for zone in [chosen_move_group.source, chosen_move_group.target]:
                self._port_graph.update_zone_occupancy_weight(
                    zone, len(current_placement_copy[zone])
                )
            # remove moves that were made
            for q, _ in chosen_qubit_indices:
                qubits_to_move2.remove(
                    (q, chosen_move_group.source, chosen_move_group.target)
                )

        # # sort based on ascending number of free places in the target zone (at beginning)
        # qubits_to_move.sort(
        #     key=lambda x: free_space(x[2])
        # )

        # while qubits_to_move:
        #     qubit, start, targ = qubits_to_move[-1]
        #     free_space_target_zone = self._mz_circ.architecture.get_zone_max_ions_gates(
        #         targ
        #     ) - len(current_placement[targ])
        #     match free_space_target_zone:
        #         case 0:
        #             self._move_qubit(qubit, start, targ, current_placement)
        #             # remove this move from list
        #             qubits_to_move.pop()
        #             # find a qubit in target zone that needs to move and put it at end
        #             # of qubits_to_move, so it comes next
        #             moves_with_start_equals_current_targ = [
        #                 i
        #                 for i, move_tup in enumerate(qubits_to_move)
        #                 if move_tup[1] == targ
        #             ]
        #             if not moves_with_start_equals_current_targ:
        #                 raise ValueError("This shouldn't happen")
        #             next_move_index = moves_with_start_equals_current_targ[0]
        #             next_move = qubits_to_move.pop(next_move_index)
        #             qubits_to_move.append(next_move)
        #         case a if a < 0:
        #             raise ValueError("Should never be negative")
        #         case _:
        #             self._move_qubit(qubit, start, targ, current_placement)
        #             # remove this move from list
        #             qubits_to_move.pop()

        return move_ops, TrapConfiguration(n_qubits, current_placement_copy)

    def _calc_move_path_cost(
        self, move_group: MoveGroup, n_move: int, edge_pos_src: int
    ) -> (
        tuple[list[tuple[int, int]], int, int, int, int] | tuple[None, None, None, None]
    ):
        src = move_group.source
        trg = move_group.target
        shortest_path_port0, path_length0, targ_port0 = (
            self._port_graph.shortest_port_path_length(src, 0, trg, n_move)
        )
        shortest_path_port1, path_length1, targ_port1 = (
            self._port_graph.shortest_port_path_length(src, 1, trg, n_move)
        )
        swap_cost_src_zone = self._port_graph.swap_costs[move_group.source]
        qubits_indx_0 = move_group.qubit_src_index[:n_move]
        qubits_indx_1 = move_group.qubit_src_index[-n_move:]
        if shortest_path_port0 is not None:
            # Cost of moving all qubits to port 0 (-i is because the ith qubit only needs to go to the
            # position to the right of the 0...i-1 qubits already moved, not all the way to the edge)
            swap_costs_0 = (
                sum(
                    [
                        pos - i
                        for i, (_, pos) in enumerate(
                            move_group.qubit_src_index[:n_move]
                        )
                    ]
                )
                * swap_cost_src_zone
            )
            total_cost_0 = path_length0 + swap_costs_0
        else:
            total_cost_0 = 0
        if shortest_path_port1 is not None:
            # Cost of moving all qubits to port 1 (-i is because the ith qubit only needs to go to the
            # position to the right of the 0...i-1 qubits already moved, not all the way to the edge)
            swap_costs_1 = (
                sum(
                    [
                        edge_pos_src - pos - i
                        for i, (_, pos) in enumerate(
                            move_group.qubit_src_index[-n_move:]
                        )
                    ]
                )
                * swap_cost_src_zone
            )
            total_cost_1 = path_length1 + swap_costs_1
        else:
            total_cost_1 = 0

        match (shortest_path_port0 is not None, shortest_path_port1 is not None):
            case (True, True):
                qubits_index, shortest_path, total_cost, src_port, target_port = (
                    (qubits_indx_0, shortest_path_port0, total_cost_0, 0, targ_port0)
                    if total_cost_0 <= total_cost_1
                    else (
                        qubits_indx_1,
                        shortest_path_port1,
                        total_cost_1,
                        1,
                        targ_port1,
                    )
                )
            case (True, False):
                qubits_index, shortest_path, total_cost, src_port, target_port = (
                    qubits_indx_0,
                    shortest_path_port0,
                    total_cost_0,
                    0,
                    targ_port0,
                )
            case (False, True):
                qubits_index, shortest_path, total_cost, src_port, target_port = (
                    qubits_indx_1,
                    shortest_path_port1,
                    total_cost_1,
                    1,
                    targ_port1,
                )
            case _:
                qubits_index, shortest_path, total_cost, src_port, target_port = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        return qubits_index, shortest_path, total_cost, src_port, target_port

    def _move_qubit(
        self,
        qubit_to_move: int,
        starting_zone: int,
        target_zone: int,
        current_placement: ZonePlacement,
    ) -> None:
        if not self._settings.ignore_swap_costs:
            shortest_path_port0, path_length0, targ_port0 = (
                self._port_graph.shortest_port_path_length(
                    starting_zone, 0, target_zone
                )
            )
            shortest_path_port1, path_length1, targ_port1 = (
                self._port_graph.shortest_port_path_length(
                    starting_zone, 1, target_zone
                )
            )
            current_spot = current_placement[starting_zone].index(qubit_to_move)
            swap_cost_start_zone = self._port_graph.swap_costs[starting_zone]
            starting_zone_occupancy = len(current_placement[starting_zone])
            initial_swap_costs0 = current_spot * swap_cost_start_zone
            initial_swap_costs1 = (
                starting_zone_occupancy - 1 - current_spot
            ) * swap_cost_start_zone
            shortest_path, target_port = (
                (shortest_path_port0, targ_port0)
                if path_length0 + initial_swap_costs0
                <= path_length1 + initial_swap_costs1
                else (shortest_path_port1, targ_port1)
            )
            self._port_graph.update_zone_occupancy_weight(
                starting_zone, starting_zone_occupancy - 1
            )
            self._port_graph.update_zone_occupancy_weight(
                target_zone, len(current_placement[target_zone]) + 1
            )
        else:
            shortest_path = self._macro_arch.shortest_path(starting_zone, target_zone)
            _, target_port = self._macro_arch.get_connected_ports(
                shortest_path[-2], shortest_path[-1]
            )

        self._mz_circ.move_qubit_precompiled(qubit_to_move, target_zone, shortest_path)
        current_placement[starting_zone].remove(qubit_to_move)
        if target_port == 1:
            current_placement[target_zone].append(qubit_to_move)
        else:
            current_placement[target_zone].insert(0, qubit_to_move)

    def move_qubits(  # noqa: PLR0912
        self,
        qubits_index_to_move: list[tuple[int, int]],
        starting_zone: int,
        target_zone: int,
        current_placement: ZonePlacement,
        path: list[int],
    ) -> list[RoutingOp]:
        """Create a list of RoutingOp's that move the qubits (with src zone index, ordered by ascending index)
        from source zone to target port

        Modifies the input current placement to reflect the movement
        """
        ops: list[RoutingOp] = []

        # first get qubits to src zone edge and shuttle to first zone in path
        src_port, trg_port = self._macro_arch.get_connected_ports(path[0], path[1])
        # all_qubits should be kept in "logical order", i.e. the order they have
        # in their current zone when viewed from port 0 to port 1. To do this,
        # anytime they are shuttled from a 0 port to a 0 port (or 1 port to 1 port)
        # their order should be reversed
        all_qubits = [qubit for qubit, _ in qubits_index_to_move]

        if src_port == PortId.p0:
            qubits_zone = list(reversed(current_placement[starting_zone]))
            last_indx = len(current_placement[starting_zone]) - 1
            # update indices to reflect reversed ordering
            qubit_src_iter = [(q, last_indx - indx) for q, indx in qubits_index_to_move]
        else:
            # reversing makes logic same for moving to port 1 instead of port 0
            qubit_src_iter = reversed(qubits_index_to_move)
            qubits_zone = deepcopy(current_placement[starting_zone])

        for qubit, index in qubit_src_iter:
            ops.extend(
                [
                    PSwap(starting_zone, left_qubit, qubit)
                    for left_qubit in qubits_zone[index + 1 :]
                ]
            )
            qubits_zone.pop(index)

        ops.append(
            Shuttle(all_qubits.copy(), starting_zone, path[1], src_port, trg_port)
        )

        # remove from starting zone (in backwards order so index is stable)
        for _, index in reversed(qubits_index_to_move):
            current_placement[starting_zone].pop(index)

        # maintain logical ordering
        if src_port == trg_port:
            all_qubits.reverse()

        # for any remaining moves, swap to other port if necessary and shuttle
        current_port = trg_port
        for current_zone, next_zone in itertools.pairwise(path[1:]):
            src_port, trg_port = self._macro_arch.get_connected_ports(
                current_zone, next_zone
            )
            current_zone_occupants = current_placement[current_zone]
            if src_port != current_port:  # otherwise no swaps needed
                # 01 -> move qubits
                # abc -> occupants
                if src_port == PortId.p1:
                    move_qubits_iter = reversed(all_qubits)
                    current_zone_iter = current_zone_occupants
                    # [01abc] -> [0a1bc] -> [0ab1c] -> [0abc1] -> [a0bc1] -> [ab0c1] -> [abc01]
                else:
                    move_qubits_iter = all_qubits
                    current_zone_iter = list(reversed(current_zone_occupants))
                    # [abc01] -> [ab0c1] -> [a0bc1] -> [0abc1] -> [0ab1c] -> [0a1bc] -> [01abc]
                for move_qubit in move_qubits_iter:
                    ops.extend(
                        [
                            PSwap(current_zone, stay_qubit, move_qubit)
                            for stay_qubit in current_zone_iter
                        ]
                    )

            ops.append(
                Shuttle(all_qubits.copy(), current_zone, next_zone, src_port, trg_port)
            )
            # update current port
            current_port = trg_port
            # maintain logical ordering
            if src_port == trg_port:
                all_qubits.reverse()

        if trg_port == PortId.p0:
            all_qubits.extend(current_placement[target_zone])
            current_placement[target_zone] = all_qubits
        else:
            current_placement[target_zone].extend(all_qubits)
        return ops
