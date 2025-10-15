from copy import deepcopy

from ...architecture_portgraph import MultiZonePortGraph
from ...circuit.helpers import TrapConfiguration, ZonePlacement, get_qubit_to_zone
from ...circuit.multizone_circuit import MultiZoneCircuit
from ...circuit_routing.settings import RoutingSettings
from .router import Router


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

    def route_source_to_target_config(
        self,
        source: TrapConfiguration,
        target: ZonePlacement,
    ) -> TrapConfiguration:
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
        # sort based on ascending number of free places in the target zone (at beginning)
        qubits_to_move.sort(
            key=lambda x: self._mz_circ.architecture.get_zone_max_ions_gates(x[2])
            - len(current_placement[x[2]])
        )

        while qubits_to_move:
            qubit, start, targ = qubits_to_move[-1]
            free_space_target_zone = self._mz_circ.architecture.get_zone_max_ions_gates(
                targ
            ) - len(current_placement[targ])
            match free_space_target_zone:
                case 0:
                    self._move_qubit(qubit, start, targ, current_placement)
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
                    self._move_qubit(qubit, start, targ, current_placement)
                    # remove this move from list
                    qubits_to_move.pop()
        return TrapConfiguration(n_qubits, current_placement)

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
            shortest_path, target_port = (
                (shortest_path_port0, targ_port0)
                if path_length0 <= path_length1
                else (shortest_path_port1, targ_port1)
            )
            self._port_graph.update_zone_occupancy_weight(
                starting_zone, len(current_placement[starting_zone]) - 1
            )
            self._port_graph.update_zone_occupancy_weight(
                target_zone, len(current_placement[target_zone]) + 1
            )
        else:
            shortest_path = self._macro_arch.shortest_path(starting_zone, target_zone)
            _, target_port = self._macro_arch.get_connected_ports(
                shortest_path[-2], shortest_path[-1]
            )

        self._mz_circ.move_qubit(
            qubit_to_move,
            target_zone,
            precompiled=True,
            use_transport_limit=True,
            path_override=shortest_path,
        )
        current_placement[starting_zone].remove(qubit_to_move)
        if target_port == 1:
            current_placement[target_zone].append(qubit_to_move)
        else:
            current_placement[target_zone].insert(0, qubit_to_move)
