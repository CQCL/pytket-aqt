from copy import deepcopy

from pytket import Circuit

from ..architecture import MultiZoneArchitectureSpec
from ..architecture_portgraph import MultiZonePortGraph
from ..circuit.helpers import TrapConfiguration
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..macro_architecture_graph import empty_macro_arch_from_architecture
from .helpers import get_qubit_to_zone
from .settings import RoutingSettings


class GeneralRouter:
    def __init__(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitectureSpec,
        settings: RoutingSettings,
    ):
        self._circuit = circuit
        self._arch = arch
        self._macro_arch = empty_macro_arch_from_architecture(arch)
        self._port_graph = MultiZonePortGraph(arch)
        self._settings = settings

    def route_source_to_target_config(
        self,
        source: TrapConfiguration,
        target: TrapConfiguration,
        mz_circ: MultiZoneCircuit,
    ) -> None:
        n_qubits = source.n_qubits
        new_place = target.zone_placement
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
            for zone, occupants in current_placement.items():
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
            key=lambda x: mz_circ.architecture.get_zone_max_ions_gates(x[2])
            - len(current_placement[x[2]])
        )

        def _move_qubit(
            qubit_to_move: int, starting_zone: int, target_zone: int
        ) -> None:
            if not self._settings.ignore_swap_costs:
                shortest_path_port0, path_length0, _ = (
                    self._port_graph.shortest_port_path_length(
                        starting_zone, 0, target_zone
                    )
                )
                shortest_path_port1, path_length1, _ = (
                    self._port_graph.shortest_port_path_length(
                        starting_zone, 1, target_zone
                    )
                )
                shortest_path = (
                    shortest_path_port0
                    if path_length0 <= path_length1
                    else shortest_path_port1
                )
                mz_circ.move_qubit(
                    qubit_to_move,
                    target_zone,
                    precompiled=True,
                    use_transport_limit=True,
                    shortest_path_override=shortest_path,
                )
                self._port_graph.update_zone_occupancy_weight(
                    starting_zone, len(current_placement[starting_zone])
                )
                self._port_graph.update_zone_occupancy_weight(
                    target_zone, len(current_placement[target_zone])
                )
            else:
                mz_circ.move_qubit(
                    qubit_to_move,
                    target_zone,
                    precompiled=True,
                    use_transport_limit=True,
                )
            current_placement[starting_zone].remove(qubit_to_move)
            current_placement[target_zone].append(qubit_to_move)

        while qubits_to_move:
            qubit, start, targ = qubits_to_move[-1]
            free_space_target_zone = mz_circ.architecture.get_zone_max_ions_gates(
                targ
            ) - len(current_placement[targ])
            match free_space_target_zone:
                case 0:
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
