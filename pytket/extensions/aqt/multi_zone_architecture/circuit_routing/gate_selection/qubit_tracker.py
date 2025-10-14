from copy import deepcopy

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import ZonePlacement


class QubitTracker:
    """Tracks which qubits are in which zones for the entire architecture"""

    def __init__(self, initial_placement: ZonePlacement) -> None:
        self._current_placement = deepcopy(initial_placement)
        self._new_placement: dict[int, list[int]] = {i: [] for i in initial_placement}
        self._current_qubit_to_zone = {}
        for zone, qubit_list in initial_placement.items():
            for qubit in qubit_list:
                self._current_qubit_to_zone[qubit] = zone

    def new_placement(self):
        return self._new_placement

    def current_zone(self, qubit: int) -> int:
        return self._current_qubit_to_zone[qubit]

    def old_zone_occupants(self, zone: int) -> list[int]:
        return self._current_placement[zone]

    def zone_new_occupants(self, zone: int) -> list[int]:
        return self._new_placement[zone]

    def n_zone_new_occupants(self, zone: int) -> int:
        return len(self._new_placement[zone])

    def move_qubit(self, qubit: int, starting_zone: int, target_zone: int) -> None:
        self._current_placement[starting_zone].remove(qubit)
        self._new_placement[target_zone].append(qubit)
        self._current_qubit_to_zone[qubit] = target_zone

    def lock_qubit(self, qubit: int, old_zone: int, lock_zone: int) -> None:
        self._current_placement[old_zone].remove(qubit)
        self._new_placement[lock_zone].append(qubit)
        self._current_qubit_to_zone[qubit] = lock_zone
