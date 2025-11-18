from copy import deepcopy

from ...circuit.helpers import ZonePlacement, get_qubit_to_zone


class QubitTracker:
    """Tracks which qubits are in which zones for the entire architecture"""

    def __init__(self, n_qubits: int, initial_placement: ZonePlacement) -> None:
        self._current_placement = deepcopy(initial_placement)
        self._new_placement: ZonePlacement = [[] for _ in initial_placement]
        self._current_qubit_to_zone = get_qubit_to_zone(n_qubits, initial_placement)

    def new_placement(self) -> ZonePlacement:
        return self._new_placement

    def current_zone(self, qubit: int) -> int:
        return int(self._current_qubit_to_zone[qubit])

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
