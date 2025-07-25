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
from copy import deepcopy

from networkx import bfs_layers  # type: ignore[import-untyped]

from pytket.circuit import Command, OpType

from ..architecture import MultiZoneArchitectureSpec
from ..circuit.helpers import TrapConfiguration
from ..depth_list.depth_list import depth_list_from_command_list
from ..macro_architecture_graph import MultiZoneArch
from .qubit_tracker import QubitTracker
from .settings import RoutingSettings


class GreedyGateSelector:
    """Uses a simple greedy algorithm for gate selection

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
        self._macro_arch = MultiZoneArch(arch)
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

        Cycle through commands and assign qubits to new zones in order to implement them on
        a "first come, first served" basis

        Rules:

        - One qubit gates do not necessitate a move (until all 2 qubit gates are accounted for).
          If they cannot be performed in the current config, they can wait until that qubit
           requires a 2 qubit gate (or at the end of the circuit if no 2 qubit gates remain)
        - Two qubit gates will cause one or more moves to place the participating qubits into
           the same memory zone. Once the placement of a qubit is changed in order to implement
           a two qubit gate, it is "locked" in place. This means at some point, no more legal
           moves will be possible. The algorithm will terminate at that point and the current
           configuration will be emitted as the next config.

           Necessary moves may require the displacement of other qubits, due to max
           qubit requirements (using the "gate operation" maximum). This displacement is
           taken into account "lazily". Any qubits that have been displaced and were not involved
           in gates before termination will be placed in the "nearest" zones will available
           capacity

        :param current_configuration: The starting configuration of ions in ion trap zones
        :param remaining_commands: The list of gate commands used to determine the next ion placement.
        """
        n_qubits = current_configuration.n_qubits
        qubit_tracker = QubitTracker(current_configuration.zone_placement)
        two_qubit_gate_depth_list = depth_list_from_command_list(
            n_qubits, remaining_commands
        )

        if two_qubit_gate_depth_list:
            self.handle_depth_list(n_qubits, two_qubit_gate_depth_list, qubit_tracker)
        else:
            handle_only_single_qubits_remaining(
                remaining_commands, qubit_tracker, self._arch, self._macro_arch
            )
        # Now move any unused qubits to vacant spots in new config
        handle_unused_qubits(self._arch, self._macro_arch, qubit_tracker)
        return TrapConfiguration(n_qubits, qubit_tracker.new_placement())

    def handle_depth_list(
        self,
        n_qubits: int,
        depth_list: list[list[tuple[int, int]]],
        qubit_tracker: QubitTracker,
    ):
        # locked qubits have already been assigned a zone in the new config and should therefore not move again
        locked_qubits = []
        # must wait qubits have had a 2 qubit gate that could not be implemented in the new config
        # thus any following 2 qubit gate involving them will not be implementable in the new config
        must_wait_qubits = []
        for depth in depth_list:
            for pair in depth:
                # If all qubits must wait or no more spots are available in gate zones, we can stop
                max_gate_zone_free_space = max(
                    [
                        self._arch.get_zone_max_ions_gates(gz)
                        - qubit_tracker.n_zone_new_occupants(gz)
                        for gz in self._macro_arch.gate_zones
                    ]
                )
                if len(must_wait_qubits) == n_qubits or max_gate_zone_free_space == 0:
                    break

                qubit0 = pair[0]
                qubit1 = pair[1]
                # handle case that one or both qubits must wait
                if handle_must_wait(qubit0, qubit1, must_wait_qubits):
                    continue

                # starting here, neither qubit must wait

                is_locked_0 = qubit0 in locked_qubits
                is_locked_1 = qubit1 in locked_qubits

                # handle both locked
                if is_locked_0 and is_locked_1:
                    zone0 = qubit_tracker.current_zone(qubit0)
                    zone1 = qubit_tracker.current_zone(qubit1)
                    if zone0 != zone1:
                        # if not in same zone, gate cannot be implemented
                        # so new gates involving these qubits must wait
                        must_wait_qubits.extend([qubit0, qubit1])
                    continue

                # handle only one locked
                if is_locked_0 or is_locked_1:
                    locked_q, other_q = (
                        (qubit0, qubit1) if is_locked_0 else (qubit1, qubit0)
                    )
                    if handle_one_locked_qubit(
                        locked_q, other_q, qubit_tracker, self._arch
                    ):
                        locked_qubits.append(other_q)
                    else:
                        must_wait_qubits.append(other_q)
                    continue

                # starting here, neither qubit is locked

                zone0 = qubit_tracker.current_zone(qubit0)
                zone1 = qubit_tracker.current_zone(qubit1)
                is_gate_zone0 = not self._arch.zones[zone0].memory_only
                is_gate_zone1 = not self._arch.zones[zone1].memory_only

                free_space_zone_0 = self._arch.get_zone_max_ions_gates(
                    zone0
                ) - qubit_tracker.n_zone_new_occupants(zone0)
                free_space_zone_1 = self._arch.get_zone_max_ions_gates(
                    zone1
                ) - qubit_tracker.n_zone_new_occupants(zone1)

                # Need to find a gate zone with 2 spots available to put the qubits, since
                # neither has been placed in the new config yet
                # Starting from and including their current zones, find the closest gate zone
                # that has two spots available

                num_spots_needed = 2

                if is_gate_zone0 and free_space_zone_0 >= num_spots_needed:
                    qubit_tracker.lock_qubit(qubit0, zone0, zone0)
                    qubit_tracker.lock_qubit(qubit1, zone1, zone0)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                if is_gate_zone1 and free_space_zone_1 >= num_spots_needed:
                    qubit_tracker.lock_qubit(qubit0, zone0, zone1)
                    qubit_tracker.lock_qubit(qubit1, zone1, zone1)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                # try to find the closest gate zone with two available spots
                closest_zone = find_best_gate_zone_to_move_to(
                    self._arch, self._macro_arch, [zone0, zone1], qubit_tracker
                )
                if closest_zone is not None:
                    qubit_tracker.lock_qubit(qubit0, zone0, closest_zone)
                    qubit_tracker.lock_qubit(qubit1, zone1, closest_zone)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                must_wait_qubits.extend([qubit0, qubit1])
                # If made it here:
                # No longer any gate zones left with two spots available
                # A gate with one locked qubit in a zone with one space
                # left could still be implemented, so continue loop


def handle_only_single_qubits_remaining(
    remaining_commands: list[Command],
    qubit_tracker: QubitTracker,
    arch: MultiZoneArchitectureSpec,
    macro_arch: MultiZoneArch,
) -> None:
    # locked qubits have already been assigned a zone in the new config and should therefore not move again
    locked_qubits = []
    for cmd in remaining_commands:
        if cmd.op.type in [OpType.Barrier]:
            continue
        qubit0 = cmd.args[0].index[0]
        is_locked_0 = qubit0 in locked_qubits
        if is_locked_0:
            continue
        zone0 = qubit_tracker.current_zone(qubit0)
        is_gate_zone0 = not arch.zones[zone0].memory_only
        free_space_zone_0 = arch.get_zone_max_ions_gates(
            zone0
        ) - qubit_tracker.n_zone_new_occupants(zone0)
        if is_gate_zone0 and free_space_zone_0 >= 1:
            qubit_tracker.lock_qubit(qubit0, zone0, zone0)
            locked_qubits.append(qubit0)
            continue

        # try to find the closest gate zone with two available spots
        closest_zone = find_best_gate_zone_to_move_to(
            arch, macro_arch, [zone0], qubit_tracker
        )
        if closest_zone is not None:
            qubit_tracker.lock_qubit(qubit0, zone0, closest_zone)
            locked_qubits.extend([qubit0])
            continue
        # if closest_zone is None, there are no more gate zone spots available
        break


def handle_must_wait(qubit0: int, qubit1: int, must_wait_qubits: list[int]) -> bool:
    must_wait_0 = qubit0 in must_wait_qubits
    must_wait_1 = qubit1 in must_wait_qubits
    if must_wait_0 or must_wait_1:
        if must_wait_0 and not must_wait_1:
            must_wait_qubits.append(qubit1)
        if must_wait_1 and not must_wait_0:
            must_wait_qubits.append(qubit0)
        return True
    return False


def handle_one_locked_qubit(
    locked_qubit: int,
    other_qubit: int,
    qubit_tracker: QubitTracker,
    arch: MultiZoneArchitectureSpec,
) -> bool:
    """Handle the case were one qubit is locked and one isn't

    Assumes other qubit is not in "must wait" qubits

    A qubit being locked implies it is in a gate zone. Move other qubit to its zone if
    space is available (return True). If not, must gate can't be implemented, so add other_qubit to must
    wait (return False).
    """
    lq_zone = qubit_tracker.current_zone(locked_qubit)
    free_space_lq_zone = arch.get_zone_max_ions_gates(
        lq_zone
    ) - qubit_tracker.n_zone_new_occupants(lq_zone)
    if free_space_lq_zone >= 1:
        oq_zone = qubit_tracker.current_zone(other_qubit)
        qubit_tracker.lock_qubit(other_qubit, oq_zone, lq_zone)
        return True
    return False


def find_best_gate_zone_to_move_to(
    arch: MultiZoneArchitectureSpec,
    macro_arch: MultiZoneArch,
    zones: list[int],
    qubit_tracker: QubitTracker,
) -> int | None:
    """Determines which gate zone to move to

    Assumes one qubit needs to move from each zone in the zones list to the gate zone
    The gate zone must have at least len(zones) places available

    Return gate zone id or None if no gate zone available with two spots
    """
    min_metric = arch.n_zones * 100  # this is strictly larger than the largest possible
    best_gate_zone = -1
    for gate_zone in macro_arch.gate_zones:
        free_space = arch.get_zone_max_ions_gates(
            gate_zone
        ) - qubit_tracker.n_zone_new_occupants(gate_zone)
        if free_space >= len(zones):
            metric = gate_zone_metric(macro_arch, gate_zone, zones)
            if metric < min_metric:
                min_metric = metric
                best_gate_zone = gate_zone
                continue
    if best_gate_zone == -1:
        return None
    return best_gate_zone


def gate_zone_metric(
    macro_arch: MultiZoneArch,
    gate_zone: int,
    zones: list[int],
) -> int:
    """Calculate metric estimating cost of moving one qubit from each of a
     list of zones to a specific gate zone

    Prefers gate zones with low total distance and high free space
    """
    distances = [len(macro_arch.shortest_path(gate_zone, zone)) for zone in zones]
    return sum(distances)


def move_qubits_to_closest_available_spots(
    arch: MultiZoneArchitectureSpec,
    macro_arch: MultiZoneArch,
    zone_qubits: list[int],
    starting_zone: int,
    qubit_tracker: QubitTracker,
) -> None:
    """Determines which a zone to move to

    Finds closest zone with a free spot available

    Assumes one qubit needs to move from each zone in the zones list to the gate zone
    The gate zone must have at least two places available

    Return gate zone id or None if no gate zone available with two spots
    """
    qubits = deepcopy(zone_qubits)
    for layer in bfs_layers(macro_arch.zone_graph, starting_zone):
        for zone in layer:
            if zone == starting_zone:
                continue
            zone_occupancy = qubit_tracker.n_zone_new_occupants(zone)
            max_zone_occupancy = arch.get_zone_max_ions_gates(zone)
            zone_free_space = max_zone_occupancy - zone_occupancy
            while zone_free_space > 0 and qubits:
                qubit_tracker.lock_qubit(qubits.pop(), starting_zone, zone)
                zone_free_space -= 1
        if not qubits:
            break


def handle_unused_qubits(
    arch: MultiZoneArchitectureSpec,
    macro_arch: MultiZoneArch,
    qubit_tracker: QubitTracker,
) -> None:
    # Take care of qubits that don't actually need to move (free space available in their old zone)
    for zone in range(arch.n_zones):
        zone_leftovers = qubit_tracker.old_zone_occupants(zone).copy()
        free_space_zone = arch.get_zone_max_ions_gates(
            zone
        ) - qubit_tracker.n_zone_new_occupants(zone)
        for qubit in zone_leftovers:
            if free_space_zone == 0:
                break
            qubit_tracker.lock_qubit(qubit, zone, zone)
            free_space_zone -= 1
    # Take care of qubits that need to move to a new zone (no more free space in their old zone)
    for zone in range(arch.n_zones):
        zone_leftovers = qubit_tracker.old_zone_occupants(zone)
        if zone_leftovers:
            move_qubits_to_closest_available_spots(
                arch, macro_arch, zone_leftovers, zone, qubit_tracker
            )
