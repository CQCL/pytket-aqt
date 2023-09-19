from copy import deepcopy
from typing import Optional

from pytket import Qubit
from pytket.circuit import Circuit
from ..architecture import MultiZoneArchitecture
from ..circuit.multizone_circuit import MultiZoneCircuit

ZonePlacement = dict[int, list[int]]
QubitPlacement = dict[int, int]


class ZoneRoutingError(Exception):
    pass


def _make_necessary_moves(
    qubits: tuple[int, int],
    mz_circ: MultiZoneCircuit,
    current_qubit_to_zone: dict[int, int],
    current_placement: ZonePlacement,
) -> None:
    def _move_qubit(qubit_to_move: int, starting_zone: int, target_zone: int) -> None:
        mz_circ.move_qubit(qubit_to_move, target_zone, precompiled=True)
        current_placement[starting_zone].remove(qubit_to_move)
        current_placement[target_zone].append(qubit_to_move)
        current_qubit_to_zone[qubit_to_move] = target_zone

    qubit0 = qubits[0]
    qubit1 = qubits[1]

    zone0 = current_qubit_to_zone[qubit0]
    zone1 = current_qubit_to_zone[qubit1]
    if zone0 == zone1:
        return
    free_space_zone_0 = mz_circ.architecture.get_zone_max_ions(zone0) - len(
        current_placement[zone0]
    )
    free_space_zone_1 = mz_circ.architecture.get_zone_max_ions(zone1) - len(
        current_placement[zone1]
    )
    match (free_space_zone_0, free_space_zone_1):
        case (0, 0):
            raise ValueError("Should not allow two full registers")
        case (1, 1):
            # find first qubit in zone1 that isn't qubit1
            uninvolved_qubit = [
                qubit for qubit in current_placement[zone1] if qubit != qubits[1]
            ][0]
            # send it to zone0
            _move_qubit(uninvolved_qubit, zone1, zone0)
            # send qubit0 to zone1
            _move_qubit(qubits[0], zone0, zone1)
        case (a, b) if a < 0 or b < 0:
            raise ValueError("Should never be negative")
        case (free0, free1) if free0 >= free1:
            _move_qubit(qubits[1], zone1, zone0)
        case (_, _):
            _move_qubit(qubits[0], zone0, zone1)


def route_circuit(
    circuit: Circuit,
    arch: MultiZoneArchitecture,
    initial_placement: Optional[ZonePlacement] = None,
) -> MultiZoneCircuit:
    n_qubits = circuit.n_qubits
    if not initial_placement:
        initial_placement = _calc_initial_placement(n_qubits, arch)
    mz_circuit = MultiZoneCircuit(arch, initial_placement, n_qubits, circuit.n_bits)
    current_qubit_to_zone = {}
    for zone, qubit_list in initial_placement.items():
        for qubit in qubit_list:
            current_qubit_to_zone[qubit] = zone
    current_zone_to_qubits = deepcopy(initial_placement)

    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1:
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
        elif n_args == 2:
            if isinstance(cmd.args[0], Qubit) and isinstance(cmd.args[1], Qubit):
                _make_necessary_moves(
                    (cmd.args[0].index[0], cmd.args[1].index[0]),
                    mz_circuit,
                    current_qubit_to_zone,
                    current_zone_to_qubits,
                )
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
        else:
            raise ZoneRoutingError("Circuit must be rebased to the AQT gate set")
    return mz_circuit


def _calc_initial_placement(
    n_qubits: int, arch: MultiZoneArchitecture
) -> ZonePlacement:
    n_qubits_max = arch.n_qubits_max
    n_zones = arch.n_zones
    if n_qubits > n_qubits_max:
        raise ZoneRoutingError(
            f"Attempting to route circuit with {n_qubits}"
            f" qubits, but architecture only supports up to {n_qubits_max}"
        )
    initial_zone_to_qubits: ZonePlacement = {zone: [] for zone in range(n_zones)}
    current_zone = 0
    # place qubits equally across zones
    for q in range(n_qubits):
        if len(initial_zone_to_qubits[current_zone]) < arch.get_zone_max_ions(
            current_zone
        ):  # always leave at least one place empty in zone
            initial_zone_to_qubits[current_zone].append(q)
        if current_zone == n_zones - 1:
            current_zone = 0
        else:
            current_zone += 1
    # rearrange
    current_qubit = 0
    for zone in range(n_zones):
        for zone_position in range(len(initial_zone_to_qubits[zone])):
            initial_zone_to_qubits[zone][zone_position] = current_qubit
            current_qubit += 1
    assert (
        sum([len(zone_list) for zone_list in initial_zone_to_qubits.values()])
        == n_qubits
    )
    return initial_zone_to_qubits
