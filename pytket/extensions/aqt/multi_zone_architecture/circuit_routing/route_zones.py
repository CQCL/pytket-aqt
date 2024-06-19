import multiprocessing
from copy import deepcopy
from random import randint
from typing import Optional, Sequence

import importlib_resources
import kahypar
import mtkahypar
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
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move two qubits into the same zone

    :param qubits: tuple of two qubits
    :param mz_circ: the MultiZoneCircuit
    :param current_qubit_to_zone: dictionary containing the current
     mapping of qubits to zones (may be altered)
    :param current_placement: dictionary the current mapping of zones
     to lists of qubits contained within them (may be altered)
    """

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


def kahypar_edge_translation(edges: list[Sequence[int]]) -> tuple[list[int], list[int]]:
    edges_kahypar = [node for edge in edges for node in edge]
    edge_indices_kahypar = [0]
    for edge in edges:
        edge_indices_kahypar.append(edge_indices_kahypar[-1] + len(edge))
    return edge_indices_kahypar, edges_kahypar


def _kahypar(
    circuit: Circuit,
    arch: MultiZoneArchitecture,
    initial_placement: ZonePlacement,
) -> None:
    context = kahypar.Context()
    package_path = importlib_resources.files("pytket.extensions.aqt")
    default_ini = (
        f"{package_path}/multi_zone_architecture/circuit_routing/cut_kKaHyPar_sea20.ini"
    )
    context.loadINIconfiguration(default_ini)
    num_zones = arch.n_zones

    context.setK(num_zones)
    block_weights = [arch.get_zone_max_ions(i) + 1 for i, _ in enumerate(arch.zones)]
    context.setCustomTargetBlockWeights(block_weights)
    context.setEpsilon(num_zones)

    edges: list[tuple[int, int]] = []
    edge_weights: list[int] = []

    for zone, qubits in initial_placement.items():
        edges.extend([(zone, qubit + num_zones) for qubit in qubits])
        edge_weights.extend([10000 for _ in qubits])

    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1:
            continue
        elif n_args == 2:
            if isinstance(cmd.args[0], Qubit) and isinstance(cmd.args[1], Qubit):
                q_node_0 = cmd.args[0].index[0] + num_zones
                q_node_1 = cmd.args[1].index[0] + num_zones
                weight = 1
                edges.append((q_node_0, q_node_1))
                edge_weights.append(weight)
        else:
            raise ZoneRoutingError("Circuit must be rebased to the AQT gate set")

    edge_indices_kahypar, edges_kahypar = kahypar_edge_translation(edges)

    num_fake_vertices = circuit.n_qubits
    num_vertices = num_zones + circuit.n_qubits + num_fake_vertices
    vertex_weights = [1 for _ in range(num_zones)] + [
        1 for _ in range(num_zones, num_vertices)
    ]
    num_edges = len(edges)

    hypergraph = kahypar.Hypergraph(
        num_vertices,
        num_edges,
        edge_indices_kahypar,
        edges_kahypar,
        num_zones,
        edge_weights,
        vertex_weights,
    )
    # for i in range(num_zones):
    #    hypergraph.fixNodeToBlock(i, i)

    kahypar.partition(hypergraph, context)

    block_assignments = {i: [] for i in range(num_zones)}
    for vertex in range(num_vertices):
        if vertex < num_zones:
            block_assignments[hypergraph.blockID(vertex)].append(f"Z{vertex}")
        elif vertex < num_zones + circuit.n_qubits:
            block_assignments[hypergraph.blockID(vertex)].append(
                f"q{vertex - num_zones}"
            )
        else:
            block_assignments[hypergraph.blockID(vertex)].append("X")

    print(block_assignments)
    print(initial_placement)
    return


def route_circuit(
    circuit: Circuit,
    arch: MultiZoneArchitecture,
    initial_placement: Optional[ZonePlacement] = None,
) -> MultiZoneCircuit:
    """
    Route a Circuit to a given MultiZoneArchitecture by adding
     physical operations where needed

    The Circuit provided cannot have more qubits than allowed by
     the architecture. If no initial placement of qubits into
    the architecture zones is provided, the qubits will be
     placed using an internal algorithm in a "balanced" way across
    the available zones.

    :param circuit: A pytket Circuit to be routed
    :param arch: MultiZoneArchitecture to route into
    :param initial_placement: An optional initial mapping of architecture
     zones to lists of qubits to use
    """
    n_qubits = circuit.n_qubits
    if not initial_placement:
        initial_placement = _calc_initial_placement(n_qubits, arch)
        initial_placement = _initial_placement_graph_partition_alg(circuit, arch)
    mz_circuit = MultiZoneCircuit(arch, initial_placement, n_qubits, circuit.n_bits)
    current_qubit_to_zone = {}
    for zone, qubit_list in initial_placement.items():
        for qubit in qubit_list:
            current_qubit_to_zone[qubit] = zone
    current_zone_to_qubits = deepcopy(initial_placement)

    _kahypar(circuit, arch, initial_placement)

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


def _initial_placement_graph_partition_alg(
    circuit: Circuit, arch: MultiZoneArchitecture
) -> ZonePlacement:
    n_qubits = circuit.n_qubits
    n_qubits_max = arch.n_qubits_max
    if n_qubits > n_qubits_max:
        raise ZoneRoutingError(
            f"Attempting to route circuit with {n_qubits}"
            f" qubits, but architecture only supports up to {n_qubits_max}"
        )
    depth_list: list[list[tuple[int, int]]] = []
    current_depth_per_qubit: list[int] = [0] * n_qubits
    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1:
            continue
        elif n_args == 2:
            if isinstance(cmd.args[0], Qubit) and isinstance(cmd.args[1], Qubit):
                qubit0 = cmd.args[0].index[0]
                qubit1 = cmd.args[1].index[0]
                depth = max(
                    current_depth_per_qubit[qubit0], current_depth_per_qubit[qubit1]
                )
                assert len(depth_list) >= depth
                if len(depth_list) > depth:
                    depth_list[depth].append((qubit0, qubit1))
                else:
                    depth_list.append([(qubit0, qubit1)])
                current_depth_per_qubit[qubit0] = depth + 1
                current_depth_per_qubit[qubit1] = depth + 1

    num_zones = arch.n_zones
    arch_node_weights = [1] * num_zones
    arch_node_weights[0] = 5
    arch_node_weights[1] = 5
    arch_edges = []
    arch_edge_weights = []
    for i, zone in enumerate(arch.zones):
        for connected_zone in zone.connected_zones.keys():
            if (i, connected_zone) not in arch_edges and (
                connected_zone,
                i,
            ) not in arch_edges:
                arch_edges.append((i, connected_zone))
                arch_edge_weights.append(1)

    arch_graph = mtkahypar.Graph(
        num_zones, len(arch_edges), arch_edges, arch_node_weights, arch_edge_weights
    )

    mtkahypar.initializeThreadPool(multiprocessing.cpu_count())
    context = mtkahypar.Context()
    context.loadPreset(mtkahypar.PresetType.DEFAULT)
    context.setPartitioningParameters(
        num_zones,
        0.1,
        mtkahypar.Objective.CUT,
    )
    mtkahypar.setSeed(randint(0, 99))

    block_weights = [arch.get_zone_max_ions(i) for i, _ in enumerate(arch.zones)]
    num_spots = sum([m - 1 for m in block_weights])

    edges: list[tuple[int, int]] = []
    edge_weights: list[int] = []

    max_depth = len(depth_list)
    for i, pairs in enumerate(depth_list):
        edges.extend(pairs)
        weight = max_depth - i
        edge_weights.extend([weight] * len(pairs))

    num_vertices = num_spots
    vertex_weights = [1 for _ in range(num_vertices)]
    num_edges = len(edges)

    graph = mtkahypar.Graph(
        num_vertices,
        num_edges,
        edges,
        vertex_weights,
        edge_weights,
    )

    partioned_graph = graph.mapOntoGraph(arch_graph, context)

    initial_placement = {i: [] for i in range(num_zones)}
    block_assignments = {i: [] for i in range(num_zones)}
    for vertex in range(n_qubits):
        block_assignments[partioned_graph.blockID(vertex)].append(f"q{vertex}")
        initial_placement[partioned_graph.blockID(vertex)].append(vertex)
    for vertex in range(n_qubits, num_vertices):
        block_assignments[partioned_graph.blockID(vertex)].append(f"X")

    print(block_assignments)
    print(initial_placement)
    return initial_placement


def _calc_initial_placement(
    n_qubits: int, arch: MultiZoneArchitecture
) -> ZonePlacement:
    """
    Calculate an initial mapping of zones to qubit lists

    :param n_qubits: number of qubits to be placed
    :param arch: MultiZoneArchitecture to place into
    """
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
    # rearrange initial_zone_to_qubits so that qubit label integers are in numerical
    # order, and qubit label integers increase with increasing zone number
    # i.e., for two zones and four total qubits: [1,3]-[2,4] -> [1,2]-[3,4]
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
