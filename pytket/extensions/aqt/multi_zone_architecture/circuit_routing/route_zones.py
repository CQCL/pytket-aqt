import math

# import multiprocessing
from copy import deepcopy
import time
from typing import Optional, Sequence

import importlib_resources
import kahypar
import mtkahypar
from pytket import Qubit
from pytket.circuit import Circuit
from ..architecture import MultiZoneArchitecture
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..macro_architecture_graph import empty_macro_arch_from_architecture, ZoneId

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


def _make_necessary_config_moves(
    configs: tuple[ZonePlacement, ZonePlacement],
    mz_circ: MultiZoneCircuit,
) -> None:
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move from one zone placement to another

    :param configs: tuple of two ZonePlacements [Old, New]
    :param mz_circ: the MultiZoneCircuit
    :param current_qubit_to_zone: dictionary containing the current
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


def _placement_generator(
    n_qubits: int,
    arch: MultiZoneArchitecture,
    initial_placement: ZonePlacement,
    depth_list: list[list[tuple[int, int]]],
) -> tuple[ZonePlacement, ZonePlacement]:
    # generates pairs of configs representing one shuttling step
    current_placement = deepcopy(initial_placement)
    depth_list = _get_updated_depth_list(n_qubits, initial_placement, depth_list)
    max_iter = len(depth_list)
    iteration = 0
    while depth_list:
        printy = False
        new_placement = _new_placement_graph_partition_alg(
            n_qubits, arch, depth_list, current_placement, printy
        )
        depth_list = _get_updated_depth_list(n_qubits, new_placement, depth_list)
        yield current_placement, new_placement
        current_placement = new_placement
        if iteration > max_iter:
            raise Exception("placement alg is not converging")
        iteration += 1


def route_circuit(
    circuit: Circuit,
    arch: MultiZoneArchitecture,
    initial_placement: Optional[ZonePlacement] = None,
    routing_alg: str = "partitioning",
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
    :param routing_alg: routing algorithm "partitioning" or "greedy"
     zones to lists of qubits to use
    """
    n_qubits = circuit.n_qubits
    gate_pairs = _get_2q_gate_pairs_from_circuit(circuit)
    depth_list = _get_depth_list(n_qubits, gate_pairs)
    if not initial_placement:
        initial_placement = _initial_placement_graph_partition_alg(
            n_qubits, arch, depth_list
        )

    mz_circuit = MultiZoneCircuit(arch, initial_placement, n_qubits, circuit.n_bits)

    start = time.time()

    if routing_alg == "partitioning":
        commands = circuit.get_commands().copy()
        for old_place, new_place in _placement_generator(
            n_qubits, arch, initial_placement, depth_list
        ):
            for zone in old_place.keys():
                print("zone ", zone, ": ", old_place[zone], new_place[zone])
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

    else:
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

    end = time.time()
    print("routing time: ", end - start)
    return mz_circuit


def _get_qubit_to_zone(n_qubits: int, placement: ZonePlacement) -> list[int]:
    qubit_to_zone: list[int] = [-1] * n_qubits
    for zone, qubits in placement.items():
        for qubit in qubits:
            qubit_to_zone[qubit] = zone
    return qubit_to_zone


def _get_updated_depth_list(
    n_qubits: int, placement: ZonePlacement, depth_list: list[list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    pruned_depth_list = [depth.copy() for depth in depth_list]
    qubit_to_zone = _get_qubit_to_zone(n_qubits, placement)
    # prune current depth list
    prune_stage = False
    prune_touched = set()
    for i, depth in enumerate(depth_list):
        for qubit_pair in depth:
            if qubit_to_zone[qubit_pair[0]] == qubit_to_zone[qubit_pair[1]]:
                if not prune_stage:
                    pruned_depth_list[i].remove(qubit_pair)
                elif (
                    qubit_pair[0] not in prune_touched
                    and qubit_pair[1] not in prune_touched
                ):
                    pruned_depth_list[i].remove(qubit_pair)
                else:
                    prune_touched.update({qubit_pair[0], qubit_pair[1]})
            else:
                prune_touched.update({qubit_pair[0], qubit_pair[1]})
        if pruned_depth_list[i]:
            prune_stage = True
        if len(prune_touched) >= n_qubits - 1:
            break
    # flatten depth list
    flattened_depth_list = [pair for depth in pruned_depth_list for pair in depth]
    # new depth list
    new_depth_list = _get_depth_list(n_qubits, flattened_depth_list)
    return new_depth_list


def _get_2q_gate_pairs_from_circuit(circuit: Circuit) -> list[tuple[int, int]]:
    pair_list: list[tuple[int, int]] = []
    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1:
            continue
        elif (
            n_args == 2
            and isinstance(cmd.args[0], Qubit)
            and isinstance(cmd.args[1], Qubit)
        ):
            qubit0 = cmd.args[0].index[0]
            qubit1 = cmd.args[1].index[0]
            pair_list.append((qubit0, qubit1))
    return pair_list


def _get_depth_list(
    n_qubits, gate_pairs: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    depth_list: list[list[tuple[int, int]]] = []
    current_depth_per_qubit: list[int] = [0] * n_qubits
    for pair in gate_pairs:
        qubit0 = pair[0]
        qubit1 = pair[1]
        depth = max(current_depth_per_qubit[qubit0], current_depth_per_qubit[qubit1])
        assert len(depth_list) >= depth
        if depth > 0:
            if (qubit0, qubit1) in depth_list[depth - 1] or (
                qubit1,
                qubit0,
            ) in depth_list[depth - 1]:
                depth_list[depth - 1].append((qubit0, qubit1))
                continue
        if len(depth_list) > depth:
            depth_list[depth].append((qubit0, qubit1))
        else:
            depth_list.append([(qubit0, qubit1)])
        current_depth_per_qubit[qubit0] = depth + 1
        current_depth_per_qubit[qubit1] = depth + 1
    return depth_list


def _initial_placement_graph_partition_alg(
    n_qubits: int,
    arch: MultiZoneArchitecture,
    depth_list: list[list[tuple[int, int]]],
) -> ZonePlacement:
    # n_threads = multiprocessing.cpu_count()
    mtkahypar.initializeThreadPool(1)
    n_qubits_max = arch.n_qubits_max
    if n_qubits > n_qubits_max:
        raise ZoneRoutingError(
            f"Attempting to route circuit with {n_qubits}"
            f" qubits, but architecture only supports up to {n_qubits_max}"
        )

    num_zones = arch.n_zones
    arch_node_weights = [1] * num_zones
    arch_edges = []
    arch_edge_weights = []
    for i, zone in enumerate(arch.zones):
        for connected_zone in zone.connected_zones.keys():
            if (i, connected_zone) not in arch_edges and (
                connected_zone,
                i,
            ) not in arch_edges:
                arch_edges.append((i, connected_zone))
                # TODO: Replace with connectivity cost
                arch_edge_weights.append(1)

    num_spots = sum([arch.get_zone_max_ions(i) - 1 for i, _ in enumerate(arch.zones)])
    avg_block_weight = num_spots / num_zones

    context = mtkahypar.Context()
    context.loadPreset(mtkahypar.PresetType.DEFAULT)
    context.setPartitioningParameters(
        num_zones,
        0.5 / avg_block_weight,
        mtkahypar.Objective.CUT,
    )
    context.logging = False
    mtkahypar.setSeed(10)

    arch_graph = mtkahypar.Graph(
        num_zones, len(arch_edges), arch_edges, arch_node_weights, arch_edge_weights
    )

    block_weights = [arch.get_zone_max_ions(i) - 1 for i, _ in enumerate(arch.zones)]
    num_spots = sum([m for m in block_weights])

    edges: list[tuple[int, int]] = []
    edge_weights: list[int] = []

    start = time.time()
    max_considered_depth = min(200, len(depth_list))
    max_weight = math.pow(2, 20)
    for i, pairs in enumerate(depth_list):
        if i > max_considered_depth:
            break
        # weight = math.ceil(math.exp(-3/avg_block_weight * i) * max_weight)
        weight = math.ceil(math.exp(-1 * i) * max_weight)
        for pair in pairs:
            if pair in edges:
                index = edges.index(pair)
                edge_weights[index] = edge_weights[index] + weight
            else:
                edges.append(pair)
                edge_weights.append(weight)

    end = time.time()
    # edge_weight_map = {edges[i]: edge_weights[i] for i in range(0, len(edges))}
    print("edges time: ", end - start)

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

    start = time.time()
    # partioned_graph = graph.mapOntoGraph(arch_graph, context)
    partioned_graph = graph.partition(context)
    end = time.time()
    print("mapping time: ", end - start)

    initial_placement = {i: [] for i in range(num_zones)}
    block_assignments = {i: [] for i in range(num_zones)}
    qubits_in_blocks: list[int] = [-1] * n_qubits
    for vertex in range(n_qubits):
        block_assignments[partioned_graph.blockID(vertex)].append(f"q{vertex}")
        initial_placement[partioned_graph.blockID(vertex)].append(vertex)
        qubits_in_blocks[vertex] = partioned_graph.blockID(vertex)
    for vertex in range(n_qubits, num_vertices):
        block_assignments[partioned_graph.blockID(vertex)].append(f"X")

    block_edges = []
    block_edge_weights = []
    for block in range(num_zones):
        for block2 in range(block + 1, num_zones):
            weight = 0
            for qubit in initial_placement[block]:
                for qubit2 in initial_placement[block2]:
                    if (qubit, qubit2) in edges:
                        index = edges.index((qubit, qubit2))
                        weight += edge_weights[index]
                    if (qubit2, qubit) in edges:
                        index = edges.index((qubit2, qubit))
                        weight += edge_weights[index]
            if weight > 0:
                block_edges.append((block, block2))
                block_edge_weights.append(weight)
    # block_edge_weight_map = {
    #     block_edges[i]: block_edge_weights[i] for i in range(0, len(block_edges))
    # }

    block_graph = mtkahypar.Graph(
        num_zones,
        len(block_edges),
        block_edges,
        [1] * num_zones,
        block_edge_weights,
    )

    sortgr = block_graph.mapOntoGraph(arch_graph, context)
    map_zone = set()
    for zone in range(num_zones):
        new_zone = sortgr.blockID(zone)
        if new_zone in map_zone:
            print("duplicate", zone, new_zone)
            raise Exception("Process mapping did not create a 1 to 1 map, thats bad!")
        map_zone.add(new_zone)

    initial_placement2 = {i: [] for i in range(num_zones)}
    mapping = {}
    for zone in range(num_zones):
        mapping[zone] = sortgr.blockID(zone)
        initial_placement2[sortgr.blockID(zone)] = initial_placement[zone]

    # print(block_assignments)
    # print(initial_placement)
    print(initial_placement2)
    all_qubits = []
    # check for duplicate qubits
    for zone, qubits in initial_placement2.items():
        for qubit in qubits:
            if qubit in all_qubits:
                raise ValueError("duplicate qubit")
            all_qubits.append(qubit)

    return initial_placement2


def _new_placement_graph_partition_alg(
    n_qubits: int,
    arch: MultiZoneArchitecture,
    depth_list: list[list[tuple[int, int]]],
    initial_placement: ZonePlacement,
    printy: bool = False,
) -> ZonePlacement:
    n_qubits_max = arch.n_qubits_max
    if n_qubits > n_qubits_max:
        raise ZoneRoutingError(
            f"Attempting to route circuit with {n_qubits}"
            f" qubits, but architecture only supports up to {n_qubits_max}"
        )

    num_zones = arch.n_zones

    num_spots = sum([arch.get_zone_max_ions(i) for i, _ in enumerate(arch.zones)])
    avg_block_weight = num_spots / num_zones

    mtkahypar.initializeThreadPool(1)
    context = mtkahypar.Context()
    context.loadPreset(mtkahypar.PresetType.DEFAULT)
    context.setPartitioningParameters(
        num_zones,
        0.5 / avg_block_weight,
        mtkahypar.Objective.CUT,
    )
    context.logging = False
    mtkahypar.setSeed(100)

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

    # add edges to ensure zones are separated
    # zone_zone_edges = [
    #    (zone1, zone2)
    #    for zone1 in range(n_qubits, n_qubits + num_zones)
    #    for zone2 in range(n_qubits, n_qubits + num_zones)
    #    if zone1 != zone2
    # ]
    # edges.extend(zone_zone_edges)
    # edge_weights.extend([-max_weight * 2] * len(zone_zone_edges))

    # add shuttling penalty (just distance between zones for now,
    # should later be dependent on shuttling cost)

    macro_arch = empty_macro_arch_from_architecture(arch)

    def shuttling_penalty(zone1: int, other_zone1: int):
        shortest_path = macro_arch.shortest_path(ZoneId(zone1), ZoneId(other_zone1))
        return len(shortest_path)

    # max_shuttle_weight = math.ceil(math.exp(-3/avg_block_weight * 5) * max_weight)
    max_shuttle_weight = math.ceil(max_weight - 10000)
    for zone, qubits in initial_placement.items():
        for other_zone in range(num_zones):
            weight = math.ceil(
                math.exp(-0.8 * (shuttling_penalty(zone, other_zone) + 4))
                * max_shuttle_weight
            )
            if weight < 1:
                continue
            edges.extend([(other_zone + n_qubits, qubit) for qubit in qubits])
            edge_weights.extend([weight for _ in qubits])

    if printy:
        print("numspots: ", num_spots)
        for i in range(len(edges)):
            print(
                "edges.emplace_back(",
                edges[i][0],
                ",",
                edges[i][1],
                ",",
                edge_weights[i],
                ");",
            )

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
    fixed_list = (
        [-1] * n_qubits
        + [zone for zone in range(num_zones)]
        + [-1] * (num_vertices - n_qubits - num_zones)
    )
    graph.addFixedVertices(fixed_list, num_zones)

    partioned_graph = graph.partition(context)

    new_placement = {i: [] for i in range(num_zones)}
    block_assignments = {i: [] for i in range(num_zones)}
    mapping = [-1] * num_zones
    for vertex in range(n_qubits, n_qubits + num_zones):
        block_assignments[partioned_graph.blockID(vertex)].append(f"Z{vertex-n_qubits}")
        mapping[partioned_graph.blockID(vertex)] = vertex - n_qubits
    for vertex in range(n_qubits):
        block_assignments[partioned_graph.blockID(vertex)].append(f"q{vertex}")
        new_placement[mapping[partioned_graph.blockID(vertex)]].append(vertex)
    for vertex in range(n_qubits + num_zones, num_vertices):
        block_assignments[partioned_graph.blockID(vertex)].append(f"X")

    # check for duplicate qubits
    all_qubits = []
    for zone, qubits in new_placement.items():
        for qubit in qubits:
            if qubit in all_qubits:
                raise ValueError("duplicate qubit")
            all_qubits.append(qubit)
    return new_placement


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
