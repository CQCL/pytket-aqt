import pytest
from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    four_zones_in_a_line,
    grid7,
)

from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementSettings,
    InitialPlacementAlg,
)

from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.settings import (
    RoutingSettings,
    RoutingAlg,
)

from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)


@pytest.fixture
def ghz_circuit() -> Circuit:
    circuit = Circuit(16)
    circuit.H(0)
    for i in range(circuit.n_qubits - 1):
        circuit.CX(i, i + 1)
    circuit.measure_all()
    return circuit


manual_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.manual,
    manual_placement={
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7],
        2: [8, 9, 10, 11, 12],
        3: [13, 14, 15],
    },
)

graph_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.graph_partition, zone_free_space=2
)

ordered_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.qubit_order, zone_free_space=2
)

graph_routing = RoutingSettings(algorithm=RoutingAlg.graph_partition, debug_level=0)

greedy_routing = RoutingSettings(algorithm=RoutingAlg.greedy)


@pytest.mark.parametrize(
    "opt_level, initial_pl_settings, routing_settings",
    [
        (0, manual_placement, greedy_routing),
        (0, manual_placement, graph_routing),
        (0, ordered_placement, greedy_routing),
        (0, ordered_placement, graph_routing),
        (0, graph_placement, greedy_routing),
        (0, graph_placement, graph_routing),
    ],
)
def test_compilation_settings_linearch(
    opt_level: int,
    initial_pl_settings: InitialPlacementSettings,
    routing_settings: RoutingSettings,
    ghz_circuit: Circuit,
) -> None:
    backend = AQTMultiZoneBackend(
        architecture=four_zones_in_a_line, access_token="invalid"
    )
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=opt_level,
        initial_placement=initial_pl_settings,
        routing=routing_settings,
    )
    compiled = backend.compile_circuit_with_routing(ghz_circuit, compilation_settings)
    print("Shuttles: ", compiled.get_n_shuttles())


manual_placement_grid = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.manual,
    manual_placement={
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [11, 12, 13],
        3: [8, 9, 10],
        4: [14, 15],
        5: [6, 7],
    },
)


@pytest.mark.parametrize(
    "opt_level, initial_pl_settings, routing_settings",
    [
        (0, manual_placement_grid, greedy_routing),
        (0, manual_placement_grid, graph_routing),
        (0, ordered_placement, greedy_routing),
        (0, ordered_placement, graph_routing),
        (0, graph_placement, greedy_routing),
        (0, graph_placement, graph_routing),
    ],
)
def test_compilation_settings_gridarch(
    opt_level: int,
    initial_pl_settings: InitialPlacementSettings,
    routing_settings: RoutingSettings,
    ghz_circuit: Circuit,
) -> None:
    backend = AQTMultiZoneBackend(architecture=grid7, access_token="invalid")
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=opt_level,
        initial_placement=initial_pl_settings,
        routing=routing_settings,
    )
    compiled = backend.compile_circuit_with_routing(ghz_circuit, compilation_settings)
    print("Shuttles: ", compiled.get_n_shuttles())
