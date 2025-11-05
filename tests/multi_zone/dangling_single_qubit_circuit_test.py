import pytest

from pytket import Circuit
from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.settings import (
    RoutingAlg,
    RoutingSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar import (
    MtKahyparConfig,
    configure_mtkahypar,
)
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
)
from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementAlg,
    InitialPlacementSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)


def build_test_circuit(n_qubits: int) -> Circuit:
    circ = Circuit(n_qubits + 1, name="QFT")
    for i in range(n_qubits):
        circ.H(i)
        for j in range(i + 1, n_qubits):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(n_qubits // 2):
        circ.SWAP(k, n_qubits - k - 1)
    circ.H(n_qubits)
    circ.S(n_qubits)
    circ.X(n_qubits)
    return circ


test_circ = build_test_circuit(15)

line_backend = AQTMultiZoneBackend(
    architecture=four_zones_in_a_line, access_token="invalid"
)

# This can be used to configure the number of threads used and random seed
# for mt-kahypar. It is not required (then default will be used) and can only
# be set once
configure_mtkahypar(MtKahyparConfig(n_threads=1, random_seed=13))
order_init = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.qubit_order,
    zone_free_space=2,
    max_depth=200,
)
graph_routing = RoutingSettings(
    algorithm=RoutingAlg.graph_partition,
)
graph_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=graph_routing,
)

greedy_routing = RoutingSettings(
    algorithm=RoutingAlg.greedy,
)
greedy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=greedy_routing,
)

qft_precompiled = line_backend.precompile_circuit(test_circ, graph_compilation_settings)

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)


@pytest.mark.parametrize(
    "compilation_settings, use_legacy",
    [
        pytest.param(greedy_compilation_settings, True),
        pytest.param(greedy_compilation_settings, False),
        pytest.param(graph_compilation_settings, False, marks=graph_skipif),
    ],
)
def test_circuit_with_dangling_single_qubit_gates(
    compilation_settings: CompilationSettings, use_legacy: bool
) -> None:
    line_backend.route_precompiled(
        qft_precompiled, compilation_settings, use_legacy_routing=use_legacy
    )
