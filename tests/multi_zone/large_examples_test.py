import os

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
from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    grid12,
    grid12_mod,
    racetrack,
    racetrack_4_gatezones,
)

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)
skip_if_no_run_long_tests = pytest.mark.skipif(
    not os.getenv("PYTKET_AQT_RUN_LONG_TESTS"),
    reason="Long test is skipped if env var PYTKET_AQT_RUN_LONG_TESTS not set",
)


def quantum_advantage_circuit(Lx: int, Ly: int) -> Circuit:
    N = Lx * Ly
    Ntrot = 30
    dt = 0.2
    hx = 2
    J = 1

    def c(x, y):
        if (x - y) % N == Lx or (y - x) % N == Lx:
            return 1
        if x // Lx == y // Lx and x == y - (y % Lx) + (((y % Lx) + 1) % Lx):
            return 1
        if x // Lx == y // Lx and y == x - (x % Lx) + (((x % Lx) + 1) % Lx):
            return 1
        return 0

    coupling_list: list[tuple[int, int]] = []
    coupling_list.extend(
        [(i, j) for i in range(N) for j in range(i + 1, N) if c(i, j) == 1]
    )

    U = Circuit(N)

    for t in range(Ntrot):
        n_trotter_steps = t

        if n_trotter_steps == 0:
            for j in range(N):
                U.Rx(-2 * dt * hx / 2, j)

        for coupling in coupling_list:
            U.ZZPhase(-2 * dt * J, coupling[0], coupling[1])
        for j in range(N):
            U.Rx(-2 * dt * hx, j)

        if n_trotter_steps == Ntrot - 1:
            for j in range(N):
                U.Rx(2 * dt * hx / 2, j)
    return U


## 56 qubit quantum advantage circuit
advantage_circuit_56 = quantum_advantage_circuit(7, 8)

## 30 qubit quantum advantage circuit
advantage_circuit_30 = quantum_advantage_circuit(5, 6)

racetrack_backend = AQTMultiZoneBackend(architecture=racetrack, access_token="invalid")

racetrack_4_gatezones_backend = AQTMultiZoneBackend(
    architecture=racetrack_4_gatezones, access_token="invalid"
)

grid_backend = AQTMultiZoneBackend(architecture=grid12, access_token="invalid")

grid_mod_backend = AQTMultiZoneBackend(architecture=grid12_mod, access_token="invalid")

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
    ignore_swap_costs=False,
    debug_level=0,
)
graph_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=graph_routing,
)

greedy_routing = RoutingSettings(
    algorithm=RoutingAlg.greedy,
    debug_level=0,
)
greedy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=greedy_routing,
)


adv_precomp = racetrack_backend.precompile_circuit(
    advantage_circuit_56, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings, use_legacy",
    [
        pytest.param(
            greedy_compilation_settings, True, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            greedy_compilation_settings, False, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            graph_compilation_settings,
            False,
            marks=[graph_skipif, skip_if_no_run_long_tests],
        ),
    ],
)
def test_quantum_advantage_racetrack_all_gate_zone(
    compilation_settings: CompilationSettings, use_legacy: bool
):
    racetrack_backend.route_precompiled(
        adv_precomp, compilation_settings, use_legacy_routing=use_legacy
    )


adv_precomp2 = racetrack_4_gatezones_backend.precompile_circuit(
    advantage_circuit_56, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings, use_legacy",
    [
        pytest.param(
            greedy_compilation_settings, True, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            greedy_compilation_settings, False, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            graph_compilation_settings,
            False,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_racetrack_4_gate_zone(
    compilation_settings: CompilationSettings, use_legacy: bool
):
    racetrack_4_gatezones_backend.route_precompiled(
        adv_precomp2, compilation_settings, use_legacy_routing=use_legacy
    )


adv_precomp3 = grid_backend.precompile_circuit(
    advantage_circuit_30, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings, use_legacy",
    [
        pytest.param(
            greedy_compilation_settings, True, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            greedy_compilation_settings, False, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            graph_compilation_settings,
            False,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_grid_12_gate_zone(
    compilation_settings: CompilationSettings, use_legacy: bool
):
    grid_backend.route_precompiled(
        adv_precomp3, compilation_settings, use_legacy_routing=use_legacy
    )


adv_precomp4 = grid_mod_backend.precompile_circuit(
    advantage_circuit_30, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings, use_legacy",
    [
        pytest.param(
            greedy_compilation_settings, True, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            greedy_compilation_settings, False, marks=skip_if_no_run_long_tests
        ),
        pytest.param(
            graph_compilation_settings,
            False,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_grid_4_gate_zone(
    compilation_settings: CompilationSettings, use_legacy: bool
):
    grid_mod_backend.route_precompiled(
        adv_precomp4, compilation_settings, use_legacy_routing=use_legacy
    )
