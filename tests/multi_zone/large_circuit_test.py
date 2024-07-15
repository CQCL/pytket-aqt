import time

import pytest
from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    racetrack,
)

from pytket.extensions.aqt.multi_zone_architecture.architecture import (
    MultiZoneArchitecture,
)


@pytest.fixture
def large_circuit() -> Circuit:
    start = time.time()
    Lx = 7
    Ly = 8

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

    coupling_list = []
    for i in range(0, N):
        for j in range(i + 1, N):
            if c(i, j) == 1:
                coupling_list.append([i, j])

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
    end = time.time()
    print("prep time: ", end - start)
    return U


@pytest.fixture
def test_arch() -> MultiZoneArchitecture:
    return racetrack


def test_large_circuit(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    mz_circuit = backend.compile_circuit_with_routing(large_circuit, None, 0)
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)


def test_large_circuit2(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    initial_placement = {i: [] for i in range(test_arch.n_zones)}
    # "concentrated" initial placement
    max_ion_per_zone = test_arch.get_zone_max_ions(0) - 1
    for i in range(large_circuit.n_qubits):
        initial_placement[i // max_ion_per_zone].append(i)
    mz_circuit = backend.compile_circuit_with_routing(
        large_circuit, initial_placement, 0
    )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)


def test_large_circuit3(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    initial_placement = {i: [] for i in range(test_arch.n_zones)}
    # "spread out" initial placement
    for i in range(large_circuit.n_qubits):
        initial_placement[i % test_arch.n_zones].append(i)
    mz_circuit = backend.compile_circuit_with_routing(
        large_circuit, initial_placement, 0
    )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)


def test_large_circuit_greedy(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    mz_circuit = backend.compile_circuit_with_routing(large_circuit, None, 0, "greedy")
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)


def test_large_circuit_greedy2(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    initial_placement = {i: [] for i in range(test_arch.n_zones)}
    max_ion_per_zone = test_arch.get_zone_max_ions(0) - 1
    for i in range(large_circuit.n_qubits):
        initial_placement[i // max_ion_per_zone].append(i)
    mz_circuit = backend.compile_circuit_with_routing(
        large_circuit, initial_placement, 0, "greedy"
    )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)


def test_large_circuit_greedy3(
    large_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", large_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    initial_placement = {i: [] for i in range(test_arch.n_zones)}
    for i in range(large_circuit.n_qubits):
        initial_placement[i % test_arch.n_zones].append(i)
    mz_circuit = backend.compile_circuit_with_routing(
        large_circuit, initial_placement, 0, "greedy"
    )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
