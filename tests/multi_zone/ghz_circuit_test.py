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
def ghz_circuit() -> Circuit:
    circuit = Circuit(16)
    circuit.H(0)
    for i in range(circuit.n_qubits - 1):
        circuit.CX(i, i + 1)
    circuit.measure_all()
    return circuit


@pytest.fixture
def test_arch() -> MultiZoneArchitecture:
    return racetrack


def test_ghz_circuit(ghz_circuit: Circuit, test_arch: MultiZoneArchitecture) -> None:
    total_t = time.time()
    print("depth uncompiled", ghz_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    mz_circuit = backend.compile_circuit_with_routing(ghz_circuit, None, 0)
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
    start = time.time()
    mz_circuit.validate()
    end = time.time()
    print("validate time: ", end - start)


def test_ghz_circuit_hand_optimized_intitial_placement(
    ghz_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", ghz_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    inital = {
        0: [0, 1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
        4: [9, 10],
        5: [11, 12],
        6: [13, 14],
        7: [15],
    }
    for i in range(8, test_arch.n_zones):
        inital[i] = []
    mz_circuit = backend.compile_circuit_with_routing(ghz_circuit, inital, 0)
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
    start = time.time()
    mz_circuit.validate()
    end = time.time()
    print("validate time: ", end - start)


def test_ghz_circuit_greedy(
    ghz_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", ghz_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    inital = {
        0: [0, 1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
        4: [9, 10],
        5: [11, 12],
        6: [13, 14],
        7: [15],
    }
    for i in range(8, test_arch.n_zones):
        inital[i] = []
    mz_circuit = backend.compile_circuit_with_routing(ghz_circuit, inital, 0, "greedy")
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
    mz_circuit.validate()


def test_ghz_circuit_greedy_hand_optimized_intitial_placement(
    ghz_circuit: Circuit, test_arch: MultiZoneArchitecture
) -> None:
    total_t = time.time()
    print("depth uncompiled", ghz_circuit.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=test_arch, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    mz_circuit = backend.compile_circuit_with_routing(ghz_circuit, None, 0, "greedy")
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
    mz_circuit.validate()
