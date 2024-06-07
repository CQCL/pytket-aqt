# Copyright 2020-2024 Quantinuum
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
import importlib_resources
import kahypar
import pytest
from pytket import Circuit
from pytket.backends import ResultHandle
from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.backends.aqt_multi_zone import (
    get_aqt_json_syntax_for_compiled_circuit,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit import (
    MultiZoneCircuit,
)
from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    four_zones_in_a_line,
)

from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.route_zones import (
    kahypar_edge_translation,
)


@pytest.fixture()
def backend() -> AQTMultiZoneBackend:
    return AQTMultiZoneBackend(
        architecture=four_zones_in_a_line, access_token="invalid"
    )


def test_not_implemented_functionality_throws(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
        2: [8, 9, 10, 11],
        3: [12, 13, 14, 15],
    }
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 16)
    with pytest.raises(NotImplementedError):
        backend.process_circuits([circuit])  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.process_circuit(circuit)  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.run_circuits([circuit])  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.run_circuit(circuit)  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.circuit_status(ResultHandle())
    with pytest.raises(NotImplementedError):
        backend.get_result(ResultHandle())
    with pytest.raises(NotImplementedError):
        backend.cancel(ResultHandle())


def test_valid_circuit_compiles(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    circuit = backend.compile_manually_routed_multi_zone_circuit(circuit)


def test_circuit_compiles(backend: AQTMultiZoneBackend) -> None:
    circuit = Circuit(8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    backend.compile_circuit_with_routing(circuit)


def test_invalid_circuit_does_not_compile(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    with pytest.raises(Exception):
        backend.compile_manually_routed_multi_zone_circuit(circuit)


def test_try_get_aqt_syntax_on_uncompiled_circuit_raises(
    backend: AQTMultiZoneBackend,
) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    with pytest.raises(Exception):
        get_aqt_json_syntax_for_compiled_circuit(circuit)


def test_compiled_circuit_has_correct_syntax(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    circuit = backend.compile_manually_routed_multi_zone_circuit(circuit)
    aqt_operation_list = get_aqt_json_syntax_for_compiled_circuit(circuit)

    initialized_zones: list[int] = []
    number_initialized_qubits: int = 0
    for i, operation in enumerate(aqt_operation_list):
        if i < 2:
            assert operation[0] == "INIT"
        else:
            assert operation[0] != "INIT"
        if operation[0] == "INIT":
            initialized_zones.append(operation[1][0])
            number_initialized_qubits += operation[1][1]
        elif operation[0] in ["X", "Y", "Z"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 1
            assert _is_valid_zop(operation[2][0], initialized_zones)
        elif operation[0] in ["MS"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_same_zone(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["SHUTTLE"]:
            assert len(operation) == 3
            assert isinstance(operation[1], int)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_different_zones(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["PSWAP"]:
            assert len(operation) == 2
            assert len(operation[1]) == 2
            assert _zop_addresses_in_same_zone(operation[1][0], operation[1][1])
            assert _is_valid_zop(operation[1][0], initialized_zones)
            assert _is_valid_zop(operation[1][1], initialized_zones)
        else:
            raise Exception(f"Detected invalid operation type: {operation[0]}")
    assert initialized_zones == [zone for zone in initial_placement]
    assert number_initialized_qubits == 8


def test_kahypar() -> None:
    context = kahypar.Context()
    package_path = importlib_resources.files("pytket.extensions.aqt")
    default_ini = (
        f"{package_path}/multi_zone_architecture/circuit_routing/cut_kKaHyPar_sea20.ini"
    )
    context.loadINIconfiguration(default_ini)

    num_nodes = 8

    edges = [(0, 1), (0, 2), (3, 4), (3, 5)]

    hyperedge_indices, hyperedges = kahypar_edge_translation(edges)

    node_weights = [1] * num_nodes
    edge_weights = [1] * len(edges)

    k = 2

    hypergraph = kahypar.Hypergraph(
        num_nodes,
        len(edges),
        hyperedge_indices,
        hyperedges,
        k,
        edge_weights,
        node_weights,
    )
    hypergraph.fixNodeToBlock(0, 0)
    hypergraph.fixNodeToBlock(3, 1)

    print("\n num_fixed", hypergraph.numFixedNodes())
    context.setK(k)
    context.setEpsilon(k)
    context.setCustomTargetBlockWeights([5, 5])

    kahypar.partition(hypergraph, context)

    block_assignments = {i: [] for i in range(k)}
    for vertex in range(num_nodes):
        block_assignments[hypergraph.blockID(vertex)].append(vertex)

    print(block_assignments)


def test_automatically_routed_circuit_has_correct_syntax(
    backend: AQTMultiZoneBackend,
) -> None:
    initial_placement = None  # {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = Circuit(8)
    (circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7))
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    mz_circuit = backend.compile_circuit_with_routing(circuit, initial_placement)

    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()

    aqt_operation_list = get_aqt_json_syntax_for_compiled_circuit(mz_circuit)

    initialized_zones: list[int] = []
    number_initialized_qubits: int = 0
    aqt_shuttles = 0
    aqt_pswaps = 0
    for i, operation in enumerate(aqt_operation_list):
        if i < 2:
            assert operation[0] == "INIT"
        else:
            assert operation[0] != "INIT"
        if operation[0] == "INIT":
            initialized_zones.append(operation[1][0])
            number_initialized_qubits += operation[1][1]
        elif operation[0] in ["X", "Y", "Z"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 1
            assert _is_valid_zop(operation[2][0], initialized_zones)
        elif operation[0] in ["MS"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_same_zone(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["SHUTTLE"]:
            assert len(operation) == 3
            assert isinstance(operation[1], int)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_different_zones(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
            aqt_shuttles += 1
        elif operation[0] in ["PSWAP"]:
            assert len(operation) == 2
            assert len(operation[1]) == 2
            assert _zop_addresses_in_same_zone(operation[1][0], operation[1][1])
            assert _is_valid_zop(operation[1][0], initialized_zones)
            assert _is_valid_zop(operation[1][1], initialized_zones)
            aqt_pswaps += 1
        else:
            raise Exception(f"Detected invalid operation type: {operation[0]}")
    assert n_pswaps == aqt_pswaps
    assert n_shuttles == aqt_shuttles
    assert initialized_zones == [zone for zone in initial_placement]
    assert number_initialized_qubits == 8


def _is_valid_zop(zop: list, zone_list: list[int]) -> bool:
    return all(
        [
            len(zop) == 3,
            zop[0] in zone_list,
            zop[1] > 0,
            zop[2] >= 0,
            zop[2] < zop[1],
        ]
    )


def _zop_addresses_in_same_zone(zop1: list, zop2: list) -> bool:
    result: bool = zop1[0] == zop2[0]
    return result


def _zop_addresses_in_different_zones(zop1: list, zop2: list) -> bool:
    result: bool = zop1[0] != zop2[0]
    return result
