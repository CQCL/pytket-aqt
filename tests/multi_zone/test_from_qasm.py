import time

import sys
import pytket
import pytket.qasm

from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    racetrack,
#    six_zones_in_a_line_102,
)


def test_circuit(qasm_filename, graph_init, routing_alg) -> None:
    total_t = time.time()
    start = time.time()
    print("loading circuit :", qasm_filename)
    C = pytket.qasm.circuit_from_qasm(qasm_filename)
    assert pytket.passes.FlattenRelabelRegistersPass().apply(C)

    end = time.time()
    print("load time: ", end - start)

    print("depth uncompiled", C.depth_2q())
    start = time.time()
    # backend = AQTMultiZoneBackend(architecture=six_zones_in_a_line_102, #racetrack,
    backend = AQTMultiZoneBackend(architecture=racetrack,
                                  access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)

    if graph_init:
        initial_placement=None
    else:
        initial_placement = {}
        n_qubits = C.n_qubits
        n_zones = 28
        n_per_trap = 3
        n_alloc = 0
        # # Placement of L6 for n-qubits
        # initial_placement = {}
        # n_qubits =  C.n_qubits
        # n_zones = 6
        # n_per_trap = 15
        # n_alloc = 0
        for i in range(n_zones):
            initial_placement[i] = list(range(n_alloc, min(n_alloc+n_per_trap, n_qubits)))
            n_alloc = min(n_alloc+n_per_trap, n_qubits)

        print(initial_placement)


    mz_circuit = backend.compile_circuit_with_routing(C, optimisation_level=0,
                                                      initial_placement=initial_placement,
                                                      routing_alg=routing_alg,
                                                      )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    # print("commands: ", mz_circuit.pytket_circuit.get_commands())
    print("n_2qb_gates: ", mz_circuit.pytket_circuit.n_2qb_gates())
    print("n_1qb_gates: ", mz_circuit.pytket_circuit.n_1qb_gates())
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)

if __name__ == "__main__":
    qasm_filename = sys.argv[1]
    graph_init = bool(int(sys.argv[2]))
    routing_alg = sys.argv[3]
    test_circuit(qasm_filename, graph_init, routing_alg)

