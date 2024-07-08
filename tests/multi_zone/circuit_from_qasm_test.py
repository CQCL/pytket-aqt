import time

import sys
import pytket
import pytket.qasm

from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    racetrack,
    six_zones_in_a_line_102,
)


def test_large_circuit(qasm_filename) -> None:
    total_t = time.time()
    start = time.time()
    print("loading circuit :", qasm_filename)
    C = pytket.qasm.circuit_from_qasm(qasm_filename)
    pytket.passes.FlattenRelabelRegistersPass().apply(C)

    end = time.time()
    print("load time: ", end - start)

    print("depth uncompiled", C.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=six_zones_in_a_line_102, #racetrack,
                                  access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)

    # initial_placement = {}
    # for i in range(16):
    #     initial_placement[i] = [i*4, i*4+1, i*4+2, i*4+3,]

    # Placement of L6 for 64 qubits
    spread = False
    if spread:
        initial_placement = {0: list(range(11)),
                             1: list(range(11,22)),
                             2: list(range(22,33)),
                             3: list(range(33,44)),
                             4: list(range(44,55)),
                             5: list(range(55,64)),
                             }
    else:
        initial_placement = {0: list(range(15)),
                             1: list(range(15,30)),
                             2: list(range(30,45)),
                             3: list(range(45,60)),
                             4: list(range(60,64)),
                             5: []
                             }

    initial_placement=None

    mz_circuit = backend.compile_circuit_with_routing(C, optimisation_level=0, initial_placement=initial_placement)
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)

if __name__ == "__main__":
    qasm_filename = sys.argv[1]
    test_large_circuit(qasm_filename)

