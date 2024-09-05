import time
import matplotlib.pyplot as plt

import sys
import pytket
import pytket.qasm
import pytket.extensions.qiskit

from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import CompilationSettings
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.settings import RoutingSettings, RoutingAlg

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    racetrack,
    six_zones_in_a_line_102,
)

from mqt.bench import get_benchmark
def named_circuit(circuit_name, num_qubits) -> Circuit:
    qcirc = get_benchmark(benchmark_name=circuit_name, level='alg', circuit_size=num_qubits)
    # qcirc.draw(output="mpl")
    print(qcirc)
    pytket_circ = pytket.extensions.qiskit.qiskit_to_tk(qcirc, preserve_param_uuid=False)
    return pytket_circ

def print_circuit(circuit_name, num_qubits) -> None:
    total_t = time.time()
    start = time.time()
    C = named_circuit(circuit_name, num_qubits)
    C.flatten_registers()
    assert C.is_simple
    assert pytket.passes.RemoveBarriers().apply(C)

    backend = AQTMultiZoneBackend(architecture=racetrack,
                                  access_token="invalid")
    C = backend.get_compiled_circuit(C)
    qcirc = pytket.extensions.qiskit.tk_to_qiskit(C,)
    print(qcirc)

    from pytket.circuit.display import get_circuit_renderer
    circuit_renderer = get_circuit_renderer() # Instantiate a circuit renderer
    circuit_renderer.set_render_options(zx_style=True) # Configure render options
    circuit_renderer.condense_c_bits = False # You can also set the properties on the instance directly
    print("Render options:")
    print(circuit_renderer.get_render_options()) # View currently set render options

    circuit_renderer.min_height = "300px" # Change the display height
    circuit_renderer.render_circuit_jupyter(C) # Render interactive display
    plt.show()



if __name__ == "__main__":
    import sys
    print_circuit(sys.argv[1], int(sys.argv[2]))

