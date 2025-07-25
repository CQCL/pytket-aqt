from ..circuit.helpers import ZonePlacement


def get_qubit_to_zone(n_qubits: int, placement: ZonePlacement) -> list[int]:
    qubit_to_zone: list[int] = [-1] * n_qubits
    for zone, qubits in placement.items():
        for qubit in qubits:
            qubit_to_zone[qubit] = zone
    return qubit_to_zone
