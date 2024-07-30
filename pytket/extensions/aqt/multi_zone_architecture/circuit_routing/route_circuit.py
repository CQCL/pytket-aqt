from pytket.circuit import Circuit

from .greedy_routing import GreedyCircuitRouter
from .partition_routing import PartitionCircuitRouter
from .settings import RoutingSettings, RoutingAlg
from ..architecture import MultiZoneArchitecture
from ..circuit.helpers import ZonePlacement
from ..circuit.multizone_circuit import MultiZoneCircuit


def route_circuit(
    settings: RoutingSettings,
    circuit: Circuit,
    arch: MultiZoneArchitecture,
    initial_placement: ZonePlacement,
) -> MultiZoneCircuit:
    """
    Route a Circuit to a given MultiZoneArchitecture by adding
     physical operations where needed

    The Circuit provided cannot have more qubits than allowed by
     the architecture. If no initial placement of qubits into
    the architecture zones is provided, the qubits will be
     placed using an internal algorithm in a "balanced" way across
    the available zones.

    :param settings: Settings used to Route Circuit
    :param circuit: A pytket Circuit to be routed
    :param arch: MultiZoneArchitecture to route into
    :param initial_placement: The initial mapping of architecture
     zones to lists of qubits
    """
    match settings.algorithm:
        case RoutingAlg.graph_partition:
            return PartitionCircuitRouter(
                circuit, arch, initial_placement, settings
            ).get_routed_circuit()
        case RoutingAlg.greedy:
            return GreedyCircuitRouter(
                circuit, arch, initial_placement, settings
            ).get_routed_circuit()
        case _:
            raise ValueError("Unknown routing algorithm")
