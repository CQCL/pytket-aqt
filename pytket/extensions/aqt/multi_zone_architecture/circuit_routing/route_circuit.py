# Copyright Quantinuum
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
from pytket.circuit import Circuit

from ..architecture import MultiZoneArchitectureSpec
from ..circuit.helpers import ZonePlacement
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
    MissingMtKahyparInstallError,
)
from .greedy_routing import GreedyCircuitRouter
from .settings import RoutingAlg, RoutingSettings

if MT_KAHYPAR_INSTALLED:
    from .partition_routing import PartitionCircuitRouter


def route_circuit(
    settings: RoutingSettings,
    circuit: Circuit,
    arch: MultiZoneArchitectureSpec,
    initial_placement: ZonePlacement,
) -> MultiZoneCircuit:
    """
    Route a Circuit to a given MultiZoneArchitecture by adding
     physical operations where needed

    The Circuit provided cannot have more qubits than allowed by
     the architecture.

    :param settings: Settings used to Route Circuit
    :param circuit: A pytket Circuit to be routed
    :param arch: MultiZoneArchitecture to route into
    :param initial_placement: The initial mapping of architecture
     zones to lists of qubits
    """
    match settings.algorithm:
        case RoutingAlg.graph_partition:
            if MT_KAHYPAR_INSTALLED:
                return PartitionCircuitRouter(
                    circuit, arch, initial_placement, settings
                ).get_routed_circuit()
            raise MissingMtKahyparInstallError()  # noqa: RSE102
        case RoutingAlg.greedy:
            return GreedyCircuitRouter(
                circuit, arch, initial_placement, settings
            ).get_routed_circuit()
        case _:
            raise ValueError("Unknown routing algorithm")
