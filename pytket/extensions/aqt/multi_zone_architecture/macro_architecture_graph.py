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

from dataclasses import dataclass
from typing import cast

from networkx import (  # type: ignore
    Graph,
    shortest_path,
)

from .architecture import MultiZoneArchitectureSpec, PortId


@dataclass(frozen=True)
class MacroZoneConfig:
    max_occupancy: int


@dataclass
class MacroZoneData:
    qubits: set[int]
    zone_config: MacroZoneConfig


class MultiZoneArch:
    def __init__(self, spec: MultiZoneArchitectureSpec):
        self.zones = Graph()
        self.shortest_paths: dict[tuple[int, int], list[int] | None] = {}
        self.zone_connections: list[list[int]] = [[]] * spec.n_zones
        self.connection_ports: dict[tuple[int, int], tuple[PortId, PortId]] = {}
        self.memory_zones: list[int] = []
        self.gate_zones: list[int] = []

        for zone_id, zone in enumerate(spec.zones):
            zone_data = MacroZoneData(
                qubits=set(),
                zone_config=MacroZoneConfig(max_occupancy=zone.max_ions),
            )
            self.zones.add_node(int(zone_id), zone_data=zone_data)
            if zone.memory_only:
                self.memory_zones.append(zone_id)
            else:
                self.gate_zones.append(zone_id)
            self.has_memory_zones = len(self.memory_zones) > 0

        for connection in spec.connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id

            self.zone_connections[zone0].append(zone1)
            self.zone_connections[zone1].append(zone0)
            if (zone0, zone1) not in self.connection_ports:
                self.connection_ports[(zone0, zone1)] = (port0, port1)
                self.connection_ports[(zone1, zone0)] = (port1, port0)
            else:
                raise ValueError(
                    f"Two connections between zones {zone0} and {zone1}"
                    f" specified, but only 1 connection between two zones is allowed"
                )
            self.zones.add_edge(int(zone0), int(zone1))

    def shortest_path(self, zone_1: int, zone_2: int) -> list[int]:
        cached_path = self.shortest_paths.get((zone_1, zone_2))
        if cached_path:
            return cached_path
        path = cast("list[int]", shortest_path(self.zones, zone_1, zone_2))
        self.shortest_paths[(zone_1, zone_2)] = path
        self.shortest_paths[(zone_2, zone_1)] = path[::-1]
        return path

    def get_connected_ports(
        self, source_zone: int, target_zone: int
    ) -> tuple[PortId, PortId]:
        try:
            return self.connection_ports[(source_zone, target_zone)]
        except KeyError:
            raise ValueError(  # noqa: B904
                f"No connection exists between zones {source_zone} and {target_zone}"
            )


def empty_macro_arch_from_architecture(
    architecture: MultiZoneArchitectureSpec,
) -> MultiZoneArch:
    return MultiZoneArch(architecture)
