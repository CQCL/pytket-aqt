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

from dataclasses import dataclass, field
from typing import NewType

from networkx import (  # type: ignore
    Graph,
    shortest_path,
)

from .architecture import MultiZoneArchitecture

QubitId = NewType("QubitId", int)
ZoneId = NewType("ZoneId", int)


@dataclass(frozen=True)
class MacroZoneConfig:
    max_occupancy: int
    min_occupancy: int


@dataclass
class MacroZoneData:
    qubits: set[QubitId]
    zone_config: MacroZoneConfig


@dataclass
class MultiZoneMacroArch:
    zones: Graph
    qubit_to_zone_map: dict[QubitId, ZoneId]
    shortest_paths: dict[tuple[ZoneId, ZoneId], list[ZoneId] | None] = field(
        default_factory=dict
    )

    def shortest_path(self, zone_1: ZoneId, zone_2: ZoneId) -> list[ZoneId] | None:
        path = self.shortest_paths.get((zone_1, zone_2))
        if path:
            return path
        path = shortest_path(self.zones, zone_1, zone_2)
        self.shortest_paths[(zone_1, zone_2)] = path
        return path


def empty_macro_arch_from_architecture(
    architecture: MultiZoneArchitecture,
) -> MultiZoneMacroArch:
    zones = Graph()
    for zone_id, zone in enumerate(architecture.zones):
        zone_type = architecture.zone_types[zone.zone_type_id]
        zone_data = MacroZoneData(
            qubits=set(),
            zone_config=MacroZoneConfig(
                max_occupancy=zone_type.max_ions, min_occupancy=zone_type.min_ions
            ),
        )
        zones.add_node(ZoneId(zone_id), zone_data=zone_data)
    for zone_id, zone in enumerate(architecture.zones):
        for connected_zone_id, _ in zone.connected_zones.items():
            zones.add_edge(ZoneId(zone_id), ZoneId(connected_zone_id))
    return MultiZoneMacroArch(zones=zones, qubit_to_zone_map={})
