from dataclasses import dataclass
from typing import NewType

from networkx import Graph  # type: ignore

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


def empty_macro_arch_from_backend(
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
        for _, connected_zone_id in zone.connected_zones.items():
            zones.add_edge(ZoneId(zone_id), ZoneId(connected_zone_id))
    return MultiZoneMacroArch(zones=zones, qubit_to_zone_map={})
