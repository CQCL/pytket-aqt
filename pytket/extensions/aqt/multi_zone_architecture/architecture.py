from enum import Enum
from typing import Union, Dict, List

from pydantic import BaseModel


class EdgeType(str, Enum):
    Right = "Right"
    Left = "Left"


class ConnectionType(str, Enum):
    RightToRight = "RightToRight"
    RightToLeft = "RightToLeft"
    LeftToRight = "LeftToRight"
    LeftToLeft = "LeftToLeft"


def source_edge_type(connection_type: ConnectionType) -> EdgeType:
    if (
        connection_type == ConnectionType.RightToLeft
        or connection_type == ConnectionType.RightToRight
    ):
        return EdgeType.Right
    return EdgeType.Left


def target_edge_type(connection_type: ConnectionType) -> EdgeType:
    if (
        connection_type == ConnectionType.LeftToRight
        or connection_type == ConnectionType.RightToRight
    ):
        return EdgeType.Right
    return EdgeType.Left


class ZoneConnection(BaseModel):
    connection_type: ConnectionType
    max_transfer: int

    class Config:
        use_enum_values = True


class Operation(BaseModel):
    operation_spec: str
    fidelity: Union[float, str]


class ZoneType(BaseModel):
    id: int
    name: str
    max_ions: int
    min_ions: int
    zone_connections: Dict[str, ZoneConnection]
    operations: List[Operation]


class Zone(BaseModel):
    id: int
    name: str
    zone_type_id: int
    connected_zones: Dict[int, str]


class MultiZoneArchitecture(BaseModel):
    n_qubits_max: int
    n_zones: int
    zone_types: List[ZoneType]
    zones: List[Zone]

    def get_connection_type(
        self, zone_index_source: int, zone_index_target: int
    ) -> ConnectionType:
        source_zone = self.zones[zone_index_source]
        source_zone_type = self.zone_types[source_zone.zone_type_id]
        connection_name = source_zone.connected_zones[zone_index_target]
        return source_zone_type.zone_connections[connection_name].connection_type

    def get_zone_max_ions(self, zone_index: int) -> int:
        zone = self.zones[zone_index]
        zone_type = self.zone_types[zone.zone_type_id]
        return zone_type.max_ions
