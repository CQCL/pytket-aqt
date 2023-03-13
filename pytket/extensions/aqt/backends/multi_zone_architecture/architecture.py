from enum import Enum
from typing import Union, Dict, List

from pydantic import BaseModel


class ConnectionType(str, Enum):
    RightToRight = "RightToRight"
    RightToLeft = "RightToLeft"
    LeftToRight = "LeftToRight"
    LeftToLeft = "LeftToLeft"


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
    connected_zones: Dict[str, int]


class MultiZoneArchitecture(BaseModel):
    n_qubits_max: int
    n_zones: int
    zone_types: List[ZoneType]
    zones: List[Zone]
