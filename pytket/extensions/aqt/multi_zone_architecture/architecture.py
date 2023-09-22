# Copyright 2020-2023 Cambridge Quantum Computing
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

from enum import Enum
from typing import Dict
from typing import List
from typing import Union

from pydantic import ConfigDict, BaseModel


class EdgeType(str, Enum):
    """Type of given zone edge

    Each zone has two edges that support connections,
    with a linear arrangement of ions between these edges

    This enum class identifies an edge as being a "right" or "left" type.
    The use of the terms "right" and "left" is not significant, only that
    the two edges are distinct. The terms come from the picture of the zone
    being a linear arrangement of ions on a horizontal line, with shuttling
    capabilities from the left or right side.

    """

    Right = "Right"
    Left = "Left"


class ConnectionType(str, Enum):
    """Type of connection between zones

    This enum class is used to classify a connection as connecting
    a right or left edge of the "source" zone to the right or left edge of
    the "target" zone
    """

    RightToRight = "RightToRight"
    RightToLeft = "RightToLeft"
    LeftToRight = "LeftToRight"
    LeftToLeft = "LeftToLeft"


def source_edge_type(connection_type: ConnectionType) -> EdgeType:
    """Retrieves the "source" EdgeType from the ConnectionType"""
    if (
        connection_type == ConnectionType.RightToLeft
        or connection_type == ConnectionType.RightToRight
    ):
        return EdgeType.Right
    return EdgeType.Left


def target_edge_type(connection_type: ConnectionType) -> EdgeType:
    """Retrieves the "target" EdgeType from the ConnectionType"""
    if (
        connection_type == ConnectionType.LeftToRight
        or connection_type == ConnectionType.RightToRight
    ):
        return EdgeType.Right
    return EdgeType.Left


class ZoneConnection(BaseModel):
    """A connection between two zones

    The connection allows shuttling between the zones
    according to the connection type and transfer limit (max
    number of ions per shuttle)
    """

    connection_type: ConnectionType
    max_transfer: int
    model_config = ConfigDict(use_enum_values=True)


class Operation(BaseModel):
    """Describes an allowed operation and its associated fidelity

    Currently not used
    """

    operation_spec: str
    fidelity: Union[float, str]


class ZoneType(BaseModel):
    """A general zone type

    A zone type is a template that can be used to "instantiate" an
    actual zone.

    The connections are placeholders for connections to actual zones
    """

    id: int
    name: str
    max_ions: int
    min_ions: int
    zone_connections: Dict[str, ZoneConnection]
    operations: List[Operation]


class Zone(BaseModel):
    """Processor Zone within the architecture"""

    id: int
    name: str
    zone_type_id: int
    connected_zones: Dict[int, str]


class MultiZoneArchitecture(BaseModel):
    """Class that determines the entire Multi-Zone Architecture"""

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
