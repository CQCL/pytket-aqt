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
import os
from typing import Union

from pydantic import BaseModel


class PortSpec(BaseModel):
    """Describes Zones port (a.k.a. shuttling edge)

    The zone_id identifies the zone and the port_id identifies
    the port. port_id can be either 0 (the shuttling port of position 0)
    or 1 (the shuttling port of the current last position)

    """

    zone_id: int
    port_id: int


class ZoneConnection(BaseModel):
    """A connection between two zones

    The connection allows shuttling between the zones
    according to the connection type and transfer limit (max
    number of ions per shuttle)
    """

    zone_port_spec0: PortSpec
    zone_port_spec1: PortSpec


class Operation(BaseModel):
    """Describes an allowed operation and its associated fidelity

    Currently not used
    """

    operation_spec: str
    fidelity: Union[float, str]


class Zone(BaseModel):
    """Processor Zone within the architecture"""

    max_ions: int
    memory_only: bool = False


class MultiZoneArchitectureSpec(BaseModel):
    """Class that determines the entire Multi-Zone Architecture"""

    n_qubits_max: int
    n_zones: int
    zones: list[Zone]
    connections: list[ZoneConnection]

    def get_zone_max_ions(self, zone_index: int) -> int:
        zone = self.zones[zone_index]
        return zone.max_ions

    def __str__(self) -> str:
        arch_spec_lines = [
            f"Max number of qubits: {self.n_qubits_max}",
            f"Number of zones: {self.n_zones}",
            "",
        ]
        connections_per_zone_port = [[[], []]] * self.n_zones
        for connection in self.connections:
            zone_0 = connection.zone_port_spec0.zone_id
            zone_1 = connection.zone_port_spec1.zone_id
            port_0 = connection.zone_port_spec0.port_id
            port_1 = connection.zone_port_spec1.port_id
            connections_per_zone_port[zone_0][port_0].append((zone_1, port_1))
            connections_per_zone_port[zone_1][port_1].append((zone_0, port_0))

        for zone_id, zone in enumerate(self.zones):
            connections_port_0 = connections_per_zone_port[zone_id][0]
            connections_port_1 = connections_per_zone_port[zone_id][1]
            arch_spec_lines.extend(
                [
                    f"Zone {zone_id}:",
                    f"    Max qubits {zone.max_ions}",
                    "    Connections:",
                ]
            )
            arch_spec_lines.append(f"       Port 0: Zone {connections_port_0}")
            arch_spec_lines.append(f"       Port 1: Zone {connections_port_1}")
        return f"{os.linesep}".join(arch_spec_lines)
