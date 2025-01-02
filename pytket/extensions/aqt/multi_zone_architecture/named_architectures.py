# Copyright 2020-2024 Quantinuum
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

"""Pre-defined named multi-zone architectures for use in multi-zone circuits"""
from itertools import combinations

from .architecture import (
    MultiZoneArchitectureSpec,
    Operation,
    PortSpec,
    Zone,
    ZoneConnection,
)

standardOperations = [
    Operation(operation_spec="[X, t, [self, o, p]]", fidelity="0.993"),
    Operation(operation_spec="[MS, t, [[self, o, p], [self, o, p]]]", fidelity="0.983"),
]


four_zones_in_a_line = MultiZoneArchitectureSpec(
    n_qubits_max=16,
    n_zones=4,
    zones=[
        Zone(max_ions=mi, memory_only=mem)
        for mi, mem in [(6, False), (8, True), (8, False), (6, True)]
    ],
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i, port_id=1),
            zone_port_spec1=PortSpec(zone_id=i + 1, port_id=0),
        )
        for i in range(3)
    ],
)


racetrack = MultiZoneArchitectureSpec(
    n_qubits_max=84,
    n_zones=28,
    zones=[Zone(max_ions=6) for _ in range(28)],
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i % 28, port_id=1),
            zone_port_spec1=PortSpec(zone_id=(i + 1) % 28, port_id=0),
        )
        for i in range(28)
    ],
)


def get_all_to_all_port_connections(
    zone_ports: list[tuple[int, int]]
) -> list[ZoneConnection]:
    """Return a list of ZoneConnections connecting
    all the zone ports in the given list"""
    return [
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=zone_port0[0], port_id=zone_port0[1]),
            zone_port_spec1=PortSpec(zone_id=zone_port1[0], port_id=zone_port1[1]),
        )
        for zone_port0, zone_port1 in combinations(zone_ports, 2)
    ]


"""
grid12:

|- 0 -|- 1 -|
2     3     4
|- 5 -|- 6 -|
7     8     9
|- 10-|- 11-|

for horizontal zones port 0 is left, port 1 is right
for vertical zones port 0 is up, port 1 is down
"""
grid_zone_max_ion = 8
grid12 = MultiZoneArchitectureSpec(
    n_qubits_max=32,
    n_zones=12,
    zones=[Zone(max_ions=grid_zone_max_ion) for _ in range(12)],
    connections=get_all_to_all_port_connections([(0, 0), (2, 0)])
    + get_all_to_all_port_connections([(0, 1), (1, 0), (3, 0)])
    + get_all_to_all_port_connections([(1, 1), (4, 0)])
    + get_all_to_all_port_connections([(2, 1), (5, 0), (7, 0)])
    + get_all_to_all_port_connections([(3, 1), (6, 0), (5, 1), (8, 0)])
    + get_all_to_all_port_connections([(6, 1), (4, 1), (9, 0)])
    + get_all_to_all_port_connections([(7, 1), (10, 0)])
    + get_all_to_all_port_connections([(8, 1), (10, 1), (11, 0)])
    + get_all_to_all_port_connections([(9, 1), (11, 1)]),
)
