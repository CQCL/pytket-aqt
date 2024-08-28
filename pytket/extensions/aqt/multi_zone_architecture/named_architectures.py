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
from .architecture import ConnectionType
from .architecture import MultiZoneArchitecture
from .architecture import Operation
from .architecture import Zone
from .architecture import ZoneConnection
from .architecture import ZoneType

standardOperations = [
    Operation(operation_spec="[X, t, [self, o, p]]", fidelity="0.993"),
    Operation(operation_spec="[MS, t, [[self, o, p], [self, o, p]]]", fidelity="0.983"),
]

four_zones_in_a_line = MultiZoneArchitecture(
    n_qubits_max=16,
    n_zones=4,
    zone_types=[
        ZoneType(
            name="LeftEdge",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                )
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                )
            ],
        ),
        ZoneType(
            name="Middle",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
        ZoneType(
            name="RightEdge",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
    ],
    zones=[
        Zone(name="LeftEdge", zone_type_id=0, connected_zones={1: "RL"}),
        Zone(name="Interior1", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}),
        Zone(name="Interior2", zone_type_id=1, connected_zones={1: "LR", 3: "RL"}),
        Zone(
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                2: "LR",
            },
        ),
    ],
)

four_zones_in_a_line_56 = MultiZoneArchitecture(
    n_qubits_max=56,
    n_zones=4,
    zone_types=[
        ZoneType(
            name="LeftEdge",
            max_ions=18,
            min_ions=0,
            zone_connections={
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                )
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                )
            ],
        ),
        ZoneType(
            name="Middle",
            max_ions=18,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
        ZoneType(
            name="RightEdge",
            max_ions=18,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
    ],
    zones=[
        Zone(name="LeftEdge", zone_type_id=0, connected_zones={1: "RL"}),
        Zone(name="Interior1", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}),
        Zone(name="Interior2", zone_type_id=1, connected_zones={1: "LR", 3: "RL"}),
        Zone(
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                2: "LR",
            },
        ),
    ],
)


four_zones_diamond_pattern = MultiZoneArchitecture(
    n_qubits_max=16,
    n_zones=4,
    zone_types=[
        ZoneType(
            name="LeftEdge",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
                "RL2": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                )
            ],
        ),
        ZoneType(
            name="Middle",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
        ZoneType(
            name="RightEdge",
            max_ions=6,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "LR2": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
    ],
    zones=[
        Zone(name="LeftEdge", zone_type_id=0, connected_zones={1: "RL", 3: "RL2"}),
        Zone(name="Interior1", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}),
        Zone(
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                1: "LR",
                3: "LR2",
            },
        ),
        Zone(name="Interior2", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}),
    ],
)

six_zones_in_a_line_102 = MultiZoneArchitecture(
    n_qubits_max=102,
    n_zones=6,
    zone_types=[
        ZoneType(
            name="LeftEdge",
            max_ions=17,
            min_ions=0,
            zone_connections={
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                )
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                )
            ],
        ),
        ZoneType(
            name="Middle",
            max_ions=17,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
        ZoneType(
            name="RightEdge",
            max_ions=17,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
    ],
    zones=[
        Zone(name="LeftEdge", zone_type_id=0, connected_zones={1: "RL"}),
        Zone(name="Interior1", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}),
        Zone(name="Interior2", zone_type_id=1, connected_zones={1: "LR", 3: "RL"}),
        Zone(name="Interior3", zone_type_id=1, connected_zones={2: "LR", 4: "RL"}),
        Zone(name="Interior4", zone_type_id=1, connected_zones={3: "LR", 5: "RL"}),
        Zone(
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                4: "LR",
            },
        ),
    ],
)



racetrack = MultiZoneArchitecture(
    n_qubits_max=84,
    n_zones=28,
    zone_types=[
        ZoneType(
            name="Middle",
            max_ions=4,
            min_ions=0,
            zone_connections={
                "LR": ZoneConnection(
                    connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
                "RL": ZoneConnection(
                    connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
            operations=standardOperations
            + [
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                    fidelity="0.999",
                ),
                Operation(
                    operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                    fidelity="0.999",
                ),
            ],
        ),
    ],
    zones=[
        Zone(name="Zone00", zone_type_id=0, connected_zones={27: "LR", 1: "RL"}),
        Zone(name="Zone01", zone_type_id=0, connected_zones={0: "LR", 2: "RL"}),
        Zone(name="Zone02", zone_type_id=0, connected_zones={1: "LR", 3: "RL"}),
        Zone(name="Zone03", zone_type_id=0, connected_zones={2: "LR", 4: "RL"}),
        Zone(name="Zone04", zone_type_id=0, connected_zones={3: "LR", 5: "RL"}),
        Zone(name="Zone05", zone_type_id=0, connected_zones={4: "LR", 6: "RL"}),
        Zone(name="Zone06", zone_type_id=0, connected_zones={5: "LR", 7: "RL"}),
        Zone(name="Zone07", zone_type_id=0, connected_zones={6: "LR", 8: "RL"}),
        Zone(name="Zone08", zone_type_id=0, connected_zones={7: "LR", 9: "RL"}),
        Zone(name="Zone09", zone_type_id=0, connected_zones={8: "LR", 10: "RL"}),
        Zone(name="Zone10", zone_type_id=0, connected_zones={9: "LR", 11: "RL"}),
        Zone(name="Zone11", zone_type_id=0, connected_zones={10: "LR", 12: "RL"}),
        Zone(name="Zone12", zone_type_id=0, connected_zones={11: "LR", 13: "RL"}),
        Zone(name="Zone13", zone_type_id=0, connected_zones={12: "LR", 14: "RL"}),
        Zone(name="Zone14", zone_type_id=0, connected_zones={13: "LR", 15: "RL"}),
        Zone(name="Zone15", zone_type_id=0, connected_zones={14: "LR", 16: "RL"}),
        Zone(name="Zone16", zone_type_id=0, connected_zones={15: "LR", 17: "RL"}),
        Zone(name="Zone17", zone_type_id=0, connected_zones={16: "LR", 18: "RL"}),
        Zone(name="Zone18", zone_type_id=0, connected_zones={17: "LR", 19: "RL"}),
        Zone(name="Zone19", zone_type_id=0, connected_zones={18: "LR", 20: "RL"}),
        Zone(name="Zone20", zone_type_id=0, connected_zones={19: "LR", 21: "RL"}),
        Zone(name="Zone21", zone_type_id=0, connected_zones={20: "LR", 22: "RL"}),
        Zone(name="Zone22", zone_type_id=0, connected_zones={21: "LR", 23: "RL"}),
        Zone(name="Zone23", zone_type_id=0, connected_zones={22: "LR", 24: "RL"}),
        Zone(name="Zone24", zone_type_id=0, connected_zones={23: "LR", 25: "RL"}),
        Zone(name="Zone25", zone_type_id=0, connected_zones={24: "LR", 26: "RL"}),
        Zone(name="Zone26", zone_type_id=0, connected_zones={25: "LR", 27: "RL"}),
        Zone(name="Zone27", zone_type_id=0, connected_zones={26: "LR", 0: "RL"}),
    ],
)

HorizontalGridZone = ZoneType(
        name="Horizontal",
        max_ions=4,
        min_ions=0,
        zone_connections={
            "LD": ZoneConnection(
                connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            "LR": ZoneConnection(
                connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            "LU": ZoneConnection(
                connection_type=ConnectionType.LeftToLeft, max_transfer=2
                ),
            "RD": ZoneConnection(
                connection_type=ConnectionType.RightToRight, max_transfer=2
                ),
            "RL": ZoneConnection(
                connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            "RU": ZoneConnection(
                connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
        operations=standardOperations
        + [
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [LD, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [LR, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [LU, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [RD, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [RL, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [RU, o, p]]]",
                fidelity="0.999",
                ),
            ],
        )

# We identify Up as Left, Down as Right.
VerticalGridZone = ZoneType(
        name="Vertical",
        max_ions=4,
        min_ions=0,
        zone_connections={
            "UR": ZoneConnection(
                connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            "UD": ZoneConnection(
                connection_type=ConnectionType.LeftToRight, max_transfer=2
                ),
            "UL": ZoneConnection(
                connection_type=ConnectionType.LeftToLeft, max_transfer=2
                ),
            "DR": ZoneConnection(
                connection_type=ConnectionType.RightToRight, max_transfer=2
                ),
            "DU": ZoneConnection(
                connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            "DL": ZoneConnection(
                connection_type=ConnectionType.RightToLeft, max_transfer=2
                ),
            },
        operations=standardOperations
        + [
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [UR, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [UD, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [UL, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [DR, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [DU, o, p]]]",
                fidelity="0.999",
                ),
            Operation(
                operation_spec="[SHUTTLE, n, [[self, o, p], [DL, o, p]]]",
                fidelity="0.999",
                ),
            ],
        )

X_grid = MultiZoneArchitecture(
        n_qubits_max=8,
        n_zones=4,
        zone_types=[
            HorizontalGridZone,
            VerticalGridZone,
            ],
        zones=[# Top
                Zone(name="Zone00", zone_type_id=1, connected_zones={1: "DR",
                                                                     3: "DU",
                                                                     2: "DL",
                                                                     }),
               # Left
                Zone(name="Zone01", zone_type_id=0, connected_zones={0: "RD",
                                                                     2: "RL",
                                                                     3: "RU",
                                                                     }),
               # Right
                Zone(name="Zone02", zone_type_id=0, connected_zones={0: "LD",
                                                                     1: "LR",
                                                                     3: "LU",
                                                                     }),
               # Bottom
                Zone(name="Zone03", zone_type_id=1, connected_zones={1: "UR",
                                                                     0: "UD",
                                                                     2: "UL",
                                                                     }),
                ],
        )

def gen_lattice(Lx, Ly):
    """
    Generate a Lx by Ly rectangular lattice.
    The nodes (zones) live on the edge of the lattice.
    The position of each vertex is given by a pair of integers.
    The position of the edge is defined as the average position
    between two vertices.
    For example:
        x - x
        |   |
        x - x
    The vertices have positions:
        [(0, 0), (1, 0), (0, 1), (1, 1)]
    The edges have positions:
        [(0.5, 0), (0, 0.5), (1, 0.5), (0.5, 1)]

    """
    pos_to_idx = {}
    idx = 0
    for y in range(Ly):
        for x in range(Lx-1):
            # print(x+0.5, y)
            pos_to_idx[(x+0.5, y)] = idx
            idx += 1

        if y != Ly -1:
            for x in range(Lx):
                # print(x, y+0.5)
                pos_to_idx[(x, y+0.5)] = idx
                idx += 1

    return pos_to_idx

def gen_plaquette_from_lattice(Lx, Ly, pos_to_idx):
    '''
    The lattice is described by the dictionary mapping
    the position to indices.
    '''
    plaquette_list = []
    for y in range(Ly-1):
        for x in range(Lx-1):
            # print("gen plaquette:", x, y)
            plaquette = [pos_to_idx[x+0.5, y],  # bottom
                         pos_to_idx[x, y+0.5],  # left
                         pos_to_idx[x+1, y+0.5],  # right
                         pos_to_idx[x+0.5, y+1],  # top
                         ]
            plaquette_list.append(plaquette)

    return plaquette_list

def gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx):
    connection_map = {}
    for plaquette in plaquette_list:
        bottom, left, right, up = plaquette

        bottom_map = connection_map.get(bottom, {})
        # print("bottom -- left:", bottom, left)
        bottom_map[left] = 'LD'
        # print("bottom -- right:", bottom, right)
        bottom_map[right] = 'RD'
        connection_map[bottom] = bottom_map

        left_map = connection_map.get(left, {})
        # print("left -- up:", left, up)
        left_map[up] = 'UL'
        # print("left -- bottom:", left, bottom)
        left_map[bottom] = 'DL'
        connection_map[left] = left_map

        right_map = connection_map.get(right, {})
        # print("right -- up", right, up)
        right_map[up] = 'UR'
        # print("right -- bottom", right, bottom)
        right_map[bottom] = 'DR'
        connection_map[right] = right_map

        up_map = connection_map.get(up, {})
        # print("up -- left:", up, left)
        up_map[left] = 'LU'
        # print("up -- right:", up, right)
        up_map[right] = 'RU'
        connection_map[up] = up_map

    # The straight connection
    for x in range(Lx-2):
        for y in range(Ly):
            left = pos_to_idx[x+0.5, y]
            right = pos_to_idx[x+1.5, y]

            left_map = connection_map.get(left, {})
            left_map[right] = 'RL'
            connection_map[left] = left_map

            right_map = connection_map.get(right, {})
            right_map[left] = 'LR'
            connection_map[right] = right_map

    for x in range(Lx):
        for y in range(Ly-2):
            bottom = pos_to_idx[x, y+0.5]
            up = pos_to_idx[x, y+1.5]

            bottom_map = connection_map.get(bottom, {})
            bottom_map[up] = 'UD'
            connection_map[bottom] = bottom_map

            up_map = connection_map.get(up, {})
            up_map[bottom] = 'DU'
            connection_map[up] = up_map

    return connection_map

# small_grid = (3, 3)
Lx = 3
Ly = 3
n_zones = (Lx-1) * Ly + Lx * (Ly-1)
n_qubits = n_zones * 2
pos_to_idx = gen_lattice(Lx, Ly)
plaquette_list = gen_plaquette_from_lattice(Lx, Ly, pos_to_idx)
connection_map = gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx)

zones = []
for y in range(Ly):
    for x in range(Lx-1):
        idx = pos_to_idx[x+0.5, y]
        zones.append(Zone(name="Zone%d" % idx,
                          zone_type_id=0,
                          connected_zones=connection_map[idx]))
    if y != Ly-1:
        for x in range(Lx):
            idx = pos_to_idx[x, y+0.5]
            zones.append(Zone(name="Zone%d" % idx,
                              zone_type_id=1,
                              connected_zones=connection_map[idx]))


small_grid = MultiZoneArchitecture(
        n_qubits_max=n_qubits,
        n_zones=n_zones,
        zone_types=[
            HorizontalGridZone,
            VerticalGridZone,
            ],
        zones=zones,
        )

# mid_grid = (5, 5)
Lx = 7
Ly = 7
n_zones = (Lx-1) * Ly + Lx * (Ly-1) # 84
n_qubits = n_zones * 2 # 168
pos_to_idx = gen_lattice(Lx, Ly)
plaquette_list = gen_plaquette_from_lattice(Lx, Ly, pos_to_idx)
connection_map = gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx)

zones = []
for y in range(Ly):
    for x in range(Lx-1):
        idx = pos_to_idx[x+0.5, y]
        zones.append(Zone(name="Zone%d" % idx,
                          zone_type_id=0,
                          connected_zones=connection_map[idx]))
    if y != Ly-1:
        for x in range(Lx):
            idx = pos_to_idx[x, y+0.5]
            zones.append(Zone(name="Zone%d" % idx,
                              zone_type_id=1,
                              connected_zones=connection_map[idx]))


mid_grid = MultiZoneArchitecture(
        n_qubits_max=n_qubits,
        n_zones=n_zones,
        zone_types=[
            HorizontalGridZone,
            VerticalGridZone,
            ],
        zones=zones,
        )


# large_grid = (11, 11)
Lx = 11
Ly = 11
n_zones = (Lx-1) * Ly + Lx * (Ly-1) # 220
n_qubits = n_zones * 2 # 440
pos_to_idx = gen_lattice(Lx, Ly)
plaquette_list = gen_plaquette_from_lattice(Lx, Ly, pos_to_idx)
connection_map = gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx)

zones = []
for y in range(Ly):
    for x in range(Lx-1):
        idx = pos_to_idx[x+0.5, y]
        zones.append(Zone(name="Zone%d" % idx,
                          zone_type_id=0,
                          connected_zones=connection_map[idx]))
    if y != Ly-1:
        for x in range(Lx):
            idx = pos_to_idx[x, y+0.5]
            zones.append(Zone(name="Zone%d" % idx,
                              zone_type_id=1,
                              connected_zones=connection_map[idx]))


large_grid = MultiZoneArchitecture(
        n_qubits_max=n_qubits,
        n_zones=n_zones,
        zone_types=[
            HorizontalGridZone,
            VerticalGridZone,
            ],
        zones=zones,
        )

