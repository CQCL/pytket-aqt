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


racetrack = MultiZoneArchitecture(
    n_qubits_max=56,
    n_zones=28,
    zone_types=[
        ZoneType(
            name="Middle",
            max_ions=5,
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
