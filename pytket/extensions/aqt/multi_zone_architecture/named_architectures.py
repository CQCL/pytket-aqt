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
        Zone(name="LeftEdge", zone_type_id=0, connected_zones={1: "RL", 2: "RL2"}),
        Zone(name="Interior1", zone_type_id=1, connected_zones={0: "LR", 3: "RL"}),
        Zone(name="Interior2", zone_type_id=1, connected_zones={0: "LR", 3: "RL"}),
        Zone(
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                1: "LR",
                2: "LR2",
            },
        ),
    ],
)
