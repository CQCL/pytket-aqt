from .architecture import (
    MultiZoneArchitecture,
    ZoneType,
    ZoneConnection,
    ConnectionType,
    Operation,
    Zone,
)

standardOperations = [
    Operation(operation_spec="[X, t, [self, o, p]]", fidelity="0.993"),
    Operation(operation_spec="[MS, t, [[self, o, p], [self, o, p]]]", fidelity="0.983"),
]

four_in_a_line = MultiZoneArchitecture(
    n_qubits_max=16,
    n_zones=4,
    zone_types=[
        ZoneType(
            id=0,
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
            id=1,
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
            id=2,
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
        Zone(id=0, name="LeftEdge", zone_type_id=0, connected_zones={1: "RL"}),
        Zone(
            id=1, name="Interior1", zone_type_id=1, connected_zones={0: "LR", 2: "RL"}
        ),
        Zone(
            id=2, name="Interior2", zone_type_id=1, connected_zones={1: "LR", 3: "RL"}
        ),
        Zone(
            id=3,
            name="RightEdge",
            zone_type_id=2,
            connected_zones={
                2: "LR",
            },
        ),
    ],
)
