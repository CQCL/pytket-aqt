from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..circuit_routing.route_circuit import ZonePlacement


class InitialPlacementSettingsError(Exception):
    pass


class InitialPlacementAlg(Enum):
    qubit_order = 0
    manual = 1
    graph_partition = 2


MIN_ZONE_FREE_SPACE: Final = 1


@dataclass
class InitialPlacementSettings:
    algorithm: InitialPlacementAlg = InitialPlacementAlg.qubit_order
    zone_free_space: int = 2
    manual_placement: ZonePlacement | None = None
    n_threads: int = 1
    max_depth: int = 200

    def __post_init__(self) -> None:
        if self.zone_free_space < MIN_ZONE_FREE_SPACE:
            raise InitialPlacementSettingsError(
                f"zone_free_space must be larger than {MIN_ZONE_FREE_SPACE}"
            )
        if self.algorithm == InitialPlacementAlg.manual and not self.manual_placement:
            raise InitialPlacementSettingsError(
                "Specified manual placement, but no manual placement " "provided"
            )

    @classmethod
    def default(cls) -> InitialPlacementSettings:
        return InitialPlacementSettings()
