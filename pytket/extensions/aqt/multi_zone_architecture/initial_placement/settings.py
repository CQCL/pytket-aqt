from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..circuit_routing.route_zones import ZonePlacement


class InitialPlacementSettingsError(Exception):
    pass


class InitialPlacementAlg(Enum):
    qubit_order = 0
    manual = 1
    graph_partition = 2


MIN_ZONE_FREE_SPACE: Final = 2


@dataclass
class InitialPlacementSettings:
    algorithm: InitialPlacementAlg = InitialPlacementAlg.graph_partition
    zone_free_space: int = 2
    manual_placement: ZonePlacement | None = None

    def __post_init__(self):
        if self.zone_free_space < MIN_ZONE_FREE_SPACE:
            raise InitialPlacementSettingsError(
                f"{self.zone_free_space.__name__}"
                f" must be larger than {MIN_ZONE_FREE_SPACE}"
            )
        if self.algorithm == InitialPlacementAlg.manual and not self.manual_placement:
            raise InitialPlacementSettingsError(
                "Specified manual placement, but no manual placement " "provided"
            )

    @classmethod
    def default(cls) -> InitialPlacementSettings:
        return InitialPlacementSettings()
