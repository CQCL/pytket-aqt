from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RoutingSettingsError(Exception):
    pass


class RoutingAlg(Enum):
    greedy = 0
    graph_partition = 1


@dataclass
class RoutingSettings:
    algorithm: RoutingAlg = RoutingAlg.graph_partition

    def __post_init__(self):
        if not isinstance(self.algorithm, RoutingAlg):
            raise RoutingSettingsError(
                f"{self.algorithm.__name__}" f" must be of type {RoutingAlg.__name__}"
            )

    @classmethod
    def default(cls) -> RoutingSettings:
        return RoutingSettings()
