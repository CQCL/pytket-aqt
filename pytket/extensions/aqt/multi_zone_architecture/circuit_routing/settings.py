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
    routing_alg: RoutingAlg

    def __post_init__(self):
        if not isinstance(self.routing_alg, RoutingAlg):
            raise RoutingSettingsError(
                f"{self.routing_alg.__name__}" f" must be of type {RoutingAlg.__name__}"
            )

    @classmethod
    def default(cls) -> RoutingSettings:
        return RoutingSettings(routing_alg=RoutingAlg.graph_partition)
