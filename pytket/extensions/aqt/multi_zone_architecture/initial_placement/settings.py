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
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
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
    max_depth: int = 200

    def __post_init__(self) -> None:
        if self.zone_free_space < MIN_ZONE_FREE_SPACE:
            raise InitialPlacementSettingsError(
                f"zone_free_space must be larger than {MIN_ZONE_FREE_SPACE}"
            )
        if self.algorithm == InitialPlacementAlg.manual and not self.manual_placement:
            raise InitialPlacementSettingsError(
                "Specified manual placement, but no manual placement provided"
            )

    @classmethod
    def default(cls) -> InitialPlacementSettings:
        return InitialPlacementSettings()
