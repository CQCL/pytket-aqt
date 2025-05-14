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


class RoutingSettingsError(Exception):
    pass


class RoutingAlg(Enum):
    greedy = 0
    graph_partition = 1


@dataclass
class RoutingSettings:
    algorithm: RoutingAlg = RoutingAlg.greedy
    max_depth: int = 50
    debug_level: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, RoutingAlg):
            raise RoutingSettingsError(
                f"{self.algorithm.__name__} must be of type {RoutingAlg.__name__}"
            )

    @classmethod
    def default(cls) -> RoutingSettings:
        return RoutingSettings()
