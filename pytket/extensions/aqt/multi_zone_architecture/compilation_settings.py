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

from dataclasses import dataclass, field
from typing import Final

from .circuit_routing.settings import RoutingSettings
from .initial_placement.settings import InitialPlacementSettings


class CompilationSettingsError(Exception):
    pass


PYTKET_OPTIMISATION_LEVELS: Final = [0, 1, 2, 3]


@dataclass
class CompilationSettings:
    pytket_optimisation_level: int = 2
    initial_placement: InitialPlacementSettings = field(
        default_factory=InitialPlacementSettings.default
    )
    routing: RoutingSettings = field(default_factory=RoutingSettings.default)

    def __post_init__(self) -> None:
        if self.pytket_optimisation_level not in PYTKET_OPTIMISATION_LEVELS:
            raise CompilationSettingsError(
                f"pytket_optimisation_level must be one of {PYTKET_OPTIMISATION_LEVELS}"
            )

    @classmethod
    def default(cls) -> CompilationSettings:
        return CompilationSettings()
