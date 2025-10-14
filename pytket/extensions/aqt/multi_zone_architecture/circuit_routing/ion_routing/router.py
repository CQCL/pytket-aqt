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
from typing import Protocol

from pytket.circuit import Command
from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit import (
    MZAOperation,
)


class ConfigSelector(Protocol):
    def next_config(
        self, current_placement: TrapConfiguration, remaining_circuit: list[Command]
    ) -> TrapConfiguration: ...


class Router(Protocol):
    def route_source_to_target_config(
        self, source: TrapConfiguration, target: TrapConfiguration
    ) -> list[MZAOperation]: ...
