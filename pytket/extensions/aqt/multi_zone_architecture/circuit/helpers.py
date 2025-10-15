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

ZonePlacement = list[list[int]]


def get_qubit_to_zone(n_qubits: int, placement: ZonePlacement) -> list[int]:
    qubit_to_zone: list[int] = [-1] * n_qubits
    for zone, qubits in enumerate(placement):
        for qubit in qubits:
            qubit_to_zone[qubit] = zone
    return qubit_to_zone


class ZoneRoutingError(Exception):
    pass


@dataclass
class TrapConfiguration:
    n_qubits: int
    zone_placement: ZonePlacement
