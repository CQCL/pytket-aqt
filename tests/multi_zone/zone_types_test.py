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

from pytket.extensions.aqt.multi_zone_architecture.architecture import (
    MultiZoneArchitecture,
)
from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    four_zones_in_a_line,
)

multizone = four_zones_in_a_line


def test_zone_types() -> None:
    assert MultiZoneArchitecture(**multizone.model_dump()) == multizone
