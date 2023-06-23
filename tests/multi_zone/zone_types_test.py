from pytket.extensions.aqt.multi_zone_architecture.architecture import (
    MultiZoneArchitecture,
)
from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    four_zones_in_a_line,
)

multizone = four_zones_in_a_line


def test_zone_types() -> None:
    assert MultiZoneArchitecture(**multizone.dict()) == multizone
