from pytket.extensions.aqt.backends.multi_zone_architecture.architecture import (
    MultiZoneArchitecture,
)

from pytket.extensions.aqt.backends.multi_zone_architecture.named_architectures import (
    four_in_a_line,
)

multizone = four_in_a_line


def test_zone_types() -> None:
    assert MultiZoneArchitecture(**multizone.dict()) == multizone
