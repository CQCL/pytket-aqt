from dataclasses import dataclass
from typing import Protocol

from pytket import Circuit
from ..circuit_routing.route_zones import ZonePlacement
from .settings import InitialPlacementSettings, InitialPlacementAlg

from ..architecture import MultiZoneArchitecture


class InitialPlacementError(Exception):
    pass


class InitialPlacementGenerator(Protocol):
    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitecture
    ) -> ZonePlacement: ...


@dataclass
class ManualInitialPlacement(InitialPlacementGenerator):
    placement: ZonePlacement

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitecture
    ) -> ZonePlacement:
        placed_qubits = []
        for zone, qubits in self.placement.items():
            placed_qubits.extend(qubits)
            if len(qubits) > arch.get_zone_max_ions(zone):
                raise InitialPlacementError(
                    f"Specified manual initial placement is faulty, zone {zone},"
                    f" can hold {arch.get_zone_max_ions(zone)} qubits, but"
                    f" {len(qubits)} were placed"
                )
        counts = [placed_qubits.count(i) for i in range(circuit.n_qubits)]
        duplicates = ".".join(
            [
                f"Qubit {i} placed {count} times"
                for i, count in enumerate(counts)
                if count > 1
            ]
        )
        if duplicates:
            raise InitialPlacementError(
                f"Duplicate placements detected in manual"
                f" initial placement. {duplicates}"
            )
        unplaced_qubits = {i for i in range(circuit.n_qubits)}.difference_update(
            placed_qubits
        )
        if unplaced_qubits:
            raise InitialPlacementError(
                f"Some qubits missing in manual initial placement."
                f" Missing qubits: {unplaced_qubits}"
            )
        for zone in range(arch.n_zones):
            if zone not in self.placement.keys():
                self.placement[zone] = []
        return self.placement


@dataclass
class QubitOrderInitialPlacement(InitialPlacementGenerator):
    zone_free_space: int

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitecture
    ) -> ZonePlacement:
        placement: ZonePlacement = {}
        i_start = 0
        for zone in range(arch.n_zones):
            places_avail = arch.get_zone_max_ions(zone) - self.zone_free_space
            i_end = min(i_start + places_avail, circuit.n_qubits)
            placement[zone] = [i for i in range(i_start, i_end)]
            i_start = i_end
        return placement


@dataclass
class GraphMapInitialPlacement(InitialPlacementGenerator):
    zone_free_space: int

    def initial_placement(
        self, circuit: Circuit, arch: MultiZoneArchitecture
    ) -> ZonePlacement:
        placement: ZonePlacement = {}
        i_start = 0
        for zone in range(arch.n_zones):
            places_avail = arch.get_zone_max_ions(zone) - self.zone_free_space
            i_end = min(i_start + places_avail, circuit.n_qubits)
            placement[zone] = [i for i in range(i_start, i_end)]
            i_start = i_end
        return placement


def get_initial_placement_generator(
    settings: InitialPlacementSettings,
) -> InitialPlacementGenerator:
    match settings.algorithm:
        case InitialPlacementAlg.graph_partition:
            return GraphMapInitialPlacement(zone_free_space=settings.zone_free_space)
        case InitialPlacementAlg.qubit_order:
            return QubitOrderInitialPlacement(zone_free_space=settings.zone_free_space)
        case InitialPlacementAlg.manual:
            assert settings.manual_placement is not None
            return ManualInitialPlacement(placement=settings.manual_placement)
