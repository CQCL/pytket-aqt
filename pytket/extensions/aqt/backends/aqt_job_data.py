import json
from dataclasses import dataclass
from typing import Optional, Sequence
from qiskit_aqt_provider import api_models

from pytket import Circuit
from pytket.backends import ResultHandle


@dataclass
class PytketAqtJobCircuitData:
    circuit: Circuit
    n_shots: int
    postprocess_json: str = json.dumps(None)
    aqt_circuit: Optional[api_models.Circuit] = None
    measures: Optional[str] = None
    handle: Optional[ResultHandle] = None


@dataclass
class PytketAqtJob:
    circuits_data: Sequence[PytketAqtJobCircuitData]
