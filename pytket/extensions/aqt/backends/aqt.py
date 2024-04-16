# Copyright 2020-2024 Cambridge Quantum Computing
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
import json
import time
from ast import literal_eval
from typing import Any, Iterator
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import httpx
from qiskit_aqt_provider import api_models
from requests import put

from pytket.backends import Backend
from pytket.backends import CircuitStatus
from pytket.backends import ResultHandle
from pytket.backends import StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendinfo import fully_connected_backendinfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit
from pytket.circuit import OpType
from pytket.circuit import Qubit
from pytket.passes import auto_rebase_pass
from pytket.passes import BasePass
from pytket.passes import DecomposeBoxes
from pytket.passes import EulerAngleReduction
from pytket.passes import FlattenRegisters
from pytket.passes import FullPeepholeOptimise
from pytket.passes import RenameQubitsPass
from pytket.passes import SequencePass
from pytket.passes import SimplifyInitial
from pytket.passes import SynthesiseTket
from pytket.predicates import GateSetPredicate
from pytket.predicates import MaxNQubitsPredicate
from pytket.predicates import NoClassicalControlPredicate
from pytket.predicates import NoFastFeedforwardPredicate
from pytket.predicates import NoMidMeasurePredicate
from pytket.predicates import NoSymbolsPredicate
from pytket.predicates import Predicate
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray

from .aqt_api import AqtApi

from ..extension_version import __extension_version__

AQT_PORTAL_URL = "https://arnica.aqt.euf/api/v1"

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"

# Hard-coded for now as there is no API to retrieve these.
# All devices are fully connected.
_AQT_MAX_QUBITS = 20

_GATE_SET = {
    OpType.Rz,
    OpType.PhasedX,
    OpType.XXPhase,
    OpType.Measure,
    OpType.Barrier,
}

AQTResult = Tuple[int, List[int]]  # (n_qubits, list of readouts)

# TODO add more
_STATUS_MAP = {
    "finished": StatusEnum.COMPLETED,
    "error": StatusEnum.ERROR,
    "queued": StatusEnum.QUEUED,
}


class AqtAuthenticationError(Exception):
    """Raised when there is no AQT access token available."""

    def __init__(self) -> None:
        super().__init__("No AQT access token provided or found in config file.")


class AQTBackend(Backend):
    """
    Interface to an AQT device or simulator.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        aqt_workspace_id: str = "default",
        aqt_resource_id: str = "offline_simulator_no_noise",
        access_token: Optional[str] = None,
        label: str = "",
    ):
        """
        Construct a new AQT backend.

        Requires a valid API key/access token, this can either be provided as a
        parameter or set in config using :py:meth:`pytket.extensions.aqt.set_aqt_config`

        :param      aqt_workspace_id:  the aqt workspace
        :type       aqt_workspace_id:  string
        :param      aqt_resource_id:  the aqt resource id
        :type       aqt_resource_id:  string
        :param      access_token: AQT access token, default None
        :type       access_token: string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        """
        super().__init__()
        self._aqt_workspace_id = aqt_workspace_id
        self._aqt_resource_id = aqt_resource_id
        self._portal_url = AQT_PORTAL_URL
        self._aqt_api = AqtApi(self._portal_url, access_token)
        self._label = label
        self._backend_info = fully_connected_backendinfo(
            type(self).__name__,
            aqt_resource_id,
            __extension_version__,
            _AQT_MAX_QUBITS,
            _GATE_SET,
            misc={
                "aqt_workspace_id": aqt_workspace_id,
                "aqt_resource_id": aqt_resource_id,
            },
        )
        self._qm = {
            Qubit(i): cast(Qubit, node)
            for i, node in enumerate(self._backend_info.nodes)
        }
        self._MACHINE_DEBUG = False

    def rebase_pass(self) -> BasePass:
        return _aqt_rebase()

    @property
    def _http_client(self) -> httpx.Client:
        """HTTP client for communicating with the AQT cloud service."""
        return api_models.http_client(
            base_url=self._portal_url, token=self._access_token
        )

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        return self._backend_info

    @classmethod
    def available_devices(
        cls, access_token: Optional[str] = None, **kwargs: Any
    ) -> List[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: none.
        """
        aqt_api = AqtApi(AQT_PORTAL_URL, access_token)
        aqt_devices = aqt_api.get_devices()
        return [
            fully_connected_backendinfo(
                cls.__name__,
                aqt_device.device_id,
                __extension_version__,
                _AQT_MAX_QUBITS,
                _GATE_SET,
                misc={
                    "aqt_workspace_id": aqt_device.workspace_id,
                    "aqt_resource_id": aqt_device.device_id,
                },
            )
            for aqt_device in aqt_devices
        ]

    @property
    def required_predicates(self) -> List[Predicate]:
        preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(_GATE_SET),
        ]
        if self._backend_info is not None:
            preds.append(MaxNQubitsPredicate(self._backend_info.n_nodes))
        return preds

    def default_compilation_pass(self, optimisation_level: int = 2) -> BasePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    DecomposeBoxes(),
                    self.rebase_pass(),
                ]
            )
        elif optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseTket(),
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    self.rebase_pass(),
                    EulerAngleReduction(OpType.Ry, OpType.Rx),
                ]
            )
        else:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    self.rebase_pass(),
                    EulerAngleReduction(OpType.Ry, OpType.Rx),
                ]
            )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, str, str)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported kwargs:
        - `postprocess`: apply end-of-circuit simplifications and classical
          postprocessing to improve fidelity of results (bool, default False)
        - `simplify_initial`: apply the pytket ``SimplifyInitial`` pass to improve
          fidelity of results assuming all qubits initialized to zero (bool, default
          False)
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = kwargs.get("postprocess", False)
        simplify_initial = kwargs.get("postprocess", False)

        handles = []
        for i, (c, n_shots) in enumerate(zip(circuits, n_shots_list)):
            if postprocess:
                c0, ppcirc = prepare_circuit(c, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = c, None
            if simplify_initial:
                SimplifyInitial(
                    allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                ).apply(c0)
            (aqt_circ, measures) = _translate_aqt(c0)
            if self._MACHINE_DEBUG:
                handles.append(
                    ResultHandle(
                        _DEBUG_HANDLE_PREFIX + str((c.n_qubits, n_shots)),
                        measures,
                        json.dumps(ppcirc_rep),
                    )
                )
            else:
                resp = put(
                    self._url,
                    data={
                        "data": json.dumps(aqt_circ),
                        "repetitions": n_shots,
                        "no_qubits": c.n_qubits,
                        "label": c.name if c.name else f"{self._label}_{i}",
                    },
                    headers=self._header,
                ).json()
                if "status" not in resp:
                    raise RuntimeError(resp["message"])
                if resp["status"] == "error":
                    raise RuntimeError(resp["ERROR"])
                handles.append(
                    ResultHandle(resp["id"], measures, json.dumps(ppcirc_rep))
                )
        for handle in handles:
            self._cache[handle] = dict()
        return handles

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: Dict[str, BackendResult]
    ) -> None:
        if handle in self._cache:
            self._cache[handle].update(result_dict)
        else:
            self._cache[handle] = result_dict

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = handle[0]
        message = ""
        measure_permutations = json.loads(handle[1])  # type: ignore
        ppcirc_rep = json.loads(cast(str, handle[2]))
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        if self._MACHINE_DEBUG:
            n_qubits, n_shots = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])  # type: ignore
            empty_ar = OutcomeArray.from_ints([0] * n_shots, n_qubits, big_endian=True)
            self._update_cache_result(
                handle, {"result": BackendResult(shots=empty_ar, ppcirc=ppcirc)}
            )
            statenum = StatusEnum.COMPLETED
        else:
            data = put(self._url, data={"id": jobid}, headers=self._header).json()
            status = data["status"]
            if "ERROR" in data:
                message = data["ERROR"]
            statenum = _STATUS_MAP.get(status, StatusEnum.ERROR)
            if statenum is StatusEnum.COMPLETED:
                shots = OutcomeArray.from_ints(
                    data["samples"], data["no_qubits"], big_endian=True
                )
                shots = shots.choose_indices(measure_permutations)
                self._update_cache_result(
                    handle, {"result": BackendResult(shots=shots, ppcirc=ppcirc)}
                )
        return CircuitStatus(statenum, message)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            timeout = cast(float, kwargs.get("timeout"))
            wait = kwargs.get("wait", 1.0)
            # Wait for job to finish; result will then be in the cache.
            end_time = (time.time() + timeout) if (timeout is not None) else None
            while (end_time is None) or (time.time() < end_time):
                circuit_status = self.circuit_status(handle)
                if circuit_status.status is StatusEnum.COMPLETED:
                    return cast(BackendResult, self._cache[handle]["result"])
                if circuit_status.status is StatusEnum.ERROR:
                    raise RuntimeError(circuit_status.message)
                time.sleep(cast(float, wait))
            raise RuntimeError(f"Timed out: no results after {timeout} seconds.")


def _translate_aqt(circ: Circuit) -> api_models.Circuit:
    """Convert a circuit in the AQT gate set to AQT list representation,
    along with a JSON string describing the measure result permutations."""
    ops: list[api_models.OperationModel] = []
    num_measurements = 0
    measures: List = list()
    for cmd in circ.get_commands():
        op = cmd.op
        optype = op.type
        # https://arnica.aqt.eu/api/v1/docs
        if optype == OpType.Rz:
            ops.append(
                api_models.Operation.rz(
                    phi=op.params[0],
                    qubit=cmd.args[0].index[0],
                )
            )
        elif optype == OpType.PhasedX:
            ops.append(
                api_models.Operation.r(
                    theta=op.params[0],
                    phi=op.params[1],
                    qubit=cmd.args[0].index[0],
                )
            )
        elif optype == OpType.XXPhase:
            ops.append(
                api_models.Operation.rxx(
                    theta=op.params[0],
                    qubits=[cmd.args[0].index[0], cmd.args[1].index[0]],
                )
            )
        elif optype == OpType.Measure:
            # predicate has already checked format is correct, so
            # errors are not handled here
            num_measurements += 1
            qb_id = cmd.qubits[0].index[0]
            bit_id = cmd.bits[0].index[0]
            while len(measures) <= bit_id:
                measures.append(None)
            measures[bit_id] = qb_id
        else:
            if optype not in {OpType.noop, OpType.Barrier}:
                message = f"Gate {optype} is not in the allowed AQT gate set"
                raise ValueError(message)
    if num_measurements == 0:
        raise ValueError("Circuit must contain at least one measurement.")

    ops.append(api_models.Operation.measure())
    return api_models.Circuit(root=ops)


def _aqt_job_from_circuits(
    circuits: Sequence[Circuit],
    n_shots: Sequence[int] = None,
) -> api_models.JobSubmission:
    """Create AQT JobSubmission from a list of circuits
    and corresponding numbers of shots"""
    circ_shots: Iterator[Tuple[Circuit, int]] = zip(circuits, n_shots)
    return api_models.JobSubmission(
        job_type="quantum_circuit",
        label="pytket",
        payload=api_models.QuantumCircuits(
            circuits=[
                api_models.QuantumCircuit(
                    repetitions=shots,
                    quantum_circuit=_translate_aqt(circuit),
                    number_of_qubits=circuit.n_qubits,
                )
                for circuit, shots in circ_shots
            ]
        ),
    )


def _aqt_rebase() -> BasePass:
    return auto_rebase_pass({OpType.XXPhase, OpType.Rz, OpType.PhasedX})


_xcirc = Circuit(1).Rx(1, 0)
_xcirc.add_phase(0.5)
