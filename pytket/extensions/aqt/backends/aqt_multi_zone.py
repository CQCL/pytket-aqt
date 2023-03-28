# Copyright 2020-2023 Cambridge Quantum Computing
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
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast, Any

from requests import put
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo, fully_connected_backendinfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.circuit import Circuit, Node, OpType, Qubit  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    SynthesiseTket,
    FullPeepholeOptimise,
    FlattenRegisters,
    RebaseCustom,
    EulerAngleReduction,
    DecomposeBoxes,
    RenameQubitsPass,
    SimplifyInitial,
    auto_rebase_pass,
)
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray

from .multi_zone_architecture.architecture import MultiZoneArchitecture
from .multi_zone_architecture.circuit.multizone_circuit import (
    MultiZoneCircuit,
    move_barrier_gate,
)
from .._metadata import __extension_version__
from .config import AQTConfig

AQT_URL_PREFIX = "https://gateway.aqt.eu/marmot/"

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"

_GATE_SET = {
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
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


class AQTMultiZoneBackend(Backend):
    """
    Interface to an AQT device with multiple compute zones.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        architecture: MultiZoneArchitecture,
        device_name: str = "multi_zone",
        access_token: Optional[str] = None,
        label: str = "",
    ):
        """
        Construct a new AQT backend.

        Requires a valid API key/access token, this can either be provided as a
        parameter or set in config using :py:meth:`pytket.extensions.aqt.set_aqt_config`

        :param      device_name:  device name (suffix of URL, e.g. "sim/noise-model-1")
        :type       device_name:  string
        :param      access_token: AQT access token, default None
        :type       access_token: string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        """
        super().__init__()
        self._url = AQT_URL_PREFIX + device_name
        self._label = label
        config = AQTConfig.from_default_config_file()

        if access_token is None:
            access_token = config.access_token
        if access_token is None:
            raise AqtAuthenticationError()

        self._header = {"Ocp-Apim-Subscription-Key": access_token, "SDK": "pytket"}
        self._backend_info: Optional[BackendInfo] = None
        self._qm: Dict[Qubit, Node] = {}
        self._backend_info = fully_connected_backendinfo(
            type(self).__name__,
            device_name,
            __extension_version__,
            architecture.n_qubits_max,
            _GATE_SET,
        )
        self._qm = {Qubit(i): node for i, node in enumerate(self._backend_info.nodes)}
        self._MACHINE_DEBUG = True

    def rebase_pass(self) -> BasePass:
        return _aqt_rebase()

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        return self._backend_info

    @classmethod
    def available_devices(cls, **kwargs: Any) -> List[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: none.
        """
        return []

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

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
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
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
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
                    SimplifyInitial(
                        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                    ),
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
        Supported kwargs: none.
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

        handles = []
        for i, (c, n_shots) in enumerate(zip(circuits, n_shots_list)):
            if postprocess:
                c0, ppcirc = prepare_circuit(c, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = c, None
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

    def get_compiled_circuit(
        self, circuit: MultiZoneCircuit, optimisation_level: int = 2
    ) -> MultiZoneCircuit:

        new_circuit = MultiZoneCircuit(
            circuit.architecture, circuit.n_qubits, circuit.n_bits
        )
        compiled_circuit = super().get_compiled_circuit(circuit)

        new_circuit.qubit_to_zones = deepcopy(circuit.qubit_to_zones)
        new_circuit.zone_to_qubits = deepcopy(circuit.zone_to_qubits)
        new_circuit.multi_zone_operations = deepcopy(circuit.multi_zone_operations)

        current_multiop_index_per_qubit: dict[int, int] = {
            k: 0 for k in new_circuit.multi_zone_operations.keys()
        }
        for cmd in compiled_circuit:
            op = cmd.op
            if op.type == OpType.Barrier:
                qubit = cmd.args[0].index[0]
                new_circuit.add_custom_gate(move_barrier_gate, [], [qubit])
                current_multiop_index = current_multiop_index_per_qubit[qubit]
                mult_op_bundles = new_circuit.multi_zone_operations[qubit]
                multi_ops = mult_op_bundles[current_multiop_index]
                for multi_op in multi_ops:
                    multi_op.append_to_circuit(new_circuit)
                current_multiop_index_per_qubit[qubit] = current_multiop_index + 1
            else:
                qubits = [q.index[0] for q in cmd.args]
                new_circuit.add_gate(cmd.op.type, op.params, qubits)

        return new_circuit


def _translate_aqt(circ: Circuit) -> Tuple[List[List], str]:
    """Convert a circuit in the AQT gate set to AQT list representation,
    along with a JSON string describing the measure result permutations."""
    gates: List = list()
    measures: List = list()
    for cmd in circ.get_commands():
        op = cmd.op
        optype = op.type
        # https://www.aqt.eu/aqt-gate-definitions/
        if optype == OpType.Rx:
            gates.append(["X", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.Ry:
            gates.append(["Y", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.XXPhase:
            gates.append(["MS", op.params[0], [q.index[0] for q in cmd.args]])
        elif optype == OpType.Measure:
            # predicate has already checked format is correct, so
            # errors are not handled here
            qb_id = cmd.qubits[0].index[0]
            bit_id = cmd.bits[0].index[0]
            while len(measures) <= bit_id:
                measures.append(None)
            measures[bit_id] = qb_id
        else:
            assert optype in {OpType.noop, OpType.Barrier}
    if None in measures:
        raise IndexError("Bit index not written to by a measurement.")
    return (gates, json.dumps(measures))


def _aqt_rebase() -> RebaseCustom:
    return auto_rebase_pass({OpType.XXPhase, OpType.Rx, OpType.Ry})


_xcirc = Circuit(1).Rx(1, 0)
_xcirc.add_phase(0.5)
