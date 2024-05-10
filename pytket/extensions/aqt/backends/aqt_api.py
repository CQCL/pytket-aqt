import json
import uuid
from dataclasses import dataclass
from typing import Protocol

import httpx
from qiskit_aqt_provider import AQTProvider, api_models
from qiskit_aqt_provider.api_models_generated import (
    JobUser,
    RRFinished,
    ResultItem,
    JobResponseRRFinished,
)
from qiskit_aqt_provider.aqt_provider import OfflineSimulator
from qiskit_aqt_provider.aqt_resource import AQTResource, OfflineSimulatorResource

from .aqt_job_data import PytketAqtJob
from .config import AQTAccessToken


@dataclass
class AqtDevice:
    workspace_id: str
    resource_id: str
    description: str
    resource_type: str

    @classmethod
    def from_aqt_resource(cls, aqt_resource: AQTResource):
        return AqtDevice(
            workspace_id=aqt_resource.resource_id.workspace_id,
            resource_id=aqt_resource.resource_id.resource_id,
            resource_type=aqt_resource.resource_id.resource_type,
            description=aqt_resource.resource_id.resource_name,
        )


class AqtApi(Protocol):
    def get_devices(self) -> list[AqtDevice]: ...

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str: ...

    def get_job_result(self, job_id: str) -> api_models.JobResponse: ...

    def cancel_job(self, job_id: str): ...


class AqtRemoteApi(AqtApi):
    def __init__(self, base_url: str, access_token: str):
        self._base_url = base_url
        self._access_token = AQTAccessToken.resolve(access_token)

    @property
    def _http_client(self) -> httpx.Client:
        """HTTP client for communicating with the AQT cloud service."""
        return api_models.http_client(base_url=self._base_url, token=self._access_token)

    def get_devices(self) -> list[AqtDevice]:
        aqt_provider = AQTProvider(access_token=self._access_token)
        backend_table = aqt_provider.backends()
        return [
            AqtDevice.from_aqt_resource(aqt_resource)
            for aqt_resource in backend_table.backends
        ]

    def post_aqt_job(self, job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        aqt_job = _aqt_job_from_pytket_aqt_job(job)
        resp = self._http_client.post(
            f"/submit/{aqt_device.workspace_id}/{aqt_device.resource_id}",
            json=aqt_job.model_dump(),
        )
        resp.raise_for_status()
        return str(api_models.Response.model_validate(resp.json()).job.job_id)

    def get_job_result(self, job_id: str) -> api_models.JobResponse:
        resp = self._http_client.get(f"/result/{job_id}")
        resp.raise_for_status()
        return api_models.Response.model_validate(resp.json())

    def cancel_job(self, job_id: str):
        resp = self._http_client.delete(f"/jobs/{job_id}")
        resp.raise_for_status()


class AqtOfflineApi(AqtApi):
    def __init__(self, simulator: OfflineSimulator):
        self._aqt_provider = AQTProvider(access_token="offline")
        self._offline_sim = OfflineSimulatorResource(
            self._aqt_provider,
            resource_id=api_models.ResourceId(
                workspace_id="default",
                resource_id=simulator.id,
                resource_name=simulator.name,
                resource_type="offline_simulator",
            ),
            with_noise_model=simulator.noisy,
        )

    def get_devices(self) -> list[AqtDevice]:
        backend_table = self._aqt_provider.backends()
        return [
            AqtDevice.from_aqt_resource(aqt_resource)
            for aqt_resource in backend_table.backends
        ]

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        pass

    def get_job_result(self, job_id: str) -> api_models.JobResponse:
        pass

    def cancel_job(self, job_id: str):
        pass


class AqtMockApi(AqtApi):
    def __init__(self) -> None:
        self._jobs: dict[str, PytketAqtJob] = dict()

    def get_devices(self) -> list[AqtDevice]:
        aqt_provider = AQTProvider(access_token="offline")
        backend_table = aqt_provider.backends()
        return [
            AqtDevice.from_aqt_resource(aqt_resource)
            for aqt_resource in backend_table.backends
        ]

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = aqt_job
        return job_id

    def get_job_result(self, job_id: str) -> api_models.JobResponse:
        job = self._jobs[job_id]
        results: dict[str, list[list[ResultItem]]] = dict()
        for i, circ_spec in enumerate(job.circuits_data):
            circ_measure_permutations = json.loads(circ_spec.measures)
            results[str(i)] = [
                [ResultItem(0) for _ in circ_measure_permutations]
                for _ in range(circ_spec.n_shots)
            ]
        return JobResponseRRFinished(
            job=JobUser(job_id=job_id), response=RRFinished(result=results)
        )

    def cancel_job(self, job_id: str):
        pass


def _aqt_job_from_pytket_aqt_job(
    job: PytketAqtJob,
) -> api_models.JobSubmission:
    """Create AQT JobSubmission from a list of circuits
    and corresponding numbers of shots"""
    return api_models.JobSubmission(
        job_type="quantum_circuit",
        label="pytket",
        payload=api_models.QuantumCircuits(
            circuits=[
                api_models.QuantumCircuit(
                    repetitions=spec.n_shots,
                    quantum_circuit=spec.aqt_circuit,
                    number_of_qubits=spec.circuit.n_qubits,
                )
                for spec in job.circuits_data
            ]
        ),
    )
