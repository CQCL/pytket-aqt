from dataclasses import dataclass
from uuid import UUID

import httpx
from qiskit_aqt_provider import AQTProvider, api_models
from qiskit_aqt_provider.aqt_resource import AQTResource

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


@dataclass
class AqtJob:
    id: UUID
    workspace_id: str
    device_id: str


class AqtApi:
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

    def post_aqt_job(
        self, aqt_job: api_models.JobSubmission, aqt_device: AqtDevice
    ) -> UUID:
        resp = self._http_client.post(
            f"/submit/{aqt_device.workspace_id}/{aqt_device.resource_id}",
            json=aqt_job.model_dump(),
        )
        resp.raise_for_status()
        return api_models.Response.model_validate(resp.json()).job.job_id

    def get_job_result(self, job_id: UUID) -> api_models.JobResponse:
        resp = self._http_client.get(f"/result/{job_id}")
        resp.raise_for_status()
        return api_models.Response.model_validate(resp.json())

    def cancel_job(self, job_id: UUID):
        resp = self._http_client.delete(f"/jobs/{job_id}")
        resp.raise_for_status()
