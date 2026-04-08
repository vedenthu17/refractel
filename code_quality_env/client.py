from __future__ import annotations

import asyncio
import atexit
import contextlib
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from .models import CodeReviewAction, CodeReviewObservation, CodeReviewStepResult


class _RawStepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class _DockerHandle:
    container_id: str
    port: int


class CodeReviewEnv:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        self._docker_handle: Optional[_DockerHandle] = None

    async def __aenter__(self) -> "CodeReviewEnv":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @classmethod
    async def from_docker_image(cls, image_name: str | None = None) -> "CodeReviewEnv":
        image = image_name or "code-review-quality-env:latest"
        port = _find_free_port()

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:7860",
            image,
        ]
        proc = subprocess.run(run_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker image {image}: {proc.stderr.strip()}")

        container_id = proc.stdout.strip()
        env = cls(base_url=f"http://127.0.0.1:{port}")
        env._docker_handle = _DockerHandle(container_id=container_id, port=port)
        await env.__aenter__()

        # Wait for service readiness with bounded retries.
        for _ in range(40):
            with contextlib.suppress(Exception):
                r = await env._client.get("/health")  # type: ignore[union-attr]
                if r.status_code == 200:
                    break
            await asyncio.sleep(0.25)
        else:
            await env.close()
            raise RuntimeError("Container started but /health did not become ready in time")

        atexit.register(_stop_container, container_id)
        return env

    async def reset(self, task_name: str | None = None) -> CodeReviewStepResult:
        client = self._require_client()
        payload = {"task_name": task_name}
        response = await client.post("/reset", json=payload)
        response.raise_for_status()
        return _RawStepResult.model_validate(response.json())

    async def step(self, action: CodeReviewAction) -> CodeReviewStepResult:
        client = self._require_client()
        response = await client.post("/step", json=action.model_dump())
        response.raise_for_status()
        return _RawStepResult.model_validate(response.json())

    async def state(self) -> dict[str, Any]:
        client = self._require_client()
        response = await client.get("/state")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._docker_handle is not None:
            _stop_container(self._docker_handle.container_id)
            self._docker_handle = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s)
        return self._client


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stop_container(container_id: str) -> None:
    subprocess.run(["docker", "stop", container_id], capture_output=True, text=True, check=False)
