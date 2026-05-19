"""PrusaLink HTTP client.

PrusaLink exposes a REST API on the printer over LAN. Endpoints used here are documented
in Prusa-Link-Web's OpenAPI spec (https://github.com/prusa3d/Prusa-Link-Web).

Two auth modes are supported:
  - **Legacy `X-Api-Key`** (older Buddy firmware): pass `api_key=...` only.
  - **Digest `user`/`password`** (Core One and current Buddy): pass `password=...`
    and optionally a `user` (defaults to `maker`). Used when the printer's Settings
    → Network → PrusaLink screen shows a generated password instead of an API key.

We use the v1 paths because they're stable on Buddy 5.x and on Core One:
  - POST   /api/v1/files/usb/<filename>     upload a .gcode (multipart)
  - POST   /api/v1/files/usb/<filename>     with `Print-After-Upload: ?1` to auto-start
  - GET    /api/v1/status                   printer state
  - GET    /api/v1/job                      current job
  - DELETE /api/v1/job/<id>                 cancel
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)


class PrusaLinkError(RuntimeError):
    pass


@dataclass(slots=True)
class JobStatus:
    state: str
    progress_pct: float
    time_remaining_s: int | None
    raw: dict[str, Any]


class PrusaLinkClient:
    def __init__(
        self,
        host: str,
        api_key: str = "",
        *,
        password: str = "",
        user: str = "maker",
        port: int = 80,
        timeout: float = 15.0,
    ):
        if not host:
            raise ValueError("host is required")
        if not (api_key or password):
            raise ValueError("either api_key or password is required")
        self.base = f"http://{host}:{port}"
        self.headers: dict[str, str] = {}
        auth: httpx.Auth | None = None
        if password:
            # Core One: Digest auth using the PrusaLink-generated password
            auth = httpx.DigestAuth(user, password)
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base, headers=self.headers, auth=auth, timeout=timeout
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "PrusaLinkClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def info(self) -> dict[str, Any]:
        r = await self._client.get("/api/v1/info")
        r.raise_for_status()
        return r.json()

    async def status(self) -> dict[str, Any]:
        r = await self._client.get("/api/v1/status")
        r.raise_for_status()
        return r.json()

    async def job(self) -> dict[str, Any] | None:
        r = await self._client.get("/api/v1/job")
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    async def upload_and_print(
        self,
        filename: str,
        gcode: str | bytes,
        *,
        target: str = "usb",
        print_after_upload: bool = True,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Upload a .gcode and (optionally) start printing it.

        On Buddy/Core One firmware, the upload endpoint is:
            PUT /api/v1/files/<target>/<filename>
        with headers controlling overwrite and auto-print. We use PUT with the raw body —
        the multipart variant works too but is harder to reason about.
        """
        if isinstance(gcode, str):
            body = gcode.encode("utf-8")
        else:
            body = gcode

        # PrusaLink expects path-style filename; strip path separators defensively
        safe_name = filename.replace("/", "_").replace("\\", "_")
        if not safe_name.endswith(".gcode") and not safe_name.endswith(".bgcode"):
            safe_name += ".gcode"

        url = f"/api/v1/files/{target}/{safe_name}"
        headers = {
            "Content-Type": "text/x.gcode",
            "Overwrite": "?1" if overwrite else "?0",
        }
        if print_after_upload:
            headers["Print-After-Upload"] = "?1"

        r = await self._client.put(url, content=body, headers=headers)
        if r.status_code >= 400:
            raise PrusaLinkError(
                f"upload failed: HTTP {r.status_code} — {r.text[:200]}"
            )
        # 201 Created on success
        try:
            return r.json()
        except ValueError:
            return {"status_code": r.status_code, "filename": safe_name}

    async def cancel_job(self, job_id: int) -> None:
        r = await self._client.delete(f"/api/v1/job/{job_id}")
        if r.status_code not in (200, 204):
            raise PrusaLinkError(f"cancel failed: HTTP {r.status_code} — {r.text[:200]}")

    async def get_job_status(self) -> JobStatus | None:
        data = await self.job()
        if not data:
            return None
        state = data.get("state", "UNKNOWN")
        progress = float(data.get("progress", 0.0))
        rem = data.get("time_remaining")
        return JobStatus(state=state, progress_pct=progress, time_remaining_s=rem, raw=data)
