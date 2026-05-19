"""FastAPI application — REST + WebSocket for the web UI."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import AppConfig, load_config, save_config
from .gcode_gen import build_sweep
from .netutil import local_ip_toward
from .replay import list_runs, replay
from .runner import TuningRun, _analysis_to_dict, params_from_config, run_tuning
from .udp_metrics import MetricStream

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    cfg: AppConfig
    stream: MetricStream
    current_run: TuningRun | None = None
    run_task: asyncio.Task | None = None
    ws_clients: set[WebSocket]

    def __init__(self):
        self.cfg = load_config()
        self.stream = MetricStream(port=self.cfg.udp_port)
        self.ws_clients = set()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.stream.start()
    log.info("PrusaPATuner v%s started — UDP on port %d", __version__, state.cfg.udp_port)
    try:
        yield
    finally:
        state.stream.stop()


app = FastAPI(title="PrusaPATuner", version=__version__, lifespan=lifespan)

# Serve the bundled JS/CSS/assets under /static. The index.html the root route
# returns references /static/styles.css and /static/app.js, so this mount is
# what makes the page actually render with styling and behaviour.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- models ----------

class ConfigModel(BaseModel):
    printer_host: str = ""
    printer_api_key: str = ""
    printer_user: str = "maker"
    printer_password: str = ""
    udp_port: int = 8514
    nozzle_temp: float = 215.0
    preheat_temp: float = 225.0
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    slow_flow_mm3_s: float = Field(1.92, gt=0)
    fast_flow_mm3_s: float = Field(19.24, gt=0)
    slow_volume_mm3: float = Field(1.92, gt=0)
    fast_volume_mm3: float = Field(4.81, gt=0)
    cycles_per_K: int = Field(14, ge=1, le=64)
    accel_mm_s2: float = Field(5000.0, gt=0)
    k_min: float = Field(0.0, ge=0)
    k_max: float = Field(0.10, ge=0)
    k_step: float = Field(0.002, gt=0)
    purge_x: float = 30.0
    purge_y: float = 30.0
    purge_z: float = 50.0
    coupled_dx_mm: float = Field(0.05, ge=0)
    coupled_dy_mm: float = Field(0.0, ge=0)
    coupled_dz_mm: float = Field(0.0, ge=0)
    first_slow_leg_factor: float = Field(10.0, ge=1)
    filament_label: str = "PLA"

    @classmethod
    def from_appconfig(cls, c: AppConfig) -> "ConfigModel":
        return cls(**{f: getattr(c, f) for f in cls.model_fields if hasattr(c, f)})

    def apply(self, c: AppConfig) -> AppConfig:
        for f in self.model_fields:
            if hasattr(c, f):
                setattr(c, f, getattr(self, f))
        return c


# ---------- routes ----------

@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return PlainTextResponse(
        "PrusaPATuner is running but static/index.html is missing — "
        "see the README for setup.",
        status_code=200,
    )


@app.get("/api/version")
async def get_version():
    return {"version": __version__}


@app.get("/api/config", response_model=ConfigModel)
async def get_config():
    return ConfigModel.from_appconfig(state.cfg)


@app.post("/api/config", response_model=ConfigModel)
async def post_config(model: ConfigModel):
    model.apply(state.cfg)
    save_config(state.cfg)
    return ConfigModel.from_appconfig(state.cfg)


@app.get("/api/status")
async def get_status():
    udp_stats = state.stream.stats
    run_dict = state.current_run.to_dict() if state.current_run else None
    return {
        "udp": udp_stats,
        "run": run_dict,
        "running": state.run_task is not None and not state.run_task.done(),
    }


@app.get("/api/preview")
async def get_preview():
    """Show the generated G-code without uploading anything."""
    try:
        udp_host = local_ip_toward(state.cfg.printer_host or "8.8.8.8")
    except Exception:
        udp_host = "192.168.1.10"
    plan = build_sweep(params_from_config(state.cfg, udp_host=udp_host))
    return PlainTextResponse(plan.gcode, media_type="text/x.gcode")


@app.post("/api/run")
async def post_run():
    if state.run_task is not None and not state.run_task.done():
        raise HTTPException(409, "A run is already in progress")
    if not state.cfg.printer_host:
        raise HTTPException(400, "printer_host must be configured")
    if not (state.cfg.printer_api_key or state.cfg.printer_password):
        raise HTTPException(
            400, "either printer_api_key or printer_password must be configured"
        )

    state.current_run = None

    async def _go():
        run = await run_tuning(state.cfg, state.stream, on_update=_broadcast_update)
        state.current_run = run

    state.run_task = asyncio.create_task(_go())
    return {"status": "started"}


@app.post("/api/cancel")
async def post_cancel():
    if state.run_task is not None and not state.run_task.done():
        state.run_task.cancel()
    return {"status": "ok"}


@app.get("/api/runs")
async def get_runs():
    """List `runs/run_*.npz` dumps available for replay.

    The frontend renders these in a dropdown; selecting one fires
    `POST /api/runs/<filename>/analyse` and renders the result in the
    same UI used for live sweeps.
    """
    runs = list_runs("runs")
    return {
        "runs": [
            {
                "filename": r.filename,
                "path": r.path,
                "mtime_unix": r.mtime_unix,
                "n_force": r.n_force,
                "n_pos": r.n_pos,
                "n_K": r.n_K,
                "cycles_per_K": r.cycles_per_K,
                "slow_half_s": r.slow_half_s,
                "fast_half_s": r.fast_half_s,
                "duration_s": r.duration_s,
                "filament_label": r.filament_label,
                "nozzle_temp": r.nozzle_temp,
            }
            for r in runs
        ]
    }


@app.post("/api/runs/{filename}/analyse")
async def post_run_analyse(filename: str):
    """Re-run `analyse_sweep` on a saved npz.

    Returns the same shape as the `analysis` field on `/api/status`,
    so the frontend can swap directly into its render path.
    """
    # Whitelist filename to prevent path traversal.
    if not filename.startswith("run_") or not filename.endswith(".npz"):
        raise HTTPException(400, "filename must match run_*.npz")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "filename must not contain path separators")
    path = Path("runs") / filename
    if not path.exists():
        raise HTTPException(404, f"run {filename} not found")
    try:
        plan, analysis = replay(path)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    return {
        "filename": filename,
        "k_values": [seg.k for seg in plan.segments],
        "analysis": _analysis_to_dict(analysis),
    }


@app.get("/api/metrics_seen")
async def get_metrics_seen():
    """Diagnostic: which metric names have we observed and how many samples."""
    stats = state.stream.stats
    return {
        "stats": stats,
        "names": {
            name: len(state.stream.snapshot(name))
            for name in sorted(state.stream._rings.keys())  # noqa: SLF001 — diagnostic only
        },
    }


@app.get("/api/diagnostics")
async def get_diagnostics(window_s: float = 5.0):
    """Live diagnostics: packet stats + per-metric sample rate.

    Use this to verify each metric is actually streaming and at what rate.
    If `loadcell_value` reads zero or far below ~100 Hz the M331 enable
    almost certainly silently failed (most likely cause: the firmware on
    your printer doesn't expose that metric name -- the M331 handler
    writes "Metric not found" to serial, which PrusaLink discards). The
    Buddy throttle is COMPILE-TIME per `METRIC_DEF` -- gcode can't change
    it, only enable/disable.
    """
    rates = state.stream.metric_rates(window_s=window_s)
    return {
        "stats": state.stream.stats,
        "rates_hz": rates,
        "window_s": window_s,
    }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    state.ws_clients.add(ws)
    # bootstrap with current run state
    if state.current_run is not None:
        try:
            await ws.send_json({"type": "run", "data": state.current_run.to_dict()})
        except Exception:
            pass
    # Live loadcell fan-out: single subscriber to loadcell_value (the only
    # loadcell metric this firmware emits). We BATCH samples instead of
    # sending one WS frame per sample -- at ~180 Hz, per-sample frames
    # overflowed the subscriber queue. Batching at ~50 ms windows
    # (~9 samples per frame) drops WebSocket overhead and lets the queue
    # drain. Sample timestamps are already spread within each UDP packet
    # by MetricStream._on_packet, so the receiver sees a continuous time
    # series rather than vertical clusters at packet-arrival times.
    BATCH_S = 0.05  # 20 Hz flush rate

    async def forward(metric: str, msg_type: str) -> None:
        import time as _time
        batch_t: list[float] = []
        batch_v: list[float] = []
        last_flush = _time.monotonic()
        try:
            async for sample in state.stream.subscribe(metric):
                v = _first_numeric(sample.fields)
                if v is None:
                    continue
                batch_t.append(sample.recv_monotonic)
                batch_v.append(v)
                now = _time.monotonic()
                if now - last_flush >= BATCH_S or len(batch_t) >= 64:
                    try:
                        await ws.send_json(
                            {
                                "type": msg_type,
                                "metric": metric,
                                "t": batch_t,
                                "v": batch_v,
                            }
                        )
                    except WebSocketDisconnect:
                        return
                    except Exception:
                        return
                    batch_t = []
                    batch_v = []
                    last_flush = now
        except Exception:
            return

    # Two parallel forwarders: the force trace and the X-position trace.
    # Each rides its own batched WebSocket message type so the client can
    # plot them on independent y-axes without metric-name disambiguation.
    tasks = [
        asyncio.create_task(forward("loadcell_value", "force_batch")),
        asyncio.create_task(forward("pos_x", "pos_batch")),
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in tasks:
            t.cancel()
        state.ws_clients.discard(ws)


def _first_numeric(fields: dict) -> float | None:
    import math
    for v in fields.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                continue
            return x
    return None


async def _broadcast_update(run: TuningRun) -> None:
    payload = {"type": "run", "data": run.to_dict()}
    dead: list[WebSocket] = []
    for ws in list(state.ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.discard(ws)
