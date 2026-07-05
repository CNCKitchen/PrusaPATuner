"""End-to-end orchestration for a single PA tuning run."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .analysis import SweepAnalysis, analyse_sweep
from .config import AppConfig
from .gcode_gen import SweepParams, SweepPlan, build_sweep
from .netutil import local_ip_toward
from .prusalink import PrusaLinkClient
from .udp_metrics import MetricSample, MetricStream

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RunState:
    state: str = "idle"  # idle | preparing | running | analyzing | done | error
    message: str = ""
    progress_pct: float = 0.0
    current_k: float | None = None
    started_at: float = 0.0
    # First-loadcell-sample monotonic time, used as a seed for the
    # analyser's auto-detect-t0 search. Not a precise anchor -- it lands
    # somewhere in the heat-up / homing phase, well before the actual
    # burst pattern begins. The analyser refines it via rolling-std +
    # model correlation; see `_detect_sweep_start` in analysis.py.
    sweep_t0: float | None = None
    analysis: SweepAnalysis | None = None
    plan: SweepPlan | None = None
    error: str | None = None


@dataclass(slots=True)
class TuningRun:
    """One PA tuning run with live state + analysis result."""

    cfg: AppConfig
    plan: SweepPlan
    state: RunState = field(default_factory=RunState)
    force_t: list[float] = field(default_factory=list)
    force_y: list[float] = field(default_factory=list)
    # Toolhead position streams used to anchor sweep_t0 precisely. pos_x is
    # the primary anchor (coupled_dx_mm is the default non-zero axis);
    # pos_y / pos_z are populated for diagnostic UI and to support runs
    # where the user moved the coupling onto dy or dz.
    pos_t: list[float] = field(default_factory=list)
    pos_x: list[float] = field(default_factory=list)
    pos_y_t: list[float] = field(default_factory=list)
    pos_y: list[float] = field(default_factory=list)
    pos_z_t: list[float] = field(default_factory=list)
    pos_z: list[float] = field(default_factory=list)
    on_update: Callable[["TuningRun"], None] | None = None

    def emit(self) -> None:
        if self.on_update is None:
            return
        try:
            result = self.on_update(self)
            # If the callback is async (e.g. the FastAPI WebSocket broadcaster),
            # schedule it as a task so the coroutine actually runs. Without this
            # the call site just creates an unawaited coroutine and Python emits
            # "coroutine X was never awaited" while no client ever sees updates.
            if asyncio.iscoroutine(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    # no running loop (e.g. called from a sync test) -- drop it
                    result.close()
        except Exception:  # broadcast must never fail the run
            log.exception("on_update callback failed")

    def to_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "state": s.state,
            "message": s.message,
            "progress_pct": s.progress_pct,
            "current_k": s.current_k,
            "started_at": s.started_at,
            "error": s.error,
            "n_force_samples": len(self.force_t),
            "n_k": len(self.plan.segments),
            "k_values": [seg.k for seg in self.plan.segments],
            "analysis": _analysis_to_dict(s.analysis) if s.analysis else None,
        }


def _bd_weights_payload(shipped: dict[str, float]) -> dict[str, Any]:
    """Pick which weights to ship to the UI: optimised (if present) or shipped.

    Reads `runs/weights_opt.json` once per call (cheap; file is tiny).
    The UI uses `bd_default_weights` to initialise the sliders and
    `bd_weights_source` to render a badge ("defaults" vs
    "optimised (N runs, date)"). When no optimiser file exists, the
    shipped BD_DEFAULT_WEIGHTS go out unchanged.
    """
    # Local import to avoid a runner→optimiser→replay cycle at module load
    # (replay imports from analysis, runner imports from gcode_gen; the
    # late binding here keeps the cycle from materialising).
    from .optimiser import read_weights_json
    import datetime as _dt

    data = read_weights_json()
    if not data:
        return {
            "bd_default_weights": shipped,
            "bd_weights_source": {"kind": "defaults"},
        }
    opt = data.get("weights") or {}
    # Merge: any metric the optimiser didn't write keeps its shipped
    # value (so a partial optimiser file doesn't silently drop a metric
    # from the cost). The UI sees the *effective* weights either way.
    merged = dict(shipped)
    for k, v in opt.items():
        if isinstance(v, (int, float)):
            merged[k] = float(v)
    ts = data.get("timestamp_unix")
    when = ""
    if isinstance(ts, (int, float)):
        try:
            when = _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
        except Exception:
            when = ""
    return {
        "bd_default_weights": merged,
        "bd_weights_source": {
            "kind": "optimised",
            "n_runs": data.get("n_runs"),
            "rms_error": data.get("rms_error"),
            "median_abs_error": data.get("median_abs_error"),
            "alpha": data.get("alpha"),
            "excluded_metrics": data.get("excluded_metrics") or [],
            "date": when,
        },
    }


def _analysis_to_dict(a: SweepAnalysis) -> dict[str, Any]:
    return {
        "per_k": [
            {
                "k": r.k,
                "n_samples": r.n_samples,
                "phase_lag_ms": _safe_float(r.phase_lag_ms),
                "integral_area": _safe_float(r.integral_area),
                "integral_area_legacy": _safe_float(r.integral_area_legacy),
                "integral_n_included": r.integral_n_included,
                "integral_n_total": r.integral_n_total,
                "force_mean": _safe_float(r.force_mean),
                "force_std": _safe_float(r.force_std),
                "coverage": _safe_float(r.coverage),
                "dropouts": r.dropouts,
            }
            for r in a.per_k
        ],
        "phase_fit": _fit_to_dict(a.phase_fit),
        "integral_fit": _fit_to_dict(a.integral_fit),
        "integral_legacy_fit": _fit_to_dict(a.integral_legacy_fit),
        "baseline": (
            {
                "mean": _safe_float(a.baseline.mean),
                "std": _safe_float(a.baseline.std),
                "drift": _safe_float(a.baseline.drift),
                "n_samples": a.baseline.n_samples,
                "t_start": _safe_float(a.baseline.t_start),
                "t_end": _safe_float(a.baseline.t_end),
            }
            if a.baseline is not None
            else None
        ),
        "sample_rate_hz": a.sample_rate_hz,
        "notes": a.notes,
        "windows": [
            {
                "k": w.k,
                "t": w.t,
                "force": w.force,
                "command": w.command,
                "ground_truth_force": w.ground_truth_force,
                "dropout_t": w.dropout_t,
            }
            for w in a.windows
        ],
        "force_baselines": (
            {
                "slow_plateau": _safe_float(a.force_baselines.slow_plateau),
                "fast_plateau": _safe_float(a.force_baselines.fast_plateau),
                "n_slow": a.force_baselines.n_slow,
                "n_fast": a.force_baselines.n_fast,
            }
            if a.force_baselines is not None
            else None
        ),
        # bd_pressure step-response analysis. `bd_segments` is one row
        # per (K, cycle) with all 12 region metrics + exclusion flag;
        # `bd_per_k` is one row per K with medians + sweep-normalised
        # values + segment counts; `bd_k_opt` is the recommended K from
        # the default-weight composite cost; `bd_default_weights` are
        # the slider defaults. If `runs/weights_opt.json` exists (the
        # optimiser persisted user-tuned weights), those override the
        # shipped defaults so the UI sliders pre-fill with the optimised
        # values; the bd_weights_source field tells the UI which set is
        # active for the on-screen badge.
        "bd_k_opt": (
            _safe_float(a.bd_k_opt) if a.bd_k_opt is not None else None
        ),
        # Signed-area zero-crossing fits (overshoot↔lag discriminator).
        # Each holds {k_opt, slope, intercept, r_squared, method}; the
        # backing per-K medians ride along in bd_per_k[*].medians under
        # rise_signed_area / fall_signed_area.
        "bd_signed_rise_fit": _fit_to_dict(a.bd_signed_rise_fit),
        "bd_signed_fall_fit": _fit_to_dict(a.bd_signed_fall_fit),
        **_bd_weights_payload(a.bd_default_weights),
        "bd_per_k": [
            {
                "k": r.k,
                "n_segments_total": r.n_segments_total,
                "n_segments_included": r.n_segments_included,
                "medians": {n: _safe_float(v) for n, v in r.medians.items()},
                "normalised": {
                    n: _safe_float(v) for n, v in r.normalised.items()
                },
                "mads": {n: _safe_float(v) for n, v in r.mads.items()},
                "iqrs": {n: _safe_float(v) for n, v in r.iqrs.items()},
            }
            for r in a.bd_per_k
        ],
        "bd_segments": [
            {
                "k": s.k,
                "seg_idx": s.seg_idx,
                "t_start": _safe_float(s.t_start),
                "t_rise": _safe_float(s.t_rise),
                "t_fall": _safe_float(s.t_fall),
                "t_end": _safe_float(s.t_end),
                "t_lo_display": _safe_float(s.t_lo_display),
                "t_hi_display": _safe_float(s.t_hi_display),
                "t_rise_end": (
                    _safe_float(s.t_rise_end) if s.t_rise_end is not None else None
                ),
                "t_fall_start": (
                    _safe_float(s.t_fall_start) if s.t_fall_start is not None else None
                ),
                "t_fall_end": (
                    _safe_float(s.t_fall_end) if s.t_fall_end is not None else None
                ),
                "t_peak": _safe_float(s.t_peak) if s.t_peak is not None else None,
                "t_trough": (
                    _safe_float(s.t_trough) if s.t_trough is not None else None
                ),
                "n_samples": s.n_samples,
                "metrics": {n: _safe_float(v) for n, v in s.metrics.items()},
                "excluded": s.excluded,
                "exclusion_reasons": s.exclusion_reasons,
            }
            for s in a.bd_segments
        ],
    }


def _fit_to_dict(f) -> dict[str, Any] | None:
    if f is None:
        return None
    return {
        "k_opt": _safe_float(f.k_opt),
        "slope": _safe_float(f.slope),
        "intercept": _safe_float(f.intercept),
        "r_squared": _safe_float(f.r_squared),
        "method": f.method,
    }


def _safe_float(x: float) -> float | None:
    try:
        if np.isfinite(x):
            return float(x)
        return None
    except (TypeError, ValueError):
        return None


def _sort_by_time(
    t: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) sorted by t ascending. Uses argsort with kind='stable'
    so samples with equal timestamps keep their original order.

    Necessary because the UDP stream can deliver per-metric samples
    with overlapping timestamps when consecutive packets cover
    overlapping firmware-time spans (the firmware-offset spread in
    udp_metrics anchors each batch at its own host arrival time, so
    when host inter-packet gap is shorter than the batch's firmware
    span, the second batch's earliest samples land before the first
    batch's latest). udp_metrics also clips to enforce monotonicity,
    but a defensive sort here means any future bug or network reorder
    won't break the plots / segment slicing.
    """
    if len(t) <= 1:
        return t, y
    # Skip the sort when already monotonic (the common case).
    if bool(np.all(np.diff(t) >= 0)):
        return t, y
    order = np.argsort(t, kind="stable")
    return t[order], y[order]


def _k_range(k_min: float, k_max: float, k_step: float) -> tuple[float, ...]:
    """Inclusive K grid from min..max in step increments.

    Uses an integer index loop with `round(..., 4)` so successive sums of
    `k_step` don't accumulate float drift across the 50-ish samples of a
    typical fine sweep (e.g. `0.0 + 50 * 0.002` should read 0.1000 exactly,
    not 0.09999999...). The end is inclusive when `k_max` lands on the
    grid (within 1e-9 of an integer multiple); otherwise the sweep stops
    at the last full step below `k_max`.
    """
    if k_step <= 0 or k_max < k_min:
        return (round(k_min, 4),)
    n = int(round((k_max - k_min) / k_step)) + 1
    return tuple(round(k_min + i * k_step, 4) for i in range(n))


def _volumetric_to_time_domain(
    flow_mm3_s: float, volume_mm3: float, filament_diameter: float,
) -> tuple[float, float]:
    """Convert (volumetric flow, volume per leg) into the time-domain pair
    that SweepParams / the analyzer expect: filament feed velocity (mm/s)
    and leg duration (s).

    feed_mm_s = flow / area;  duration_s = volume / flow.
    """
    import math
    area = math.pi * (filament_diameter / 2.0) ** 2  # mm²
    feed_mm_s = flow_mm3_s / max(area, 1e-9)
    duration_s = volume_mm3 / max(flow_mm3_s, 1e-9)
    return feed_mm_s, duration_s


def params_from_config(cfg: AppConfig, udp_host: str) -> SweepParams:
    slow_feed_mm_s, slow_half_s = _volumetric_to_time_domain(
        cfg.slow_flow_mm3_s, cfg.slow_volume_mm3, cfg.filament_diameter,
    )
    fast_feed_mm_s, fast_half_s = _volumetric_to_time_domain(
        cfg.fast_flow_mm3_s, cfg.fast_volume_mm3, cfg.filament_diameter,
    )
    return SweepParams(
        nozzle_temp=cfg.nozzle_temp,
        preheat_temp=cfg.preheat_temp,
        nozzle_diameter=cfg.nozzle_diameter,
        filament_diameter=cfg.filament_diameter,
        filament_label=cfg.filament_label,
        slow_feed_mm_s=slow_feed_mm_s,
        fast_feed_mm_s=fast_feed_mm_s,
        slow_half_s=slow_half_s,
        fast_half_s=fast_half_s,
        cycles_per_K=cfg.cycles_per_K,
        accel_mm_s2=cfg.accel_mm_s2,
        K_values=_k_range(cfg.k_min, cfg.k_max, cfg.k_step),
        purge_x=cfg.purge_x,
        purge_y=cfg.purge_y,
        purge_z=cfg.purge_z,
        coupled_dx_mm=cfg.coupled_dx_mm,
        coupled_dy_mm=cfg.coupled_dy_mm,
        coupled_dz_mm=cfg.coupled_dz_mm,
        first_slow_leg_factor=cfg.first_slow_leg_factor,
        udp_host=udp_host,
        udp_port=cfg.udp_port,
        label=f"PA tuning -- {cfg.filament_label}",
    )


def _extract_numeric(sample: MetricSample) -> float | None:
    """Pull the first finite numeric value out of a sample.

    Used for position metrics (pos_x / pos_y / pos_z) and any other plain-
    value telemetry. Unlike _extract_force we look at `v` first (the canonical
    field name on this firmware), then fall through to any other numeric.
    """
    import math
    f = sample.fields
    v = f.get("v")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if math.isfinite(x):
            return x
    for val in f.values():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            x = float(val)
            if math.isfinite(x):
                return x
    return None


def _extract_force(sample: MetricSample) -> float | None:
    """Try the well-known field names from probe_load_line / loadcell metrics.

    The exact field name depends on what's actually streaming. We accept any of:
      - `v` (single value)
      - `load` / `force` / `z` (named field)
      - the first numeric field in the sample as a last resort.

    NaN values are rejected (return None). loadcell_hp streams `v=nan` while
    the loadcell is idle -- if we let those through, the "first metric to
    deliver wins" race would lock onto loadcell_hp and flood force_y with
    NaNs, breaking downstream analysis.
    """
    import math
    f = sample.fields
    for key in ("load", "force", "z", "v"):
        if key in f and isinstance(f[key], (int, float)) and not isinstance(f[key], bool):
            v = float(f[key])
            if math.isnan(v) or math.isinf(v):
                return None
            return v
    for v in f.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                continue
            return x
    return None


async def run_tuning(
    cfg: AppConfig,
    stream: MetricStream,
    *,
    on_update: Callable[["TuningRun"], None] | None = None,
    loadcell_metric: str = "loadcell_value",
    poll_interval_s: float = 1.0,
    job_timeout_s: float | None = None,
) -> TuningRun:
    """Execute one tuning run from start to finish.

    The caller must have already started `stream` and have a network reachable printer.

    `job_timeout_s` defaults to `None` which means "compute from the plan".
    A previous hardcoded 600s (10 min) silently killed any sweep longer
    than that: the user's 21-K × 10-cycle sweep ran ~12 minutes on the
    printer, the timeout fired BEFORE PrusaLink reported FINISHED, the
    run transitioned to `error`, and no analysis or NPZ dump was
    produced -- even though the print itself had succeeded. With the
    dynamic default the timeout scales with sweep length plus a 5-min
    margin for heat-up / homing / unexpected slowdowns.
    """
    udp_host = local_ip_toward(cfg.printer_host, port=80)
    params = params_from_config(cfg, udp_host=udp_host)
    plan = build_sweep(params)

    if job_timeout_s is None:
        # plan.segments[-1].start_offset_s + duration_s = sweep duration
        # in seconds (relative to "sweep start" marker). Add a 2x safety
        # factor plus a fixed 5-minute headroom for the heat-up / home /
        # Z-marker phase that runs BEFORE the sweep timer starts.
        if plan.segments:
            sweep_dur = (
                plan.segments[-1].start_offset_s
                + plan.segments[-1].duration_s
            )
        else:
            sweep_dur = 0.0
        job_timeout_s = max(900.0, 2.0 * sweep_dur + 300.0)
        log.info(
            "job_timeout_s=%.0f auto-derived from %d K values × %d cycles "
            "(estimated sweep duration %.0f s)",
            job_timeout_s, len(plan.segments), plan.params.cycles_per_K,
            sweep_dur,
        )

    run = TuningRun(cfg=cfg, plan=plan, on_update=on_update)
    run.state.started_at = time.time()
    run.state.state = "preparing"
    run.state.message = (
        f"Generated sweep ({len(plan.segments)} K values, "
        f"timeout {job_timeout_s/60:.0f} min), uploading…"
    )
    run.emit()

    # Collectors:
    #   * loadcell_value -- the primary force trace fed to the analyser
    #   * pos_x / pos_y / pos_z -- toolhead position. Used by the analyser
    #     to anchor sweep_t0 from the first X transition (much more
    #     precise than the loadcell auto-detect). y and z are captured too
    #     for diagnostic completeness.
    stop_collect = asyncio.Event()

    async def collect_loadcell() -> None:
        async for sample in stream.subscribe(loadcell_metric):
            if stop_collect.is_set():
                return
            v = _extract_force(sample)
            if v is None:
                continue
            run.force_t.append(sample.recv_monotonic)
            run.force_y.append(v)
            if run.state.sweep_t0 is None and run.state.state == "running":
                run.state.sweep_t0 = sample.recv_monotonic

    async def collect_pos(
        metric: str, t_list: list[float], y_list: list[float],
    ) -> None:
        async for sample in stream.subscribe(metric):
            if stop_collect.is_set():
                return
            v = _extract_numeric(sample)
            if v is None:
                continue
            t_list.append(sample.recv_monotonic)
            y_list.append(v)

    collector_primary = asyncio.create_task(collect_loadcell())
    collector_pos_x = asyncio.create_task(collect_pos("pos_x", run.pos_t, run.pos_x))
    collector_pos_y = asyncio.create_task(collect_pos("pos_y", run.pos_y_t, run.pos_y))
    collector_pos_z = asyncio.create_task(collect_pos("pos_z", run.pos_z_t, run.pos_z))

    try:
        async with PrusaLinkClient(
            cfg.printer_host,
            cfg.printer_api_key,
            password=cfg.printer_password,
            user=cfg.printer_user or "maker",
        ) as pl:
            # upload + auto-print
            filename = f"pa_tuner_{int(time.time())}.gcode"
            await pl.upload_and_print(filename, plan.gcode)
            run.state.state = "running"
            run.state.message = "Job started; capturing loadcell stream…"
            run.emit()

            # poll job progress. The Core One occasionally drops HTTP
            # connections during long heatups / homing -- httpx surfaces
            # this as ReadError / ReadTimeout / ConnectError. None of those
            # mean the job actually failed; just back off and retry.
            import httpx
            transient = (
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
            )
            t_start = time.monotonic()
            last_progress = -1.0
            last_status_state: str | None = None
            consecutive_failures = 0
            while True:
                if time.monotonic() - t_start > job_timeout_s:
                    # Before giving up, do one final blocking status
                    # check. The user's machine sometimes finishes the
                    # gcode while we're between polls; if a final poll
                    # confirms FINISHED/STOPPED/IDLE, the print is
                    # actually done and we should analyse, not error.
                    try:
                        final_status = await pl.get_job_status()
                    except transient:
                        final_status = None
                    final_state = (
                        final_status.state.upper() if final_status is not None
                        else (last_status_state or "")
                    )
                    if final_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE", ""):
                        log.warning(
                            "job_timeout_s=%.0f reached but final status "
                            "is %r -- treating as completed, continuing "
                            "to analysis with the data captured so far",
                            job_timeout_s, final_state,
                        )
                        break
                    raise TimeoutError(
                        f"job exceeded timeout ({job_timeout_s:.0f}s); "
                        f"last known state: {final_state}"
                    )
                try:
                    status = await pl.get_job_status()
                    consecutive_failures = 0
                except transient as exc:
                    consecutive_failures += 1
                    # Tolerate a handful of transient drops; only fail if
                    # they pile up (suggesting the printer is actually gone).
                    if consecutive_failures >= 10:
                        raise RuntimeError(
                            f"PrusaLink unresponsive after {consecutive_failures} "
                            f"consecutive failures: {type(exc).__name__}: {exc}"
                        )
                    log.warning(
                        "transient PrusaLink error %s (%s/10); retrying",
                        type(exc).__name__, consecutive_failures,
                    )
                    run.state.message = (
                        f"Printing… (PrusaLink busy, retrying "
                        f"{consecutive_failures}/10)"
                    )
                    run.emit()
                    await asyncio.sleep(poll_interval_s)
                    continue
                if status is None:
                    # job already gone — assume finished
                    break
                last_status_state = status.state.upper()
                if last_status_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE"):
                    break
                if last_status_state in ("ERROR",):
                    raise RuntimeError(f"printer reported ERROR: {status.raw}")
                if status.progress_pct != last_progress:
                    run.state.progress_pct = status.progress_pct
                    run.state.message = f"Printing… {status.progress_pct:.0f}%"
                    run.emit()
                    last_progress = status.progress_pct
                await asyncio.sleep(poll_interval_s)

        # analyse
        run.state.state = "analyzing"
        run.state.message = "Crunching loadcell signal…"
        run.emit()

        if run.state.sweep_t0 is None and run.force_t:
            run.state.sweep_t0 = run.force_t[0]
        if run.state.sweep_t0 is None:
            raise RuntimeError(
                "No loadcell samples received — verify the metric streams during print"
            )

        force_t = np.asarray(run.force_t, dtype=float)
        force_y = np.asarray(run.force_y, dtype=float)
        pos_t = np.asarray(run.pos_t, dtype=float) if run.pos_t else None
        pos_x = np.asarray(run.pos_x, dtype=float) if run.pos_x else None
        pos_z_t = np.asarray(run.pos_z_t, dtype=float) if run.pos_z_t else None
        pos_z_arr = np.asarray(run.pos_z, dtype=float) if run.pos_z else None
        # Defensive sort: even with the udp_metrics monotonic-clip,
        # mixed-up packet ordering on the network can leave per-stream
        # timestamps out of order. Sort each stream by time before
        # analysis so window slicing and segment metrics see a
        # well-ordered timeline.
        force_t, force_y = _sort_by_time(force_t, force_y)
        if pos_t is not None and pos_x is not None:
            pos_t, pos_x = _sort_by_time(pos_t, pos_x)
        if pos_z_t is not None and pos_z_arr is not None:
            pos_z_t, pos_z_arr = _sort_by_time(pos_z_t, pos_z_arr)
        analysis = analyse_sweep(
            sweep_t0=run.state.sweep_t0,
            force_t=force_t,
            force_y=force_y,
            plan=plan,
            pos_t=pos_t,
            pos_x=pos_x,
            pos_z_t=pos_z_t,
            pos_z=pos_z_arr,
            z_marker_lift_mm=plan.params.z_marker_lift_mm,
        )
        run.state.analysis = analysis

        # Dump raw data to disk so the user can inspect / re-analyse
        # the run offline. NPZ is loadable with `np.load(path)` and
        # carries all the timestamped streams. The path is appended to
        # the analyser notes so it surfaces in the UI / on the API.
        try:
            from pathlib import Path
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            dump_path = runs_dir / f"run_{int(run.state.started_at)}.npz"
            # Wall-clock anchor: pair the current monotonic instant
            # with the corresponding wall-clock unix timestamp. With
            # both, every monotonic sample can be converted to a UTC
            # datetime in post-processing:
            #   wall_unix(sample) = mono_anchor_unix + (sample.mono - mono_anchor_mono)
            mono_anchor_mono = time.monotonic()
            mono_anchor_unix = time.time()
            # Deterministic packet-loss log from msg= sequence gaps
            # (see MetricStream.loss_events). Empty arrays when the
            # stream never saw syslog-wrapped packets.
            loss = stream.loss_events()
            loss_t = np.array([t for t, _ in loss], dtype=float)
            loss_n = np.array([n for _, n in loss], dtype=float)
            np.savez(
                dump_path,
                force_t=force_t,
                force_y=force_y,
                loss_t=loss_t,
                loss_n=loss_n,
                pos_t=pos_t if pos_t is not None else np.array([]),
                pos_x=pos_x if pos_x is not None else np.array([]),
                pos_y_t=np.asarray(run.pos_y_t, dtype=float),
                pos_y=np.asarray(run.pos_y, dtype=float),
                pos_z_t=pos_z_t if pos_z_t is not None else np.array([]),
                pos_z=pos_z_arr if pos_z_arr is not None else np.array([]),
                sweep_t0=np.array([float(run.state.sweep_t0)]),
                # Time-domain anchor for monotonic → wall-clock conversion.
                # mono_anchor_mono is the monotonic instant we captured;
                # mono_anchor_unix is the wall-clock unix time at that
                # same instant. Subtract mono_anchor_mono from any
                # *_t array, add mono_anchor_unix, and you have a unix
                # timestamp (seconds since 1970-01-01 UTC).
                mono_anchor_mono=np.array([float(mono_anchor_mono)]),
                mono_anchor_unix=np.array([float(mono_anchor_unix)]),
                started_at_unix=np.array([float(run.state.started_at)]),
                k_values=np.array([seg.k for seg in plan.segments], dtype=float),
                cycle_period_s=np.array([plan.segments[0].cycle_period_s]) if plan.segments else np.array([]),
                cycles_per_K=np.array([plan.params.cycles_per_K]),
                slow_half_s=np.array([plan.params.slow_half_s]),
                fast_half_s=np.array([plan.params.fast_half_s]),
                slow_feed_mm_s=np.array([plan.params.slow_feed_mm_s]),
                fast_feed_mm_s=np.array([plan.params.fast_feed_mm_s]),
                # Save the sweep-shape parameters needed for correct
                # replay analysis. Without these, replay rebuilds the
                # plan with defaults (e.g. first_slow_leg_factor=10)
                # and the pos_x-transition slicer's K[0] warm-up
                # detection breaks because the plan's predicted slow
                # leg duration disagrees with the data. coupled_d*_mm
                # determine the pos_x transition deadband; purge_*
                # define the toolhead idle position.
                first_slow_leg_factor=np.array(
                    [plan.params.first_slow_leg_factor]
                ),
                coupled_dx_mm=np.array([plan.params.coupled_dx_mm]),
                coupled_dy_mm=np.array([plan.params.coupled_dy_mm]),
                coupled_dz_mm=np.array([plan.params.coupled_dz_mm]),
                purge_x=np.array([plan.params.purge_x]),
                purge_y=np.array([plan.params.purge_y]),
                purge_z=np.array([plan.params.purge_z]),
                z_marker_lift_mm=np.array([plan.params.z_marker_lift_mm]),
                # Filament + test temperature. Use a fixed-width Unicode
                # dtype (NOT object) so `np.load` without `allow_pickle`
                # can read the file back. An object-dtype array forces
                # pickle, and `list_runs` was silently dropping runs that
                # included it because the default `np.load` raises
                # ValueError when it sees one -- two NPZ files were on
                # disk but invisible in the UI dropdown until the loader
                # was fixed.
                filament_label=np.array(
                    [plan.params.filament_label], dtype="U128"
                ),
                nozzle_temp=np.array([plan.params.nozzle_temp]),
            )
            analysis.notes.append(
                f"raw data dumped to {dump_path.resolve()} "
                f"(np.load() to inspect arrays: force_t, force_y, "
                f"pos_t/x/y/z, sweep_t0, k_values, plan params)"
            )
            log.info("dumped raw run data to %s", dump_path)
        except Exception:
            log.exception("raw data dump failed (analysis still succeeded)")
        run.state.state = "done"
        run.state.message = "Done."
        run.state.progress_pct = 100.0
        run.state.current_k = None  # clear the live K display now that we're done
        run.emit()

    except Exception as exc:
        run.state.state = "error"
        run.state.error = f"{type(exc).__name__}: {exc}"
        run.state.message = run.state.error
        log.exception("tuning run failed")
        run.emit()
    finally:
        stop_collect.set()
        for task in (collector_primary, collector_pos_x, collector_pos_y, collector_pos_z):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        log.info(
            "captured: %d loadcell, %d pos_x, %d pos_y, %d pos_z samples",
            len(run.force_t), len(run.pos_x), len(run.pos_y), len(run.pos_z),
        )

    return run
