"""Analysis algorithms for PA tuning.

Two independent algorithms are run on every sweep, side-by-side, so we can compare them
empirically:

1. **Phase-lag (cross-correlation)** — for each K, build the commanded extrusion-velocity
   square wave and the measured loadcell trace on a common time grid, cross-correlate
   them, and find the sub-sample peak via parabolic interpolation. The lag at peak
   correlation is the "PA error". Linear-fit lag-vs-K → solve for lag = 0 → optimal K.

2. **Integral / area fit (Snapmaker-style)** — for each K, integrate
   `(loadcell - mean) * direction_of_transition` over a window CENTERED on each
   velocity transition. With perfect tracking, half the window sits in each leg's
   plateau and the contributions cancel → total = 0. When force lags, the
   post-transition half still reads the pre-transition value → negative. When force
   overshoots/leads, positive. Linear-fit area-vs-K → solve for area = 0 → optimal K.

   The legacy `sign(d_command/dt)` formulation (only ~5% duty cycle under our
   accel-limited ramp) is computed in parallel as `integral_area_legacy` so we can
   visually verify the centered-window form actually responds to K differences.

Both methods reduce to a 1-D linear regression on K, so we share the same fit step.

Sweep-t0 anchoring and per-K window slicing live in `alignment.py`; this module
imports them back (and re-exports the private helper names for existing callers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from .gcode_gen import KSegment, SweepPlan
from .alignment import (  # noqa: F401 -- re-exported: tests/callers import via analysis
    AnchorResolution,
    _anchor_and_slice_from_pos,
    _detect_first_pos_motion,
    _detect_force_cycle_starts,
    _detect_pos_cycle_starts,
    _detect_pos_transitions,
    _detect_sweep_start,
    _detect_z_marker_anchor,
    _slice_from_force_cycles,
    _slice_from_plan,
    _slice_from_pos_transitions,
    _slice_from_pos_with_known_anchor,
    resolve_anchor_and_windows,
)


@dataclass(slots=True)
class KResult:
    k: float
    n_samples: int
    phase_lag_ms: float
    integral_area: float
    integral_area_legacy: float
    force_mean: float
    force_std: float
    # Data-quality flag in [0, 1]: actual loadcell samples in the burst
    # window divided by what we expected at the run's average incoming
    # sample rate. < ~0.5 means UDP packet loss / firmware throttle /
    # subscriber overflow dropped enough samples that the per-transition
    # integrals are unreliable. The K_opt extractors exclude windows
    # below `MIN_COVERAGE_FOR_FIT` so a few bad K values don't poison the
    # whole sweep.
    coverage: float = 1.0
    # Count of detected dropouts in the K window (samples preceded by
    # an inter-sample force jump > 0.5·plateau_delta -- a tell-tale
    # for missing samples). When `dropouts / n_samples > 0.05` the K
    # is also excluded from the K_opt fits, even if `coverage` looks
    # high (a window with 1000 samples and 100 dropouts has nominal
    # coverage 1.0 but the per-transition integrals are still junk).
    dropouts: int = 0
    # bd-cycle accounting backing the integral_area value: how many of
    # `_bd_segment_metrics`-built cycles for this K survived the shared
    # auto-exclusion gate (dropouts in critical zone, sample rate < 40 Hz,
    # signal-below-noise, ...). When `n < 4` the integral_area itself is
    # set NaN and the K is dropped from the linear fit.
    integral_n_included: int = 0
    integral_n_total: int = 0


@dataclass(slots=True)
class KWindow:
    """Per-K resampled timeseries kept around for plotting / inspection.

    `t` is sweep-relative seconds; `force` is the (centered) loadcell trace on
    the analyser's uniform grid; `command` is the reconstructed commanded
    extrusion velocity (mm/s) on the same grid. These are exactly what the
    fitters consume, so plotting them lets the user see what the algorithm
    actually saw.

    `ground_truth_force` is a separate, force-units, pos_x-derived square
    wave at the run's measured slow/fast plateau levels (centered the
    same way as `force`). When non-empty it is the UI's preferred dashed
    reference: overlaying it on `force` makes over/undershoot visible at
    a glance. When empty (no pos_x, no transitions, or baseline plateau
    medians weren't computable), the UI falls back to plotting `command`
    on a secondary mm/s axis.
    """
    k: float
    t: list[float]
    force: list[float]
    command: list[float]
    ground_truth_force: list[float] = field(default_factory=list)
    # Sample-grade dropout markers: timestamps (sweep-relative seconds)
    # of force samples that arrived after a gap large enough to be a
    # missing-sample / UDP packet-loss. Detected as |force[i] −
    # force[i−1]| > 0.5·plateau_delta where plateau_delta is the
    # window's own (P90 − P10) spread. Plotting these as red markers
    # on the per-K trace lets the user see exactly where the loadcell
    # stream went sparse. When the fraction of dropouts exceeds
    # ~5%, the K is excluded from the K_opt fits.
    dropout_t: list[float] = field(default_factory=list)


@dataclass(slots=True)
class FitResult:
    k_opt: float
    slope: float
    intercept: float
    r_squared: float
    method: str  # "phase_lag" or "integral"


@dataclass(slots=True)
class Baseline:
    """Loadcell zero captured during the held-idle dwell after heat-up,
    before any extrusion. Used as a diagnostic (drift / noise floor) and as
    a reference line in the per-K segment plots.
    """
    mean: float
    std: float
    drift: float  # signed mean-of-second-half minus mean-of-first-half
    n_samples: int
    t_start: float
    t_end: float


@dataclass(slots=True)
class ForceBaselines:
    """Steady-state loadcell readings during the slow and fast plateaus,
    extracted from the actual sweep data and used as the per-K plot's
    ground-truth reference.

    Each cycle has a settle margin (we skip the first ~150 ms after every
    transition, where the loadcell is still responding to the velocity
    change); the remaining samples are the plateau itself. The MEDIAN of
    plateau samples across all detected cycles is robust to outliers
    (e.g. the single 12 k loadcell glitch the user saw in one K=0.08
    plot) and converges on the true steady-state response at slow_v and
    fast_v respectively. With these in hand, the per-K plots overlay a
    square wave at exactly (slow_plateau, fast_plateau) -- if PA is
    perfect, the force trace snaps cleanly between those two levels;
    over-PA exceeds them, under-PA falls short.
    """
    slow_plateau: float
    fast_plateau: float
    n_slow: int
    n_fast: int


# bd_pressure-style per-segment metrics. Each low-high-low segment in the
# burst gets all 12 of these numbers measured against eight conceptual
# regions of the step response (region map mirrors the bd_pressure
# reference image: R1 low baseline, R2 rising edge, R3 overshoot spike,
# R4 high plateau, R5 plateau slope, R6 falling edge, R7 undershoot,
# R8 recovery tail). The composite cost is built from a subset of these
# (rise_error_area, overshoot, undershoot, tail_area, plateau_slope) with
# user-tunable weights, but every metric is exposed so the per-segment
# debug UI can show what the analyser saw.
BD_METRIC_NAMES: tuple[str, ...] = (
    "baseline_median",
    "baseline_noise_std",
    "rise_delay",
    "rise_error_area",
    "rise_signed_area",
    "rise_slope",
    "overshoot",
    "high_level",
    "plateau_slope",
    "plateau_creep",
    "fall_delay",
    "fall_error_area",
    "fall_signed_area",
    "undershoot",
    "tail_area",
    "settling_time",
    # Per-segment estimate of the OPTIMAL K, from first-order theory:
    # the fall response to a velocity step with mistuned PA is an
    # exponential with residual time constant τ_res ≈ (K_true − K_set),
    # and its signed area over the fall window is amplitude · τ_res.
    # So K_true ≈ K_set + fall_signed_area / (fast−slow plateau delta).
    # Every segment at every K yields an independent estimate of the
    # answer; the sweep-wide median (bd_k_from_segments) uses ALL the
    # data instead of only the neighbourhood of a zero crossing.
    "k_estimate",
)

# Default weights for the composite cost. Each metric is normalised to
# its sweep-wide max before this weighting, so the units don't need to
# match. Weights are also shipped to the UI as the initial slider
# positions; the JS recomputes cost + K_opt as the user drags.
#
# The mix of "area" and "delay" metrics is intentional:
#   * AREA metrics (rise_error_area, overshoot, undershoot, tail_area,
#     plateau_slope) measure HOW BIG the response deviation is. They're
#     small near K_opt and grow on EITHER side. They identify the
#     bottom of the valley.
#   * DELAY metrics (rise_delay, fall_delay, settling_time) measure HOW
#     LONG the response takes. They form a CLEAN STEP: high at low K,
#     drop to a floor at K_opt, stay flat (or barely rise) past K_opt.
#     They identify the LEFT edge of the valley -- where the response
#     first becomes fast enough.
# Combining both gives a tighter cost minimum than either alone:
# overshoot/undershoot push K_opt down (penalise too-high K), the
# delays push it up (penalise too-low K), and the minimum lands at
# the elbow where both are low.
BD_DEFAULT_WEIGHTS: dict[str, float] = {
    "rise_error_area": 1.0,
    "overshoot": 2.0,
    "undershoot": 2.0,
    "tail_area": 1.0,
    "plateau_slope": 0.5,
    # Delay/timing metrics added 2026-05 after the user's
    # run_1779015193 showed these were the CLEANEST step responses in
    # the metric grid -- much less segment-to-segment noise than the
    # absolute overshoot/undershoot magnitudes -- and reinforce the
    # cost valley from the "K-too-low" side.
    "rise_delay": 1.0,
    "fall_delay": 1.0,
    "settling_time": 0.5,
}


@dataclass(slots=True)
class BdSegment:
    """One low-high-low step response (low_n + high_n + low_{n+1}).

    `t_start`/`t_rise`/`t_fall`/`t_end` are sweep-relative seconds and
    define the four boundaries the regions live inside. `t_peak` /
    `t_trough` are the feature-detected times of the post-rise max and
    post-fall min (regions R3 / R7); when no detectable extremum was
    found they are None and the corresponding overshoot / undershoot
    metric is NaN.

    `metrics` is a flat dict keyed by `BD_METRIC_NAMES`; values may be NaN
    when the underlying window had too few samples or no detectable
    feature. NaNs flow through `np.nanmedian` in the K aggregation.

    `excluded` + `exclusion_reasons` are set by the auto-quality gate.
    Excluded segments are skipped during median aggregation but still
    surfaced in the per-segment UI with a banner explaining why.
    """
    k: float
    seg_idx: int                          # 0..cycles_per_K-1
    t_start: float                        # start of low_n (sweep-rel)
    t_rise: float                         # detected slow→fast transition
    t_fall: float                         # detected fast→slow transition
    t_end: float                          # end of low_{n+1} (sweep-rel)
    # Display-only crop, inset from [t_start, t_end] by ~10% of slow_half
    # on each side. The segment shares boundaries with its neighbours
    # (`t_start` is the previous segment's `t_end - slow_half`, and
    # `t_end` is the next segment's `t_start + slow_half`), and at the
    # boundary a single firmware-throttle gap can drop a sample whose
    # neighbour is already in the NEXT cycle's fast leg -- plotly then
    # draws a vertical "rising edge" right at the segment edge. Cropping
    # the display by a small margin hides those edge-of-cycle artifacts.
    # Metrics are still computed on the FULL [t_start, t_end] window.
    t_lo_display: float = 0.0             # display crop start (sweep-rel)
    t_hi_display: float = 0.0             # display crop end (sweep-rel)
    # Rise-completion and fall-completion times: where the force first
    # crosses the 90% / 10% level threshold relative to (high_level −
    # baseline_median). These delimit R2/R4 and R6/R8 robustly even
    # when the high plateau is creeping upward (so that the legacy
    # argmax-based t_peak is way to the right and would otherwise
    # swallow most of the plateau into R2). When the crossing isn't
    # found within a sane window the value is None and the UI falls
    # back to the argmax/argmin t_peak/t_trough.
    t_rise_end: float | None = None
    # t_fall_start: time the force ACTUALLY begins falling (first
    # sustained drop below the 90% threshold, detected from data).
    # Often ~20–40 ms earlier than the commanded `t_fall` because of
    # PA lag. R4 plateau ends here, R6 fall begins here, and the
    # rise_error / fall_error windows split here -- so the early-fall
    # transient doesn't contaminate plateau-region metrics.
    t_fall_start: float | None = None
    t_fall_end: float | None = None
    t_peak: float | None = None           # R3 marker (overshoot location)
    t_trough: float | None = None         # R7 marker (undershoot location)
    n_samples: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    excluded: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BdKResult:
    """Per-K aggregate of the BdSegment metrics.

    `medians` holds the np.nanmedian over INCLUDED segments only, one
    entry per `BD_METRIC_NAMES`. `n_segments_total` is the count built
    for this K (= cycles_per_K) and `n_segments_included` is how many
    passed the auto-quality gate. When `n_segments_included < 4` the
    K is treated as too unreliable for the K_opt search and is
    excluded from `bd_k_opt` (its medians still ship to the UI for
    inspection, just flagged).

    `normalised` is `medians[m] / max(|medians[m]|)` across the sweep —
    used so the composite weighted sum is dimensionally consistent.
    Computed by `_bd_compute_normalised` after all per-K medians are in
    hand.
    """
    k: float
    n_segments_total: int
    n_segments_included: int
    medians: dict[str, float] = field(default_factory=dict)
    normalised: dict[str, float] = field(default_factory=dict)
    # Spread of each metric across INCLUDED segments at this K. Used by
    # the UI to draw error bars on the per-metric K-vs-value plots so
    # the user can tell which metrics are reliable (tight bar = low
    # segment-to-segment variance) from which are noise-dominated
    # (huge bar = the median is meaningless). `mads` = median absolute
    # deviation from the median (×1.4826 ≈ σ-equivalent for normal
    # data, but robust to outliers). `iqrs` = inter-quartile range
    # (75th − 25th percentile). Both ship; the UI uses MAD by default.
    mads: dict[str, float] = field(default_factory=dict)
    iqrs: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SweepAnalysis:
    per_k: list[KResult]
    phase_fit: FitResult | None
    integral_fit: FitResult | None
    integral_legacy_fit: FitResult | None
    sample_rate_hz: float
    baseline: Baseline | None = None
    notes: list[str] = field(default_factory=list)
    windows: list[KWindow] = field(default_factory=list)
    force_baselines: ForceBaselines | None = None
    # bd_pressure step-response analysis: one BdSegment per (K, cycle),
    # one BdKResult per K (medians + normalised), plus the composite
    # cost's argmin K. The default weights are shipped so the UI can
    # pre-fill the sliders; the K_opt here is computed with those
    # defaults but the UI recomputes live as the user adjusts.
    bd_segments: list[BdSegment] = field(default_factory=list)
    bd_per_k: list[BdKResult] = field(default_factory=list)
    bd_k_opt: float | None = None
    # Signed-area zero-crossing estimators (overshoot↔lag discriminator).
    # Fit the per-K median of {rise,fall}_signed_area vs K and solve for
    # area = 0. Independent of the composite-cost argmin (`bd_k_opt`) — a
    # cross-check that exploits the SIGN of the step-response asymmetry
    # rather than its magnitude. `bd_signed_fall_fit` is the cleaner of the
    # two (no rise-transient bias); see the per-segment metric notes.
    bd_signed_rise_fit: FitResult | None = None
    bd_signed_fall_fit: FitResult | None = None
    # Sweep-wide median of the per-segment k_estimate metric (see
    # BD_METRIC_NAMES): every included segment independently estimates
    # K_true = K_set + fall_signed_area / amplitude, so this estimator
    # uses ALL the data rather than only the zero-crossing
    # neighbourhood. `_mad` is the 1.4826-scaled median absolute
    # deviation (σ-equivalent) across those estimates; `_n` is how many
    # segments contributed.
    bd_k_from_segments: float | None = None
    bd_k_from_segments_mad: float | None = None
    bd_k_from_segments_n: int = 0
    bd_default_weights: dict[str, float] = field(
        default_factory=lambda: dict(BD_DEFAULT_WEIGHTS)
    )


def _resample_uniform(
    t: np.ndarray, y: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate (t, y) onto a uniform grid with spacing dt."""
    if len(t) < 2:
        return np.array([]), np.array([])
    t0 = t[0]
    t1 = t[-1]
    n = max(2, int(np.floor((t1 - t0) / dt)) + 1)
    grid = t0 + np.arange(n) * dt
    return grid, np.interp(grid, t, y)


_re = __import__("re")
# Match a G1 line and pull out the E target and (separately) the F feedrate.
# Two independent searches so the order of fields in the gcode line doesn't
# matter and "other axis fields" (X, Y, Z, A...) don't interfere with parsing
# E or F. We deliberately do NOT consume them in a single pattern -- the old
# combined regex included F in the "skip other axes" group and swallowed it
# before the F-capture could match, returning F=None for every line.
_RE_G1_LINE = _re.compile(r"^\s*G[01]\b", _re.IGNORECASE)
_RE_E_FIELD = _re.compile(r"(?:^|\s)E([-+]?\d+(?:\.\d+)?)", _re.IGNORECASE)
_RE_F_FIELD = _re.compile(r"(?:^|\s)F(\d+(?:\.\d+)?)", _re.IGNORECASE)


def _parse_g1_e_f(line: str) -> tuple[float | None, float | None] | None:
    """Return (E_target, F_mm_per_min) for a G1/G0 line, or None if not G0/1."""
    if not _RE_G1_LINE.match(line):
        return None
    e_m = _RE_E_FIELD.search(line)
    f_m = _RE_F_FIELD.search(line)
    e_val: float | None = None
    f_val: float | None = None
    if e_m is not None:
        try: e_val = float(e_m.group(1))
        except (ValueError, TypeError): pass
    if f_m is not None:
        try: f_val = float(f_m.group(1))
        except (ValueError, TypeError): pass
    if e_val is None and f_val is None:
        return None
    return (e_val, f_val)


# Legacy alias kept for any callers still doing _G1E_RE.match() -- not used
# internally anymore.
_G1E_RE = _re.compile(
    r"^\s*G[01]\b.*?E([-+]?\d+(?:\.\d+)?).*?(?:F(\d+(?:\.\d+)?))?",
    _re.IGNORECASE,
)


def _command_wave_from_gcode(
    t_grid: np.ndarray,
    gcode_t: np.ndarray,
    gcode_lines: list[str],
) -> np.ndarray:
    """Ground-truth commanded extrusion velocity, parsed from the firmware's
    `gcode` metric stream (one STRING per processed gcode line, timestamped
    at the moment of execution).

    A `gcode` event fires when the firmware STARTS executing a line. At
    that moment E is at the TARGET of the previous G1 E (the move that
    just ended), and the firmware begins moving toward the new target. So
    during the interval [t_i, t_{i+1}], E travels from `events[i-1].target`
    to `events[i].target`, and the average extrusion velocity over the
    interval is `(e_i - e_{i-1}) / (t_{i+1} - t_i)`.

    We deliberately do NOT derive velocity from the gcode `F` field. For
    pure-E moves F is the extrusion feedrate (mm/min), but for composite
    XYZ+E moves Marlin interprets F as the XYZ trajectory velocity and
    slaves E to fit the same duration. The timestamp-based formula above
    is firmware-agnostic: it gives the correct E velocity for both cases
    without needing to know which axes participated.

    The first interval (i=0) has no `events[-1]`, so we seed e_prev=0 -- the
    sweep gcode always starts with a fresh `G92 E0` before any G1 E, so
    that's the actual pre-state.
    """
    if not gcode_lines or len(gcode_t) != len(gcode_lines):
        return np.array([])

    # Extract (timestamp, target_E) for every G1 line that touches E. F is
    # not used (see docstring), but we still rely on _parse_g1_e_f to
    # screen out non-G0/G1 lines.
    events: list[tuple[float, float]] = []
    for t, ln in zip(gcode_t, gcode_lines):
        parsed = _parse_g1_e_f(ln)
        if parsed is None:
            continue
        e_target, _f = parsed
        if e_target is None:
            continue  # pure travel, no extruder motion
        events.append((float(t), float(e_target)))
    if len(events) < 2:
        return np.array([])

    out = np.zeros_like(t_grid)
    e_prev = 0.0  # G92 E0 always precedes the first burst
    for i in range(len(events)):
        t_start, e_target = events[i]
        # End of this move = start of next move (or a tiny epsilon if
        # we're already at the last event).
        t_end = events[i + 1][0] if i + 1 < len(events) else t_start + 1e-3
        duration = t_end - t_start
        if duration > 1e-6:
            v = (e_target - e_prev) / duration  # mm/s
            mask = (t_grid >= t_start) & (t_grid < t_end)
            out[mask] = v
        e_prev = e_target
    return out


def _build_command_wave(
    seg: KSegment,
    t_grid: np.ndarray,
    slow_v: float,
    fast_v: float,
    slow_half_s: float,
    accel_mm_s2: float = 5000.0,
    burst_start_override: float | None = None,
) -> np.ndarray:
    """Reconstruct the commanded extrusion velocity at each timestamp in `t_grid`.

    Assumes the burst starts at seg.start_offset_s (relative to the sweep-start marker)
    and consists of seg.cycles cycles of (slow_half_s, cycle_period_s - slow_half_s).

    The transition between slow and fast is NOT an instantaneous step --
    the firmware accel-limits every velocity change. With accel=5000 mm/s^2
    and slow_v=0.8, fast_v=8.0, the transition takes (8.0 - 0.8) / 5000 =
    1.44 ms. Modeling that ramp makes the command wave match what the
    printer actually executes; without it, the analyzer reads ~half a
    ramp-duration of phantom lag at every transition. The earlier 200
    mm/s² default produced a 36 ms ramp that the analyser had to model
    explicitly; the new ramp is so short it's effectively a step at our
    1 kHz analysis grid, but we keep the ramp logic for correctness in
    case a user dials accel back down.

    Asymmetric halves matter too: our cycles are 1.0 s slow + 0.25 s fast,
    NOT 50/50. Earlier versions split the cycle in half and produced ~375 ms
    of fake lag in every K-vs-K comparison.
    """
    out = np.zeros_like(t_grid)
    burst_start = (
        seg.start_offset_s if burst_start_override is None else float(burst_start_override)
    )
    burst_end = burst_start + seg.duration_s
    period = seg.cycle_period_s
    fast_half_s = period - slow_half_s
    # Accel-limited transition duration -- same for slow->fast and fast->slow.
    ramp_s = max(0.0, abs(fast_v - slow_v) / max(accel_mm_s2, 1e-6))
    # If the leg is shorter than the ramp the firmware never reaches steady
    # state -- still apply the ramp, clipped to the leg duration.
    ramp_s = min(ramp_s, slow_half_s * 0.5, fast_half_s * 0.5)

    for i, t in enumerate(t_grid):
        if t < burst_start or t >= burst_end:
            continue
        phase = (t - burst_start) % period
        if phase < slow_half_s:
            # Inside the slow leg; check whether we're still ramping from
            # the previous fast leg.
            if phase < ramp_s and burst_start + (t - burst_start) > burst_start:
                # ramp from fast_v down to slow_v
                frac = phase / ramp_s if ramp_s > 0 else 1.0
                out[i] = fast_v + (slow_v - fast_v) * frac
            else:
                out[i] = slow_v
        else:
            # Inside the fast leg; check whether we're still ramping up.
            into_fast = phase - slow_half_s
            if into_fast < ramp_s:
                frac = into_fast / ramp_s if ramp_s > 0 else 1.0
                out[i] = slow_v + (fast_v - slow_v) * frac
            else:
                out[i] = fast_v
    return out


def _detrend(y: np.ndarray) -> np.ndarray:
    if len(y) < 4:
        return y - np.mean(y) if len(y) else y
    return sps.detrend(y, type="linear")


def _parabolic_peak(corr: np.ndarray, lags: np.ndarray) -> float:
    """Sub-sample interpolation of the cross-correlation peak.

    Fits a parabola to the peak and its two neighbours, returns the interpolated lag.
    """
    if len(corr) < 3:
        return 0.0
    i = int(np.argmax(corr))
    if i == 0 or i == len(corr) - 1:
        return float(lags[i])
    y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(lags[i])
    delta = 0.5 * (y0 - y2) / denom
    return float(lags[i] + delta * (lags[1] - lags[0]))


def _phase_lag_ms(
    force: np.ndarray, command: np.ndarray, dt: float, max_lag_s: float = 1.0
) -> float:
    """Cross-correlate force vs command, return peak lag in milliseconds.

    Positive lag => force lags the command (over-PA, K too high in some sign conventions;
    we just present the raw lag and let the linear fit pick zero-crossing).

    Returns NaN when the cross-correlation peak hits the ±max_lag boundary --
    that's a sign the search range was too narrow or the signals don't share
    a peak inside it (typical when the cycle period >= max_lag, which makes
    the correlation alias to the next cycle and pin against the boundary).
    Letting boundary-saturated values into the linear fit produces the
    non-monotonic K-vs-lag scatter the user observed.
    """
    if len(force) < 8 or len(command) < 8:
        return float("nan")
    f = _detrend(force)
    c = _detrend(command)
    # Normalise to make r^2 ~comparable across K values
    f = f / (np.std(f) + 1e-12)
    c = c / (np.std(c) + 1e-12)
    corr = sps.correlate(f, c, mode="full")
    lags_idx = sps.correlation_lags(len(f), len(c), mode="full")
    lags_s = lags_idx * dt
    mask = np.abs(lags_s) <= max_lag_s
    if not mask.any():
        return float("nan")
    masked_corr = corr[mask]
    masked_lags = lags_s[mask]
    peak_i = int(np.argmax(masked_corr))
    # Reject boundary hits: they're aliased or noise.
    if peak_i == 0 or peak_i == len(masked_corr) - 1:
        return float("nan")
    peak_lag_s = _parabolic_peak(masked_corr, masked_lags)
    return peak_lag_s * 1000.0


def _integral_area_legacy(force: np.ndarray, command: np.ndarray) -> float:
    """Legacy: integrate (F - mean) * sign(dCommand/dt) over the whole window.

    Equivalent to the original Snapmaker-style formulation. Kept around so the
    UI can plot it side-by-side with the centered-window form -- on real hardware
    this version's "sign window" is just the ~36 ms accel ramp at each transition
    (well under the melt-pressure τ), so it's dominated by an intrinsic-τ
    baseline and barely moves with K. The new `_integral_area` should show a
    visibly steeper slope vs K and a less wild zero-crossing.
    """
    if len(force) < 4 or len(command) < 4:
        return float("nan")
    f = force - np.mean(force)
    dc = np.diff(command, prepend=command[0])
    sign = np.sign(dc)
    return float(np.sum(f * sign))


def _integral_area(
    force: np.ndarray,
    command: np.ndarray,
    dt: float,
    window_s: float | None = None,
    transition_idx: np.ndarray | None = None,
    directions: np.ndarray | None = None,
) -> float:
    """Snapmaker-style PA error, integrated over a window CENTERED on each
    velocity transition.

    For each slow→fast (direction +1) and fast→slow (direction −1) crossing
    of the command's mid-level, integrate (F − local_mean) over half_win
    samples on either side. Each cycle uses its OWN local mean (over the
    [t_rise-hw, t_fall+hw] span around that cycle's two transitions), so
    slow loadcell drift across a 17-min sweep doesn't bias the per-K
    metric. Two windows that bracket the same transition cannot overlap:
    each is clipped at the midpoint between adjacent transitions.

    Why centered, not forward-only:

    - Forward-only windows make leads (force-leads-command, K < K_opt) and
      lags (K > K_opt) asymmetric: a small lead "costs" nothing (window
      stays inside the post-transition plateau) while a matching lag eats
      window time at the start. The fit slope then biases K_opt toward
      higher K. Centered windows are symmetric in lead/lag and produce a
      clean zero-crossing at K_opt.

    Why a NARROW window (≈ melt-pressure τ, not "as wide as fits"):

    - The U1 reference cycles 1.0 s slow + 0.25 s fast, so 0.5·fast_half
      naturally caps the U1 window at ~125 ms half-width. Their algorithm
      is implicitly tuned for that physical scale: ~1-2 melt-pressure
      time-constants, dominated by the transient response.
    - On our 2.0 s + 1.0 s cycle, "0.5·fast_half" is 500 ms — 4× too wide.
      A 600 ms total window is filled mostly with plateau samples (the
      transient is over by ~150 ms), so the metric reads "average plateau
      bias", drifts with loadcell zero, and reports K_opt ~30-50% too high.
      Replaying run_1779016571 with a 300 ms cap: K_opt=0.061; with a
      150 ms cap: K_opt=0.057 — much closer to the bd_pressure
      overshoot/undershoot crossover (~0.051), the user's expected range,
      and less sensitive to high-K saturation tails.
    - Setting the cap at 150 ms ties the window to physics (one to two
      hotend τ) instead of geometry, so the metric behaves the same way
      on U1-style short cycles and our longer cycles.

    `window_s` defaults to whichever is smaller: 150 ms or half the
    shortest leg duration (so windows never bleed into adjacent
    transitions).

    `transition_idx` / `directions` (optional): when supplied, these are
    used directly instead of mid-level crossings of the command wave. The
    intended supplier is pos_x velocity sign-flips -- the printer's actual
    leg-transition timestamps, which beat the model command wave's
    transitions because the model assumes perfect periodicity from the
    burst start while reality has per-cycle planner / accel-limit jitter.
    Direction convention: +1 = slow→fast (E rises, pos_x peak), -1 =
    fast→slow (E falls, pos_x trough).
    """
    n = len(force)
    if n < 4 or len(command) < 4:
        return float("nan")

    if (
        transition_idx is not None
        and directions is not None
        and len(transition_idx) >= 2
    ):
        transition_idx = np.asarray(transition_idx, dtype=int)
        directions = np.asarray(directions, dtype=np.float64)
    else:
        # Fallback: mid-level crossings of the command wave (robust to
        # whatever ramp shape the firmware actually uses).
        c_lo = float(np.percentile(command, 10))
        c_hi = float(np.percentile(command, 90))
        if c_hi - c_lo < 1e-6:
            return float("nan")
        c_mid = 0.5 * (c_lo + c_hi)
        above = command > c_mid
        crossings = np.diff(above.astype(np.int8))
        transition_idx = np.where(crossings != 0)[0] + 1
        if len(transition_idx) < 2:
            return float("nan")
        directions = crossings[transition_idx - 1].astype(np.float64)

    # Auto-pick the window: cap at 150 ms (~one melt-pressure τ at typical
    # hotend response — see docstring), and never wider than half the
    # shortest gap between transitions so the ±half_win at one transition
    # can't overlap the next transition's window.
    if window_s is None:
        gaps = np.diff(transition_idx)
        min_leg_s = float(gaps.min()) * dt if len(gaps) else 0.0
        if min_leg_s <= 0:
            return float("nan")
        window_s = min(0.15, 0.5 * min_leg_s)
    half_win = max(1, int(round(0.5 * window_s / max(dt, 1e-9))))

    # Pair adjacent transitions into cycles (slow→fast paired with the
    # following fast→slow, and vice versa). For each pair, compute a
    # LOCAL mean over the cycle's combined window [start_pair, end_pair]
    # so loadcell zero drift across the K window doesn't bias the
    # per-cycle area. Unpaired transitions at the head/tail of the
    # transition list still get processed with their own ±half_win mean.
    total = 0.0
    i = 0
    while i < len(transition_idx):
        idx_a = int(transition_idx[i])
        start_a = max(0, idx_a - half_win)
        end_a = min(n, idx_a + half_win)
        if i > 0:
            mid_prev = (int(transition_idx[i - 1]) + idx_a) // 2
            start_a = max(start_a, mid_prev)
        if i + 1 < len(transition_idx):
            mid_nxt_a = (idx_a + int(transition_idx[i + 1])) // 2
            end_a = min(end_a, mid_nxt_a)

        has_pair = i + 1 < len(transition_idx)
        if has_pair:
            idx_b = int(transition_idx[i + 1])
            start_b = max(0, idx_b - half_win)
            end_b = min(n, idx_b + half_win)
            mid_between = (idx_a + idx_b) // 2
            start_b = max(start_b, mid_between)
            if i + 2 < len(transition_idx):
                mid_nxt_b = (idx_b + int(transition_idx[i + 2])) // 2
                end_b = min(end_b, mid_nxt_b)

            local_lo = min(start_a, start_b)
            local_hi = max(end_a, end_b)
            if local_hi - local_lo >= 2:
                local_mean = float(np.mean(force[local_lo:local_hi]))
                if end_a > start_a:
                    total += (
                        float(directions[i])
                        * float(np.sum(force[start_a:end_a] - local_mean))
                        * dt
                    )
                if end_b > start_b:
                    total += (
                        float(directions[i + 1])
                        * float(np.sum(force[start_b:end_b] - local_mean))
                        * dt
                    )
            i += 2
        else:
            if end_a - start_a >= 2:
                local_mean = float(np.mean(force[start_a:end_a]))
                total += (
                    float(directions[i])
                    * float(np.sum(force[start_a:end_a] - local_mean))
                    * dt
                )
            i += 1
    return total


def _integral_area_from_segments(
    t_rel: np.ndarray,
    force_y: np.ndarray,
    segments: "list[BdSegment]",
    half_win_s: float,
) -> tuple[float, int, int]:
    """U1/Snapmaker-style integral-area, but consuming the same BdSegment
    cycles the bd_pressure analysis uses — so the auto-exclusion gate is
    a single source of truth.

    `_bd_segment_metrics` already flags each (rise, fall) cycle with
    `excluded=True` when the cycle has a dropout in the critical zone,
    a sample rate below 40 Hz, signal-below-noise, etc. Until now the
    integral_area summed over every pos_x sign-flip blindly, so a
    cycle the user could see was junk on the bd plot still contributed
    to the linear fit's zero-crossing. Routing through here, the two
    algorithms agree on which cycles are good:

      * bd_pressure aggregates metrics over INCLUDED segments;
      * integral_area integrates force over the SAME included segments.

    For each kept cycle, integrate `(F − local_mean)` over
    `[t_rise − hw, t_rise + hw]` (direction +1) and
    `[t_fall − hw, t_fall + hw]` (direction −1). `local_mean` is taken
    over the cycle's combined window `[t_rise − hw, t_fall + hw]` so
    slow loadcell drift across the sweep doesn't bias the per-cycle
    contribution. Sums are accumulated with signs.

    Returns `(total_signed_area, n_included, n_total)`.
    """
    n_tot = len(segments)
    if half_win_s <= 0 or n_tot == 0:
        return float("nan"), 0, n_tot
    t_arr = np.asarray(t_rel, dtype=float)
    y_arr = np.asarray(force_y, dtype=float)
    total = 0.0
    n_inc = 0
    for seg in segments:
        if seg.excluded:
            continue
        t_r = float(seg.t_rise)
        t_f = float(seg.t_fall)
        if not np.isfinite(t_r) or not np.isfinite(t_f) or t_f <= t_r:
            continue
        m_pair = (t_arr >= t_r - half_win_s) & (t_arr <= t_f + half_win_s)
        m_sf = (t_arr >= t_r - half_win_s) & (t_arr <= t_r + half_win_s)
        m_fs = (t_arr >= t_f - half_win_s) & (t_arr <= t_f + half_win_s)
        if int(m_pair.sum()) < 4 or int(m_sf.sum()) < 2 or int(m_fs.sum()) < 2:
            continue
        local_mean = float(np.mean(y_arr[m_pair]))
        a_sf = float(np.trapezoid(y_arr[m_sf] - local_mean, t_arr[m_sf]))
        a_fs = float(np.trapezoid(y_arr[m_fs] - local_mean, t_arr[m_fs]))
        total += a_sf - a_fs
        n_inc += 1
    if n_inc == 0:
        return float("nan"), 0, n_tot
    return float(total), n_inc, n_tot


def _nan_metrics() -> dict[str, float]:
    """All 14 BD metrics filled with NaN. Used when a segment can't be
    measured (no peak, no trough, too few samples). NaNs propagate cleanly
    through the per-K nanmedian aggregation."""
    return {name: float("nan") for name in BD_METRIC_NAMES}


def _bd_segment_metrics(
    force_t: np.ndarray,
    force_y: np.ndarray,
    k: float,
    seg_idx: int,
    t_start: float,
    t_rise: float,
    t_fall: float,
    t_end: float,
    slow_half_s: float,
    fast_half_s: float,
    dropout_t: np.ndarray,
) -> BdSegment:
    """Compute the 12 region metrics for one low-high-low segment.

    Region boundaries follow the Q5 hybrid spec:

      - R1 baseline window  = `[t_start + 0.15·slow_half, t_rise - 0.05·slow_half]`
      - t_peak              = argmax(force[t_rise : t_rise + 0.5·fast_half])
      - R2 rising window    = `[t_rise, t_peak]`
      - R3 overshoot        = force[t_peak] − high_level
      - R4 plateau window   = `[t_peak + min(50ms, 0.2·fast_half), t_fall - 0.02·fast_half]`
      - R5 slope            = linregress slope over the plateau window
      - t_trough            = argmin(force[t_fall : t_fall + 0.5·slow_half])
      - R6 falling window   = `[t_fall, t_trough]`
      - R7 undershoot       = baseline_median − force[t_trough]
      - R8 recovery window  = `[t_trough + 0.1·slow_half, t_end - 0.05·slow_half]`

    All metrics are computed AFTER per-segment tare: subtract the R1 median
    from the whole segment's force trace so overshoot / undershoot / plateau
    levels are read against a local zero. This neutralises the slow head-load
    drift over a ~17 min sweep.

    Auto-exclusion (any single match excludes the segment from K's median):
      - any element of `dropout_t` falls inside `[t_start, t_end]`
      - actual sample count < 60% of expected (60Hz nominal)
      - `high_level - baseline_median < 3 * baseline_noise_std`
      - `baseline_low - min(R6 ∪ R7) < 3 * baseline_noise_std`  (no detectable fall)
      - `baseline_noise_std > 0.30 * (high_level - baseline_median)`
    """
    metrics = _nan_metrics()
    excluded = False
    reasons: list[str] = []

    # Display crop (sweep-rel). Inset by ~10% of slow_half on each side
    # so the plot doesn't show the shared boundary samples that can
    # bridge into the next cycle's fast leg via a single firmware-
    # throttle gap.
    display_margin = 0.10 * slow_half_s
    t_lo_display = t_start + display_margin
    t_hi_display = t_end - display_margin

    mask = (force_t >= t_start) & (force_t <= t_end)
    t = force_t[mask]
    y = force_y[mask]
    n = int(len(t))

    # Hard floor: with fewer than ~20 samples in a >2s window the metrics
    # are dominated by noise. The 60Hz nominal incoming rate × 2s = ~120;
    # 20 is a generous floor that still rejects obviously sparse windows.
    if n < 20:
        excluded = True
        reasons.append(f"only {n} samples in segment window")
        return BdSegment(
            k=k, seg_idx=seg_idx, t_start=t_start, t_rise=t_rise,
            t_fall=t_fall, t_end=t_end,
            t_lo_display=t_lo_display, t_hi_display=t_hi_display,
            t_rise_end=None, t_fall_start=None, t_fall_end=None,
            t_peak=None, t_trough=None,
            n_samples=n, metrics=metrics, excluded=excluded,
            exclusion_reasons=reasons,
        )

    # In-window dropout check (uses the per-K dropout_t already detected
    # upstream). Only excludes when the dropout falls in the CRITICAL
    # zone where it would actually corrupt a metric:
    # `[t_rise - 0.1·slow_half, t_fall + 0.5·slow_half]` covers the rise
    # transition, the high plateau, the fall transition, and the
    # immediate trough/early-recovery window. Dropouts in the bulk of
    # low_n or late in low_{n+1}'s recovery tail are still SHOWN as
    # red Xs on the plot but don't auto-exclude — their impact on
    # rise/overshoot/undershoot/plateau metrics is minimal.
    if len(dropout_t):
        in_seg = (dropout_t >= t_start) & (dropout_t <= t_end)
        if in_seg.any():
            crit_lo = t_rise - 0.1 * slow_half_s
            crit_hi = t_fall + 0.5 * slow_half_s
            crit_mask = in_seg & (dropout_t >= crit_lo) & (dropout_t <= crit_hi)
            if crit_mask.any():
                t_first = float(dropout_t[crit_mask][0])
                reasons.append(
                    f"dropout at t={t_first - t_start:.2f}s (segment-rel) "
                    f"in critical region"
                )
                excluded = True

    # Coverage check using the global sample rate within the window.
    seg_duration = max(t_end - t_start, 1e-9)
    seg_rate_hz = n / seg_duration
    # Expected ~60Hz on Buddy; threshold at 40Hz = ~66% of nominal.
    if seg_rate_hz < 40.0:
        reasons.append(
            f"low sample rate {seg_rate_hz:.0f}Hz (expected ≥40Hz)"
        )
        excluded = True

    # R1 baseline window: avoid the first/last slivers of low_n so the
    # tail of any prior transient doesn't contaminate the zero reference.
    t0_r1 = t_start + 0.15 * slow_half_s
    t1_r1 = t_rise - 0.05 * slow_half_s
    r1_mask = (t >= t0_r1) & (t <= t1_r1)
    if int(r1_mask.sum()) < 4:
        reasons.append("baseline window too narrow")
        excluded = True
        return BdSegment(
            k=k, seg_idx=seg_idx, t_start=t_start, t_rise=t_rise,
            t_fall=t_fall, t_end=t_end,
            t_lo_display=t_lo_display, t_hi_display=t_hi_display,
            t_rise_end=None, t_fall_start=None, t_fall_end=None,
            t_peak=None, t_trough=None,
            n_samples=n, metrics=metrics, excluded=excluded,
            exclusion_reasons=reasons,
        )
    baseline_y = y[r1_mask]
    baseline_median = float(np.median(baseline_y))
    baseline_noise_std = float(np.std(baseline_y))
    metrics["baseline_median"] = baseline_median
    metrics["baseline_noise_std"] = baseline_noise_std

    # Per-segment tare: from here on we work in (force − baseline_median),
    # so the low plateau sits at zero, high plateau at the leg delta,
    # overshoot above the plateau, undershoot below zero.
    y_tared = y - baseline_median

    # Coarse high-level estimate: median of force over the LATTER half of
    # the fast leg [t_rise + 0.5·fast_half, t_fall − 0.05·fast_half].
    # This window is far enough past the rise transition that even on a
    # creeping plateau we get a reasonable level estimate; we use this
    # ONLY to locate the rise/fall threshold crossings. The final R4
    # plateau level (`high_level`) is re-computed below on the refined
    # plateau window.
    coarse_high_lo = t_rise + 0.5 * fast_half_s
    coarse_high_hi = t_fall - 0.05 * fast_half_s
    coarse_high_mask = (t >= coarse_high_lo) & (t <= coarse_high_hi)
    coarse_high_tared = (
        float(np.median(y_tared[coarse_high_mask]))
        if int(coarse_high_mask.sum()) >= 3
        else float("nan")
    )

    # Threshold-based rise / fall completion. The "rise" region (R2)
    # ends when force first crosses 90% of (high − baseline). The
    # "fall" region (R6) ends when force first drops back to 10% of
    # (high − baseline). The fall START is detected the same way:
    # walk forward from the late plateau and find the first SUSTAINED
    # drop below 90% (force "leaves" the plateau). This is needed
    # because on user's run_1778962189.npz the actual force begins
    # falling ~20-40 ms before the commanded t_fall (PA-lag), so the
    # last samples before t_fall are already in the fall transient
    # and were corrupting both R4 (plateau) and rise_error_area
    # (which integrates up to t_fall). Detecting t_fall_start from
    # the force trace itself lets the regions track the actual
    # response, not the commanded timing.
    RISE_COMPLETION_FRAC = 0.90
    FALL_COMPLETION_FRAC = 0.10
    FALL_SUSTAIN_S = 0.05  # need 50ms below 90% to confirm the actual fall
    t_rise_end: float | None = None
    t_fall_start: float | None = None
    t_fall_end: float | None = None
    if np.isfinite(coarse_high_tared) and coarse_high_tared > 0:
        rise_thr = RISE_COMPLETION_FRAC * coarse_high_tared
        fall_thr = FALL_COMPLETION_FRAC * coarse_high_tared
        # R2 end: walk forward from t_rise, capped at 50% into fast_half.
        rise_window = np.where(
            (t >= t_rise) & (t <= t_rise + 0.5 * fast_half_s)
        )[0]
        for idx in rise_window:
            if y_tared[idx] >= rise_thr:
                t_rise_end = float(t[idx])
                break
        # t_fall_start: walk forward in a window centred on the
        # commanded t_fall. We start the search well into the plateau
        # (so transient overshoot dips below 90% don't trigger) and end
        # a small grace past t_fall (so a slightly-early actual fall
        # is still caught). Require a SUSTAINED drop -- the next
        # FALL_SUSTAIN_S of samples must also stay below rise_thr.
        # Without sustain-check, a brief overshoot return below 90%
        # during early plateau would mis-fire as fall start.
        fall_start_search_lo = max(
            t_rise_end + 0.05 if t_rise_end is not None
            else t_rise + 0.3 * fast_half_s,
            t_fall - 0.3 * fast_half_s,
        )
        fall_start_search_hi = t_fall + 0.1 * fast_half_s
        fall_start_window = np.where(
            (t >= fall_start_search_lo) & (t <= fall_start_search_hi)
        )[0]
        for i_idx, idx in enumerate(fall_start_window):
            if y_tared[idx] >= rise_thr:
                continue
            # Check sustained-below: next ~50ms must stay below rise_thr.
            sustain_end_t = float(t[idx]) + FALL_SUSTAIN_S
            future_mask = (t > t[idx]) & (t <= sustain_end_t)
            future_y = y_tared[future_mask]
            if len(future_y) == 0 or not np.any(future_y >= rise_thr):
                t_fall_start = float(t[idx])
                break
        # R6 end: walk forward from t_fall_start (or t_fall if no
        # explicit fall-start was detected), capped at 50% into slow_half.
        fall_end_anchor = t_fall_start if t_fall_start is not None else t_fall
        fall_window = np.where(
            (t >= fall_end_anchor) & (t <= fall_end_anchor + 0.5 * slow_half_s)
        )[0]
        for idx in fall_window:
            if y_tared[idx] <= fall_thr:
                t_fall_end = float(t[idx])
                break

    # R3 peak (overshoot location): max within a TIGHT transient window
    # right after t_rise_end. Real PA overshoot is a fast transient that
    # decays back to the plateau within ~50-100 ms; anything past that
    # is plateau noise, NOT overshoot. The previous window
    # (t_rise_end + 30% of fast_half) was too wide -- for the user's
    # 2026-05 1-cycle/K sweep, low-K segments had argmax landing
    # 200-380 ms after t_rise (mid-plateau noise) instead of at the
    # initial transient. The tighter window of 10% of fast_half (=100 ms
    # for a 1 s fast leg) keeps the search inside the actual transient
    # while still tolerating slow PA at low K (peaks lag a bit).
    peak_search_lo = t_rise
    peak_search_hi = (
        (t_rise_end + 0.10 * fast_half_s)
        if t_rise_end is not None
        else (t_rise + 0.20 * fast_half_s)
    )
    peak_mask = (t >= peak_search_lo) & (t <= peak_search_hi)
    t_peak: float | None = None
    peak_idx_global: int | None = None
    if int(peak_mask.sum()) >= 2:
        local = np.where(peak_mask)[0]
        peak_local_off = int(np.argmax(y_tared[local]))
        peak_idx_global = int(local[peak_local_off])
        t_peak = float(t[peak_idx_global])

    # R4 plateau window. Starts at the THRESHOLD-BASED rise completion
    # (with a tiny settle margin), ends well before t_fall so the early
    # ramp-down isn't counted as plateau. When t_rise_end is missing,
    # use a generous 0.5·fast_half offset from t_rise. Falls back to
    # t_peak only if neither is available.
    plateau_settle_margin = max(0.030, 0.10 * fast_half_s)
    plateau_pre_fall_margin = max(0.020, 0.10 * fast_half_s)
    if t_rise_end is not None:
        plateau_t_lo = t_rise_end + plateau_settle_margin
    elif t_peak is not None:
        plateau_t_lo = max(t_peak, t_rise) + plateau_settle_margin
    else:
        plateau_t_lo = t_rise + 0.5 * fast_half_s
    # Plateau end: use the DETECTED fall start (where force actually
    # leaves the plateau) when available, falling back to t_fall − margin.
    # This avoids letting the start of the fall transient corrupt the
    # plateau median / slope when the actual fall begins slightly before
    # the commanded t_fall (PA-lag observed on user's 2026-05 NPZs).
    if t_fall_start is not None:
        plateau_t_hi = t_fall_start - plateau_pre_fall_margin
    else:
        plateau_t_hi = t_fall - plateau_pre_fall_margin
    plateau_mask = (t >= plateau_t_lo) & (t <= plateau_t_hi)
    n_plat = int(plateau_mask.sum())
    high_level_tared = float("nan")
    if n_plat >= 3:
        high_level_tared = float(np.median(y_tared[plateau_mask]))
        metrics["high_level"] = high_level_tared + baseline_median
        # R5: linear fit slope over the plateau window. Force units per
        # second (raw loadcell units / s).
        plateau_t = t[plateau_mask]
        plateau_y = y_tared[plateau_mask]
        if n_plat >= 4 and (plateau_t[-1] - plateau_t[0]) > 1e-6:
            slope, _intercept = np.polyfit(plateau_t, plateau_y, 1)
            metrics["plateau_slope"] = float(slope)
            metrics["plateau_creep"] = float(
                abs(slope) * (plateau_t[-1] - plateau_t[0])
            )

    # R3 overshoot: peak_value − high_level − E[max of noise], UNCLAMPED.
    #
    # Two failure modes to avoid, from opposite directions:
    #   * Plateau noise gives argmax a positive bias — a noise-only
    #     segment reports a "peak" of E[max of n samples] ≈ σ·√(2·ln n)
    #     above the plateau (user report run_1779125302: 80-180 raw
    #     units of phantom overshoot). Subtracting that EXPECTED bias
    #     centres noise-only segments near 0.
    #   * The earlier max(0, … − 2σ) clamp CENSORED the distribution:
    #     near K_opt most segments clamp to exactly 0, the per-K median
    #     of a clamped distribution is biased, and the cost valley gets
    #     a flat, noise-positioned bottom right where the argmin needs
    #     resolution. So the value stays signed — small negatives are
    #     real information ("peak was below expectation"), and the cost
    #     composition clips negatives (see _bd_compute_cost) so "no
    #     spike" isn't rewarded below zero.
    if t_peak is not None and np.isfinite(high_level_tared):
        peak_value = float(y_tared[peak_idx_global])  # type: ignore[index]
        # Plateau noise: residual std around high_level over the
        # plateau window. Fall back to baseline_noise_std when the
        # plateau window was too short for a stable std.
        plateau_noise_std = baseline_noise_std
        if n_plat >= 6:
            plateau_residual = y_tared[plateau_mask] - high_level_tared
            plateau_noise_std = float(np.std(plateau_residual))
        sigma = max(plateau_noise_std, baseline_noise_std)
        n_win = max(2, int(peak_mask.sum()))
        noise_bias = sigma * math.sqrt(2.0 * math.log(n_win))
        metrics["overshoot"] = peak_value - high_level_tared - noise_bias

    # R2 rising-edge metrics. rise_delay = time from t_rise to peak. The
    # rise_error_area integrates |target − force| from t_rise to end of
    # high_n; target = high_level once past t_peak, ramping linearly
    # from baseline (=0) to high_level over [t_rise, t_peak]. With the
    # tare in place, baseline is 0 and high_level is `high_level_tared`.
    if t_peak is not None:
        metrics["rise_delay"] = max(0.0, float(t_peak) - t_rise)
        # rise_slope = (peak - baseline=0) / (t_peak - t_rise)
        denom = max(float(t_peak) - t_rise, 1e-9)
        peak_value = float(y_tared[peak_idx_global])  # type: ignore[index]
        metrics["rise_slope"] = peak_value / denom
    # rise_error_area: integrate |high_level − force| over the high_n
    # window. At K_opt the force snaps cleanly to high_level and this
    # integral → 0; under-PA leaves a slow ramp, over-PA leaves a tall
    # spike. End the integration at the DETECTED fall start (or t_fall
    # if not detected) so the early fall transient -- which is NOT a
    # "rise tracking error" -- doesn't inflate this metric.
    if np.isfinite(high_level_tared):
        rise_window_hi = t_fall_start if t_fall_start is not None else t_fall
        rise_window_mask = (t >= t_rise) & (t <= rise_window_hi)
        if int(rise_window_mask.sum()) >= 2:
            rw_t = t[rise_window_mask]
            rw_y = y_tared[rise_window_mask]
            err = np.abs(high_level_tared - rw_y)
            metrics["rise_error_area"] = float(np.trapezoid(err, rw_t))
            # Signed twin of rise_error_area: ∫(force − high_level) over
            # the SAME window. The abs version only has a VALLEY at K_opt
            # (you must argmin it); the signed integral CROSSES ZERO at
            # K_opt, so its sign tells you which side you're on:
            #   force above the plateau (overshoot spike) → POSITIVE → over-PA
            #   force below the plateau (still ramping/lag) → NEGATIVE → under-PA
            # Caveat vs the fall twin: this window opens at t_rise, so even
            # at K_opt the unavoidable rise transient contributes a small
            # negative (sub-plateau) bias — its zero-crossing sits a touch
            # high. fall_signed_area (target = baseline) has no such bias and
            # is the preferred zero-crossing estimator.
            metrics["rise_signed_area"] = float(
                np.trapezoid(rw_y - high_level_tared, rw_t)
            )

    # R7 trough (undershoot location): min within [t_fall_start (or
    # t_fall), t_fall_end + 30% of slow_half]. Mirrors R3's bounding:
    # cap the search so a late noise dip in the recovery tail doesn't
    # get mis-attributed as undershoot.
    trough_search_lo = t_fall_start if t_fall_start is not None else t_fall
    trough_search_hi = (
        (t_fall_end + 0.30 * slow_half_s)
        if t_fall_end is not None
        else (t_fall + 0.5 * slow_half_s)
    )
    trough_mask = (t >= trough_search_lo) & (t <= trough_search_hi)
    t_trough: float | None = None
    trough_idx_global: int | None = None
    if int(trough_mask.sum()) >= 2:
        local = np.where(trough_mask)[0]
        trough_local_off = int(np.argmin(y_tared[local]))
        trough_idx_global = int(local[trough_local_off])
        t_trough = float(t[trough_idx_global])

    # R6 falling-edge: fall_delay = time from t_fall_start (actual fall
    # start, not commanded t_fall) to trough. fall_error_area integrates
    # |0 − force| over low_{n+1}, starting at the DETECTED fall start
    # so the last few samples of the high plateau (which would otherwise
    # be ~high_level above the target) don't dominate.
    fall_anchor = t_fall_start if t_fall_start is not None else t_fall
    if t_trough is not None:
        metrics["fall_delay"] = max(0.0, float(t_trough) - fall_anchor)
    fall_window_mask = (t >= fall_anchor) & (t <= t_end)
    if int(fall_window_mask.sum()) >= 2:
        fw_t = t[fall_window_mask]
        fw_y = y_tared[fall_window_mask]
        err = np.abs(fw_y)
        metrics["fall_error_area"] = float(np.trapezoid(err, fw_t))
        # Signed twin of fall_error_area: ∫(force − baseline) over the SAME
        # window (baseline = tared 0). The fall target IS the baseline, so
        # unlike the rise window there's no built-in transient deficit — the
        # integral is a clean zero-crossing discriminator:
        #   force lingers ABOVE baseline (slow pressure bleed / drool tail)
        #     → POSITIVE → under-PA (K too low)
        #   force dips BELOW baseline (retraction-like undershoot dip)
        #     → NEGATIVE → over-PA (K too high)
        # This is the metric backing bd_signed_fall_fit's K_opt.
        metrics["fall_signed_area"] = float(np.trapezoid(fw_y, fw_t))
        # k_estimate: first-order theory says the fall's signed area is
        # amplitude · (K_true − K_set) — the exponential tail of a
        # mistuned step integrates to amplitude · τ_residual, and the
        # residual advance error IS K_true − K_set. Dividing by the
        # measured leg amplitude turns every segment into an
        # independent estimate of the optimal K:
        #   drool tail (area > 0)  → K_true above this segment's K
        #   undershoot (area < 0)  → K_true below it
        if np.isfinite(high_level_tared) and high_level_tared > 0:
            metrics["k_estimate"] = (
                k + metrics["fall_signed_area"] / high_level_tared
            )

    # R7 undershoot: how far the trough sits below the baseline (tared
    # zero), minus the expected argmin-noise bias. Signed and unclamped
    # for the same censoring reason as overshoot above (the cost clips
    # negatives; the metric doesn't).
    if t_trough is not None:
        trough_value = float(y_tared[trough_idx_global])  # type: ignore[index]
        n_win = max(2, int(trough_mask.sum()))
        noise_bias = baseline_noise_std * math.sqrt(2.0 * math.log(n_win))
        metrics["undershoot"] = -trough_value - noise_bias

    # R8 recovery: integrate |force| over the recovery window. Starts
    # at t_fall_end (threshold-based, force back within 10% of plateau
    # delta) plus a small settle margin, ends just before the
    # neighbouring segment's domain. Skipping by t_fall_end (instead of
    # by t_trough) keeps the recovery window from being eaten when the
    # trough sits very early (the old logic used trough + 10% slow_half
    # as the floor, which on a slow recovery left only ~60% of R8 to
    # integrate over). Falls back to t_trough when t_fall_end is None.
    recov_settle_margin = max(0.030, 0.05 * slow_half_s)
    if t_fall_end is not None:
        recov_t_lo = t_fall_end + recov_settle_margin
    elif t_trough is not None:
        recov_t_lo = float(t_trough) + 0.10 * slow_half_s
    else:
        recov_t_lo = float("inf")
    if np.isfinite(recov_t_lo):
        recov_t_hi = t_end - 0.05 * slow_half_s
        recov_mask = (t >= recov_t_lo) & (t <= recov_t_hi)
        if int(recov_mask.sum()) >= 2:
            rt = t[recov_mask]
            ry = y_tared[recov_mask]
            metrics["tail_area"] = float(np.trapezoid(np.abs(ry), rt))
            # settling_time: walk forward from the trough (or t_fall_end
            # when no trough was found) and find the first sample whose
            # |y_tared| stays under 3·noise_std for ≥100ms.
            tol = max(3.0 * baseline_noise_std, 1e-9)
            settle_t = float("nan")
            settle_anchor = (
                float(t_trough) if t_trough is not None
                else (t_fall_end if t_fall_end is not None else t_fall)
            )
            post_mask = (t >= settle_anchor) & (t <= t_end)
            pt = t[post_mask]
            py = y_tared[post_mask]
            if len(pt) >= 2:
                ok = np.abs(py) < tol
                # Find earliest run-of-≥100ms-of-True.
                start_idx = None
                for i in range(len(pt)):
                    if not ok[i]:
                        start_idx = None
                        continue
                    if start_idx is None:
                        start_idx = i
                    if pt[i] - pt[start_idx] >= 0.10:
                        settle_t = float(pt[start_idx]) - t_fall
                        break
            metrics["settling_time"] = settle_t

    # Auto-exclusion: rise/fall amplitude vs noise floor + drift check.
    if np.isfinite(high_level_tared) and baseline_noise_std > 0:
        leg_delta = high_level_tared  # in tared units, baseline = 0
        if leg_delta < 3.0 * baseline_noise_std:
            reasons.append(
                f"rise {leg_delta:.1f} below 3·noise {3*baseline_noise_std:.1f}"
            )
            excluded = True
        # Drift gate: if the noise we measured ON the baseline window is a
        # substantial fraction of the high-low delta, the baseline isn't
        # stable enough to anchor anything.
        if leg_delta > 0 and baseline_noise_std > 0.30 * leg_delta:
            reasons.append(
                f"baseline noise {baseline_noise_std:.1f} > 30% of leg delta {leg_delta:.1f}"
            )
            excluded = True
    if t_trough is not None and baseline_noise_std > 0:
        trough_value = float(y_tared[trough_idx_global])  # type: ignore[index]
        # If the trough sits within noise of the baseline AND the rise
        # was also weak, we'd have flagged above. Here we specifically
        # catch the case where the fall amplitude is too small to even
        # detect (force didn't drop).
        if (
            np.isfinite(high_level_tared)
            and high_level_tared > 3.0 * baseline_noise_std
            and abs(trough_value) > 5.0 * high_level_tared
        ):
            reasons.append(
                f"fall to {trough_value:.1f} more than 5x leg delta — likely noise spike"
            )
            excluded = True

    return BdSegment(
        k=k, seg_idx=seg_idx,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall, t_end=t_end,
        t_lo_display=t_lo_display, t_hi_display=t_hi_display,
        t_rise_end=t_rise_end, t_fall_start=t_fall_start, t_fall_end=t_fall_end,
        t_peak=t_peak, t_trough=t_trough,
        n_samples=n, metrics=metrics,
        excluded=excluded, exclusion_reasons=reasons,
    )


def _bd_aggregate_per_k(
    segments_by_k: dict[float, list[BdSegment]],
) -> list[BdKResult]:
    """Median over included segments → per-K aggregates. Excluded segments
    are skipped; if fewer than 4 included segments remain for a K, all of
    that K's medians stay finite but the K is flagged by setting
    `n_segments_included < 4` (caller checks that to drop it from
    `bd_k_opt`).
    """
    out: list[BdKResult] = []
    for k, segs in segments_by_k.items():
        included = [s for s in segs if not s.excluded]
        medians: dict[str, float] = {}
        mads: dict[str, float] = {}
        iqrs: dict[str, float] = {}
        for name in BD_METRIC_NAMES:
            vals = [s.metrics.get(name, float("nan")) for s in included]
            if not vals:
                medians[name] = float("nan")
                mads[name] = float("nan")
                iqrs[name] = float("nan")
                continue
            arr = np.asarray(vals, dtype=float)
            if np.isnan(arr).all():
                medians[name] = float("nan")
                mads[name] = float("nan")
                iqrs[name] = float("nan")
                continue
            med = float(np.nanmedian(arr))
            medians[name] = med
            # MAD scaled to σ-equivalent for normally distributed data
            # (constant 1.4826). Robust to outliers.
            finite = arr[np.isfinite(arr)]
            if len(finite) >= 2:
                mads[name] = float(1.4826 * np.median(np.abs(finite - med)))
                q75, q25 = np.percentile(finite, [75, 25])
                iqrs[name] = float(q75 - q25)
            else:
                mads[name] = float("nan")
                iqrs[name] = float("nan")
        out.append(
            BdKResult(
                k=float(k),
                n_segments_total=len(segs),
                n_segments_included=len(included),
                medians=medians,
                mads=mads,
                iqrs=iqrs,
            )
        )
    return out


def _bd_compute_normalised(per_k: list[BdKResult]) -> None:
    """In-place: fill `normalised` on each BdKResult.

    For each metric, divide by a ROBUST sweep-wide scale — the 90th
    percentile of |per-K medians| — so the cost composition is
    dimensionally consistent. The previous scale (plain max) let a
    single outlier K rescale the metric's entire contribution to the
    composite cost, which also made optimiser-learned weights depend on
    whichever run had the worst outlier. With the p90 scale, values may
    slightly exceed ±1 (by design: outliers stick out instead of
    compressing everything else). If the scale is 0 or all values NaN,
    the metric normalises to NaN for every K.
    """
    if not per_k:
        return
    for name in BD_METRIC_NAMES:
        vals = np.asarray(
            [r.medians.get(name, float("nan")) for r in per_k], dtype=float
        )
        finite_abs = np.abs(vals[np.isfinite(vals)])
        if len(finite_abs) == 0:
            denom = 0.0
        else:
            p90 = float(np.percentile(finite_abs, 90))
            mx = float(finite_abs.max())
            # Sparse-signal guard: when a metric is ~zero at most Ks
            # (e.g. fall_delay on a snappy response), p90 collapses to
            # ~0 and would blow the division up / NaN the whole cost.
            # In that regime the robust scale carries no information --
            # fall back to the max.
            denom = p90 if p90 > 0.05 * mx else mx
        for r, v in zip(per_k, vals):
            if denom > 0 and np.isfinite(v):
                r.normalised[name] = float(v / denom)
            else:
                r.normalised[name] = float("nan")


def _bd_compute_cost(
    per_k: list[BdKResult], weights: dict[str, float],
) -> np.ndarray:
    """Composite cost per K using the supplied weights over normalised
    metrics. NaN-safe: any NaN normalised metric makes the K's cost NaN
    (so it falls out of argmin).

    Special handling per Q4: `overshoot` and `undershoot` are clipped at
    zero on the negative side before contributing (so a K with no spike
    is rewarded, not double-penalised).
    """
    cost = np.zeros(len(per_k), dtype=float)
    for i, r in enumerate(per_k):
        total = 0.0
        any_nan = False
        for name, w in weights.items():
            v = r.normalised.get(name, float("nan"))
            if not np.isfinite(v):
                any_nan = True
                break
            if name in ("overshoot", "undershoot"):
                v = max(0.0, v)
            total += w * v
        cost[i] = float("nan") if any_nan else total
    return cost


def _argmin_with_parabolic(
    k_values: np.ndarray, cost: np.ndarray
) -> float | None:
    """K at the minimum of cost(K), with sub-step parabolic interpolation.

    Returns None if fewer than 1 finite cost value is available. Returns
    K[argmin] (no interpolation) when the minimum is at a boundary, or when
    the local 3-point fit is non-concave-up (noise-dominated). The
    interpolated vertex is clamped to the local 3-point K range so we never
    extrapolate beyond the sweep.
    """
    if len(k_values) == 0 or len(cost) == 0:
        return None
    finite = np.isfinite(cost)
    if not finite.any():
        return None
    k_f = k_values[finite].astype(float)
    c_f = cost[finite].astype(float)
    i = int(np.argmin(c_f))
    if len(c_f) < 3 or i == 0 or i == len(c_f) - 1:
        return float(k_f[i])
    x0, x1, x2 = float(k_f[i - 1]), float(k_f[i]), float(k_f[i + 1])
    y0, y1, y2 = float(c_f[i - 1]), float(c_f[i]), float(c_f[i + 1])
    # Solve y = a*x^2 + b*x + c through (x0,y0),(x1,y1),(x2,y2)
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return float(x1)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (
        x2 * x2 * (y0 - y1)
        + x1 * x1 * (y2 - y0)
        + x0 * x0 * (y1 - y2)
    ) / denom
    if a <= 0:
        # Not a U-shape locally -- trust the discrete argmin instead of
        # extrapolating off a flat / inverted parabola.
        return float(x1)
    vertex = -b / (2.0 * a)
    # Clamp to the 3-point bracket so noise can't push the interpolated K
    # outside the sweep range.
    return float(max(min(vertex, x2), x0))


def _linear_fit_zero_crossing(
    k_values: np.ndarray, y_values: np.ndarray, method: str
) -> FitResult | None:
    """Fit y = m*k + b, return k where y = 0 and the R²."""
    mask = np.isfinite(y_values)
    if mask.sum() < 2:
        return None
    k = k_values[mask].astype(float)
    y = y_values[mask].astype(float)
    if np.allclose(np.std(k), 0):
        return None
    slope, intercept = np.polyfit(k, y, 1)
    if slope == 0:
        return None
    k_opt = -intercept / slope
    # R²
    y_pred = slope * k + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return FitResult(
        k_opt=float(k_opt),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r2,
        method=method,
    )


def _signed_area_zero_crossing_fit(
    k_values: np.ndarray,
    y_values: np.ndarray,
    method: str,
    half_window: int = 5,
) -> FitResult | None:
    """Zero-crossing K_opt from a signed-area-vs-K curve, fitted LOCALLY
    around the sign change with a Theil–Sen robust line.

    Why not a global OLS fit: the signed-area response is only locally
    linear around K_opt; the high-K tail saturates and is noisy, and a
    global `polyfit` lets that tail drag the zero crossing around
    (observed on real runs). The Snapmaker U1's production algorithm
    mitigates the same problem by fitting only 4–5 points bracketing
    the crossing inside a narrow K window — this is the grid-sweep
    equivalent: find the (+ → −) sign change of the per-K medians and
    fit only ±`half_window` grid points around it. Theil–Sen (median of
    pairwise slopes) keeps a single bad K in the local window from
    steering the fit the way OLS would.

    Falls back to the global linear fit when no (+ → −) crossing exists
    in the data (all-positive = sweep ended below K_opt; all-negative =
    started above it) — extrapolation is then the only option and the
    global trend is the best available.
    """
    mask = np.isfinite(y_values)
    if mask.sum() < 3:
        return _linear_fit_zero_crossing(k_values, y_values, method)
    k = k_values[mask].astype(float)
    y = y_values[mask].astype(float)

    # Locate the (+ → −) crossing: signed areas run positive (under-PA)
    # at low K and negative (over-PA) at high K. With noise there may
    # be several sign flips; pick the one closest to the median K of
    # ALL flips (robust to isolated noise flips at the tails).
    flips = [
        i for i in range(len(y) - 1)
        if y[i] > 0 >= y[i + 1]
    ]
    if not flips:
        return _linear_fit_zero_crossing(k_values, y_values, method)
    mid_flip_k = float(np.median([k[i] for i in flips]))
    cross = min(flips, key=lambda i: abs(k[i] - mid_flip_k))

    lo = max(0, cross - half_window + 1)
    hi = min(len(y), cross + 1 + half_window)
    kw = k[lo:hi]
    yw = y[lo:hi]
    if len(kw) < 3:
        return _linear_fit_zero_crossing(k_values, y_values, method)

    # Theil–Sen: slope = median over all pairwise slopes, intercept =
    # median residual.
    slopes = [
        (yw[j] - yw[i]) / (kw[j] - kw[i])
        for i in range(len(kw))
        for j in range(i + 1, len(kw))
        if kw[j] != kw[i]
    ]
    if not slopes:
        return _linear_fit_zero_crossing(k_values, y_values, method)
    slope = float(np.median(slopes))
    if slope == 0:
        return _linear_fit_zero_crossing(k_values, y_values, method)
    intercept = float(np.median(yw - slope * kw))
    k_opt = -intercept / slope
    # Clamp to the local window: the crossing is IN this window by
    # construction, so a fit escaping it is noise-driven.
    k_opt = float(min(max(k_opt, kw[0]), kw[-1]))
    # R² of the robust line over the local window (display only).
    y_pred = slope * kw + intercept
    ss_res = float(np.sum((yw - y_pred) ** 2))
    ss_tot = float(np.sum((yw - np.mean(yw)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return FitResult(
        k_opt=k_opt,
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        method=f"{method}_theil_sen_local",
    )


def _square_wave_at_transitions(
    t_grid: np.ndarray,
    transitions_t: np.ndarray,
    transitions_dirs: np.ndarray,
    low_val: float,
    high_val: float,
) -> np.ndarray:
    """Build a piecewise-constant square wave on `t_grid` whose edges sit
    exactly at the supplied transition timestamps.

    Convention matches `_detect_pos_transitions`:
      * direction +1 = pos_x peak = slow→fast in E → wave RISES (low→high)
      * direction −1 = pos_x trough = fast→slow in E → wave FALLS (high→low)

    Before the first transition, the wave holds the value OPPOSITE the
    first direction (i.e. the value the printer is leaving).

    Used in two flavours: (low=slow_v, high=fast_v) to produce the
    velocity command wave with pos_x-driven timing; (low=slow_plateau,
    high=fast_plateau) to produce the force-units ground-truth overlay.
    """
    out = np.empty_like(t_grid, dtype=float)
    if len(transitions_t) == 0:
        out[:] = low_val
        return out
    initial = low_val if float(transitions_dirs[0]) > 0 else high_val
    out[:] = initial
    for t_x, d in zip(transitions_t, transitions_dirs):
        mask = t_grid >= float(t_x)
        out[mask] = high_val if float(d) > 0 else low_val
    return out


def analyse_sweep(
    sweep_t0: float,
    force_t: np.ndarray,
    force_y: np.ndarray,
    plan: SweepPlan,
    resample_hz: float = 1000.0,
    auto_detect_t0: bool = True,
    gcode_t: np.ndarray | None = None,
    gcode_lines: list[str] | None = None,
    t0_is_anchored: bool = False,
    baseline_t_range: tuple[float, float] | None = None,
    pos_t: np.ndarray | None = None,
    pos_x: np.ndarray | None = None,
    pos_z_t: np.ndarray | None = None,
    pos_z: np.ndarray | None = None,
    z_marker_lift_mm: float = 2.0,
) -> SweepAnalysis:
    """Run both analyses given the loadcell timeseries and the sweep plan.

    Parameters
    ----------
    sweep_t0
        Monotonic timestamp (seconds) of the SWEEP_START marker — segment offsets are
        relative to this.
    force_t
        Array of monotonic timestamps for each loadcell sample.
    force_y
        Array of loadcell values (any unit — we only care about waveform shape).
    plan
        The SweepPlan returned by gcode_gen.build_sweep().
    gcode_t, gcode_lines
        Optional ground-truth command-timing log. The Buddy `gcode` metric
        streams every processed gcode line as a STRING with the firmware's
        timestamp at the moment of execution. When supplied, the analyzer
        derives the commanded velocity wave from consecutive G1 E lines
        (velocity per leg = dE / dt between adjacent events), bypassing the
        model square wave entirely. Buddy does NOT expose a pos_e metric;
        the `gcode` line stream is the only ground-truth signal available.
    """
    notes: list[str] = []
    if len(force_t) != len(force_y):
        raise ValueError("force_t and force_y length mismatch")
    if len(force_t) < 10:
        notes.append("Too few loadcell samples -- check UDP and metric is enabled.")
        return SweepAnalysis(
            per_k=[], phase_fit=None, integral_fit=None,
            integral_legacy_fit=None, sample_rate_hz=0.0, notes=notes,
        )

    # Defensively drop non-finite samples. loadcell_hp emits NaN while
    # idle; the collector tries to reject them but if anything slipped
    # through, scipy.signal.detrend would raise ValueError downstream.
    force_t = np.asarray(force_t, dtype=float)
    force_y = np.asarray(force_y, dtype=float)
    finite = np.isfinite(force_t) & np.isfinite(force_y)
    if not finite.all():
        dropped = int((~finite).sum())
        notes.append(
            f"dropped {dropped} non-finite loadcell sample(s) "
            f"(NaN/inf -- usually loadcell_hp at idle)"
        )
        force_t = force_t[finite]
        force_y = force_y[finite]
    if len(force_t) < 10:
        notes.append("After NaN/inf filtering, too few samples remain.")
        return SweepAnalysis(
            per_k=[], phase_fit=None, integral_fit=None,
            integral_legacy_fit=None, sample_rate_hz=0.0, notes=notes,
        )

    # Pre-sweep baseline: the runner records timestamps of a held dwell
    # right after heat-up, BEFORE any priming/extrusion. The loadcell during
    # that window is reading the static head load only -- a clean zero we
    # surface in the UI as a diagnostic (is the cell tared? is it drifting?
    # is the noise floor sensible?). Per-K analysis still mean-subtracts
    # inside each window, so the baseline is informative, not a global tare.
    baseline: Baseline | None = None
    if baseline_t_range is not None:
        bt0, bt1 = baseline_t_range
        if bt1 > bt0:
            bm = (force_t >= bt0) & (force_t <= bt1)
            n_b = int(bm.sum())
            if n_b >= 8:
                by = force_y[bm]
                half = n_b // 2
                drift = float(np.mean(by[half:])) - float(np.mean(by[:half]))
                baseline = Baseline(
                    mean=float(np.mean(by)),
                    std=float(np.std(by)),
                    drift=drift,
                    n_samples=n_b,
                    t_start=float(bt0),
                    t_end=float(bt1),
                )
                notes.append(
                    f"baseline: mean={baseline.mean:.1f} std={baseline.std:.2f} "
                    f"drift={baseline.drift:+.2f} (n={n_b})"
                )
            else:
                notes.append(
                    f"baseline window had only {n_b} samples -- skipping"
                )

    # Anchor sweep_t0 AND derive per-K window bounds. The full precedence
    # cascade (gcode-event anchor, Z-marker pulse, pos_x periodic cycles,
    # pos_x first-motion, loadcell rolling-std, force-trace re-anchor)
    # lives in `alignment.resolve_anchor_and_windows` -- see its docstring
    # for the decision table.
    resolution = resolve_anchor_and_windows(
        sweep_t0, force_t, force_y, plan,
        t0_is_anchored=t0_is_anchored,
        auto_detect_t0=auto_detect_t0,
        pos_t=pos_t, pos_x=pos_x,
        pos_z_t=pos_z_t, pos_z=pos_z,
        z_marker_lift_mm=z_marker_lift_mm,
        notes=notes,
    )
    sweep_t0 = resolution.sweep_t0
    t0_is_anchored = resolution.t0_is_anchored
    pre_burst_data_windows = resolution.windows

    # shift to sweep-relative time
    t_rel = force_t - sweep_t0
    dt = 1.0 / resample_hz

    # Pre-compute pos_x leg transitions (every slow→fast and fast→slow
    # crossing across the entire sweep) once, so the per-K loop can slice
    # them per window. These are the ACTUAL cycle-by-cycle transition
    # times the toolhead executed -- they replace the model command
    # wave's mid-level crossings as the integrator's reference clock.
    # Per-cycle drift (planner overhead, accel-limit slop) accumulates
    # over 14 cycles to ~50-100 ms by end-of-window, enough to push the
    # ±half_win=300 ms integration windows off the real transitions.
    # Anchoring to pos_x removes that drift.
    pos_transitions_t: np.ndarray | None = None
    pos_transitions_dirs: np.ndarray | None = None
    if (
        pos_t is not None
        and pos_x is not None
        and len(pos_t) >= 10
        and plan.params.coupled_dx_mm > 0
    ):
        tt, td = _detect_pos_transitions(
            pos_t, pos_x,
            expected_amplitude_mm=plan.params.coupled_dx_mm,
        )
        if len(tt) >= 2:
            pos_transitions_t = tt
            pos_transitions_dirs = td
            notes.append(
                f"per-cycle integration windows anchored to pos_x "
                f"velocity sign-flips ({len(tt)} transitions detected "
                f"across the sweep)"
            )

    # If the firmware streamed the gcode-line log, prepare it on
    # sweep-relative time so each per-K window can derive its commanded
    # velocity from real per-line transition timestamps rather than a
    # reconstructed model.
    gcode_t_rel: np.ndarray | None = None
    gcode_lines_local: list[str] | None = None
    if (
        gcode_t is not None
        and gcode_lines is not None
        and len(gcode_t) >= 4
        and len(gcode_t) == len(gcode_lines)
    ):
        gt = np.asarray(gcode_t, dtype=float)
        finite_g = np.isfinite(gt)
        if finite_g.sum() >= 4:
            gcode_t_rel = gt[finite_g] - sweep_t0
            gcode_lines_local = [gcode_lines[i] for i in np.where(finite_g)[0]]
            notes.append(
                f"using gcode-log ground-truth command wave "
                f"({len(gcode_lines_local)} lines)"
            )

    # estimate true incoming sample rate (for diagnostics)
    if len(force_t) >= 2:
        in_rate = (len(force_t) - 1) / (force_t[-1] - force_t[0])
    else:
        in_rate = 0.0

    per_k: list[KResult] = []
    windows: list[KWindow] = []
    lags: list[float] = []
    areas: list[float] = []
    areas_legacy: list[float] = []
    coverages: list[float] = []
    ks: list[float] = []
    # bd_pressure step-response analysis: build one BdSegment per (K, cycle)
    # using the pos_x-detected rising/falling transition times. Segments
    # are computed inside the per-K loop so they share the same dropout
    # metadata and window slicing as the existing per-K analysis.
    bd_segments: list[BdSegment] = []
    bd_segments_by_k: dict[float, list[BdSegment]] = {}
    p = plan.params
    # Per-K caches needed AFTER the loop to compute force baselines and
    # inject the pos_x-driven ground-truth force overlay.
    per_k_pos_transitions_rel: list[np.ndarray] = []  # sweep-rel times
    per_k_pos_transitions_dirs: list[np.ndarray] = []
    per_k_force_offset: list[float] = []  # mean(force) we subtracted in centering
    # Plateau force values, collected globally across the sweep so the
    # median is robust to single-K outliers. We deliberately skip the
    # first SETTLE_S after every transition because the loadcell hasn't
    # finished tracking the velocity change yet -- including that ramp
    # would bias the baselines toward the inter-plateau values.
    plateau_settle_s = 0.15
    plateau_slow_values: list[float] = []
    plateau_fast_values: list[float] = []
    # Per-K data-quality floor. Below this fraction of expected samples the
    # window is excluded from the K_opt extractors so a single dropout-
    # ridden K can't yank the cost curve's minimum off the real one.
    MIN_COVERAGE_FOR_FIT = 0.5

    # Per-K window source. Resolved in three tiers from most-preferred to
    # least-preferred:
    #   1. `pre_burst_data_windows` set by the anchor block above. This is
    #      EITHER plan-direct slicing (when the Z-marker anchored, the
    #      safest case) OR data-derived bounds (when pos_x periodic or
    #      force-trace re-anchor produced them, which historically snapped
    #      window boundaries to the exact slow-leg starts of each cycle).
    #   2. Plan-direct slicing as a universal fallback when (1) is None
    #      and we have a usable anchor. The plan encodes everything --
    #      including the K[0] warm-up extension -- so this works even
    #      when no edge detector has succeeded. Using plan offsets is
    #      always safer than guessing window bounds.
    #   3. The legacy guarded plan slicing only kicks in when sweep_t0
    #      itself wasn't anchored and we genuinely have no idea where
    #      the sweep is. Then we apply a 0.25 × period guard at both
    #      ends and hope.
    data_windows = pre_burst_data_windows
    if data_windows is None and plan.segments:
        data_windows = _slice_from_plan(plan)
        notes.append(
            "K windows fell back to plan-direct slicing "
            "(no edge detector produced usable windows)"
        )

    for seg_idx, seg in enumerate(plan.segments):
        # slice the timeseries for this K's burst window
        if data_windows is not None:
            data_lo, data_hi = data_windows[seg_idx]
            # No guard at the lo end -- data_lo is the EXACT slow-leg
            # start of cycle 0 (force-cycle slicer is precise), so we
            # want the window to begin right at that LOW plateau.
            # EXTEND the hi end by `slow_half_s` so the trailing slow
            # leg of the NEXT K's cycle 0 is visible -- that's the
            # "ends on low" the user expects, AND it's the same data
            # point as the next K's window opening (the shared
            # boundary). Window content: LOW, (HIGH, LOW) × cycles_per_K.
            t_lo = data_lo
            t_hi = data_hi + plan.params.slow_half_s
            burst_start_for_model: float | None = data_lo
        else:
            guard = seg.cycle_period_s * 0.25
            t_lo = seg.start_offset_s + guard
            t_hi = seg.start_offset_s + seg.duration_s - guard
            burst_start_for_model = None
        mask = (t_rel >= t_lo) & (t_rel <= t_hi)
        seg_t = t_rel[mask]
        seg_y = force_y[mask]
        # Expected sample count given the run's observed incoming rate; the
        # ratio gives us a per-K data-quality score independent of the
        # configured analyzer resample_hz.
        expected_n = max(1.0, (t_hi - t_lo) * max(in_rate, 1.0))
        coverage = min(1.0, len(seg_t) / expected_n)

        # Dropout detection. The firmware delivers samples in bursts
        # of ~3 ms intra-batch with ~31 ms gaps between batches (one
        # ADC-accumulator period). A naive 3×median threshold flags
        # every batch-boundary gap as a "dropout", which is wrong.
        # We use a STRICTER criterion: a real dropout has BOTH
        #   (a) inter-sample time gap > max(10×median_dt, 50 ms), AND
        #   (b) force jump > 0.5·plateau_spread.
        # That catches "samples missing AT a transition so the force
        # jumped suddenly" -- the actual failure mode the user cares
        # about -- and ignores the firmware's normal batch cadence.
        dropout_t_list: list[float] = []
        if len(seg_t) >= 10:
            seg_t_arr = np.asarray(seg_t, dtype=float)
            seg_y_arr = np.asarray(seg_y, dtype=float)
            dt_intervals = np.diff(seg_t_arr)
            median_dt = float(np.median(dt_intervals))
            time_gap_thresh = max(10.0 * median_dt, 0.050)
            time_gap_mask = dt_intervals > time_gap_thresh
            seg_p10 = float(np.percentile(seg_y_arr, 10))
            seg_p90 = float(np.percentile(seg_y_arr, 90))
            plateau_spread = seg_p90 - seg_p10
            if plateau_spread > 50.0:
                force_jumps = np.abs(np.diff(seg_y_arr)) > 0.5 * plateau_spread
                dropout_mask = time_gap_mask & force_jumps
            else:
                dropout_mask = np.zeros_like(time_gap_mask)
            drop_idx = np.where(dropout_mask)[0] + 1
            dropout_t_list = [float(seg_t_arr[i]) for i in drop_idx]
        if len(seg_t) < 8:
            notes.append(f"K={seg.k:.4f}: only {len(seg_t)} samples, skipping.")
            per_k.append(KResult(k=seg.k, n_samples=len(seg_t),
                                 phase_lag_ms=float("nan"),
                                 integral_area=float("nan"),
                                 integral_area_legacy=float("nan"),
                                 force_mean=float("nan"),
                                 force_std=float("nan"),
                                 coverage=coverage))
            bd_segments_by_k[seg.k] = []
            continue
        grid, force = _resample_uniform(seg_t, seg_y, dt)
        if len(grid) < 8:
            continue

        # Build per-segment transition indices on the resampled grid from
        # the global pos_x sign-flip list. Each transition_t is in
        # monotonic time; subtract sweep_t0 to get sweep-relative time
        # and quantise to the uniform grid spacing dt.
        seg_trans_idx: np.ndarray | None = None
        seg_trans_dirs: np.ndarray | None = None
        seg_trans_t_rel: np.ndarray | None = None
        if (
            pos_transitions_t is not None
            and pos_transitions_dirs is not None
            and len(grid) >= 2
        ):
            rel = pos_transitions_t - sweep_t0
            in_win = (rel >= grid[0]) & (rel <= grid[-1])
            if int(in_win.sum()) >= 2:
                rel_in = rel[in_win]
                dirs_in = pos_transitions_dirs[in_win]
                idx_arr = np.round((rel_in - grid[0]) / dt).astype(int)
                idx_arr = np.clip(idx_arr, 0, len(grid) - 1)
                unique_idx, keep = np.unique(idx_arr, return_index=True)
                seg_trans_idx = unique_idx
                seg_trans_dirs = dirs_in[keep]
                seg_trans_t_rel = rel_in[keep]

        # Pick the command wave (mm/s). Preference order:
        #   1. Pos_x-driven: real per-cycle transitions executed by the
        #      printer. Eliminates per-cycle drift (~50-100 ms over 14
        #      cycles) the model wave would have.
        #   2. Gcode-event log (silent on current firmware build; kept
        #      for back-compat with other Buddy builds).
        #   3. Model square wave (last-resort fallback).
        command = np.array([])
        if seg_trans_t_rel is not None and len(seg_trans_t_rel) >= 2:
            command = _square_wave_at_transitions(
                grid, seg_trans_t_rel, seg_trans_dirs,
                low_val=p.slow_feed_mm_s, high_val=p.fast_feed_mm_s,
            )
        if len(command) == 0 and gcode_t_rel is not None and gcode_lines_local is not None:
            pad = seg.cycle_period_s
            in_seg = (gcode_t_rel >= grid[0] - pad) & (
                gcode_t_rel <= grid[-1] + pad
            )
            if in_seg.sum() >= 2:
                seg_gt = gcode_t_rel[in_seg]
                seg_lines = [
                    gcode_lines_local[i] for i in np.where(in_seg)[0]
                ]
                command = _command_wave_from_gcode(grid, seg_gt, seg_lines)
        if len(command) == 0:
            command = _build_command_wave(
                seg, grid,
                slow_v=p.slow_feed_mm_s, fast_v=p.fast_feed_mm_s,
                slow_half_s=p.slow_half_s,
                accel_mm_s2=p.accel_mm_s2,
                burst_start_override=burst_start_for_model,
            )
        # Cap the cross-correlation search at half the cycle period.
        max_lag = 0.5 * seg.cycle_period_s
        lag_ms = _phase_lag_ms(force, command, dt, max_lag_s=max_lag)

        # Collect plateau samples for the slow/fast force baselines.
        # For every pair of consecutive transitions, the leg between
        # them is HIGH if the direction at the first transition was +1
        # (we rose), LOW if it was -1. Skip the first plateau_settle_s
        # of the plateau where the loadcell is still tracking.
        if (
            seg_trans_idx is not None
            and seg_trans_dirs is not None
            and len(seg_trans_idx) >= 2
        ):
            settle_n = max(1, int(round(plateau_settle_s / dt)))
            for i in range(len(seg_trans_idx) - 1):
                idx_start = int(seg_trans_idx[i]) + settle_n
                idx_end = int(seg_trans_idx[i + 1])
                if idx_end - idx_start < 4:
                    continue
                plateau_force = force[idx_start:idx_end]
                med = float(np.median(plateau_force))
                if float(seg_trans_dirs[i]) > 0:
                    plateau_fast_values.append(med)
                else:
                    plateau_slow_values.append(med)

        # integral_area is now computed AFTER bd_segs_this_k is built so it
        # consumes the same auto-exclusion gate the bd_pressure analysis
        # uses (see _integral_area_from_segments). The pos_x-derived
        # variant is kept around as a sanity check, but only the bd-gated
        # value flows into `areas`/the linear fit/the UI.
        area = float("nan")
        area_legacy = _integral_area_legacy(force, command)
        per_k.append(
            KResult(
                k=seg.k,
                n_samples=int(len(seg_t)),
                phase_lag_ms=lag_ms,
                integral_area=area,
                integral_area_legacy=area_legacy,
                force_mean=float(np.mean(seg_y)),
                force_std=float(np.std(seg_y)),
                coverage=coverage,
                dropouts=len(dropout_t_list),
            )
        )

        # bd_pressure step-response segments for THIS K. Three sources
        # for the rising/falling transition times, tried in order:
        #
        #   1. FORCE-derived (preferred). The loadcell is the cleanest
        #      cycle-timing signal we have — each slow→fast transition
        #      shows up as an unambiguous mid-level crossing. We use
        #      `_detect_force_cycle_starts` for the rising side and
        #      synthesize the falling side at `rising + fast_half_s`
        #      (the fast leg's commanded duration). This decisively
        #      beats pos_x in two scenarios we've observed on real
        #      hardware: (a) when the user's gcode has tiny or no
        #      XY coupling, pos_x doesn't oscillate on every cycle and
        #      pos_x transitions correspond to coarser features (K-to-K
        #      moves) instead of per-cycle boundaries; (b) when pos_x
        #      reporting is delayed relative to motion, the detected
        #      "peaks" sit late and the resulting segment windows
        #      sample the wrong leg of the cycle entirely (manifests
        #      as `baseline_median > high_level`, a giant `noise_std`,
        #      and bogus "rise below noise" exclusions on visibly
        #      clean data).
        #
        #   2. pos_x detected transitions. Useful when the user
        #      configured large XY coupling and the position trace is
        #      clean and aligned with the force signal.
        #
        #   3. Model synthesis from `cycle_period_s` + `slow_half_s`
        #      anchored at `t_lo`. Final fallback for K's the other two
        #      methods couldn't cover.
        #
        # Segment N uses (t_rise_N, t_fall_N) and
        # t_start = t_rise_N - slow_half_s,
        # t_end = t_fall_N + slow_half_s — which for K[0] cycle 0 walks
        # back into the warmup giving "last slow_half of warmup" as
        # low_0 (Q2 B=i), and for segment 13 extends t_end past data_hi
        # into the next K's first slow leg (Q2 A=i).
        bd_segs_this_k: list[BdSegment] = []
        rising_t: np.ndarray = np.array([])
        falling_t: np.ndarray = np.array([])
        # --- Source 1: force-derived ---
        # Look slightly outside the per-K window so the cycle-edge
        # crossings near the boundaries are still picked up.
        force_t_lo = sweep_t0 + t_lo - 0.5 * seg.cycle_period_s
        force_t_hi = sweep_t0 + t_hi + 0.5 * seg.cycle_period_s
        force_rising_abs = _detect_force_cycle_starts(
            force_t, force_y,
            t_lo=force_t_lo, t_hi=force_t_hi,
            min_gap_s=0.5 * seg.cycle_period_s,
        )
        # Keep only rising edges that land inside the per-K window
        # (sweep-relative). Sub-sample precision is preserved by
        # `_detect_force_cycle_starts`.
        if len(force_rising_abs) > 0:
            force_rising_rel = force_rising_abs - sweep_t0
            in_win = (force_rising_rel >= t_lo - 0.1) & (
                force_rising_rel <= t_hi + 0.1
            )
            force_rising_rel = force_rising_rel[in_win]
            # Demand at least 70% of expected cycles to trust this source —
            # otherwise the signal isn't periodic enough.
            if len(force_rising_rel) >= max(2, int(0.7 * seg.cycles)):
                # Clamp to at most seg.cycles — the look-ahead extension
                # on the t_hi window can occasionally pick up the first
                # rising edge of K[i+1]. We don't want it in this K's
                # segment list.
                rising_t = force_rising_rel[: seg.cycles]
                falling_t = rising_t + p.fast_half_s
                if len(rising_t) != seg.cycles:
                    notes.append(
                        f"K={seg.k:.4f}: force-derived transitions found "
                        f"{len(rising_t)}/{seg.cycles} cycles "
                        f"(synthesis will fill the rest if any)"
                    )
        # --- Source 2: pos_x derived (when force didn't give enough) ---
        if len(rising_t) < seg.cycles and (
            seg_trans_t_rel is not None
            and seg_trans_dirs is not None
            and len(seg_trans_t_rel) >= 2
        ):
            pos_rising = seg_trans_t_rel[seg_trans_dirs > 0]
            pos_falling = seg_trans_t_rel[seg_trans_dirs < 0]
            n_pos_pairs = min(len(pos_rising), len(pos_falling))
            if n_pos_pairs > len(rising_t):
                rising_t = pos_rising[:seg.cycles]
                falling_t = pos_falling[:seg.cycles]
        # --- Source 3: model fallback for any remaining missing cycles ---
        n_pairs = min(len(rising_t), len(falling_t))
        if n_pairs < seg.cycles:
            n_missing = seg.cycles - n_pairs
            synth_rising = []
            synth_falling = []
            # The model places cycle 0 at the actual burst start, not at
            # `t_lo`: when `data_windows is None` we shift `t_lo` forward
            # by `guard = 0.25·cycle_period` to skip start-of-burst
            # transients, and synth_rising would then sit inside the
            # WRONG leg (the user's run_1779016571 in fallback mode
            # produced t_rise=1.1 when the actual slow→fast transition
            # was at 0.8). Anchor the synthesis to the same burst start
            # the command-wave model uses.
            burst_start_bd = (
                burst_start_for_model
                if burst_start_for_model is not None
                else seg.start_offset_s
            )
            for j in range(n_pairs, seg.cycles):
                # cycle j: slow leg [burst_start + j·period,
                # burst_start + j·period + slow_half], fast leg ends at
                # burst_start + (j+1)·period.
                synth_rising.append(
                    burst_start_bd + j * seg.cycle_period_s + plan.params.slow_half_s
                )
                synth_falling.append(burst_start_bd + (j + 1) * seg.cycle_period_s)
            if synth_rising:
                rising_t = np.concatenate(
                    [rising_t, np.asarray(synth_rising, dtype=float)]
                )
                falling_t = np.concatenate(
                    [falling_t, np.asarray(synth_falling, dtype=float)]
                )
                rising_t.sort()
                falling_t.sort()
                notes.append(
                    f"K={seg.k:.4f}: synthesized {n_missing} cycle "
                    f"transition pair(s) from plan geometry"
                )
            n_pairs = min(len(rising_t), len(falling_t))
        if n_pairs >= 1:
            # Convert per-K window dropouts to sweep-relative absolute
            # times once -- _bd_segment_metrics expects absolute times.
            dropout_abs = np.asarray(dropout_t_list, dtype=float)
            for i in range(n_pairs):
                t_rise_abs = float(rising_t[i])
                t_fall_abs = float(falling_t[i])
                # Defensive: pair only if falling actually follows rising
                # (a slow→fast crossing followed by a fast→slow crossing
                # within the same cycle). Otherwise this i-th rising
                # belongs to a different cycle structure and we skip it.
                if t_fall_abs <= t_rise_abs:
                    continue
                # `t_start` is the slow leg start before t_rise =
                # t_rise - slow_half_s. We deliberately use the NORMAL
                # slow_half_s here even for K[0] cycle 0 (the warm-up
                # cycle), so the bd_segment view shows a uniform 5 s
                # low-high-low shape for every segment regardless of
                # warm-up duration. Including the full 10 s warm-up
                # slow leg in segment 1's display crowds it with
                # melt-pressure transients that aren't analytically
                # interesting -- the user explicitly asked to crop
                # the warm-up out of the first segment view.
                # The K-window (used by the per-K analysis upstream)
                # still covers the full warm-up so the integral/phase
                # fits see all the data.
                # NOTE: despite the `_abs` suffix in these names, the
                # convention in this loop is sweep-RELATIVE seconds
                # (t_rise_abs, t_fall_abs come from rising_t /
                # falling_t which are populated via seg_trans_t_rel).
                t_start_abs = t_rise_abs - p.slow_half_s
                t_end_abs = t_fall_abs + p.slow_half_s
                bd_seg = _bd_segment_metrics(
                    force_t=t_rel,
                    force_y=force_y,
                    k=float(seg.k),
                    seg_idx=i,
                    t_start=t_start_abs,
                    t_rise=t_rise_abs,
                    t_fall=t_fall_abs,
                    t_end=t_end_abs,
                    slow_half_s=float(p.slow_half_s),
                    fast_half_s=float(p.fast_half_s),
                    dropout_t=dropout_abs,
                )
                # First-cycle exclusion (Snapmaker U1's drop1=1): with
                # zero inter-K dwell, cycle 0's slow leg IS the
                # previous K's pressure relaxation, so its step
                # response carries the neighbouring K's history — a
                # bias that is directionally correlated with sweep
                # order. Only applied when the K has enough cycles to
                # spare (single-cycle-per-K sweeps keep everything).
                if i == 0 and seg.cycles >= 3 and not bd_seg.excluded:
                    bd_seg.excluded = True
                    bd_seg.exclusion_reasons.append(
                        "first cycle after K switch (carries previous "
                        "K's pressure history)"
                    )
                bd_segs_this_k.append(bd_seg)
        bd_segments_by_k[seg.k] = bd_segs_this_k
        bd_segments.extend(bd_segs_this_k)

        # ----- shared-exclusion integral_area --------------------------
        # Both algorithms (bd_pressure step-response medians and the U1/
        # Snapmaker integral) now read from the same BdSegment list and
        # honour the same `excluded` flag. A K's integral_area is the
        # signed sum over INCLUDED cycles only; if fewer than 4 cycles
        # survive the gate the value is NaN and the K is dropped from
        # the linear fit. The half-window is the physics-based 75 ms,
        # clamped to half the fast leg so it never overlaps adjacent
        # transitions on U1-style short cycles.
        bd_half_win = min(0.075, 0.5 * float(p.fast_half_s))
        area_bd, n_inc_bd, n_tot_bd = _integral_area_from_segments(
            t_rel, force_y, bd_segs_this_k, half_win_s=bd_half_win,
        )
        if n_tot_bd > 0 and n_inc_bd < 4:
            area_bd = float("nan")
        # Overwrite the placeholder NaN set when KResult was built.
        per_k[-1].integral_area = area_bd
        per_k[-1].integral_n_included = int(n_inc_bd)
        per_k[-1].integral_n_total = int(n_tot_bd)
        area = area_bd  # used by `areas.append(area)` below
        # Save the windowed timeseries for the UI to plot. We ship the
        # RAW per-segment samples (seg_t/seg_y), NOT the 1 kHz
        # resampled grid -- hovering on the plot now shows ACTUAL
        # loadcell measurements, not interpolated points. The straight
        # diagonals on fast transitions the user spotted were
        # `np.interp` artefacts of the 1 kHz resample bridging two raw
        # samples 5-6 ms apart; serving raw samples lets the user
        # judge real measurement density at every point. The analysis
        # itself still uses `grid`/`force` internally (cross-
        # correlation, integral windows) because those need a uniform
        # time base, but the UI doesn't.
        #
        # The ground-truth overlay (`ground_truth_force`, built after
        # the loop) is also evaluated on the raw timestamps so the two
        # traces share the same x-axis points.
        seg_t_rel = [float(x) for x in seg_t]
        seg_y_raw = [float(x) for x in seg_y]
        # Build the command wave on the RAW seg_t timestamps (not the
        # 1 kHz grid) so it lines up sample-for-sample with the raw
        # force trace in the UI. Cheap re-eval -- the wave generators
        # are piecewise-constant.
        if seg_trans_t_rel is not None and len(seg_trans_t_rel) >= 2:
            command_ui = _square_wave_at_transitions(
                seg_t, seg_trans_t_rel, seg_trans_dirs,
                low_val=p.slow_feed_mm_s, high_val=p.fast_feed_mm_s,
            )
        else:
            command_ui = _build_command_wave(
                seg, seg_t,
                slow_v=p.slow_feed_mm_s, fast_v=p.fast_feed_mm_s,
                slow_half_s=p.slow_half_s,
                accel_mm_s2=p.accel_mm_s2,
                burst_start_override=burst_start_for_model,
            )
        windows.append(
            KWindow(
                k=seg.k,
                t=seg_t_rel,
                force=seg_y_raw,
                command=[float(x) for x in command_ui],
                dropout_t=dropout_t_list,
            )
        )
        per_k_pos_transitions_rel.append(
            seg_trans_t_rel if seg_trans_t_rel is not None else np.array([])
        )
        per_k_pos_transitions_dirs.append(
            seg_trans_dirs if seg_trans_dirs is not None else np.array([])
        )
        per_k_force_offset.append(0.0)  # unused now; kept for arity parity
        ks.append(seg.k)
        lags.append(lag_ms)
        areas.append(area)
        areas_legacy.append(area_legacy)
        coverages.append(coverage)

    # ---- force baselines + per-K ground-truth overlay --------------------
    # Slow/fast plateau medians across the whole sweep. These are the
    # measured steady-state loadcell readings when the printer is in
    # its slow leg vs fast leg, with the loadcell ramp explicitly
    # excluded (we dropped the first plateau_settle_s of each plateau).
    # Used both as a numeric reference in the UI and as the amplitudes
    # of the pos_x-derived ground-truth wave that overlays each K plot
    # so over/undershoot is visible at a glance.
    force_baselines: ForceBaselines | None = None
    MIN_PLATEAU_SAMPLES = 4
    if (
        len(plateau_slow_values) >= MIN_PLATEAU_SAMPLES
        and len(plateau_fast_values) >= MIN_PLATEAU_SAMPLES
    ):
        # Plateau values were collected from the RAW force trace inside
        # each K window. Medians-of-medians across the whole sweep are
        # robust to single-K outliers. The result is in RAW loadcell
        # units, the same units `force` is stored in on each KWindow,
        # so the dashed overlay sits exactly where the actual force
        # plateaus do -- making over/undershoot the visible deviation.
        force_baselines = ForceBaselines(
            slow_plateau=float(np.median(plateau_slow_values)),
            fast_plateau=float(np.median(plateau_fast_values)),
            n_slow=len(plateau_slow_values),
            n_fast=len(plateau_fast_values),
        )
        notes.append(
            f"force baselines (raw units): "
            f"slow={force_baselines.slow_plateau:+.1f}, "
            f"fast={force_baselines.fast_plateau:+.1f} "
            f"(medians across {force_baselines.n_slow} slow + "
            f"{force_baselines.n_fast} fast plateaus, "
            f"first {plateau_settle_s*1000:.0f} ms of each leg dropped)"
        )
        # Inject the ground-truth force overlay into every KWindow. Use
        # each window's cached pos_x transitions; if a window has none
        # (rare, only when transitions were too sparse), leave its
        # ground_truth_force empty and the UI will keep the legacy
        # mm/s overlay for that K.
        for i, kw in enumerate(windows):
            trans_t = per_k_pos_transitions_rel[i]
            trans_d = per_k_pos_transitions_dirs[i]
            if len(trans_t) < 2:
                continue
            grid_arr = np.asarray(kw.t, dtype=float)
            gt = _square_wave_at_transitions(
                grid_arr, trans_t, trans_d,
                low_val=force_baselines.slow_plateau,
                high_val=force_baselines.fast_plateau,
            )
            kw.ground_truth_force = [float(x) for x in gt]
    elif plateau_slow_values or plateau_fast_values:
        notes.append(
            f"force baselines skipped: only {len(plateau_slow_values)} "
            f"slow + {len(plateau_fast_values)} fast plateau samples "
            f"(min {MIN_PLATEAU_SAMPLES} each required)"
        )

    # Apply the coverage gate to every fit input. Low-coverage K values
    # have so few samples that their amplitude/asymmetry/lag estimates are
    # dominated by noise -- letting them through tugs the linear fits and
    # the argmin minimum to wherever the bad K happens to land.
    ks_arr_all = np.asarray(ks, dtype=float)
    cov_arr = np.asarray(coverages, dtype=float)
    # Per-K dropout fraction gate. A K window can have nominal coverage
    # ~1.0 (right number of samples) but still be unusable if many of
    # those samples sit after time-gap+force-jump dropouts. We compute
    # `dropouts / n_samples` per K and exclude windows above 5%.
    dropout_arr = np.asarray([r.dropouts for r in per_k], dtype=float)
    nsamp_arr = np.asarray([max(1, r.n_samples) for r in per_k], dtype=float)
    drop_frac = dropout_arr / nsamp_arr
    MAX_DROPOUT_FRAC = 0.05
    quality_mask = (cov_arr >= MIN_COVERAGE_FOR_FIT) & (drop_frac <= MAX_DROPOUT_FRAC)
    n_excluded = int(np.sum(~quality_mask))
    if n_excluded:
        excluded_ks = ks_arr_all[~quality_mask]
        reasons = []
        for ki, k in enumerate(ks_arr_all):
            if not quality_mask[ki]:
                why = []
                if cov_arr[ki] < MIN_COVERAGE_FOR_FIT:
                    why.append(f"cov={cov_arr[ki]:.2f}")
                if drop_frac[ki] > MAX_DROPOUT_FRAC:
                    why.append(f"dropouts={int(dropout_arr[ki])}/{int(nsamp_arr[ki])}")
                reasons.append(f"K={k:.4f}({', '.join(why)})")
        notes.append(
            f"excluded {n_excluded} K value(s) from fit due to low data "
            f"quality (coverage < {MIN_COVERAGE_FOR_FIT:.0%} or dropouts > "
            f"{MAX_DROPOUT_FRAC:.0%}): " + ", ".join(reasons)
        )

    def _masked(arr: list[float]) -> np.ndarray:
        return np.asarray(arr, dtype=float)[quality_mask] if any(quality_mask) else np.asarray([])

    ks_fit = ks_arr_all[quality_mask] if any(quality_mask) else ks_arr_all
    phase_fit = _linear_fit_zero_crossing(ks_fit, _masked(lags), "phase_lag")
    integral_fit = _linear_fit_zero_crossing(ks_fit, _masked(areas), "integral")
    integral_legacy_fit = _linear_fit_zero_crossing(
        ks_fit, _masked(areas_legacy), "integral_legacy"
    )

    # bd_pressure step-response aggregate: median over included segments
    # per K → normalise per metric across sweep → composite cost via
    # default weights → argmin (with parabolic interp). K's with fewer
    # than 4 included segments OR that fail the per-K coverage / dropout
    # gates above are excluded from the K_opt search.
    MIN_INCLUDED_SEGS = 4
    bd_per_k = _bd_aggregate_per_k(bd_segments_by_k)
    # Align bd_per_k ordering with the per_k order so the UI can index
    # both lists the same way.
    bd_per_k_by_k = {r.k: r for r in bd_per_k}
    bd_per_k_ordered: list[BdKResult] = []
    for seg in plan.segments:
        r = bd_per_k_by_k.get(seg.k)
        if r is None:
            r = BdKResult(
                k=seg.k, n_segments_total=0, n_segments_included=0,
                medians={n: float("nan") for n in BD_METRIC_NAMES},
            )
        bd_per_k_ordered.append(r)
    bd_per_k = bd_per_k_ordered
    _bd_compute_normalised(bd_per_k)

    # K-level gate: same coverage/dropout mask used for the legacy fits,
    # plus a "≥ MIN_INCLUDED_SEGS" floor on included segments. Both must
    # pass for a K to contribute to bd_k_opt.
    bd_quality_mask = np.array(
        [
            quality_mask[i]
            and bd_per_k[i].n_segments_included >= MIN_INCLUDED_SEGS
            for i in range(len(bd_per_k))
        ],
        dtype=bool,
    )
    n_bd_excluded = int(np.sum(~bd_quality_mask))
    if n_bd_excluded:
        bd_dropped_for_seg_count = [
            f"K={bd_per_k[i].k:.4f}({bd_per_k[i].n_segments_included}/"
            f"{bd_per_k[i].n_segments_total} segs)"
            for i in range(len(bd_per_k))
            if quality_mask[i]
            and bd_per_k[i].n_segments_included < MIN_INCLUDED_SEGS
        ]
        if bd_dropped_for_seg_count:
            notes.append(
                f"bd_pressure: excluded {len(bd_dropped_for_seg_count)} "
                f"K value(s) for too few included segments "
                f"(< {MIN_INCLUDED_SEGS}): "
                + ", ".join(bd_dropped_for_seg_count)
            )

    bd_cost = _bd_compute_cost(bd_per_k, BD_DEFAULT_WEIGHTS)
    if bd_quality_mask.any():
        ks_bd_fit = np.asarray([r.k for r in bd_per_k], dtype=float)[bd_quality_mask]
        cost_bd_fit = bd_cost[bd_quality_mask]
        bd_k_opt = _argmin_with_parabolic(ks_bd_fit, cost_bd_fit)
    else:
        bd_k_opt = None

    # Signed-area zero-crossing fits. Reuse the SAME per-K quality mask as
    # bd_k_opt so all three estimators agree on which K values are usable.
    # rise_signed_area = ∫(force − plateau); fall_signed_area =
    # ∫(force − baseline). Under-PA leaves the rise below the plateau (−)
    # and the fall draining above baseline (+); over-PA overshoots the
    # plateau (+) and undershoots the baseline (−). Fit the per-K MEDIAN
    # and solve for the zero crossing. The fall fit is the cleaner
    # estimator (target = baseline → no rise-transient deficit); the rise
    # fit ships as an independent cross-check.
    bd_signed_rise_fit: FitResult | None = None
    bd_signed_fall_fit: FitResult | None = None
    if bd_quality_mask.any():
        ks_signed = np.asarray([r.k for r in bd_per_k], dtype=float)[bd_quality_mask]
        rise_signed = np.asarray(
            [r.medians.get("rise_signed_area", float("nan")) for r in bd_per_k],
            dtype=float,
        )[bd_quality_mask]
        fall_signed = np.asarray(
            [r.medians.get("fall_signed_area", float("nan")) for r in bd_per_k],
            dtype=float,
        )[bd_quality_mask]
        bd_signed_rise_fit = _signed_area_zero_crossing_fit(
            ks_signed, rise_signed, "signed_rise_area"
        )
        bd_signed_fall_fit = _signed_area_zero_crossing_fit(
            ks_signed, fall_signed, "signed_fall_area"
        )

    # Per-segment K estimator: median of k_estimate over every included
    # segment whose K passed the same quality gates as the other
    # extractors. Robust (median + MAD) and uses the whole sweep.
    bd_k_from_segments: float | None = None
    bd_k_from_segments_mad: float | None = None
    bd_k_from_segments_n = 0
    if bd_quality_mask.any():
        good_ks = {
            r.k for i, r in enumerate(bd_per_k) if bd_quality_mask[i]
        }
        k_ests = np.asarray(
            [
                s.metrics.get("k_estimate", float("nan"))
                for s in bd_segments
                if not s.excluded and s.k in good_ks
            ],
            dtype=float,
        )
        k_ests = k_ests[np.isfinite(k_ests)]
        if len(k_ests) >= 4:
            bd_k_from_segments = float(np.median(k_ests))
            bd_k_from_segments_mad = float(
                1.4826 * np.median(np.abs(k_ests - bd_k_from_segments))
            )
            bd_k_from_segments_n = int(len(k_ests))
            notes.append(
                f"per-segment K estimator (fall area / amplitude, "
                f"first-order theory): K_opt="
                f"{bd_k_from_segments:.4f} ± "
                f"{bd_k_from_segments_mad:.4f} (MAD) over "
                f"{bd_k_from_segments_n} segments"
            )

    if phase_fit is None:
        notes.append("Phase-lag fit failed — insufficient data or zero slope.")
    if integral_fit is None:
        notes.append("Integral-area fit failed — insufficient data or zero slope.")
    if bd_k_opt is None:
        notes.append(
            "bd_pressure K_opt fit failed — no K passed the segment-quality gate."
        )
    if bd_signed_fall_fit is not None:
        notes.append(
            "signed fall-area: +area=drool/under-PA, −area=undershoot/over-PA; "
            "optimum is the measured zero crossing (UI), linear-fit trend "
            f"R²={bd_signed_fall_fit.r_squared:.2f}"
        )
    if in_rate < 100:
        notes.append(f"Incoming sample rate is only {in_rate:.0f} Hz — analysis may be noisy.")

    # bd_pressure summary line: how many segments survived per K, and
    # what fraction of the total. Surfaces in the UI's notes block.
    total_segs = sum(r.n_segments_total for r in bd_per_k)
    total_incl = sum(r.n_segments_included for r in bd_per_k)
    if total_segs:
        notes.append(
            f"bd_pressure: {total_incl}/{total_segs} segments included "
            f"across {len(bd_per_k)} K values "
            f"(K_opt={bd_k_opt:.4f})"
            if bd_k_opt is not None
            else f"bd_pressure: {total_incl}/{total_segs} segments included "
            f"across {len(bd_per_k)} K values (no K_opt)"
        )

    return SweepAnalysis(
        per_k=per_k,
        phase_fit=phase_fit,
        integral_fit=integral_fit,
        integral_legacy_fit=integral_legacy_fit,
        sample_rate_hz=float(in_rate),
        baseline=baseline,
        notes=notes,
        windows=windows,
        force_baselines=force_baselines,
        bd_segments=bd_segments,
        bd_per_k=bd_per_k,
        bd_k_opt=bd_k_opt,
        bd_signed_rise_fit=bd_signed_rise_fit,
        bd_signed_fall_fit=bd_signed_fall_fit,
        bd_k_from_segments=bd_k_from_segments,
        bd_k_from_segments_mad=bd_k_from_segments_mad,
        bd_k_from_segments_n=bd_k_from_segments_n,
        bd_default_weights=dict(BD_DEFAULT_WEIGHTS),
    )
