"""Sweep-t0 anchoring and per-K window slicing.

Extracted verbatim from `analysis.py` (which re-exports these names for
backwards compatibility). This module owns everything that decides WHERE
the sweep is in the captured timeseries:

* burst-onset detectors (`_detect_sweep_start`, `_detect_first_pos_motion`,
  `_detect_z_marker_anchor`),
* cycle-boundary detectors (`_detect_pos_cycle_starts`,
  `_detect_pos_transitions`, `_detect_force_cycle_starts`),
* per-K window slicers (`_anchor_and_slice_from_pos`,
  `_slice_from_force_cycles`, `_slice_from_plan`,
  `_slice_from_pos_transitions`, `_slice_from_pos_with_known_anchor`),
* and `resolve_anchor_and_windows`, the precedence cascade that combines
  them into a single (sweep_t0, windows) resolution for `analyse_sweep`.

The allowed dependency direction is `analysis -> alignment`; this module
must never import from `analysis`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .gcode_gen import SweepPlan


def _detect_sweep_start(
    t: np.ndarray,
    y: np.ndarray,
    cycle_period_s: float,
    slow_half_s: float,
    n_model_cycles: int = 4,
    min_sustain_cycles: int = 1,
) -> float | None:
    """Find the wall-clock time the burst pattern actually begins.

    The runner's `sweep_t0` is set to the first UDP packet after job upload,
    but the printer spends 30-60+ s heating/homing/priming before the first
    burst. Using that t0 puts every per-K window inside the heatup phase
    and analysis returns NaN.

    Two-stage detection:

    1. **Coarse** -- rolling std over one cycle period. Threshold is
       relative to the GLOBAL MAX of stds (`0.25 · max_std`), not to
       the median of an arbitrary "head" window. Bursts produce std
       far larger than any pre-burst transient (homing, heating-element
       click, fan ramp-up); pegging the threshold to the run's own peak
       cleanly separates the burst region from everything before it.
       Then we require the threshold to be sustained for at least
       `min_sustain_cycles` cycles -- this rejects short loud spikes
       (loadcell tap, single mechanical event) that happen to exceed
       the threshold.

       The previous implementation pegged the threshold to "median of
       the first 25% of stds + 8·MAD", which is fooled when pre-burst
       noise (e.g. the user's case: M109-park motion at t≈22 s with
       loadcell std 130) lands inside the "head" window and pushes the
       baseline up so the burst region never registers as a clear
       outlier. Worse, the homing transient itself can exceed the
       threshold and the detector returns a t somewhere in the homing
       phase. Run inspection on the user's NPZ showed this returning
       t=7 s when the actual bursts start at t=100 s.

    2. **Fine** -- cross-correlate the (detrended) force trace against
       a model square wave, restricted to a narrow ±2-cycle window
       around the coarse estimate. The argmax there is the K=0 burst
       start, accurate to a sample.

    Sub-cycle precision matters: the per-K phase-lag fitter searches over
    ±1 s, so a `t0` error larger than one cycle aliases the lag estimate
    to the search-window boundary and the fit becomes meaningless.
    """
    n = len(t)
    if n < 200:
        return None
    sr = (n - 1) / max(1e-6, t[-1] - t[0])
    cycle_samples = max(8, int(round(cycle_period_s * sr)))
    slow_samples = max(1, int(round(slow_half_s * sr)))
    slow_samples = min(slow_samples, cycle_samples - 1)
    model_len = cycle_samples * n_model_cycles
    if model_len >= n - 4:
        return None

    yf_raw = np.asarray(y, dtype=np.float64)

    # --- stage 1: coarse rolling-std --------------------------------------
    csum = np.cumsum(yf_raw)
    csum2 = np.cumsum(yf_raw * yf_raw)
    sums = csum[cycle_samples:] - csum[:-cycle_samples]
    sums2 = csum2[cycle_samples:] - csum2[:-cycle_samples]
    means = sums / cycle_samples
    var = np.maximum(sums2 / cycle_samples - means * means, 0.0)
    stds = np.sqrt(var)
    if len(stds) < 4:
        return None
    max_std = float(np.max(stds))
    if max_std <= 0:
        return None
    # Threshold = 25% of global max. Burst std is typically 5-50× the
    # quietest pre-burst noise, so 25% of the peak comfortably exceeds
    # any pre-burst transient.
    thresh = 0.25 * max_std
    above = stds > thresh
    if not above.any():
        return None
    # Require N consecutive cycles of sustained activity above threshold
    # (was 1 cycle; now configurable, default 3). Short isolated spikes
    # like the loadcell-tap from G28 or a heater-click during M109
    # easily clear 1-cycle gating; 3 cycles essentially demands real
    # burst-pattern activity.
    min_sustain_n = cycle_samples * max(1, int(min_sustain_cycles))
    run = 0
    first_above: int | None = None
    coarse_idx: int | None = None
    for i, v in enumerate(above):
        if v:
            if run == 0:
                first_above = i
            run += 1
            if run >= min_sustain_n and first_above is not None:
                coarse_idx = first_above
                break
        else:
            run = 0
            first_above = None
    if coarse_idx is None:
        return None

    # --- stage 2: fine model-correlation in a ±2-cycle window -------------
    one_cycle = np.zeros(cycle_samples, dtype=np.float64)
    one_cycle[slow_samples:] = 1.0
    model = np.tile(one_cycle, n_model_cycles)
    model = model - model.mean()
    yf = sps.detrend(yf_raw, type="linear")

    lo = max(0, coarse_idx - 2 * cycle_samples)
    hi = min(n - model_len, coarse_idx + 2 * cycle_samples)
    if hi <= lo:
        return float(t[coarse_idx])
    # build the correlation only over the candidate window
    window = yf[lo : hi + model_len]
    corr = sps.correlate(window, model, mode="valid")
    if len(corr) < 1:
        return float(t[coarse_idx])
    peak_val = float(np.max(corr))
    if peak_val <= 0:
        return float(t[coarse_idx])
    # The model is exactly periodic, so correlating it against a clean
    # burst trace produces near-identical peaks at every cycle-offset
    # alignment (cycles 1-4 vs 2-5 of the bursts, etc.). Plain argmax
    # is then susceptible to picking a LATER cycle when noise nudges
    # it marginally higher. We want the EARLIEST high-correlation
    # alignment -- that's the actual sweep start. Take all indices
    # within 95% of the peak and return the smallest one.
    near_peak = corr >= 0.95 * peak_val
    earliest_argmax = int(np.argmax(near_peak))  # first True
    fine_idx = lo + earliest_argmax
    return float(t[fine_idx])


def _detect_pos_cycle_starts(
    pos_t: np.ndarray, pos_x: np.ndarray, quiet_frac: float = 0.1,
) -> np.ndarray:
    """Find every slow-leg start time from the pos_x oscillation pattern.

    Each sweep cycle is `slow leg (X ramps up) + fast leg (X ramps down)`,
    so the velocity of pos_x is positive during slow, negative during
    fast, ~zero during inter-burst settling. A cycle starts where the
    sign of velocity transitions from non-positive (settled or fast-leg-
    ending) to positive.

    Implementation: smooth the velocity with a ~100 ms moving average
    (removes sample-grid jitter), bucket each sample as +1/0/−1 around a
    `quiet_frac · peak_velocity` deadband, and emit a timestamp every
    time the sign transitions from {−1, 0} to +1. The deadband prevents
    noise during settled periods from masquerading as a cycle start, but
    we deliberately allow the FIRST positive sample after the initial
    quiet (sign 0 → +1) to count -- otherwise we'd lose cycle[0] of every
    burst sequence because the detector never saw a -1 before it.

    Asymmetric slow/fast legs work: even when peak fast-leg velocity is
    4× peak slow-leg velocity, the slow ramp still clears `quiet_frac ·
    peak_v` because we set the deadband loose (10%). A threshold-state
    machine that required slow velocity to clear 30% of peak failed in
    practice because slow velocity = peak_v / 4 = 25% of peak.

    Returns a 1-D array of cycle-start timestamps (host monotonic time).
    """
    if len(pos_t) < 10:
        return np.array([])
    velocity = np.gradient(pos_x.astype(float), pos_t.astype(float))
    dt = float(np.median(np.diff(pos_t)))
    if dt > 0:
        kernel_n = max(1, int(round(0.1 / dt)))
        if 1 < kernel_n < len(velocity):
            kernel = np.ones(kernel_n) / kernel_n
            velocity = np.convolve(velocity, kernel, mode="same")
    peak_v = float(np.percentile(np.abs(velocity), 95))
    if peak_v < 1e-6:
        return np.array([])
    quiet = quiet_frac * peak_v
    starts: list[float] = []
    last_sign = 0
    for i in range(len(velocity)):
        v = float(velocity[i])
        if v > quiet:
            sign = 1
        elif v < -quiet:
            sign = -1
        else:
            sign = 0
        if sign == 1 and last_sign != 1:
            starts.append(float(pos_t[i]))
        if sign != 0:
            last_sign = sign
    return np.asarray(starts, dtype=float)


def _detect_pos_transitions(
    pos_t: np.ndarray,
    pos_x: np.ndarray,
    expected_amplitude_mm: float | None = None,
    deadband_frac: float = 0.30,
    **_legacy_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Find every leg-transition time and direction from pos_x using a
    STICKY sign-of-delta algorithm — the same logic the live preview's
    JavaScript uses and that produces a clean square wave even when
    pos_x is firmware-quantized in coarse steps.

    Returns (times, directions) where:

      * direction +1 = pos_x reversed from increasing to decreasing, i.e.
        a peak. In the sweep gcode the slow leg moves X from x_base to
        x_base+dx (X increasing) and the fast leg returns (X decreasing),
        so a peak is the slow→fast transition in E (commanded velocity
        rising). Matches `_integral_area`'s "above crossings" convention.
      * direction −1 = pos_x reversed from decreasing to increasing, i.e.
        a trough = fast→slow transition = E velocity falling.

    Why sticky-delta instead of smoothed-velocity sign-flips:

    Buddy reports pos_x at ~56-80 Hz with a quantization step around
    0.05-0.1 mm. With coupled_dx_mm=1 and fast_half_s=0.8 s, the
    fast-leg velocity is ~1.25 mm/s, so the position advances ~22 µm
    between consecutive pos samples — BELOW the firmware's reporting
    resolution. The firmware emits stair-stepped pos_x: many samples
    at one value, then a sudden jump. `np.gradient` on this produces
    huge velocity spikes alternating with zero plateaus, and any
    smoothed-velocity sign-detector fires repeatedly inside one real
    leg (observed: 6 transitions in 0.84 s of what should be one
    fast leg on the user's 2026-05 NPZ).

    The sticky-delta approach is robust to quantization. We compute
    `delta = pos_x[i] − last_committed_x`. As long as delta stays
    inside ±deadband, the direction is held. Once |delta| ≥ deadband,
    the sign of delta becomes the new committed direction, and
    last_committed_x is updated. A real transition is emitted only on
    a sign change of the committed direction.

    `deadband_frac` is fraction of the observed pos_x oscillation
    amplitude (default 20%). Auto-discovered from `np.percentile(pos_x,
    95) − np.percentile(pos_x, 5)`, with a 0.02 mm minimum so a
    nearly-static pos_x doesn't fire on encoder dither.
    """
    if len(pos_t) < 5:
        return np.array([]), np.array([])
    x_arr = np.asarray(pos_x, dtype=float)
    t_arr = np.asarray(pos_t, dtype=float)
    # Deadband sizing. When the caller knows the expected oscillation
    # amplitude (`expected_amplitude_mm`, from `coupled_d{x,y,z}_mm`),
    # use it directly: deadband = 0.3·amplitude. Without that hint, we
    # auto-discover from 5%-95% percentiles -- BUT that can fail when
    # pos_x has a wide non-burst envelope (firmware park motion to
    # X=240, homing to X=0, etc.) that swamps the burst signal. So
    # only fall back to percentile when the caller didn't supply one.
    if expected_amplitude_mm is not None and expected_amplitude_mm > 0:
        amplitude = float(expected_amplitude_mm)
    else:
        x_lo = float(np.percentile(x_arr, 5))
        x_hi = float(np.percentile(x_arr, 95))
        amplitude = x_hi - x_lo
    if amplitude < 0.05:
        return np.array([]), np.array([])
    deadband = max(deadband_frac * amplitude, 0.02)

    # Peak-follower algorithm: track the running max while direction is
    # +1 (rising) and the running min while direction is -1 (falling).
    # A reversal is confirmed when pos_x moves `deadband` away from the
    # tracked extremum -- and the recorded transition timestamp is the
    # extremum's sample time, NOT the confirmation time. This eliminates
    # the detection lag the earlier "sticky-anchor" version had: the
    # anchor lagged ~0.3 mm behind pos_x through every leg, and the
    # reversal only fired after pos_x had moved another 0.3 mm past
    # the anchor in the new direction, putting the dashed wave 0.3-0.5 s
    # late vs the actual peak/trough. The peak-follower records the
    # extremum at its actual timestamp.
    times: list[float] = []
    dirs: list[float] = []
    direction = 0  # +1 rising, -1 falling, 0 not yet established
    extremum_x = float(x_arr[0])
    extremum_idx = 0
    for i in range(1, len(x_arr)):
        x = float(x_arr[i])
        if direction == 0:
            # Establish the initial direction from the first sustained
            # deadband-clearing move from the starting position.
            # extremum_x stays anchored to x_arr[0] here -- once direction
            # is established we'll start tracking the real running max/min.
            if x > extremum_x + deadband:
                direction = 1
                extremum_x = x
                extremum_idx = i
            elif x < extremum_x - deadband:
                direction = -1
                extremum_x = x
                extremum_idx = i
        elif direction > 0:
            if x > extremum_x:
                extremum_x = x
                extremum_idx = i
            elif x < extremum_x - deadband:
                # Peak confirmed -- emit transition at the extremum's time
                times.append(float(t_arr[extremum_idx]))
                dirs.append(1.0)  # peak = slow→fast in E = +1
                direction = -1
                extremum_x = x
                extremum_idx = i
        else:  # direction < 0
            if x < extremum_x:
                extremum_x = x
                extremum_idx = i
            elif x > extremum_x + deadband:
                # Trough confirmed -- emit transition at the extremum's time
                times.append(float(t_arr[extremum_idx]))
                dirs.append(-1.0)  # trough = fast→slow in E = -1
                direction = 1
                extremum_x = x
                extremum_idx = i
    return np.asarray(times, dtype=float), np.asarray(dirs, dtype=float)


def _anchor_and_slice_from_pos(
    pos_t: np.ndarray,
    pos_x: np.ndarray,
    plan: SweepPlan,
    n_validation: int = 3,
    notes: list[str] | None = None,
) -> tuple[float | None, list[tuple[float, float]] | None]:
    """One-shot: find sweep_t0 AND per-K (t_lo, t_hi) windows from pos_x.

    Three-stage pipeline:
      1. **Detect** every cycle start (slow-leg start) via velocity
         sign-flip in pos_x.
      2. **Anchor**: find the first run of `n_validation + 1` consecutive
         detected starts spaced at `cycle_period_s` (±30%). Anything
         earlier is junk (park trajectory, planner-lookahead jitter
         during M-code processing, homing transients) and discarded.
      3. **Rectify**: starting at the anchor, walk through detected
         starts and produce a clean period-spaced sequence:
           * Drop "too-close" detections (gap < 0.5·period): false
             positives, usually duplicates from a noisy velocity ramp.
           * Fill "too-far" gaps (> 1.5·period): a missed cycle. Insert
             synthetic starts at `previous + period` until the gap fits.
         This rectified sequence is what the per-K chunking consumes,
         so a single missed/extra mid-sweep detection no longer slides
         every downstream K window by one cycle (the user reported
         exactly this: late-sweep K plot looks aligned, earlier K plots
         look empty -- happens when one extra detection at the front
         pushes K[0..N-2] off by one and the last K's window happens to
         land on the real last burst's cycles by coincidence).

    Returns (sweep_t0, data_windows) on success, (None, None) when:
      * too few cycle starts detected to cover every K
      * no periodic run found (data is too noisy or no bursts present)
      * after rectification, fewer cycles remain than the plan expects.

    `notes`, when supplied, receives diagnostic lines about how many
    cycles were detected vs expected, how many were skipped/inserted,
    and the per-K start times in sweep-relative seconds. These are
    invaluable for diagnosing future misalignments from offline screenshots.
    """
    expected_total = sum(s.cycles for s in plan.segments)
    if expected_total <= 0 or not plan.segments:
        return None, None
    expected_period = plan.segments[0].cycle_period_s
    starts = _detect_pos_cycle_starts(pos_t, pos_x)
    if notes is not None:
        notes.append(
            f"raw cycle starts detected: {len(starts)} "
            f"(expected ≥ {expected_total} = sum of cycles per K)"
        )
    if len(starts) < expected_total:
        return None, None

    n_val = min(n_validation, max(1, expected_total - 1))
    tolerance = 0.3 * expected_period
    anchor_idx: int | None = None
    for i in range(len(starts) - n_val):
        diffs = np.diff(starts[i:i + n_val + 1])
        if np.all(np.abs(diffs - expected_period) <= tolerance):
            anchor_idx = i
            break
    if anchor_idx is None:
        return None, None
    if notes is not None and anchor_idx > 0:
        notes.append(
            f"anchor scan discarded {anchor_idx} non-periodic start(s) "
            f"before the first periodic run (park / planner jitter)"
        )

    # --- rectification pass ----------------------------------------------
    # Build a clean period-spaced run starting from the anchor. Drop
    # too-close false positives; insert synthetics for missed cycles.
    rectified: list[float] = [float(starts[anchor_idx])]
    n_skipped = 0
    n_inserted = 0
    for s in starts[anchor_idx + 1:]:
        gap = float(s) - rectified[-1]
        if gap < 0.5 * expected_period:
            n_skipped += 1
            continue
        n_missed = int(round(gap / expected_period)) - 1
        if n_missed > 0:
            for _ in range(n_missed):
                rectified.append(rectified[-1] + expected_period)
                n_inserted += 1
        rectified.append(float(s))
    # If after rectification we still don't have enough, extend with
    # synthetics. Past the last real detection the printer is presumably
    # still cycling at `expected_period`; this lets the per-K loop still
    # slice the tail of the sweep instead of giving up entirely.
    while len(rectified) < expected_total:
        rectified.append(rectified[-1] + expected_period)
        n_inserted += 1
    if notes is not None and (n_skipped or n_inserted):
        notes.append(
            f"cycle start rectification: skipped {n_skipped} too-close "
            f"detection(s), inserted {n_inserted} synthetic start(s) "
            f"for missed cycle(s)"
        )

    filtered = np.asarray(rectified[:expected_total], dtype=float)
    sweep_t0 = float(filtered[0] - plan.segments[0].start_offset_s)

    windows: list[tuple[float, float]] = []
    offset = 0
    for seg in plan.segments:
        n = seg.cycles
        t_start = float(filtered[offset] - sweep_t0)
        t_end = float(filtered[offset + n - 1] - sweep_t0 + seg.cycle_period_s)
        windows.append((t_start, t_end))
        offset += n

    if notes is not None:
        sample = ", ".join(
            f"K={seg.k:.4f}@{w[0]:.2f}s"
            for seg, w in list(zip(plan.segments, windows))[:6]
        )
        if len(plan.segments) > 6:
            sample += ", ..."
        notes.append(f"K window start times (sweep-rel): {sample}")
    return sweep_t0, windows


def _detect_force_cycle_starts(
    force_t: np.ndarray,
    force_y: np.ndarray,
    t_lo: float | None = None,
    t_hi: float | None = None,
    min_gap_s: float = 0.5,
) -> np.ndarray:
    """Find slow→fast rising edges in the loadcell trace.

    The burst signal has a long slow plateau and a brief fast plateau,
    cleanly separated by the loadcell response to the velocity step.
    A mid-level threshold crossing on the rising side gives one
    timestamp per cycle -- far more reliable than pos_x cycle
    detection, which on this firmware's 56 Hz pos throttle picks up
    sub-cycle features as false positives and misses real cycles
    when X amplitude is small.

    Implementation:
      * Slice the trace to [t_lo, t_hi] (default: full range). The
        caller usually passes the burst region from
        `_detect_sweep_start` so the percentile-based threshold isn't
        skewed by pre-burst noise.
      * Auto-discover plateau levels from the 10th and 90th
        percentiles of the windowed force.
      * Threshold = midpoint between plateaus.
      * Emit one timestamp per rising-edge crossing.
      * Merge edges closer than `min_gap_s` (defensive against
        loadcell ringing right after the transition crossing the
        threshold a second time).

    Returns a 1-D array of host-monotonic timestamps.
    """
    if len(force_t) < 50:
        return np.array([])
    if t_lo is None:
        t_lo = float(force_t[0])
    if t_hi is None:
        t_hi = float(force_t[-1])
    mask = (force_t >= t_lo) & (force_t <= t_hi)
    if int(mask.sum()) < 50:
        return np.array([])
    sub_t = force_t[mask]
    sub_y = force_y[mask]
    plateau_lo = float(np.percentile(sub_y, 10))
    plateau_hi = float(np.percentile(sub_y, 90))
    if plateau_hi - plateau_lo < 50.0:
        # Not enough dynamic range -- probably no bursts here.
        return np.array([])
    mid = 0.5 * (plateau_lo + plateau_hi)
    above = sub_y > mid
    rise_idx = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(rise_idx) == 0:
        return np.array([])
    # Linear-interpolate the exact threshold crossing between samples
    # for sub-sample precision.
    times: list[float] = []
    for i in rise_idx:
        t0_s, t1_s = float(sub_t[i - 1]), float(sub_t[i])
        v0, v1 = float(sub_y[i - 1]), float(sub_y[i])
        if v1 != v0:
            frac = (mid - v0) / (v1 - v0)
            frac = max(0.0, min(1.0, frac))
            t_cross = t0_s + frac * (t1_s - t0_s)
        else:
            t_cross = 0.5 * (t0_s + t1_s)
        # Skip edges too close to the previous one (loadcell ringing).
        if times and (t_cross - times[-1]) < min_gap_s:
            continue
        times.append(t_cross)
    return np.asarray(times, dtype=float)


def _slice_from_force_cycles(
    force_cycle_starts: np.ndarray,
    plan: SweepPlan,
    sweep_t0: float,
    notes: list[str] | None = None,
) -> tuple[list[tuple[float, float]], float] | None:
    """Chunk loadcell-derived cycle starts into per-K data windows.

    Trusts the force-cycle detector as ground truth: the burst signal
    is too clean for the detector to miss or double-count cycles
    (unlike pos_x, where small ~0.05 mm amplitudes can hide cycles
    below the firmware's position-reporting resolution).

    `_detect_force_cycle_starts` returns slow→fast rising edges, which
    sit at the END of each slow leg -- `slow_half_s` after the cycle
    actually begins. We shift each detected edge back by `slow_half_s`
    so the per-K window starts at the slow leg start (cycle start),
    matching the gcode's "slow leg then fast leg" cycle structure.
    Without this shift, K windows start mid-cycle at the rising edge
    and the FIRST low→high transition of each window is missing
    (observed in user's 2026-05 NPZ `run_1778939146.npz` -- K=0.0000
    began at the end-of-slow-leg rising edge instead of the slow leg
    start, so the dashed wave started HIGH and "low-high" was missing).

    The pipeline:
      1. Shift rising edges back by slow_half_s → cycle-start times.
      2. Drop starts before `sweep_t0 + start_offset_s − 0.3·period`
         (pre-burst false positives).
      3. If we have at least `cycles_per_K · n_K` remaining, slice in
         groups of `cycles_per_K`. Each K window spans from its first
         start to its last start + one cycle_period_s.
      4. If we have fewer remaining, the data ran short. Return None
         and let the caller fall back to plan offsets.
    """
    expected_total = sum(s.cycles for s in plan.segments)
    if expected_total <= 0 or not plan.segments:
        return None
    expected_period = plan.segments[0].cycle_period_s
    slow_half_s = float(plan.params.slow_half_s)
    rising_edges = np.asarray(force_cycle_starts, dtype=float)

    # Periodicity validation: walk the rising edges and find the FIRST
    # run of 3 consecutive period-spaced edges. The anchor of that run
    # is treated as a TRUSTED cycle boundary -- we don't trust any
    # other detected edges as cycle counters, only as refinements.
    # User's 2026-05 NPZ `run_1778941168.npz` had a single rising edge
    # at sweep-rel 102.74 followed by a 5.29 s gap before the real
    # bursts began at 108.03 -- the scan drops the noise edge before
    # anchoring.
    tolerance = 0.3 * expected_period
    if len(rising_edges) < 4:
        return None
    n_val = 3
    periodic_anchor_idx: int | None = None
    for i in range(len(rising_edges) - n_val):
        diffs = np.diff(rising_edges[i:i + n_val + 1])
        if np.all(np.abs(diffs - expected_period) <= tolerance):
            periodic_anchor_idx = i
            break
    if periodic_anchor_idx is None:
        if notes is not None:
            notes.append(
                "force-trace: no periodic run of 3 consecutive cycles "
                "found -- giving up on force-trace slicing"
            )
        return None

    # Refine: walk backward from the periodic anchor looking for
    # earlier detected edges that fit the same period (these are real
    # cycle starts that didn't make the run because edges before them
    # were noise). For each step back, accept the closest earlier
    # detected edge within tolerance of (anchor - k*period); stop as
    # soon as no edge fits. This handles the case where the FIRST
    # burst rising edge gets lost in noise but later cycles are clean.
    anchor_t = float(rising_edges[periodic_anchor_idx])
    earlier_edges = rising_edges[:periodic_anchor_idx]
    n_back = 0
    while len(earlier_edges):
        target = anchor_t - expected_period
        diffs_to_target = np.abs(earlier_edges - target)
        idx_closest = int(np.argmin(diffs_to_target))
        if diffs_to_target[idx_closest] <= tolerance:
            anchor_t = float(earlier_edges[idx_closest])
            earlier_edges = earlier_edges[:idx_closest]
            n_back += 1
        else:
            break
    if n_back and notes is not None:
        notes.append(
            f"force-trace: extended anchor backward {n_back} cycle(s) "
            f"by matching earlier detected edges to the periodic pattern"
        )

    # PREDICT each rising edge from the anchor at integer multiples of
    # expected_period. Then snap each predicted edge to the nearest
    # detected edge within tolerance (sub-sample refinement) or fall
    # back to the predicted value when no detected edge is close
    # enough. This is fundamentally more robust than counting every
    # detected edge as a "cycle start", because higher-K bursts emit
    # SPURIOUS extra rising edges per cycle: the PA-induced velocity-
    # reversal undershoot recovers above the mid-threshold, crossing
    # it a second time. Observed on user's run_1778962189.npz: K[1]
    # and K[2] produced ~2 rising edges per cycle, shifting every K
    # window after K[0] by one cycle period.
    # Rolling-anchor prediction: each cycle's prediction is "the
    # PREVIOUS cycle's actual time + expected_period". Snap to nearest
    # detected edge within tolerance, else use the predicted value
    # (synthetic). Re-anchoring on every snap means a small mismatch
    # between the nominal `expected_period` and the firmware's actual
    # cycle time CAN'T accumulate beyond one cycle -- as soon as the
    # next real edge appears we resync. Observed on user's
    # run_1779016571.npz: a global-anchor prediction (anchor + i ×
    # period) drifted ~2 s by cycle 170 because the firmware's actual
    # period was slightly off from the planned 3.0 s, and 44 of the
    # 210 cycles were synthetic. K=0.0850 seg 0's t_start ended up
    # 2 s BEFORE the actual cycle's slow-leg start (no data there) --
    # the displayed segment showed only high_n + low_{n+1}, missing
    # low_n entirely.
    refined_rising = np.empty(expected_total, dtype=float)
    refined_rising[0] = anchor_t
    n_snapped = 1   # the anchor itself counts as snapped
    n_synth = 0
    for k in range(1, expected_total):
        predicted = refined_rising[k - 1] + expected_period
        diffs = np.abs(rising_edges - predicted)
        if len(diffs):
            j = int(np.argmin(diffs))
            if diffs[j] <= tolerance:
                refined_rising[k] = float(rising_edges[j])
                n_snapped += 1
                continue
        refined_rising[k] = predicted
        n_synth += 1
    if notes is not None:
        notes.append(
            f"force-trace: predicted {expected_total} cycle starts via "
            f"rolling-anchor (snapped {n_snapped} to detected edges, "
            f"{n_synth} synthetic where no edge was within "
            f"{tolerance:.2f}s of the rolling prediction)"
        )

    # Shift: the rising edge is at the END of the slow leg, but the
    # cycle BEGINS at the start of the slow leg, slow_half_s earlier.
    used = refined_rising - slow_half_s
    # NB: `used` was built by shifting rising_edges back by slow_half_s
    # uniformly. For K[i] segments with `first_cycle_slow_extension_s > 0`
    # (only K[0] under the warm-up scheme), the warm-up sits BEFORE
    # `used[offset]` -- but we deliberately DON'T include it in the
    # window. The warm-up's job is to establish steady-state melt
    # pressure BEFORE the first measurable transition; showing the
    # full 20 s warm-up ramp at the front of K[0]'s plot just makes
    # it look different from K[1+] without adding analytical value.
    # We start K[0] at the LAST `slow_half_s` of the warm-up
    # (= already at slow plateau), which matches the layout of every
    # other K window: opens on a clean slow plateau, runs N cycles,
    # closes on a clean slow plateau (latter via the +slow_half
    # extension applied by the caller).
    windows: list[tuple[float, float]] = []
    offset = 0
    for seg in plan.segments:
        n = seg.cycles
        # `used[offset]` is the cycle 0 slow-leg start with the
        # uniform slow_half_s shift already applied. For K[0] this
        # is "warm-up end minus slow_half_s" = last slow_half_s of
        # the warm-up. We use that as t_start so every K's window
        # has the same length and structure.
        t_start = float(used[offset] - sweep_t0)
        # K[i] cycle N-1 slow-leg start + cycle_period = K[i+1] cycle 0
        # slow-leg start = shared boundary.
        t_end = float(used[offset + n - 1] - sweep_t0 + expected_period)
        windows.append((t_start, t_end))
        offset += n
    if notes is not None:
        sample = ", ".join(
            f"K={seg.k:.4f}@{w[0]:.2f}s"
            for seg, w in list(zip(plan.segments, windows))[:6]
        )
        if len(plan.segments) > 6:
            sample += ", ..."
        notes.append(f"K window start times (sweep-rel, force-cycle): {sample}")
    # Return both windows AND the absolute time of cycle 0's slow-leg
    # start. The caller can use the absolute anchor to refine sweep_t0
    # if its prior anchor was off: any disagreement larger than one
    # cycle period means the force-trace (high-SNR, full-trace scan)
    # disagrees with the prior anchor (pos_x park motion, loadcell
    # rolling-std, etc.), and the force-trace should win.
    cycle0_abs = float(used[0])
    return windows, cycle0_abs


def _slice_from_plan(plan: SweepPlan) -> list[tuple[float, float]]:
    """Plan-direct K windows: use `start_offset_s` and `duration_s` exactly.

    The G-code generator is the ground truth for the sweep schedule. It
    knows the per-segment start times relative to `sweep_t0` AND it
    encodes any non-uniform cycle behaviour (e.g. K[0] cycle 0 is
    `warmup_factor × slow_half + fast_half` long instead of the usual
    `slow_half + fast_half`, so the very first segment can be 11 s while
    every later K is 3 s). Slicing directly from `seg.start_offset_s`
    and `seg.duration_s` therefore handles warm-up, accel ramps, and
    any future per-segment quirks without needing the analyser to
    rediscover them from edge detection.

    This is the preferred slicer whenever we have a trusted anchor (the
    Z-marker pulse is unambiguous when emitted). Force/pos_x edge
    detection only earns its keep when the anchor itself is uncertain.

    Returns one `(t_lo, t_hi)` pair per segment in sweep-relative
    seconds, where `t_lo = seg.start_offset_s` and `t_hi = t_lo +
    seg.duration_s`. The caller may extend `t_hi` to include the
    trailing slow leg of the following K (matches existing display
    convention).
    """
    return [
        (float(seg.start_offset_s), float(seg.start_offset_s + seg.duration_s))
        for seg in plan.segments
    ]


def _slice_from_pos_transitions(
    pos_t: np.ndarray,
    pos_x: np.ndarray,
    plan: SweepPlan,
    sweep_t0_estimate: float,
    coupled_amplitude_mm: float,
    notes: list[str] | None = None,
) -> tuple[list[tuple[float, float]], float] | None:
    """Slice K windows directly from pos_x leg-direction transitions.

    pos_x is the most reliable cycle-boundary signal we have:
      * Each commanded cycle drives the toolhead a known `coupled_d*_mm`
        away from purge and back. The sign-of-delta detector emits one
        +1 (slow→fast = X reverses from rising to falling) and one -1
        (fast→slow = X reverses from falling to rising) per cycle.
      * The transitions are physically guaranteed -- the toolhead
        actually moved -- so unlike force-trace threshold crossings they
        don't fail under purge spikes or under-amplified plateaus.
      * They land at the actual cycle moment regardless of any firmware
        processing delay between gcode parse and motion execution.

    The third point is why this slicer is preferred over plan-direct
    slicing whenever pos_x oscillation is available: on the user's
    run_1779100636 the gap between Z-marker DOWN (the supposed sweep_t0)
    and the FIRST pos_x +1 transition was ~2.3 s longer than the
    plan's predicted 10.5 s, almost certainly because M572 / M83 /
    G1-with-XYZ planner-sync took ~2.3 s the plan can't see. Plan-
    direct slicing put every K window 2.3 s before the data; pos_x-
    transition slicing puts them on the data.

    Algorithm (DATA-DRIVEN for K[0], plan-driven for K[1..]):
      1. Detect all pos_x transitions via `_detect_pos_transitions`.
      2. Strip leading -1s that come BEFORE the first +1 (those are
         pre-burst rebounds, not real cycles).
      3. Require at least `2 × total_cycles` transitions remaining,
         alternating (+1, -1).
      4. K[0]'s first slow leg start is detected from the DATA, not the
         plan: scan pos_x backwards from the first +1 transition for
         the latest time pos_x sat within `0.5·amplitude` of the pre-
         burst baseline. That instant is when the toolhead actually
         started moving. We use this instead of inverting
         `K[0].duration_s` because the plan's `first_slow_leg_factor`
         may be wrong on replay (NPZ dumps did not store it), and the
         data is always authoritative.
      5. For each K[i]:
         * t_lo: start of K[i] cycle 0's slow leg. i=0 → detected K[0]
           motion start from step 4. i>0 → -1 transition of K[i-1]'s
           last cycle (slow leg of K[i] starts where K[i-1] ended).
         * t_hi: -1 transition of K[i]'s last cycle.
      6. Refine sweep_t0 from K[0]'s detected slow leg start:
         `refined_t0 = K[0]_slow_start - K[0].start_offset_s`.

    Returns `(windows, refined_sweep_t0)` on success, `None` on any
    structural mismatch (wrong number of transitions, non-alternating
    pattern, etc.).
    """
    if not plan.segments or coupled_amplitude_mm <= 0:
        return None

    # Self-calibrate the amplitude from the actual burst region. The
    # caller's `coupled_amplitude_mm` is a HINT (taken from
    # plan.params.coupled_dx_mm) but on replay that value may be wrong:
    # the NPZ format didn't store coupled_dx_mm originally, so the
    # replay reconstructor falls back to a percentile heuristic that
    # picks up the park motion (X≈240 mm) instead of the cycle
    # amplitude (X≈1 mm).
    # Strategy: take the pos_x samples in the rough burst window
    # (`sweep_t0_estimate` to `+ total sweep duration`), compute the
    # spread of the SMALL-amplitude oscillation within them, and use
    # that as the detector amplitude. The deadband then matches the
    # actual cycle motion regardless of any pre/post burst transit.
    period = plan.segments[0].cycle_period_s
    sweep_duration = (
        plan.segments[-1].start_offset_s
        + plan.segments[-1].duration_s
    )
    earliest_t = sweep_t0_estimate - 0.5 * period
    burst_lo = sweep_t0_estimate
    burst_hi = sweep_t0_estimate + sweep_duration + 2.0 * period
    pos_t_arr = np.asarray(pos_t, dtype=float)
    pos_x_arr = np.asarray(pos_x, dtype=float)
    burst_mask = (pos_t_arr >= burst_lo) & (pos_t_arr <= burst_hi)
    n_burst = int(burst_mask.sum())
    if n_burst < 20:
        return None
    burst_x = pos_x_arr[burst_mask]
    # Within the burst region the toolhead oscillates around `purge_x`
    # with the small `coupled_d*_mm` amplitude. Estimate the amplitude
    # from the spread; clamp to a sensible band so park motions that
    # leak into the burst window don't blow up the deadband and so a
    # tiny encoder dither doesn't collapse the deadband to zero.
    p5_b = float(np.percentile(burst_x, 5))
    p95_b = float(np.percentile(burst_x, 95))
    auto_amplitude = max(p95_b - p5_b, 0.05)
    auto_amplitude = min(auto_amplitude, 50.0)
    # Use whichever amplitude is SMALLER -- if the caller's hint is
    # 1 mm and the auto-discovered one is 0.5 mm, the burst is
    # likely smaller than the hint; if the caller's hint is 215 mm
    # (the broken replay heuristic) and the auto-discovered is 1 mm,
    # the burst is the small one.
    detector_amplitude = min(coupled_amplitude_mm, auto_amplitude)
    if detector_amplitude < 0.05:
        detector_amplitude = max(coupled_amplitude_mm, 0.05)

    transitions_t, transitions_d = _detect_pos_transitions(
        pos_t, pos_x, expected_amplitude_mm=detector_amplitude,
    )
    if len(transitions_t) < 2:
        return None

    mask = transitions_t >= earliest_t
    trans_t = transitions_t[mask]
    trans_d = transitions_d[mask]
    if len(trans_t) < 2:
        return None
    # Drop leading -1s. The first burst transition is ALWAYS a +1
    # (end of K[0]'s slow leg). Any -1 ahead of that is a homing
    # rebound or pre-purge artifact.
    first_plus = None
    for i, d in enumerate(trans_d):
        if d > 0:
            first_plus = i
            break
    if first_plus is None:
        return None
    trans_t = trans_t[first_plus:]
    trans_d = trans_d[first_plus:]

    total_cycles = sum(s.cycles for s in plan.segments)
    expected_n_trans = 2 * total_cycles
    if len(trans_t) < expected_n_trans:
        return None
    trans_t = trans_t[:expected_n_trans]
    trans_d = trans_d[:expected_n_trans]

    # Validate alternation. Each cycle = (+1, -1). If anything in the
    # leading 2·total_cycles transitions breaks the pattern, abandon
    # this slicer -- something is off (missed cycle, double-counted
    # transition, planner stall) and the caller's fallback is safer.
    for i in range(expected_n_trans):
        expected = 1 if (i % 2 == 0) else -1
        if int(trans_d[i]) != expected:
            if notes is not None:
                notes.append(
                    f"pos_x transitions: alternation broken at index {i} "
                    f"(expected dir={expected}, got {int(trans_d[i])}) -- "
                    f"falling through to next slicer"
                )
            return None

    plus_times = trans_t[::2]   # +1 transitions: 1 per cycle
    minus_times = trans_t[1::2]  # -1 transitions: 1 per cycle

    # DETECT K[0]'s slow leg start from the data, not the plan.
    # Strategy: walk BACKWARDS from the first +1 transition. The
    # toolhead during the slow leg ramps from purge_x to purge_x +
    # coupled_dx_mm. We define `baseline_x` as the value pos_x took
    # right before the slow leg started -- that's `purge_x`. We then
    # find the latest sample whose pos_x sat within
    # `motion_threshold` of baseline_x. The very next sample after
    # that is the slow-leg onset.
    # Walking backwards (not forwards) avoids the trap that broke an
    # earlier attempt: scanning the pre-burst window forwards would
    # latch on to the park-to-purge transit motion (X=240 → X=30)
    # which happens BEFORE the K[0] slow leg and produces a HUGE
    # rise in pos_x that the threshold-cross sees as "motion start".
    # The slow leg itself is much smaller (~1 mm) but the burst-region
    # auto-amplitude makes the deadband small enough to detect it.
    first_plus_t = float(plus_times[0])
    pos_t_arr = np.asarray(pos_t, dtype=float)
    pos_x_arr = np.asarray(pos_x, dtype=float)
    # baseline_x = the TROUGH of pos_x near the first -1 transition.
    # That trough is the toolhead's idle/purge_x position. Median over
    # a window after the -1 would already include the rising ramp of
    # K[i+1]'s slow leg (pos_x updates at ~56 Hz so within 100 ms the
    # toolhead has moved 0.05-0.10 mm up); using the minimum captures
    # the actual purge_x.
    first_minus_t = float(minus_times[0])
    base_mask = (pos_t_arr >= first_minus_t - 0.3) & (
        pos_t_arr <= first_minus_t + 0.3
    )
    if int(base_mask.sum()) < 3:
        if notes is not None:
            notes.append(
                "pos_x transitions: not enough samples around first -1 "
                "transition to find baseline -- falling through to next slicer"
            )
        return None
    baseline_x = float(np.min(pos_x_arr[base_mask]))
    # Tight motion threshold: just above pos_x quantization (~0.05 mm
    # on Buddy). 0.2·amplitude is too coarse for the warm-up slow leg,
    # which ramps SLOWLY -- the first few samples can sit 0.03..0.1 mm
    # above baseline for a couple of seconds before a coarser threshold
    # would fire. We want to catch the actual motion onset, not the
    # later "well above baseline" point. Floor at 0.03 mm so encoder
    # dither doesn't false-fire; cap at 0.2·amplitude so a large
    # coupled motion still uses a fraction-of-amplitude criterion.
    motion_threshold = min(max(0.05 * detector_amplitude, 0.03),
                           0.2 * detector_amplitude)

    # Walk BACKWARDS from the first +1 transition through pos_x. The
    # last sample where pos_x sits within motion_threshold of baseline_x
    # is the latest moment the toolhead was idle. The very next sample
    # is when motion started.
    search_lo = first_plus_t - 60.0  # cap the search at 60 s before
    pre_mask = (pos_t_arr >= search_lo) & (pos_t_arr <= first_plus_t)
    pre_idxs = np.where(pre_mask)[0]
    if len(pre_idxs) < 5:
        if notes is not None:
            notes.append(
                "pos_x transitions: too few samples in 60 s before first +1 "
                "transition -- falling through to next slicer"
            )
        return None
    # Find the LAST index where pos_x was within motion_threshold of baseline_x.
    at_baseline = (
        np.abs(pos_x_arr[pre_idxs] - baseline_x) <= motion_threshold
    )
    if not at_baseline.any():
        if notes is not None:
            notes.append(
                f"pos_x transitions: pos_x never settled within "
                f"{motion_threshold:.3f} mm of baseline {baseline_x:.3f} "
                f"in 60 s before the first +1 -- falling through to next slicer"
            )
        return None
    last_idle_local = int(np.where(at_baseline)[0][-1])
    last_idle_global = int(pre_idxs[last_idle_local])
    # The slow leg onset is somewhere between this idle sample and the
    # next sample (which is above motion_threshold).
    if last_idle_global + 1 < len(pos_t_arr):
        t_idle = float(pos_t_arr[last_idle_global])
        t_next = float(pos_t_arr[last_idle_global + 1])
        v_idle = float(pos_x_arr[last_idle_global])
        v_next = float(pos_x_arr[last_idle_global + 1])
        if v_next != v_idle:
            crossing = baseline_x + motion_threshold
            frac = (crossing - v_idle) / (v_next - v_idle)
            frac = max(0.0, min(1.0, frac))
            k0_slow_start_abs = t_idle + frac * (t_next - t_idle)
        else:
            k0_slow_start_abs = 0.5 * (t_idle + t_next)
    else:
        k0_slow_start_abs = float(pos_t_arr[last_idle_global])

    # Refine sweep_t0 from the detected K[0] slow leg start.
    refined_t0 = k0_slow_start_abs - plan.segments[0].start_offset_s

    windows: list[tuple[float, float]] = []
    cycle_offset = 0
    for seg_idx, seg in enumerate(plan.segments):
        n_cycles = seg.cycles
        last_minus_idx = cycle_offset + n_cycles - 1
        if seg_idx == 0:
            slow_start_abs = k0_slow_start_abs
        else:
            # K[i]'s first slow leg starts where K[i-1] ended.
            slow_start_abs = float(minus_times[cycle_offset - 1])
        end_abs = float(minus_times[last_minus_idx])
        t_lo = slow_start_abs - refined_t0
        t_hi = end_abs - refined_t0
        windows.append((t_lo, t_hi))
        cycle_offset += n_cycles

    if notes is not None:
        shift = refined_t0 - sweep_t0_estimate
        warmup_duration = first_plus_t - k0_slow_start_abs
        notes.append(
            f"K windows sliced from pos_x transitions ({len(trans_t)} "
            f"transitions, {total_cycles} cycles); K[0] slow leg "
            f"detected from data spans {warmup_duration:.2f}s "
            f"(plan said {plan.segments[0].duration_s - plan.params.fast_half_s:.2f}s, "
            f"data-driven detection is authoritative); sweep_t0 "
            f"refined by {shift:+.2f}s from prior anchor"
        )
        sample = ", ".join(
            f"K={seg.k:.4f}@{w[0]:.2f}s"
            for seg, w in list(zip(plan.segments, windows))[:6]
        )
        if len(plan.segments) > 6:
            sample += ", ..."
        notes.append(f"K window start times (sweep-rel, pos_x-trans): {sample}")
    return windows, refined_t0


def _slice_from_pos_with_known_anchor(
    pos_t: np.ndarray,
    pos_x: np.ndarray,
    plan: SweepPlan,
    sweep_t0: float,
    notes: list[str] | None = None,
) -> list[tuple[float, float]] | None:
    """Chunk pos_x cycle starts into per-K data windows when the
    sweep_t0 anchor is already known (e.g. from the Z marker).

    Skips the periodicity-validation step `_anchor_and_slice_from_pos`
    does for its own anchor discovery; instead, just trusts `sweep_t0`
    and rectifies the cycle starts that follow it.

    Returns a list of (t_lo, t_hi) windows in sweep-relative seconds,
    one per plan segment. Returns None when too few cycle starts
    follow the anchor.
    """
    expected_total = sum(s.cycles for s in plan.segments)
    if expected_total <= 0 or not plan.segments:
        return None
    expected_period = plan.segments[0].cycle_period_s
    starts = _detect_pos_cycle_starts(pos_t, pos_x)
    # First burst is expected at sweep_t0 + start_offset_s. Allow a
    # generous backshift (0.3·period) to catch any anchor-timing slack.
    first_expected_abs = sweep_t0 + plan.segments[0].start_offset_s
    after_anchor = starts[starts >= first_expected_abs - 0.3 * expected_period]
    if notes is not None:
        notes.append(
            f"raw cycle starts after Z-marker anchor: {len(after_anchor)} "
            f"(expected ≥ {expected_total})"
        )
    if len(after_anchor) < 1:
        return None

    # Rectify: drop too-close false positives, fill gaps with synthetics.
    rectified: list[float] = [float(after_anchor[0])]
    n_skipped = 0
    n_inserted = 0
    for s in after_anchor[1:]:
        gap = float(s) - rectified[-1]
        if gap < 0.5 * expected_period:
            n_skipped += 1
            continue
        n_missed = int(round(gap / expected_period)) - 1
        if n_missed > 0:
            for _ in range(n_missed):
                rectified.append(rectified[-1] + expected_period)
                n_inserted += 1
        rectified.append(float(s))
    while len(rectified) < expected_total:
        rectified.append(rectified[-1] + expected_period)
        n_inserted += 1
    if notes is not None and (n_skipped or n_inserted):
        notes.append(
            f"cycle-start rectification (Z-anchored): skipped {n_skipped} "
            f"too-close, inserted {n_inserted} synthetic"
        )

    filtered = np.asarray(rectified[:expected_total], dtype=float)
    windows: list[tuple[float, float]] = []
    offset = 0
    for seg in plan.segments:
        n = seg.cycles
        t_start = float(filtered[offset] - sweep_t0)
        t_end = float(filtered[offset + n - 1] - sweep_t0 + seg.cycle_period_s)
        windows.append((t_start, t_end))
        offset += n
    if notes is not None:
        sample = ", ".join(
            f"K={seg.k:.4f}@{w[0]:.2f}s"
            for seg, w in list(zip(plan.segments, windows))[:6]
        )
        if len(plan.segments) > 6:
            sample += ", ..."
        notes.append(f"K window start times (sweep-rel, Z-anchored): {sample}")
    return windows


def _detect_z_marker_anchor(
    pos_z_t: np.ndarray,
    pos_z: np.ndarray,
    expected_lift_mm: float = 2.0,
) -> float | None:
    """Find the host-clock time the toolhead returned to baseline after
    the sweep's pre-burst Z marker pulse.

    The gcode generator emits a unique-signature Z motion immediately
    before the first burst: lift the toolhead by `expected_lift_mm`,
    brief dwell, drop back. Nothing else in the run produces a Z
    excursion of this magnitude, so a single bump in pos_z anchored to
    its return-to-baseline gives us an unambiguous sweep_t0 — far more
    robust than periodicity-based cycle-start detection on a noisy
    pos_x trace where park motions, planner-lookahead jitter, and the
    bursts themselves all look superficially similar.

    Detection:

    1. Auto-discover the resting Z baseline from the first ~1 s of pos_z
       (median is robust to a few outliers).
    2. Walk forward; find the first sample where pos_z exceeds
       `baseline + 0.5·expected_lift_mm` (toolhead has clearly lifted).
    3. From there, find the first sample where pos_z is back within
       `0.3·expected_lift_mm` of the baseline. Linearly interpolate the
       threshold crossing for sub-sample precision.

    Returns the host-clock timestamp of the return, or None when no
    Z bump of the expected magnitude is found (older gcode without
    the marker, or pos_z metric not enabled).
    """
    if len(pos_z_t) < 10 or expected_lift_mm <= 0:
        return None
    dt_med = max(float(np.median(np.diff(pos_z_t))), 1e-6)

    # Buddy emits placeholder pos_z values BEFORE homing/positioning
    # completes -- 168.000 (axis-max sentinel) for tens of seconds,
    # then big negative excursions to -160+ during homing. A naive
    # global baseline = median(head) lands on the sentinel and makes
    # the lift threshold unreachable.
    #
    # Detect by SIGNATURE instead: walk a sliding LOCAL baseline
    # (median over a recent settled lookback). For each sample, ask
    # "did pos_z just lift by ≈expected_lift_mm above the local
    # baseline AND return within a short window?". The sentinel
    # itself never moves, so no lift fires. Homing excursions are
    # too LARGE to be a +expected_lift_mm marker, and they don't
    # return to baseline within the marker timescale. Only the
    # actual marker pulse matches.
    LOOKBACK_S = 0.5    # window to compute local baseline before each sample
    GUARD_S = 0.25      # gap between lookback end and candidate sample,
                        # so the lift ramp itself doesn't contaminate the
                        # baseline estimate
    SETTLE_PP_MM = 0.5  # lookback must be settled to this PP for a valid baseline
    RETURN_WINDOW_S = 1.5  # marker must return to baseline within this
    lookback_n = max(5, int(round(LOOKBACK_S / dt_med)))
    guard_n = max(1, int(round(GUARD_S / dt_med)))
    return_n = max(10, int(round(RETURN_WINDOW_S / dt_med)))
    lift_threshold_rel = 0.5 * expected_lift_mm
    return_threshold_rel = 0.3 * expected_lift_mm

    for i in range(lookback_n + guard_n, len(pos_z) - return_n):
        lookback = pos_z[i - lookback_n - guard_n:i - guard_n]
        lb_min = float(lookback.min())
        lb_max = float(lookback.max())
        if lb_max - lb_min > SETTLE_PP_MM:
            continue  # lookback not settled — could be homing or motion
        local_baseline = float(np.median(lookback))
        if float(pos_z[i]) <= local_baseline + lift_threshold_rel:
            continue
        # Candidate lift. Find the first sample within RETURN_WINDOW_S
        # that drops back to within return_threshold_rel of local_baseline.
        look_end = min(i + return_n, len(pos_z))
        return_thr = local_baseline + return_threshold_rel
        for j in range(i + 1, look_end):
            if float(pos_z[j]) < return_thr:
                t0, t1 = float(pos_z_t[j - 1]), float(pos_z_t[j])
                v0, v1 = float(pos_z[j - 1]), float(pos_z[j])
                if v1 != v0:
                    frac = (return_thr - v0) / (v1 - v0)
                    frac = max(0.0, min(1.0, frac))
                    return t0 + frac * (t1 - t0)
                return 0.5 * (t0 + t1)
        # Lift didn't return — not a marker (probably a real Z move).
    return None


def _detect_first_pos_motion(
    pos_t: np.ndarray, pos_x: np.ndarray, baseline: float, amplitude: float,
    cycle_period_s: float | None = None,
) -> float | None:
    """Find the host-clock time of the first burst-induced X step.

    AUTO-BASELINE: we do not trust the `baseline` argument (kept for API
    stability but unused). For each sample we compute the local pos_x
    range over a sliding 500 ms lookback; the first sample whose
    preceding window was stable AND that exceeds the window's minimum
    by > 0.5·amplitude is a candidate motion onset.

    OSCILLATION GATE: a "candidate" only becomes a real anchor if pos_x
    RETURNS to within `2·amplitude` of the pre-motion baseline within
    `cycle_period_s` (or 3 s if not given). This is what distinguishes
    a BURST -- a small oscillation that immediately reverses -- from a
    TRANSIT -- a large one-way move (e.g. firmware parking the
    toolhead to X=240 for heating, observed on the user's Core One).
    Without this gate, the detector triggered on the park motion at
    t≈22 s on the user's 2026-05 run and shifted every per-K window
    77 s before the actual bursts.

    Rejects park / homing trajectories two ways: large amplitudes blow
    past the lookback stability check (existing behaviour) AND moves
    that don't reverse within a cycle period are filtered (new).
    """
    del baseline  # auto-discovered; argument kept for API stability
    if len(pos_t) < 10 or amplitude <= 0:
        return None
    dt_median = float(np.median(np.diff(pos_t)))
    if dt_median <= 0:
        return None
    window_n = max(5, int(round(0.5 / dt_median)))
    if window_n >= len(pos_t):
        return None
    stability_threshold = max(0.2 * amplitude, 0.05)  # mm
    motion_threshold = 0.5 * amplitude  # mm
    reversal_window_s = float(cycle_period_s) if cycle_period_s else 3.0
    reversal_n = max(10, int(round(reversal_window_s / dt_median)))
    reversal_tol = 2.0 * amplitude

    for i in range(window_n, len(pos_t)):
        win = pos_x[i - window_n:i]
        win_min = float(win.min())
        win_max = float(win.max())
        if win_max - win_min > stability_threshold:
            continue  # not settled in the lookback
        if pos_x[i] - win_min > motion_threshold:
            # Oscillation gate: require pos_x to return close to
            # win_min within reversal_window_s, otherwise this is a
            # transit (firmware-park, home, large repositioning) not a
            # burst.
            look_end = min(i + reversal_n, len(pos_x))
            future = pos_x[i:look_end]
            if np.any(np.abs(future - win_min) <= reversal_tol):
                return float(0.5 * (pos_t[i - 1] + pos_t[i]))
            # else: not oscillating -- keep scanning forward for the
            # real burst onset.
    return None


@dataclass(slots=True)
class AnchorResolution:
    """Outcome of `resolve_anchor_and_windows`.

    `sweep_t0` is the (possibly re-anchored) sweep origin on the host
    monotonic clock; `windows` is the per-K (t_lo, t_hi) list in
    sweep-relative seconds, or None when no detector produced data-derived
    windows (caller falls back to plan-direct slicing); `source` names the
    detector that won the anchor; `t0_is_anchored` reports whether ANY
    anchor (caller-supplied or detected) is in effect.
    """
    sweep_t0: float
    windows: list[tuple[float, float]] | None
    source: str | None
    t0_is_anchored: bool


def resolve_anchor_and_windows(
    sweep_t0: float,
    force_t: np.ndarray,
    force_y: np.ndarray,
    plan: SweepPlan,
    *,
    t0_is_anchored: bool,
    auto_detect_t0: bool,
    pos_t: np.ndarray | None = None,
    pos_x: np.ndarray | None = None,
    pos_z_t: np.ndarray | None = None,
    pos_z: np.ndarray | None = None,
    z_marker_lift_mm: float = 2.0,
    notes: list[str],
) -> AnchorResolution:
    """Resolve the sweep_t0 anchor AND (where possible) per-K data windows.

    This is the anchor-precedence cascade formerly inlined in
    `analyse_sweep`; logic and diagnostic note strings are unchanged.
    Decision table -- stages are tried top to bottom, first hit wins the
    anchor (the force-trace post-pass may still override it):

    | # | source                | trigger                                    | produces           |
    |---|-----------------------|--------------------------------------------|--------------------|
    | 1 | `gcode_event`         | caller passed `t0_is_anchored=True`        | anchor only        |
    | 2 | `z_marker`            | Z lift+drop marker pulse found in pos_z    | anchor + windows   |
    |   | `pos_x_transitions`   | ... refined by pos_x leg transitions when  | anchor + windows   |
    |   |                       | pos_x is usable (preferred over 2's        | (data-derived)     |
    |   |                       | plan-direct windows; also refines t0)      |                    |
    | 3 | `pos_x_periodic`      | validated periodic run of pos_x cycle      | anchor + windows   |
    |   |                       | starts; rejected if it would place the     |                    |
    |   |                       | sweep end past the captured data           |                    |
    | 4 | `pos_x_first_motion`  | first burst-like (oscillating) X step;     | anchor only        |
    |   |                       | same past-data-end sanity check            |                    |
    | 5 | `loadcell_auto`       | rolling-std burst detector on the force    | anchor only        |
    |   |                       | trace (only when `auto_detect_t0`)         |                    |

    Post-pass (force-trace re-anchor): unless the winning source is one of
    `gcode_event` / `z_marker` / `pos_x_transitions`, the loadcell
    cycle-start detector re-slices the windows and, when its cycle-0 anchor
    disagrees with `sweep_t0` by more than half a cycle period, RE-ANCHORS
    sweep_t0 (source becomes `force_trace_reanchor`). Sources 2's refined
    forms are exempt because their anchors are already authoritative.

    "anchor only" stages leave `windows=None`; the caller falls back to
    plan-direct slicing. Diagnostic strings are appended to `notes` exactly
    as the pre-refactor inline code did.
    """
    # Anchor sweep_t0 AND derive per-K window bounds. Order of preference:
    #
    # 1. Caller's gcode-event anchor (legacy path, unused on this firmware
    #    because the `gcode` metric is silent -- kept for compat).
    # 2. pos_x periodic-cycle detection: validates a run of consecutive
    #    cycle starts spaced at cycle_period_s before treating them as
    #    real bursts. The first start of the validated run is sweep_t0
    #    (after subtracting segments[0].start_offset_s). Everything
    #    earlier -- park motion, planner-lookahead jitter during M-code
    #    processing, homing transients -- is discarded. This same scan
    #    produces the per-K (t_lo, t_hi) windows directly, so anchor and
    #    slicing always agree.
    # 3. pos_x first-motion fallback (anchor only): when periodic
    #    detection can't validate (too few cycles, very noisy data), use
    #    the first pos_x step away from a stable lookback. K-slicing
    #    then falls back to plan offsets.
    # 4. Loadcell rolling-std auto-detect (last resort).
    pre_burst_data_windows: list[tuple[float, float]] | None = None
    # Track WHICH detector produced sweep_t0. The Z-marker pulse has a
    # unique 2 mm lift+drop signature that can't be confused with anything
    # else, so when it succeeds it is treated as authoritative: subsequent
    # detectors are skipped for slicing (we go straight to plan-direct
    # windows) and the force-trace re-anchor below is disabled (it would
    # otherwise wander when the per-cycle period is non-uniform, e.g.
    # the K[0] warm-up cycle is much longer than every other K's cycle).
    anchor_source: str | None = None
    if t0_is_anchored:
        notes.append(
            f"sweep_t0={sweep_t0:.3f} taken from gcode-event anchor "
            f"(auto-detect skipped)"
        )
        anchor_source = "gcode_event"
    # FIRST: Z-marker anchor. The gcode generator emits a distinctive
    # Z-up / Z-down pulse immediately before the first burst, with a
    # known `z_marker_post_dwell_s` of dwell between marker return and
    # first burst's slow-leg start. If pos_z is streaming, this gives
    # us a one-shot, unambiguous anchor that can't be confused with
    # park motions, planner jitter, or the bursts themselves. It also
    # makes us robust to the user's complaint that K[0]'s window
    # starts before the first burst.
    if (
        not t0_is_anchored
        and pos_z_t is not None
        and pos_z is not None
        and len(pos_z_t) >= 10
        and plan.segments
    ):
        z_return_t = _detect_z_marker_anchor(
            pos_z_t, pos_z, expected_lift_mm=z_marker_lift_mm,
        )
        if z_return_t is not None:
            # Gcode-side contract: the Z marker pulse ends with pos_z
            # returning to baseline, immediately followed by the
            # `start_offset_s` pre-roll dwell, then the first burst.
            # So sweep_t0 (the "SWEEP_START" instant) is exactly the
            # Z-return time -- the pre-roll dwell IS our
            # `start_offset_s`.
            new_t0 = float(z_return_t)
            notes.append(
                f"sweep_t0 anchored via Z marker return at "
                f"recv={z_return_t:.3f} "
                f"(was {sweep_t0:.3f}, shift={new_t0 - sweep_t0:+.2f}s)"
            )
            sweep_t0 = new_t0
            t0_is_anchored = True
            anchor_source = "z_marker"
            # Z-marker gives a COARSE sweep_t0 (within a few seconds of
            # truth). On user's run_1779100636 there was still a +3.0 s
            # firmware processing delay between Z-DOWN and the actual
            # first X motion -- almost certainly M572/M83/G1 planner-
            # sync that the gcode timeline doesn't model. So we refine
            # with pos_x transitions when available: each cycle's slow→
            # fast and fast→slow transitions ARE the cycle boundaries,
            # and they are physically guaranteed (the toolhead actually
            # moved), giving us the truest possible window edges.
            if (
                pos_t is not None
                and pos_x is not None
                and len(pos_t) >= 10
                and plan.params.coupled_dx_mm > 0
            ):
                trans_result = _slice_from_pos_transitions(
                    pos_t, pos_x, plan, sweep_t0,
                    coupled_amplitude_mm=plan.params.coupled_dx_mm,
                    notes=notes,
                )
                if trans_result is not None:
                    pos_windows, refined_t0 = trans_result
                    sweep_t0 = refined_t0
                    pre_burst_data_windows = pos_windows
                    anchor_source = "pos_x_transitions"
            # Fallback: if pos_x didn't yield transition-based windows
            # (no XY coupling, or pos_x stream missing / sparse), use
            # plan-direct slicing relative to the Z-marker anchor.
            if pre_burst_data_windows is None:
                pre_burst_data_windows = _slice_from_plan(plan)
                notes.append(
                    f"K windows sliced directly from plan (Z-marker "
                    f"anchor, no pos_x transitions usable; "
                    f"{len(pre_burst_data_windows)} segments, K[0] "
                    f"window "
                    f"{pre_burst_data_windows[0][1] - pre_burst_data_windows[0][0]:.2f}s "
                    f"vs subsequent "
                    f"{pre_burst_data_windows[-1][1] - pre_burst_data_windows[-1][0]:.2f}s "
                    f"per the gcode schedule)"
                )
    if (
        not t0_is_anchored
        and pos_t is not None
        and pos_x is not None
        and len(pos_t) >= 10
        and plan.segments
        and plan.params.coupled_dx_mm > 0
    ):
        new_t0_periodic, pre_burst_data_windows_candidate = _anchor_and_slice_from_pos(
            pos_t, pos_x, plan, notes=notes,
        )
        if (
            new_t0_periodic is not None
            and pre_burst_data_windows_candidate is not None
        ):
            # SANITY: the candidate anchor must place the entire sweep
            # within the captured data range. If pos_x periodic locks
            # onto a coincidental periodic run in the middle of random
            # cycle-start noise, the resulting sweep_t0 can sit
            # arbitrarily late and push the last K segments past the
            # end of the data (observed on the user's 2026-05 NPZ: an
            # anchor at +167 s left K[5..] in the post-burst dead
            # zone). Require sweep_t0 + last burst end < max(force_t).
            sweep_end_abs = (
                new_t0_periodic
                + plan.segments[-1].start_offset_s
                + plan.segments[-1].duration_s
            )
            if sweep_end_abs > float(force_t[-1]) + plan.segments[-1].cycle_period_s:
                notes.append(
                    f"pos_x periodic anchor REJECTED: would place sweep "
                    f"end at {sweep_end_abs:.2f} past data end "
                    f"{float(force_t[-1]):.2f} -- falling through to "
                    f"loadcell auto-detect"
                )
            else:
                notes.append(
                    f"sweep_t0 anchored via pos_x periodic cycle detection "
                    f"(was {sweep_t0:.3f}, now {new_t0_periodic:.3f}, "
                    f"shift={new_t0_periodic - sweep_t0:+.2f}s; "
                    f"{len(pre_burst_data_windows_candidate)} K windows sliced from data)"
                )
                sweep_t0 = new_t0_periodic
                pre_burst_data_windows = pre_burst_data_windows_candidate
                t0_is_anchored = True
                anchor_source = "pos_x_periodic"
    if (
        not t0_is_anchored
        and pos_t is not None
        and pos_x is not None
        and len(pos_t) >= 2
        and plan.segments
        and plan.params.coupled_dx_mm > 0
    ):
        first_motion = _detect_first_pos_motion(
            pos_t, pos_x,
            baseline=plan.params.purge_x,
            amplitude=plan.params.coupled_dx_mm,
            cycle_period_s=plan.segments[0].cycle_period_s if plan.segments else None,
        )
        if first_motion is not None:
            new_t0 = first_motion - plan.segments[0].start_offset_s
            sweep_end_abs = (
                new_t0
                + plan.segments[-1].start_offset_s
                + plan.segments[-1].duration_s
            )
            if sweep_end_abs > float(force_t[-1]) + plan.segments[-1].cycle_period_s:
                notes.append(
                    f"pos_x first-motion anchor REJECTED: would place sweep "
                    f"end at {sweep_end_abs:.2f} past data end "
                    f"{float(force_t[-1]):.2f} -- falling through to "
                    f"loadcell auto-detect"
                )
            else:
                notes.append(
                    f"sweep_t0 anchored via pos_x first-motion at recv={first_motion:.3f} "
                    f"(was {sweep_t0:.3f}, now {new_t0:.3f}, "
                    f"shift={new_t0 - sweep_t0:+.2f}s)"
                )
                sweep_t0 = new_t0
                t0_is_anchored = True
                anchor_source = "pos_x_first_motion"
    # Loadcell rolling-std auto-detect: final fallback. With the new
    # global-max threshold and earliest-peak fine stage, this is the
    # most reliable detector when pos_x anchors get rejected by the
    # data-fit sanity check above.
    if auto_detect_t0 and not t0_is_anchored and plan.segments:
        cycle = plan.segments[0].cycle_period_s
        detected = _detect_sweep_start(
            force_t, force_y, cycle, plan.params.slow_half_s
        )
        if detected is not None:
            head_offset = plan.segments[0].start_offset_s
            new_t0 = detected - head_offset
            notes.append(
                f"sweep_t0 auto-detected via loadcell rolling-std "
                f"(was {sweep_t0:.3f}, now {new_t0:.3f}, "
                f"shift={new_t0 - sweep_t0:+.2f}s)"
            )
            sweep_t0 = new_t0
            t0_is_anchored = True
            anchor_source = "loadcell_auto"

    # Force-trace cycle-start slicing (used only as a fallback / refinement).
    # The loadcell signal has the highest SNR for cycle boundaries -- each
    # cycle's slow→fast transition crosses the mid-threshold cleanly --
    # but its threshold is auto-discovered from the trace percentiles and
    # its cycle prediction assumes a UNIFORM period. Both assumptions
    # break in real runs:
    #   1. The warm-up spike (post-prime melt-pressure peak) skews the
    #      p90 well above the real fast plateau, so the mid threshold sits
    #      below the slow plateau and cycles never go below it -- the
    #      detector misses most edges. Observed on user's run_1779100636:
    #      9 edges detected out of 11 expected, then 6 of those weren't
    #      cycle starts at all.
    #   2. With first_slow_leg_factor > 1, K[0] cycle 0's slow leg lasts
    #      `warmup_factor × slow_half`, so the period from cycle 0 start
    #      to cycle 1 start is much longer than `cycle_period_s`. The
    #      rolling-anchor prediction with a uniform period therefore
    #      lands cycle 0 in the wrong slot.
    # When the prior anchor came from the Z-marker pulse, we trust it
    # absolutely and slice from the plan instead -- the force-trace
    # re-anchoring would only introduce errors, not correct them. The
    # re-anchor remains active for anchor_source in {pos_x_periodic,
    # pos_x_first_motion, loadcell_auto, initial} because those can lock
    # onto park motion / planner jitter and the force trace genuinely is
    # the more reliable signal in that case.
    force_reanchor_active = anchor_source not in (
        "z_marker", "pos_x_transitions", "gcode_event",
    )
    if plan.segments and force_reanchor_active:
        force_starts = _detect_force_cycle_starts(
            force_t, force_y,
            min_gap_s=0.5 * plan.segments[0].cycle_period_s,
        )
        force_result = _slice_from_force_cycles(
            force_starts, plan, sweep_t0, notes=notes,
        )
        if force_result is not None:
            force_data_windows, cycle0_abs = force_result
            # Reconcile sweep_t0 with the force-trace anchor.
            # cycle0_abs is the absolute time of cycle 0's slow-leg
            # start; per the plan, that equals sweep_t0 + start_offset_s.
            inferred_t0 = cycle0_abs - plan.segments[0].start_offset_s
            disagree_s = inferred_t0 - sweep_t0
            cycle = plan.segments[0].cycle_period_s
            if abs(disagree_s) > 0.5 * cycle:
                notes.append(
                    f"sweep_t0 re-anchored by force-trace periodic "
                    f"detector (was {sweep_t0:.3f}, now "
                    f"{inferred_t0:.3f}, shift={disagree_s:+.2f}s) — "
                    f"prior anchor placed cycle 0 in the wrong "
                    f"cycle; the force-trace's first periodic run is "
                    f"authoritative"
                )
                # Re-slice with the corrected sweep_t0 so the relative
                # window coordinates match the new origin.
                sweep_t0 = inferred_t0
                anchor_source = "force_trace_reanchor"
                force_result2 = _slice_from_force_cycles(
                    force_starts, plan, sweep_t0, notes=notes,
                )
                if force_result2 is not None:
                    force_data_windows, _ = force_result2
                t0_is_anchored = True
            pre_burst_data_windows = force_data_windows

    return AnchorResolution(
        sweep_t0=sweep_t0,
        windows=pre_burst_data_windows,
        source=anchor_source,
        t0_is_anchored=t0_is_anchored,
    )
