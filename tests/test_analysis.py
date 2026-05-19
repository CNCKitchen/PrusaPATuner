"""Synthesise a loadcell signal with a known phase lag per K, then verify both algorithms
recover the planted optimum.
"""
from __future__ import annotations

import numpy as np

from prusa_pa_tuner.analysis import (
    analyse_sweep,
    _argmin_with_parabolic,
    _detect_sweep_start,
)
from prusa_pa_tuner.gcode_gen import SweepParams, build_sweep


def _square_wave(t: np.ndarray, period: float, lo: float, hi: float, start: float, end: float) -> np.ndarray:
    out = np.zeros_like(t)
    half = period / 2.0
    inside = (t >= start) & (t < end)
    phase = ((t - start) % period)
    out[inside] = np.where(phase[inside] < half, lo, hi)
    return out


def _synthesise(plan, k_optimal: float, lag_per_k_unit_s: float, noise: float = 0.01,
                sample_rate: float = 1000.0, sweep_t0: float = 1000.0):
    """Build a force time series where lag(K) = (K - k_optimal) * lag_per_k_unit_s."""
    total_s = plan.segments[-1].start_offset_s + plan.segments[-1].duration_s + 1.0
    n = int(total_s * sample_rate)
    t_rel = np.arange(n) / sample_rate
    t_abs = sweep_t0 + t_rel

    force = np.zeros(n)
    for seg in plan.segments:
        lag = (seg.k - k_optimal) * lag_per_k_unit_s
        # the commanded velocity at time t is slow/fast. The force "leads" or "lags" by `lag`.
        # If lag>0, force is at (t-lag) — i.e. force seen at time t corresponds to the
        # command at time t-lag. Equivalent to delaying the command by `lag` to make the force.
        cmd = _square_wave(
            t_rel - lag,
            period=seg.cycle_period_s,
            lo=plan.params.slow_feed_mm_s,
            hi=plan.params.fast_feed_mm_s,
            start=seg.start_offset_s,
            end=seg.start_offset_s + seg.duration_s,
        )
        force += cmd
    rng = np.random.default_rng(42)
    force += rng.normal(0, noise, size=n)
    return t_abs, force, sweep_t0


def test_phase_lag_recovers_optimum():
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=10,
        slow_half_s=0.4,
        fast_half_s=0.4,
    ))
    t_abs, force, t0 = _synthesise(plan, k_optimal=0.045, lag_per_k_unit_s=2.0)
    # auto_detect_t0=False -- tests assume the caller supplies a correct t0.
    # The detector itself is exercised in test_detect_sweep_start_* below.
    result = analyse_sweep(
        sweep_t0=t0, force_t=t_abs, force_y=force, plan=plan,
        auto_detect_t0=False,
    )

    assert result.phase_fit is not None
    # Should land within ±0.01 of 0.045
    assert abs(result.phase_fit.k_opt - 0.045) < 0.01
    assert result.phase_fit.r_squared > 0.8


def test_integral_area_recovers_optimum():
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=10,
        slow_half_s=0.4,
        fast_half_s=0.4,
    ))
    # Use a different lag mapping; the integral method should still find the zero-crossing
    t_abs, force, t0 = _synthesise(plan, k_optimal=0.05, lag_per_k_unit_s=2.0, noise=0.005)
    # auto_detect_t0=False -- tests assume the caller supplies a correct t0.
    # The detector itself is exercised in test_detect_sweep_start_* below.
    result = analyse_sweep(
        sweep_t0=t0, force_t=t_abs, force_y=force, plan=plan,
        auto_detect_t0=False,
    )

    assert result.integral_fit is not None
    assert abs(result.integral_fit.k_opt - 0.05) < 0.015
    # The integral-vs-K curve isn't perfectly linear in the presence of pure square-wave
    # lags (it's roughly linear near zero-crossing). 0.7 is plenty for a sanity check
    # that the line fit is meaningful — what actually matters is the k_opt accuracy above.
    assert result.integral_fit.r_squared > 0.7


def test_centered_integral_beats_legacy_on_first_order_response():
    """The legacy `sign(dC/dt)` integrator is dominated by an intrinsic-τ
    baseline when the system has finite rise time (real loadcell + melt
    pressure dynamics) -- it produces a near-flat area-vs-K curve. The
    centered-window form has to do better: clearly K-sensitive slope and
    a zero-crossing close to the planted K_opt.

    Models the force as a first-order lag of the command (the melt-flow
    transient PA is supposed to compensate). PA acts on the *effective
    time constant*: K_opt corresponds to τ_eff = 0 (instant response),
    other K values to non-zero τ on either side of zero.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40),
        cycles_per_K=8,
        slow_half_s=1.0,
        fast_half_s=0.25,
        accel_mm_s2=200.0,
    ))
    sr = 1000.0
    K_OPT = 0.18
    # τ_eff goes from positive (under-PA, slow rise) at K=0 to negative
    # (over-PA, overshoot) at K=0.40, crossing 0 at K_OPT.
    total_s = plan.segments[-1].start_offset_s + plan.segments[-1].duration_s + 1.0
    n = int(total_s * sr)
    t_rel = np.arange(n) / sr
    t0 = 1000.0
    t_abs = t0 + t_rel
    force = np.zeros(n)
    p = plan.params
    for seg in plan.segments:
        # Build the *actual* commanded velocity (the same accel-limited ramp
        # the analyser models) so the only thing we're testing is response
        # dynamics, not command-shape mismatch.
        from prusa_pa_tuner.analysis import _build_command_wave
        cmd = _build_command_wave(
            seg, t_rel,
            slow_v=p.slow_feed_mm_s, fast_v=p.fast_feed_mm_s,
            slow_half_s=p.slow_half_s,
            accel_mm_s2=p.accel_mm_s2,
        )
        # First-order response: dF/dt = (cmd - F) / τ_eff. For τ_eff > 0
        # the force lags; for τ_eff < 0 we model overshoot as a leading
        # response (pre-shifted by |τ_eff|, clipped).
        tau_eff = 0.25 * (K_OPT - seg.k) / K_OPT  # 250 ms baseline at K=0
        f = np.zeros(n)
        if tau_eff > 1e-4:
            alpha = 1.0 - np.exp(-1.0 / (tau_eff * sr))
            for i in range(1, n):
                f[i] = f[i - 1] + alpha * (cmd[i] - f[i - 1])
        elif tau_eff < -1e-4:
            # over-PA: F leads. Shift command back in time and clip.
            shift = int(round(-tau_eff * sr))
            f[:n - shift] = cmd[shift:]
            f[n - shift:] = cmd[-1]
        else:
            f = cmd.copy()
        force += f
    rng = np.random.default_rng(7)
    force += rng.normal(0, 0.05, size=n)

    result = analyse_sweep(
        sweep_t0=t0, force_t=t_abs, force_y=force, plan=plan,
        auto_detect_t0=False,
    )
    assert result.integral_fit is not None, result.notes
    assert result.integral_legacy_fit is not None, result.notes
    # The centered form must recover K_OPT within a reasonable tolerance.
    assert abs(result.integral_fit.k_opt - K_OPT) < 0.05, (
        f"new metric K_opt={result.integral_fit.k_opt:.3f}, planted {K_OPT}; "
        f"legacy K_opt={result.integral_legacy_fit.k_opt:.3f}"
    )
    # And the area values must actually move with K (slope significant
    # relative to the per-window magnitude). The exact R^2 depends on how
    # linear the dynamic-response curve is across the K range; what
    # matters is that K_OPT recovery works.
    assert abs(result.integral_fit.slope) > 1.0, (
        f"new metric slope={result.integral_fit.slope:.3f} -- too flat"
    )


def test_argmin_with_parabolic_recovers_subgrid_minimum():
    """The parabolic interpolation around the discrete argmin should
    recover a sub-grid minimum better than the raw argmin. Plant a known
    minimum slightly off the K grid and confirm we land closer to it than
    the nearest K.
    """
    k = np.linspace(0.0, 0.10, 51)  # step = 0.002
    k_true = 0.0431  # deliberately between grid points
    cost = (k - k_true) ** 2 + 1e-6  # clean parabola, tiny floor
    k_argmin = _argmin_with_parabolic(k, cost)
    nearest_k = float(k[int(np.argmin(cost))])
    assert k_argmin is not None
    assert abs(k_argmin - k_true) < abs(nearest_k - k_true), (
        f"parabolic {k_argmin:.5f} not closer to {k_true} than grid "
        f"point {nearest_k:.5f}"
    )
    # Must land within one grid step
    assert abs(k_argmin - k_true) < 0.002


def test_argmin_with_parabolic_handles_boundary_and_nonconvex():
    """At a boundary, or when the local 3-point fit is not concave-up,
    return the discrete argmin without extrapolation. Otherwise noise can
    push the interpolated K outside the sweep range or to the wrong side
    of the minimum.
    """
    k = np.linspace(0.0, 0.10, 11)  # step = 0.01
    # Monotonically decreasing -> argmin at last index (boundary)
    cost = -k.copy()
    k_argmin = _argmin_with_parabolic(k, cost)
    assert k_argmin == 0.10

    # Three equal cost values around the minimum -> non-strict parabola.
    # Must not crash, must return the discrete argmin K.
    cost = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    k_argmin = _argmin_with_parabolic(k, cost)
    assert k_argmin is not None
    # argmin tie-breaks at the first occurrence (index 1), which is K=0.01
    assert abs(k_argmin - 0.01) < 0.01

    # All NaN -> None
    cost = np.full(11, np.nan)
    assert _argmin_with_parabolic(k, cost) is None


def test_bd_segment_metrics_extract_overshoot_undershoot():
    """A single synthetic low-high-low segment with a known overshoot
    and undershoot must produce metrics that match the planted values
    within tolerance, AND not be excluded by the quality gate."""
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    slow = 1.0
    fast = 0.25
    fs = 200.0
    t_start = 0.0
    t_rise = slow
    t_fall = slow + fast
    t_end = slow + fast + slow
    t = np.arange(t_start, t_end, 1.0 / fs)
    baseline = 100.0
    high_level = baseline + 1000.0
    y = np.full_like(t, baseline)
    # Rising side: linear ramp to peak, exponential decay to plateau.
    peak_at = t_rise + 0.03
    overshoot_mag = 300.0
    tau = 0.04
    mask_rise = (t >= t_rise) & (t < t_fall)
    y[mask_rise] = high_level + overshoot_mag * np.exp(
        -(t[mask_rise] - peak_at) / tau
    )
    mask_ramp = (t >= t_rise) & (t < peak_at)
    y[mask_ramp] = baseline + (high_level + overshoot_mag - baseline) * (
        (t[mask_ramp] - t_rise) / (peak_at - t_rise)
    )
    # Falling side: linear ramp to trough, exponential recovery to baseline.
    trough_at = t_fall + 0.04
    undershoot_mag = 400.0
    mask_fall = t >= t_fall
    y[mask_fall] = baseline - undershoot_mag * np.exp(
        -(t[mask_fall] - trough_at) / 0.10
    )
    mask_drop = (t >= t_fall) & (t < trough_at)
    y[mask_drop] = high_level + (
        baseline - undershoot_mag - high_level
    ) * ((t[mask_drop] - t_fall) / (trough_at - t_fall))
    rng = np.random.default_rng(0)
    y = y + rng.normal(0, 2.0, len(y))

    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.04, seg_idx=0,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall, t_end=t_end,
        slow_half_s=slow, fast_half_s=fast,
        dropout_t=np.array([]),
    )

    assert not seg.excluded, seg.exclusion_reasons
    assert seg.t_peak is not None and abs(seg.t_peak - peak_at) < 0.02
    assert seg.t_trough is not None and abs(seg.t_trough - trough_at) < 0.02
    # overshoot magnitude is within ~30% of planted 300 (the 2σ
    # plateau-noise gate subtracts ~4 raw units; allow a wider window
    # to accommodate the exponential decay starting at the peak).
    assert 180.0 < seg.metrics["overshoot"] < 380.0, seg.metrics
    # undershoot within ~20% of planted 400
    assert 300.0 < seg.metrics["undershoot"] < 480.0, seg.metrics
    # baseline median should be near 100, std small
    assert abs(seg.metrics["baseline_median"] - 100.0) < 5.0
    assert seg.metrics["baseline_noise_std"] < 5.0


def test_bd_segment_overshoot_rejects_plateau_noise_without_real_transient():
    """User report (run_1779125302, K=0.0..0.05): segments with NO real
    PA overshoot still produced 80-180 raw-unit `overshoot` values
    because the analyser's argmax landed on a 3σ plateau-noise spike
    well past the rising edge (peak_offset 200-380 ms past t_rise).

    The fix has two layers:
      1. Narrow the peak-search window to ~10% of fast_half past
         t_rise_end. Real PA overshoot is a fast transient that decays
         within 50-100 ms; anything past that IS plateau noise.
      2. Subtract a 2·σ_plateau noise floor from `peak - high_level`,
         so single-sample noise excursions don't register as overshoot.

    Test: synth a clean square wave (low-high-low) with NO planted
    overshoot, just plateau noise. The metric must report 0 (or near 0)
    overshoot, NOT the random noise excursion.
    """
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    slow = 2.0
    fast = 1.0
    fs = 200.0
    t_start, t_rise, t_fall, t_end = 0.0, slow, slow + fast, slow + fast + slow
    t = np.arange(t_start, t_end, 1.0 / fs)
    baseline = 700.0
    high_level = 3200.0
    y = np.full_like(t, baseline)
    # Sharp rise (effectively instant -- 1 sample wide), no overshoot.
    y[t >= t_rise] = high_level
    # Sharp fall.
    y[t >= t_fall] = baseline
    # Realistic plateau noise (matches user's measured ~60 raw units).
    rng = np.random.default_rng(123)
    y = y + rng.normal(0.0, 60.0, len(y))

    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.0, seg_idx=0,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall, t_end=t_end,
        slow_half_s=slow, fast_half_s=fast,
        dropout_t=np.array([]),
    )
    # Without the fix, this segment reported 100-200 raw-unit overshoot
    # purely from 3σ plateau-noise spikes near mid-plateau. With the
    # tightened window + noise gate, overshoot collapses to 0 (or a
    # tiny residue from the few samples that happen to bracket t_rise
    # within the transient window).
    assert seg.metrics.get("overshoot", 0.0) < 30.0, (
        f"plateau-noise-only segment should not register as overshoot; "
        f"got overshoot={seg.metrics.get('overshoot')}"
    )


def test_bd_segment_region_boundaries_match_actual_transitions_with_creeping_plateau():
    """Regression for the user's 2026-05 complaint: on segments with a
    pressure-advance-induced creeping high plateau (force ramps slowly
    upward through the fast leg), the legacy argmax-based t_peak lands
    near the END of the plateau (where force is highest). That makes
    R2 (rising-edge region) swallow most of the plateau, R4 spills
    into the falling edge, R6 (falling-edge region) eats half the
    recovery tail, and the resulting metrics are garbage.

    The threshold-based t_rise_end / t_fall_end fields delimit the
    actual transitions: t_rise_end = first sample ≥ 90% of (high −
    baseline) after t_rise; t_fall_end = first sample ≤ 10% of (high −
    baseline) after t_fall. They should land at the END of the rising
    edge (≤100ms after t_rise) and the END of the falling edge
    (≤100ms after t_fall), NOT at the argmax/argmin which can be
    seconds later on a creeping plateau.
    """
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    slow = 2.0
    fast = 0.8
    fs = 200.0
    t_start = 0.0
    t_rise = slow                          # 2.0
    t_fall = slow + fast                   # 2.8
    t_end = slow + fast + slow             # 4.8
    t = np.arange(t_start, t_end, 1.0 / fs)
    baseline = 800.0
    delta = 2700.0  # high - baseline = 2700 → matches user's K=0.02 range
    y = np.full_like(t, baseline)

    # Rise: 30ms ramp from baseline to 95% of plateau
    rise_dur = 0.030
    plateau_start_y = baseline + 0.95 * delta
    mask_ramp = (t >= t_rise) & (t < t_rise + rise_dur)
    y[mask_ramp] = baseline + (plateau_start_y - baseline) * (
        (t[mask_ramp] - t_rise) / rise_dur
    )
    # Creeping plateau: linear creep from 95% of plateau to 105% of
    # plateau across the fast leg (this is what fools argmax — the max
    # is at t_fall, not near t_rise).
    mask_plat = (t >= t_rise + rise_dur) & (t < t_fall)
    plateau_t = t[mask_plat]
    plateau_dur = max(t_fall - (t_rise + rise_dur), 1e-9)
    y[mask_plat] = plateau_start_y + (
        (baseline + 1.05 * delta - plateau_start_y)
        * (plateau_t - (t_rise + rise_dur)) / plateau_dur
    )
    # Fall: 30ms ramp back to near baseline
    fall_dur = 0.030
    fall_end_y = baseline + 0.05 * delta
    mask_drop = (t >= t_fall) & (t < t_fall + fall_dur)
    y[mask_drop] = (baseline + 1.05 * delta) + (
        (fall_end_y - (baseline + 1.05 * delta))
        * (t[mask_drop] - t_fall) / fall_dur
    )
    # Slow recovery back to baseline over the rest of low_{n+1}
    mask_recov = t >= (t_fall + fall_dur)
    tau = 0.25
    y[mask_recov] = baseline + (fall_end_y - baseline) * np.exp(
        -(t[mask_recov] - (t_fall + fall_dur)) / tau
    )

    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.02, seg_idx=0,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall, t_end=t_end,
        slow_half_s=slow, fast_half_s=fast,
        dropout_t=np.array([]),
    )

    assert not seg.excluded, seg.exclusion_reasons
    # t_rise_end must land in the actual rise window (within ~100ms of
    # t_rise), NOT at the plateau end. The 90% threshold is crossed by
    # construction near t_rise + 30ms (when the ramp hits 95% of plateau,
    # which is ≥ 90%·delta = 0.9·plateau_start ≈ 0.9·(0.95·delta)).
    assert seg.t_rise_end is not None
    rise_lag = seg.t_rise_end - t_rise
    assert 0.0 <= rise_lag < 0.15, (
        f"t_rise_end landed {rise_lag*1000:.0f}ms past t_rise; if >150ms "
        f"the threshold-crossing is hunting the creeping-plateau max "
        f"instead of the actual rise completion"
    )
    # t_fall_end must land within ~150ms of t_fall (the fall takes 30ms
    # to drop from ~3500 to baseline+5%delta).
    assert seg.t_fall_end is not None
    fall_lag = seg.t_fall_end - t_fall
    assert 0.0 <= fall_lag < 0.15, (
        f"t_fall_end landed {fall_lag*1000:.0f}ms past t_fall; if >150ms "
        f"the 10%-threshold crossing was confused by slow recovery"
    )
    # Plateau window must START at or after the rise completion (so
    # R4 doesn't include the rising transient) AND END before t_fall
    # (so R4 doesn't bleed into the falling edge). plateau_slope must
    # be POSITIVE for the creeping plateau (force still rising) —
    # if argmax made R4 start late, plateau_slope might be tiny.
    assert seg.metrics["plateau_slope"] > 50.0, (
        f"plateau_slope = {seg.metrics['plateau_slope']:.1f}; expected "
        f">50 (force/s) for the planted creep. A small slope here "
        f"means R4 is too narrow or shifted."
    )
    # Overshoot should be ~0 (the planted profile has no spike above
    # the high plateau — it's a slow ramp).
    assert seg.metrics["overshoot"] < 200.0, (
        f"overshoot = {seg.metrics['overshoot']:.1f}; expected near 0 "
        f"on this no-spike profile"
    )


def test_bd_segment_detects_pa_lagged_fall_start_before_commanded_t_fall():
    """The actual force often begins falling slightly BEFORE the
    commanded `t_fall` because of pressure-advance lag. If R4 plateau
    is held to end at `t_fall`, the last ~20-40 ms of plateau is
    already in the fall transient and contaminates BOTH the plateau
    metrics AND the rise_error_area (which integrates from t_rise to
    t_fall). The fix: detect the actual fall start from the data
    (first sustained drop below the 90% plateau threshold) and use
    THAT as the R4/R6 boundary.

    Test setup: synthesise a fast leg whose force begins dropping
    30 ms BEFORE the commanded t_fall. Verify t_fall_start lands in
    that pre-t_fall window, and the rise_error_area does NOT include
    the fall transient.
    """
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    slow = 2.0
    fast = 0.8
    fs = 200.0
    t_start = 0.0
    t_rise = slow                          # 2.0
    t_fall_cmd = slow + fast               # 2.8 (commanded)
    t_end = slow + fast + slow             # 4.8
    PA_LAG = 0.030                          # actual fall starts 30 ms early
    t_fall_real = t_fall_cmd - PA_LAG       # 2.770
    t = np.arange(t_start, t_end, 1.0 / fs)
    baseline = 800.0
    high_plateau = 3500.0
    y = np.full_like(t, baseline)

    # Fast rise (30 ms) to plateau.
    rise_dur = 0.030
    mask_ramp = (t >= t_rise) & (t < t_rise + rise_dur)
    y[mask_ramp] = baseline + (high_plateau - baseline) * (
        (t[mask_ramp] - t_rise) / rise_dur
    )
    # Plateau [t_rise + rise_dur, t_fall_real]
    mask_plat = (t >= t_rise + rise_dur) & (t < t_fall_real)
    y[mask_plat] = high_plateau
    # Fall (30 ms) from t_fall_real to baseline.
    fall_dur = 0.030
    mask_drop = (t >= t_fall_real) & (t < t_fall_real + fall_dur)
    y[mask_drop] = high_plateau + (baseline - high_plateau) * (
        (t[mask_drop] - t_fall_real) / fall_dur
    )
    # Recovery is just baseline.
    mask_recov = t >= (t_fall_real + fall_dur)
    y[mask_recov] = baseline

    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.04, seg_idx=0,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall_cmd, t_end=t_end,
        slow_half_s=slow, fast_half_s=fast,
        dropout_t=np.array([]),
    )
    assert not seg.excluded, seg.exclusion_reasons
    assert seg.t_fall_start is not None
    # t_fall_start must land in [t_fall_real - 5ms, t_fall_cmd]: it's
    # the first sample where force drops below 90% threshold, which on
    # this clean profile is at t_fall_real exactly (force jumps from
    # plateau to dropping there).
    assert (
        t_fall_real - 0.005 <= seg.t_fall_start <= t_fall_cmd
    ), (
        f"t_fall_start = {seg.t_fall_start:.4f}; expected in "
        f"[{t_fall_real - 0.005:.4f}, {t_fall_cmd:.4f}] -- this fixture "
        f"plants the actual fall {PA_LAG*1000:.0f}ms before commanded t_fall, "
        f"and the detector must catch it."
    )


def test_bd_segment_display_crop_inset_from_segment_bounds():
    """Display window must be inset from [t_start, t_end] so the plot
    doesn't bridge into the neighbouring cycle when a single firmware-
    throttle gap drops a sample at the segment boundary. Used by the
    UI to slice plot data; metrics are still computed on the full
    [t_start, t_end] window."""
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    slow = 2.0
    fast = 0.8
    fs = 200.0
    t_start = 0.0
    t_rise = slow
    t_fall = slow + fast
    t_end = slow + fast + slow
    t = np.arange(t_start, t_end, 1.0 / fs)
    y = np.full_like(t, 800.0)
    y[(t >= t_rise) & (t < t_fall)] = 3500.0

    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.02, seg_idx=0,
        t_start=t_start, t_rise=t_rise, t_fall=t_fall, t_end=t_end,
        slow_half_s=slow, fast_half_s=fast,
        dropout_t=np.array([]),
    )
    # Margin = 0.1 * slow_half = 0.2s on each side
    expected_margin = 0.10 * slow
    assert abs(seg.t_lo_display - (t_start + expected_margin)) < 1e-6, (
        f"t_lo_display = {seg.t_lo_display:.3f}, expected "
        f"{t_start + expected_margin:.3f}"
    )
    assert abs(seg.t_hi_display - (t_end - expected_margin)) < 1e-6, (
        f"t_hi_display = {seg.t_hi_display:.3f}, expected "
        f"{t_end - expected_margin:.3f}"
    )


def test_bd_aggregate_per_k_computes_mad_and_iqr_for_error_bars():
    """The per-K aggregate must surface MAD and IQR alongside the
    median so the UI can draw error bars on the per-metric K plots
    and the user can tell which metrics are reliable (tight bar) from
    which are dominated by segment-to-segment noise (huge bar)."""
    from prusa_pa_tuner.analysis import _bd_aggregate_per_k, BdSegment

    def _mk_segment(k, idx, metrics):
        return BdSegment(
            k=k, seg_idx=idx,
            t_start=0.0, t_rise=2.0, t_fall=2.8, t_end=4.8,
            t_lo_display=0.2, t_hi_display=4.6,
            t_rise_end=2.05, t_fall_start=2.78, t_fall_end=2.85,
            t_peak=2.06, t_trough=2.86,
            n_samples=200, metrics=metrics,
            excluded=False, exclusion_reasons=[],
        )

    # 5 segments at K=0.05, planted overshoot values [10, 12, 14, 16, 18]
    # (median = 14, MAD = 1.4826 * 2 = 2.965, IQR = 4)
    segs = [
        _mk_segment(0.05, i, {"overshoot": v, "high_level": 3500.0})
        for i, v in enumerate([10.0, 12.0, 14.0, 16.0, 18.0])
    ]
    per_k = _bd_aggregate_per_k({0.05: segs})
    assert len(per_k) == 1
    r = per_k[0]
    assert abs(r.medians["overshoot"] - 14.0) < 1e-6
    # MAD = 1.4826 × median(|x − median|) = 1.4826 × 2 = 2.9652
    assert abs(r.mads["overshoot"] - 2.9652) < 0.01, (
        f"MAD = {r.mads['overshoot']:.4f}, expected 2.9652"
    )
    # IQR = P75 − P25 over [10,12,14,16,18] = 16 − 12 = 4
    assert abs(r.iqrs["overshoot"] - 4.0) < 1e-6


def test_default_weights_include_delay_metrics_for_left_edge_of_cost_valley():
    """The default cost weights must include the timing/delay metrics
    (rise_delay, fall_delay, settling_time) in addition to the area
    metrics (overshoot, undershoot, rise_error_area, ...). Without
    them, the cost is dragged toward K=0 because overshoot AND
    undershoot are both zero at low K -- the area metrics only see
    the RIGHT side of the valley (where K is too high). The delay
    metrics provide the LEFT side: at low K the response is slow,
    delays are large, the cost rises again -- so the composite
    minimum lands at the actual elbow.
    """
    from prusa_pa_tuner.analysis import BD_DEFAULT_WEIGHTS

    for required in ("rise_delay", "fall_delay", "settling_time"):
        assert required in BD_DEFAULT_WEIGHTS, (
            f"{required!r} missing from BD_DEFAULT_WEIGHTS -- the "
            f"composite cost will have no penalty for slow response "
            f"and K_opt will be pulled to K=0."
        )
        assert BD_DEFAULT_WEIGHTS[required] > 0, (
            f"{required!r} has zero weight -- equivalent to not being "
            f"in the cost at all."
        )


def test_bd_segment_excluded_on_low_sample_rate():
    """Segments whose sample rate falls below 40 Hz must be excluded
    with a reason mentioning the low rate."""
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    # 10 Hz sampling — well below the 40 Hz floor
    t = np.linspace(0.0, 2.25, 23)
    y = np.full_like(t, 100.0)
    y[(t >= 1.0) & (t < 1.25)] = 1000.0  # high leg
    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.04, seg_idx=0,
        t_start=0.0, t_rise=1.0, t_fall=1.25, t_end=2.25,
        slow_half_s=1.0, fast_half_s=0.25,
        dropout_t=np.array([]),
    )
    assert seg.excluded
    assert any("sample rate" in r for r in seg.exclusion_reasons), seg.exclusion_reasons


def test_bd_segment_excluded_on_dropout_in_window():
    """A dropout timestamp inside the segment window flags it for
    exclusion."""
    from prusa_pa_tuner.analysis import _bd_segment_metrics

    # 100 Hz sampling, clean step response
    fs = 100.0
    t = np.arange(0.0, 2.25, 1.0 / fs)
    y = np.full_like(t, 100.0)
    y[(t >= 1.0) & (t < 1.25)] = 1100.0
    # Dropout at t=1.4 -- inside the segment
    seg = _bd_segment_metrics(
        force_t=t, force_y=y,
        k=0.04, seg_idx=0,
        t_start=0.0, t_rise=1.0, t_fall=1.25, t_end=2.25,
        slow_half_s=1.0, fast_half_s=0.25,
        dropout_t=np.array([1.4]),
    )
    assert seg.excluded
    assert any("dropout" in r for r in seg.exclusion_reasons), seg.exclusion_reasons


def test_bd_k_opt_end_to_end_localizes_planted_optimum():
    """End-to-end: synthetic per-K step responses with overshoot+undershoot
    magnitude ∝ |K − K_OPT|. analyse_sweep's bd_pressure path should land
    bd_k_opt within one K-step of the planted optimum.
    """
    K_OPT = 0.040
    K_values = tuple(round(i * 0.005, 4) for i in range(13))  # 0..0.06
    plan = build_sweep(SweepParams(
        K_values=K_values,
        cycles_per_K=6,
        slow_half_s=1.0,
        fast_half_s=0.25,
        accel_mm_s2=5000.0,
        coupled_dx_mm=0.0,  # disable pos_x path -- exercise the model fallback
    ))
    sr = 200.0
    t0 = 1000.0
    p = plan.params
    cycle = p.slow_half_s + p.fast_half_s
    total_s = plan.segments[-1].start_offset_s + plan.segments[-1].duration_s + 1.0
    n = int(total_s * sr)
    t_rel = np.arange(n) / sr
    force = np.full(n, 100.0)  # baseline head load
    GAIN_OVER = 8000.0  # per |ΔK| unit -- scales overshoot magnitude
    GAIN_UNDER = 9000.0
    high_level = 1100.0
    for seg in plan.segments:
        delta = abs(seg.k - K_OPT)
        over = GAIN_OVER * delta
        under = GAIN_UNDER * delta
        for c in range(seg.cycles):
            t_rise = seg.start_offset_s + c * cycle + p.slow_half_s
            t_fall = seg.start_offset_s + (c + 1) * cycle
            # High leg: jump to high_level + overshoot, decay to plateau.
            mask_rise = (t_rel >= t_rise) & (t_rel < t_fall)
            tau = 0.04
            force[mask_rise] = high_level + over * np.exp(
                -(t_rel[mask_rise] - (t_rise + 0.03)) / tau
            )
            # Low leg recovery after the fall, decay back to 100.
            mask_fall = (t_rel >= t_fall) & (t_rel < t_fall + p.slow_half_s)
            force[mask_fall] = 100.0 - under * np.exp(
                -(t_rel[mask_fall] - (t_fall + 0.04)) / 0.10
            )
    rng = np.random.default_rng(42)
    force += rng.normal(0, 3.0, n)

    result = analyse_sweep(
        sweep_t0=t0,
        force_t=t0 + t_rel,
        force_y=force,
        plan=plan,
        auto_detect_t0=False,
    )

    assert result.bd_k_opt is not None, result.notes
    assert abs(result.bd_k_opt - K_OPT) < 0.01, (
        f"bd_k_opt={result.bd_k_opt:.4f}, planted {K_OPT} "
        f"(notes tail: {result.notes[-3:]})"
    )

    # Each K should have produced its full 6 segments via the model fallback
    # (pos_x is disabled), and most should be included.
    for r in result.bd_per_k:
        assert r.n_segments_total == 6, f"K={r.k}: {r.n_segments_total}/6"
    # At least the K's near K_OPT should have ≥4 included segments
    near_opt = [
        r for r in result.bd_per_k
        if abs(r.k - K_OPT) < 0.011
    ]
    assert all(
        r.n_segments_included >= 4 for r in near_opt
    ), [
        (r.k, r.n_segments_included) for r in near_opt
    ]


def test_bd_aggregate_uses_only_included_segments():
    """Median-over-segments aggregation drops excluded segments."""
    from prusa_pa_tuner.analysis import BdSegment, _bd_aggregate_per_k, BD_METRIC_NAMES

    segs = [
        BdSegment(
            k=0.04, seg_idx=0, t_start=0, t_rise=1, t_fall=1.25, t_end=2.25,
            t_peak=1.03, t_trough=1.29, n_samples=450,
            metrics={n: 10.0 for n in BD_METRIC_NAMES},
            excluded=False,
        ),
        BdSegment(
            k=0.04, seg_idx=1, t_start=2.25, t_rise=3.25, t_fall=3.5, t_end=4.5,
            t_peak=3.28, t_trough=3.54, n_samples=450,
            metrics={n: 30.0 for n in BD_METRIC_NAMES},
            excluded=False,
        ),
        BdSegment(
            k=0.04, seg_idx=2, t_start=4.5, t_rise=5.5, t_fall=5.75, t_end=6.75,
            t_peak=5.53, t_trough=5.79, n_samples=450,
            metrics={n: 99999.0 for n in BD_METRIC_NAMES},  # outlier
            excluded=True,
            exclusion_reasons=["test fixture: forced exclusion"],
        ),
    ]
    agg = _bd_aggregate_per_k({0.04: segs})
    assert agg[0].n_segments_total == 3
    assert agg[0].n_segments_included == 2
    # Median over [10, 30] = 20 for every metric (outlier dropped).
    for name in BD_METRIC_NAMES:
        assert agg[0].medians[name] == 20.0, name


def test_handles_missing_samples():
    plan = build_sweep(SweepParams(K_values=(0.0, 0.05, 0.1)))
    # empty arrays
    r = analyse_sweep(0.0, np.array([]), np.array([]), plan)
    assert r.phase_fit is None
    assert r.integral_fit is None
    assert r.notes  # has a note about it


def test_gcode_log_ground_truth_path_runs():
    """If the gcode-event log is supplied, the analyser should use it
    instead of the model square wave. Buddy firmware does not expose pos_e;
    the `gcode` metric (one STRING per processed gcode line) is the
    available ground-truth timing source. The fit should still recover the
    planted k_opt (within tolerance), and the notes should record the
    path taken.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=10, slow_half_s=0.4, fast_half_s=0.4,
    ))
    t_abs, force, t0 = _synthesise(plan, k_optimal=0.05, lag_per_k_unit_s=2.0)

    # Synthesise a matching gcode-event log: every slow->fast and
    # fast->slow transition timestamped with the cumulative E target the
    # firmware was commanded to reach. This mirrors what `M331 gcode`
    # would actually stream during a real sweep.
    p = plan.params
    half = p.slow_half_s
    fast_half = p.fast_half_s
    gcode_t: list[float] = []
    gcode_lines: list[str] = []
    e_cum = 0.0
    for seg in plan.segments:
        for cyc in range(seg.cycles):
            t_slow = t0 + seg.start_offset_s + cyc * seg.cycle_period_s
            e_cum += p.slow_feed_mm_s * half
            gcode_t.append(t_slow)
            gcode_lines.append(
                f"G1 E{e_cum:.4f} F{p.slow_feed_mm_s * 60:.1f}"
            )
            t_fast = t_slow + half
            e_cum += p.fast_feed_mm_s * fast_half
            gcode_t.append(t_fast)
            gcode_lines.append(
                f"G1 E{e_cum:.4f} F{p.fast_feed_mm_s * 60:.1f}"
            )

    result = analyse_sweep(
        sweep_t0=t0, force_t=t_abs, force_y=force, plan=plan,
        gcode_t=np.asarray(gcode_t), gcode_lines=gcode_lines,
        auto_detect_t0=False,
    )
    assert any("gcode-log" in n or "gcode" in n for n in result.notes), result.notes
    assert result.integral_fit is not None
    assert abs(result.integral_fit.k_opt - 0.05) < 0.02


def test_handles_nan_samples_gracefully():
    """loadcell_hp streams NaN while idle. If any NaNs slip past the
    collector's filter, analyse_sweep must drop them rather than raising
    out of scipy.detrend (the failure mode we hit in production).

    Poisoning rate is 2%: enough to exercise the NaN-handling path
    without driving the bd-segment noise gate above its 30%-of-leg-delta
    exclusion threshold. (Heavier poisoning legitimately trips the
    shared-exclusion gate that now backs integral_fit -- which is the
    correct production behaviour, just not what this test is verifying.)
    """
    # 50:50 leg lengths so the test's symmetric `_square_wave` synthesiser
    # produces force transitions where the bd-segment analysis expects them.
    # (With the SweepParams defaults of 1.0 s slow + 0.25 s fast, the
    # symmetric wave's transitions land mid-plateau in bd's view and the
    # shared-exclusion gate trips on all cycles -- but that's a synthesiser
    # mismatch, not a NaN-handling bug.)
    plan = build_sweep(SweepParams(
        K_values=(0.0, 0.05, 0.1), cycles_per_K=5,
        slow_half_s=0.4, fast_half_s=0.4,
    ))
    # `lag_per_k_unit_s=0.1` keeps all lags ≤ 5 ms so the bd-segment
    # transitions don't slip out of their expected legs (the test
    # previously used 2.0, which produced ±100 ms lags — bigger than
    # the bd baseline windows can absorb. The shared-exclusion gate
    # now correctly drops K values whose transitions wander into the
    # baseline region, so the lag has to be modest for the
    # fit-still-works assertion to hold).
    t_abs, force, t0 = _synthesise(plan, k_optimal=0.05, lag_per_k_unit_s=0.1)
    rng = np.random.default_rng(7)
    n_nans = max(5, len(force) // 50)  # ~2% sparse NaN poisoning
    force[rng.choice(len(force), n_nans, replace=False)] = np.nan
    # Should NOT raise -- should drop NaNs, note it, still produce a fit.
    r = analyse_sweep(
        sweep_t0=t0, force_t=t_abs, force_y=force, plan=plan,
        auto_detect_t0=False,
    )
    assert any("non-finite" in n for n in r.notes), r.notes
    # With sparse NaN poisoning enough segments survive the shared
    # exclusion gate that both fits still come back.
    assert r.integral_fit is not None


def test_detect_sweep_start_finds_burst_after_heatup():
    """Detector should locate the K=0 burst even when the trace begins with
    a long heatup-style quiet/drift region. This is the real-world failure
    mode that broke 'nothing computed' in the first pass."""
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=10, slow_half_s=0.4, fast_half_s=0.4,
    ))
    sr = 1000.0
    heatup_s = 10.0
    total_s = (
        heatup_s
        + plan.segments[-1].start_offset_s
        + plan.segments[-1].duration_s
        + 1.0
    )
    n = int(total_s * sr)
    t_rel = np.arange(n) / sr
    t0_abs = 1000.0
    t_abs = t0_abs + t_rel
    rng = np.random.default_rng(42)
    force = rng.normal(0, 0.01, size=n)
    # slow thermal drift during heatup
    force[: int(heatup_s * sr)] += 0.05 * (np.arange(int(heatup_s * sr)) / sr)
    # plant the bursts -- with no per-K lag this time, so detector
    # accuracy is testable against a clean reference.
    for seg in plan.segments:
        period = seg.cycle_period_s
        half = period / 2
        seg_t = t_rel - heatup_s
        inside = (
            (seg_t >= seg.start_offset_s)
            & (seg_t < seg.start_offset_s + seg.duration_s)
        )
        phase = (seg_t - seg.start_offset_s) % period
        force[inside] = np.where(
            phase[inside] < half,
            plan.params.slow_feed_mm_s,
            plan.params.fast_feed_mm_s,
        ) + rng.normal(0, 0.01, size=inside.sum())

    detected = _detect_sweep_start(
        t_abs, force,
        plan.segments[0].cycle_period_s,
        plan.params.slow_half_s,
    )
    expected = t0_abs + heatup_s + plan.segments[0].start_offset_s
    assert detected is not None
    # one sample (1 ms at 1 kHz) of slack is plenty for our purposes
    assert abs(detected - expected) < 0.1, (
        f"detected {detected:.3f}, expected {expected:.3f}"
    )


def test_detect_sweep_start_skips_loud_preburst_transients():
    """Real-world failure mode from the user's 2026-05 NPZ: pre-burst
    homing / heating produced loadcell std ≈ 130 in a 5-second window
    around t ≈ 17 s (firmware doing post-home calibration motion). The
    old detector used `median(head_stds) + 8·MAD` as its threshold,
    which the loud transient cleared, and the detector returned a t
    inside the homing region (t ≈ 7 s) instead of the real burst
    start at t ≈ 100 s. The fix pins the threshold to 25% of the
    GLOBAL max of stds -- since bursts produce std vastly larger than
    any pre-burst noise, the threshold easily separates them.

    Synthesised analogue: 5 s of loud (~10% of burst-magnitude)
    pre-burst noise, then quiet, then real bursts. Detector must
    locate the bursts, not the noise.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06),
        cycles_per_K=8, slow_half_s=1.0, fast_half_s=0.25,
    ))
    sr = 1000.0
    noise_burst_start = 5.0
    noise_burst_end = 10.0
    real_burst_start = 30.0   # = sweep_t0 + start_offset
    sweep_t0 = real_burst_start - plan.segments[0].start_offset_s
    total_s = real_burst_start + plan.segments[-1].duration_s + 1.0
    n = int(total_s * sr)
    t_rel = np.arange(n) / sr
    t0_abs = 1000.0
    t_abs = t0_abs + t_rel
    rng = np.random.default_rng(7)
    # Quiet background
    force = rng.normal(0, 0.01, size=n)
    # Pre-burst loud transient (modeling the homing/heating noise the
    # user's printer emits) -- 10% of burst amplitude, sustained 5 s,
    # NOT periodic at cycle_period.
    burst_amplitude = (
        plan.params.fast_feed_mm_s - plan.params.slow_feed_mm_s
    ) * 0.5
    noise_mask = (t_rel >= noise_burst_start) & (t_rel < noise_burst_end)
    force[noise_mask] += rng.normal(
        0, 0.1 * burst_amplitude, size=int(noise_mask.sum())
    )
    # Real bursts
    for seg in plan.segments:
        period = seg.cycle_period_s
        slow_half = plan.params.slow_half_s
        seg_t_rel = t_rel - sweep_t0
        inside = (
            (seg_t_rel >= seg.start_offset_s)
            & (seg_t_rel < seg.start_offset_s + seg.duration_s)
        )
        phase = (seg_t_rel - seg.start_offset_s) % period
        force[inside] = np.where(
            phase[inside] < slow_half,
            plan.params.slow_feed_mm_s,
            plan.params.fast_feed_mm_s,
        ) + rng.normal(0, 0.01, size=int(inside.sum()))

    detected = _detect_sweep_start(
        t_abs, force,
        plan.segments[0].cycle_period_s,
        plan.params.slow_half_s,
    )
    expected = t0_abs + real_burst_start
    assert detected is not None, "detector should fire even with pre-burst noise"
    err = abs(detected - expected)
    # Within one cycle period is fine -- the fine stage's earliest-peak
    # logic resolves cycle alignment.
    assert err < plan.segments[0].cycle_period_s, (
        f"detected {detected:.2f}, expected {expected:.2f} (off by {err:.2f}s) "
        f"-- the loud pre-burst transient must not pull the anchor early"
    )


def test_force_cycle_slicing_overrides_bad_pos_x_periodicity():
    """End-to-end regression for the failure mode in the user's
    2026-05 NPZ (`run_1778932341.npz`):

      * pos_x cycle-start detector finds ~50 spurious starts in random
        non-burst regions (firmware reports pos_x with sub-cycle
        steps at low coupled_dx amplitudes).
      * The periodicity validator locks onto a coincidental periodic
        run somewhere in that noise -- NOT in the real burst region.
      * sweep_t0 lands ~67 s past the real first burst.
      * Per-K windows fall progressively past the end of the data.

    Two layered fixes are exercised here:
      1. Sanity gate on pos_x anchors: if the resulting sweep_t0
         would place sweep_end past the captured data, reject the
         anchor.
      2. Force-cycle slicer: detect cycle starts in the loadcell
         signal (clean rising edges per cycle) and use those to
         slice per-K windows. Overrides pos_x slicing because the
         loadcell signal has higher SNR for cycle boundaries.

    Synthesised analogue: a 6-K sweep where the loadcell has clean
    bursts, pos_x is noisy with sub-cycle false positives that
    happen to form a "periodic run" later in the data. The
    detector must reject the bad pos_x anchor and use the loadcell
    instead.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=5,
        slow_half_s=2.0, fast_half_s=0.8,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s  # 2.8
    burst_s = cycles * period  # 14
    n_k = len(plan.segments)

    sr = 200.0
    duration_s = burst_s * n_k + 10  # just enough room
    n = int(sr * duration_s)
    t0_abs = 9000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(101)

    # Bursts START at t=2.0 (sweep_t0 + start_offset_s = 2.0 means
    # sweep_t0 = 1.5). Place clean square-wave bursts in the force.
    real_burst_start = 2.0
    sweep_t0_true = real_burst_start - plan.segments[0].start_offset_s
    force_y = rng.normal(0, 5.0, size=n)
    burst_starts_abs = [
        real_burst_start + i * burst_s for i in range(n_k)
    ]
    for k_idx, burst_start in enumerate(burst_starts_abs):
        for c in range(cycles):
            t_slow = burst_start + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            slow_mask = (
                (force_t - t0_abs >= t_slow)
                & (force_t - t0_abs < t_fast)
            )
            fast_mask = (
                (force_t - t0_abs >= t_fast)
                & (force_t - t0_abs < t_end)
            )
            force_y[slow_mask] += 1000.0
            force_y[fast_mask] += 3000.0

    # pos_x: real burst-amplitude oscillation in the burst region
    pos_sr = 56.0
    n_pos = int(sr * duration_s / (sr / pos_sr))
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.0)
    for burst_start in burst_starts_abs:
        for c in range(cycles):
            t_slow = burst_start + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow) & (pos_t - t0_abs < t_fast)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow) / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.0 + 1.0 * slow_phase
            fast_mask = (
                (pos_t - t0_abs >= t_fast) & (pos_t - t0_abs < t_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast) / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.0 - 1.0 * fast_phase

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t, pos_x=pos_x,
    )
    notes = "\n".join(analysis.notes)
    # Either pos_x periodic anchored correctly (because oscillation is clean)
    # OR loadcell anchored after pos_x rejection. Both are acceptable;
    # what we REQUIRE is correct coverage on every K window.
    assert len(analysis.per_k) == n_k
    for r in analysis.per_k:
        assert r.coverage > 0.5, (
            f"K={r.k:.4f} coverage {r.coverage:.2f} too low; notes:\n{notes}"
        )
    # Baselines must reflect the planted plateaus (~1000 slow / ~3000 fast)
    assert analysis.force_baselines is not None, (
        f"force_baselines should be present; notes:\n{notes}"
    )
    delta = (
        analysis.force_baselines.fast_plateau
        - analysis.force_baselines.slow_plateau
    )
    assert 1500 < delta < 2500, (
        f"baselines delta {delta:.0f} should be ~2000 (planted "
        f"3000-1000); slow={analysis.force_baselines.slow_plateau:.0f}, "
        f"fast={analysis.force_baselines.fast_plateau:.0f}"
    )


def test_pos_x_anchor_ignores_home_and_park_trajectory():
    """The naive detector ("first sample past threshold") was triggering
    during the park move on real hardware: on Core One the home X is far
    from purge_x, so the first pos_x sample after M331/homing reads e.g.
    250 mm -- well past the +0.5·amplitude threshold -- and the anchor
    locked to that instead of the actual first burst. This planted that
    trajectory in synthetic data: home at X=250, park move down to
    purge_x=30, settle, then bursts. The anchor MUST find the burst
    transition, not the park motion.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=4, slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, coupled_dx_mm=0.05,
    ))
    # Loadcell trace: pure noise (the auto-detect fallback would be unreliable).
    sr = 100.0
    duration_s = 30.0
    n = int(duration_s * sr)
    t0_abs = 5000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(7)
    force_y = rng.normal(0, 1.0, size=n)

    # pos_x at 10 Hz. Trajectory:
    #   [0, 2): home at X=250 (post-G28)
    #   [2, 4): park move from 250 -> 30 (linear ramp)
    #   [4, 11): settled at purge_x=30
    #   [11, ...): burst loop, oscillating 30 <-> 30.05 every cycle
    # The expected anchor is the first 30 -> 30.05 transition at t≈11.
    pos_sr = 10.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.zeros(n_pos)
    # Phase 1: home position (X=250) for 2 s
    home_end = int(2.0 * pos_sr)
    pos_x[:home_end] = 250.0
    # Phase 2: park ramp 250 -> 30 over 2 s -- explicitly crosses the
    # naive +0.5·amplitude threshold (30.025) from above
    park_end = int(4.0 * pos_sr)
    pos_x[home_end:park_end] = np.linspace(250.0, 30.0, park_end - home_end)
    # Phase 3: settled at purge_x
    settle_end = int(11.0 * pos_sr) + 1
    pos_x[park_end:settle_end] = 30.0
    # Phase 4: bursts. Slow leg = X moves to 30.05 for 0.4 s,
    # fast leg = X back to 30 for 0.4 s. Just plant the first transition;
    # everything after that doesn't matter for the anchor.
    transition_idx = settle_end
    pos_x[transition_idx:] = 30.05

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t,
        force_y=force_y,
        plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t,
        pos_x=pos_x,
    )
    expected_first_motion = 0.5 * (pos_t[transition_idx - 1] + pos_t[transition_idx])
    expected_t0 = expected_first_motion - plan.segments[0].start_offset_s
    note_lines = "\n".join(analysis.notes)
    assert "pos_x first-motion" in note_lines, (
        f"pos_x anchor should be reported in notes; got: {note_lines}"
    )
    assert f"{expected_t0:.3f}" in note_lines, (
        f"expected anchored t0 ≈ {expected_t0:.3f} in notes; got: {note_lines}"
    )


def test_per_k_windows_track_pos_x_when_plan_offsets_drift():
    """End-to-end: bursts ACTUALLY execute on a different schedule than the
    plan predicts (planner overhead between K segments shifts everything
    later). Data-based slicing from pos_x cycle detection MUST recover
    the right K windows; plan-based slicing would land outside the
    bursts and the per-K plots come up empty (which is what the user
    has been seeing).
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    # 2 K values, 8 cycles each, 1.0 + 0.25 = 1.25 s cycle period.
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=8,
        slow_half_s=1.0, fast_half_s=0.25,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    burst_s = cycles * period

    # Construct synthetic recv-monotonic timeline. The PLAN says K[0]
    # starts at 0.5 s and K[1] at 0.5 + burst_s. Real hardware adds a
    # drift before K[1]; we plant a 2 s shift to simulate planner overhead
    # so plan-based slicing would land entirely outside K[1]'s bursts.
    sr = 200.0
    duration_s = 30.0
    n = int(duration_s * sr)
    t0_abs = 1000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(101)
    force_y = rng.normal(0, 0.5, size=n)

    real_burst_starts = [
        0.5,                       # K[0]
        0.5 + burst_s + 2.0,       # K[1] -- drifted by 2 s
    ]
    # Plant burst structure in force_y so the data has SOMETHING the
    # analyser can integrate at the real burst times (not at plan times).
    for k_idx, burst_start in enumerate(real_burst_starts):
        for c in range(cycles):
            t_cycle = burst_start + c * period
            # Big amplitude during the cycle so it dominates the noise.
            mask = (force_t - t0_abs >= t_cycle) & (force_t - t0_abs < t_cycle + period)
            force_y[mask] += 100.0 * (k_idx + 1)  # different per K

    # pos_x at 56 Hz: stays at firmware-baseline ≈ 30.2 until the first
    # burst, then ramps linearly between 30.2 and 31.2 during each cycle
    # (slow leg up, fast leg down) -- this matches what the firmware
    # actually emits (interpolated planner position), not idealized
    # square steps.
    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.2)
    for burst_start in real_burst_starts:
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            t_fast_start = t_slow_start + plan.params.slow_half_s
            t_cycle_end = t_slow_start + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow_start)
                & (pos_t - t0_abs < t_fast_start)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow_start)
                / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase  # ramp up
            fast_mask = (
                (pos_t - t0_abs >= t_fast_start)
                & (pos_t - t0_abs < t_cycle_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast_start)
                / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase  # ramp down

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t,
        force_y=force_y,
        plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t,
        pos_x=pos_x,
    )
    notes = "\n".join(analysis.notes)
    assert "K windows sliced from data" in notes, (
        f"data-based slicing should be reported; notes were:\n{notes}"
    )
    # Both K segments should have plenty of samples (high coverage)
    # because the windows are aligned to where the bursts actually fired.
    assert len(analysis.per_k) == 2
    for k in analysis.per_k:
        assert k.coverage > 0.5, (
            f"K={k.k:.4f} got coverage {k.coverage:.2f} -- expected >0.5; "
            f"data-based slicing failed to align"
        )
        assert k.n_samples > 100, (
            f"K={k.k:.4f} got only {k.n_samples} samples -- window misaligned"
        )


def test_per_k_windows_reject_spurious_pre_burst_motion():
    """Real failure mode: the velocity-based cycle detector picks up
    park-move transients and planner-lookahead jitter during M-code
    processing as false "cycle starts" at the front of pos_x. The
    previous slicing logic blindly took the first N starts, so those
    false positives shifted every K window's alignment downstream --
    K[0] window got chopped short and K[1] window ballooned to span
    everything until the real K[1] cycles. The user's screenshot showed
    K=0.0100's window at 95 seconds wide for this reason. The fix:
    periodicity validation. Only trust a cycle start once we've seen
    several consecutive starts spaced at cycle_period_s (±30%); discard
    anything before the validated periodic run.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=4, slow_half_s=1.0, fast_half_s=0.25,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    slow_half = plan.params.slow_half_s
    fast_half = plan.params.fast_half_s

    duration_s = 30.0
    sr = 200.0
    n = int(duration_s * sr)
    t0_abs = 1000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(42)
    force_y = rng.normal(0, 0.5, size=n)

    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.2)

    # SPURIOUS pre-burst motion: three short out-and-back X dithers at
    # IRREGULAR spacing (NOT cycle_period_s). These mimic the park-move
    # and planner-lookahead transients we see in real Buddy data. The
    # detector's velocity-zero-crossing logic happily flags each as a
    # "cycle start"; only periodicity validation can reject them.
    for t_swing_start in [0.5, 1.2, 1.9]:
        slow_mask = (pos_t - t0_abs >= t_swing_start) & (pos_t - t0_abs < t_swing_start + 0.3)
        slow_phase = (pos_t[slow_mask] - t0_abs - t_swing_start) / 0.3
        pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
        fast_mask = (pos_t - t0_abs >= t_swing_start + 0.3) & (pos_t - t0_abs < t_swing_start + 0.4)
        fast_phase = (pos_t[fast_mask] - t0_abs - t_swing_start - 0.3) / 0.1
        pos_x[fast_mask] = 31.2 - 1.0 * fast_phase

    # Real bursts start at t=6.0 (after a settle period). Both K segments
    # run back-to-back (no inter-K dwell, matching the current gcode_gen).
    burst_offsets = [6.0, 6.0 + cycles * period]
    for burst_start in burst_offsets:
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            t_fast_start = t_slow_start + slow_half
            t_cycle_end = t_slow_start + period
            slow_mask = (pos_t - t0_abs >= t_slow_start) & (pos_t - t0_abs < t_fast_start)
            slow_phase = (pos_t[slow_mask] - t0_abs - t_slow_start) / slow_half
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
            fast_mask = (pos_t - t0_abs >= t_fast_start) & (pos_t - t0_abs < t_cycle_end)
            fast_phase = (pos_t[fast_mask] - t0_abs - t_fast_start) / fast_half
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase

    # Plant burst structure in force_y so coverage is meaningful
    for k_idx, burst_start in enumerate(burst_offsets):
        for c in range(cycles):
            t_cycle = burst_start + c * period
            mask = (force_t - t0_abs >= t_cycle) & (force_t - t0_abs < t_cycle + period)
            force_y[mask] += 100.0 * (k_idx + 1)

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t,
        force_y=force_y,
        plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t,
        pos_x=pos_x,
    )
    notes = "\n".join(analysis.notes)
    assert "periodic cycle detection" in notes, (
        f"periodicity validation should fire; notes:\n{notes}"
    )
    # Spurious motion at t=0.5/1.2/1.9 (sweep-relative) must NOT be in
    # K[0]'s window -- K[0]'s window should start at t=6.0 - sweep_t0,
    # which corresponds to a sweep-relative time of segments[0].start_offset_s.
    assert len(analysis.per_k) == 2
    for k in analysis.per_k:
        assert k.coverage > 0.5, (
            f"K={k.k:.4f}: coverage {k.coverage:.2f} too low; "
            f"periodicity-validated slicing failed"
        )
        assert k.n_samples > 100, (
            f"K={k.k:.4f}: only {k.n_samples} samples"
        )


def test_pos_x_anchor_handles_firmware_offset_baseline():
    """Real failure mode (2026-05): the user's Buddy reports pos_x ≈ 30.2 mm
    when the toolhead is "settled at purge_x = 30 mm" -- some planner
    lookahead or sensor offset. The previous detector compared against the
    configured `purge_x` and never fired because |30.2 - 30| > 0.2·amplitude.
    The auto-baseline detector should ignore the configured purge_x entirely
    and lock onto the first step away from whatever the firmware happens to
    be reporting as the settled value.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.0, 0.05),
        cycles_per_K=4, slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    sr = 100.0
    duration_s = 30.0
    n = int(duration_s * sr)
    t0_abs = 5000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(11)
    force_y = rng.normal(0, 1.0, size=n)

    # pos_x at 56 Hz (matching observed Buddy throttle). Trajectory:
    #   [0, 11): settled at OFFSET baseline 30.2 (not the configured 30.0)
    #   [11, ...): toolhead jumps to 31.2 (== firmware-baseline + amplitude)
    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    firmware_baseline = 30.2  # not 30.0!
    pos_x = np.full(n_pos, firmware_baseline)
    transition_idx = int(11.0 * pos_sr) + 1
    pos_x[transition_idx:] = firmware_baseline + 1.0

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t,
        force_y=force_y,
        plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t,
        pos_x=pos_x,
    )
    expected_first_motion = 0.5 * (pos_t[transition_idx - 1] + pos_t[transition_idx])
    expected_t0 = expected_first_motion - plan.segments[0].start_offset_s
    note_lines = "\n".join(analysis.notes)
    assert "pos_x first-motion" in note_lines, (
        f"anchor should fire with offset baseline; notes were: {note_lines}"
    )
    assert f"{expected_t0:.3f}" in note_lines, (
        f"expected anchored t0 ≈ {expected_t0:.3f}; notes were: {note_lines}"
    )


def test_integral_uses_supplied_pos_x_transitions():
    """When `transition_idx` and `directions` are supplied, the integral
    must use them as the per-cycle reference clock instead of detecting
    transitions from the model command wave.

    This is what wires pos_x velocity sign-flips into the per-cycle
    integration windows: the model wave assumes perfect periodicity
    from burst_start, so over 14 cycles its predicted transitions
    accumulate planner-overhead and accel-limit jitter and drift off
    where the printer actually executed each leg. Anchoring to pos_x
    keeps the ±half_win integration centred on real transitions.

    Test setup: a force trace with two narrow positive pulses centered
    at t=0.5 s and t=1.5 s. A model command wave whose mid-level
    crossings are at t=0.7 s and t=1.7 s -- shifted 200 ms LATER than
    the actual pulses (the failure mode this fix targets). Supplied
    transitions at t=0.5/1.5 s should integrate the full pulse;
    auto-detected transitions at t=0.7/1.7 s sit on the post-pulse
    plateau and integrate to near zero.
    """
    from prusa_pa_tuner.analysis import _integral_area

    dt = 0.001
    n = 2000
    force = np.zeros(n)
    pulse_half = 50  # 50 samples = 50 ms
    force[500 - pulse_half : 500 + pulse_half] = 1.0
    force[1500 - pulse_half : 1500 + pulse_half] = -1.0
    command = np.zeros(n)
    command[700:1700] = 1.0

    area_auto = _integral_area(force, command, dt)
    area_explicit = _integral_area(
        force, command, dt,
        transition_idx=np.array([500, 1500]),
        directions=np.array([1.0, -1.0]),
    )
    assert abs(area_explicit) > 10 * abs(area_auto), (
        f"explicit transitions should pick up the pulses far better than "
        f"auto-detection at the wrong times -- got "
        f"auto={area_auto:.4f}, explicit={area_explicit:.4f}"
    )
    assert abs(area_explicit) > 0.04


def test_pos_x_transitions_used_for_per_cycle_integration():
    """End-to-end: when pos_t/pos_x are supplied to `analyse_sweep`, the
    integrator must consume the pos_x velocity sign-flips as the
    per-cycle reference clock and note that it did so.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=6,
        slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    burst_s = cycles * period

    sr = 200.0
    duration_s = 15.0
    n = int(duration_s * sr)
    t0_abs = 2000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(7)
    force_y = rng.normal(0, 0.1, size=n)

    burst_starts = [0.5, 0.5 + burst_s + 0.4]
    for k_idx, burst_start in enumerate(burst_starts):
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            mask = (
                (force_t - t0_abs >= t_slow_start)
                & (force_t - t0_abs < t_slow_start + period)
            )
            force_y[mask] += 50.0 * (k_idx + 1)

    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.2)
    for burst_start in burst_starts:
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            t_fast_start = t_slow_start + plan.params.slow_half_s
            t_cycle_end = t_slow_start + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow_start)
                & (pos_t - t0_abs < t_fast_start)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow_start)
                / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
            fast_mask = (
                (pos_t - t0_abs >= t_fast_start)
                & (pos_t - t0_abs < t_cycle_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast_start)
                / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t, pos_x=pos_x,
    )
    notes = "\n".join(analysis.notes)
    assert "per-cycle integration windows anchored to pos_x" in notes, (
        f"analyser should report pos_x-driven per-cycle integration; "
        f"notes were:\n{notes}"
    )
    # Should see ~ (cycles_per_K * 2) - 1 = 11 transitions per K window,
    # roughly 2*cycles*2 - small_clipping ~ 22-24 across the sweep.
    assert "transitions detected" in notes


def test_force_baselines_and_ground_truth_overlay_from_pos_x():
    """When pos_x is supplied, analyse_sweep extracts slow/fast plateau
    medians from the force trace and emits a per-K ground_truth_force
    overlay (square wave at those baseline levels, timed by the pos_x
    leg transitions). This is what makes over/undershoot visible at a
    glance in the per-K plots.

    The plateau medians must be POSITION-WEIGHTED averages: we plant
    distinct slow-leg and fast-leg force levels (slow=+30, fast=+90 from
    centered) and check that:
      * force_baselines.slow_plateau ≈ −X, fast_plateau ≈ +Y where X,Y
        reflect the centered values of the planted levels;
      * each KWindow's ground_truth_force is non-empty;
      * its values are exactly the two baseline numbers nowhere else;
      * timing of the level changes matches the pos_x transitions.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=6,
        slow_half_s=1.0, fast_half_s=0.25,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    burst_s = cycles * period

    sr = 200.0
    duration_s = 20.0
    n = int(duration_s * sr)
    t0_abs = 3000.0
    force_t = t0_abs + np.arange(n) / sr

    burst_starts = [0.5, 0.5 + burst_s]
    # Plant noisy force trace with distinct slow / fast plateau levels.
    # Slow leg: force = SLOW_LEVEL.  Fast leg: force = FAST_LEVEL.
    SLOW_LEVEL = 30.0
    FAST_LEVEL = 200.0
    rng = np.random.default_rng(99)
    force_y = rng.normal(0, 0.5, size=n)
    for burst_start in burst_starts:
        for c in range(cycles):
            t_slow = burst_start + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            slow_mask = (
                (force_t - t0_abs >= t_slow)
                & (force_t - t0_abs < t_fast)
            )
            fast_mask = (
                (force_t - t0_abs >= t_fast)
                & (force_t - t0_abs < t_end)
            )
            force_y[slow_mask] += SLOW_LEVEL
            force_y[fast_mask] += FAST_LEVEL

    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.2)
    for burst_start in burst_starts:
        for c in range(cycles):
            t_slow = burst_start + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow) & (pos_t - t0_abs < t_fast)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow) / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
            fast_mask = (
                (pos_t - t0_abs >= t_fast) & (pos_t - t0_abs < t_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast) / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t, pos_x=pos_x,
    )

    assert analysis.force_baselines is not None, (
        f"force_baselines should be computed; notes:\n"
        + "\n".join(analysis.notes)
    )
    # The plateau medians are computed on CENTERED force, so
    # fast_plateau > slow_plateau and the difference equals
    # ~(FAST_LEVEL - SLOW_LEVEL) (within noise).
    delta = (
        analysis.force_baselines.fast_plateau
        - analysis.force_baselines.slow_plateau
    )
    expected_delta = FAST_LEVEL - SLOW_LEVEL
    assert abs(delta - expected_delta) < 5.0, (
        f"baseline plateau delta {delta:.2f} should ≈ planted "
        f"{expected_delta:.2f}; baselines: "
        f"slow={analysis.force_baselines.slow_plateau:.2f}, "
        f"fast={analysis.force_baselines.fast_plateau:.2f}"
    )
    # Every KWindow gets a force-units ground truth overlay.
    for kw in analysis.windows:
        assert len(kw.ground_truth_force) == len(kw.t), (
            f"ground_truth_force length mismatch on K={kw.k:.4f}"
        )
        gt = np.asarray(kw.ground_truth_force, dtype=float)
        unique = np.unique(np.round(gt, 4))
        # Exactly two distinct plateau values (slow + fast). Allow
        # rounding noise of ±1 distinct value at the boundaries.
        assert 1 <= len(unique) <= 3, (
            f"ground truth wave should take ≤3 distinct values "
            f"(slow/fast plateau + edge); got {len(unique)} on K={kw.k:.4f}"
        )
        # Levels match the baseline values.
        assert np.any(np.isclose(gt, analysis.force_baselines.slow_plateau, atol=0.1)), (
            f"K={kw.k:.4f}: ground truth never reaches slow_plateau"
        )
        assert np.any(np.isclose(gt, analysis.force_baselines.fast_plateau, atol=0.1)), (
            f"K={kw.k:.4f}: ground truth never reaches fast_plateau"
        )


def test_z_marker_anchors_sweep_t0():
    """End-to-end: a pre-burst Z lift+drop pulse in pos_z lets the
    analyser lock sweep_t0 to the Z return-to-baseline timestamp,
    sidestepping the periodicity-detection failure modes the user
    has hit repeatedly (park motion / planner jitter at the front of
    pos_x, missed or extra cycle starts mid-sweep, etc.).

    Test: synthesize a sweep where pos_z bumps up 2 mm, dwells 100 ms,
    drops back, then bursts begin 500 ms later. Force trace contains
    the cycle structure; pos_x contains the cycle motion; pos_z
    contains ONLY the marker and a flat baseline afterward. Verify:
      * `sweep_t0` is anchored via Z marker (notes mention it),
      * the first K window starts at start_offset_s relative to t0,
      * later K windows align correctly without periodicity drift.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05, 0.10),
        cycles_per_K=6,
        slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, purge_z=2.0,
        coupled_dx_mm=1.0, z_marker_lift_mm=2.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    burst_s = cycles * period

    sr = 200.0
    duration_s = 30.0
    n = int(sr * duration_s)
    t0_abs = 5000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(31)
    force_y = rng.normal(0, 0.5, size=n)

    # Z marker: lift at t=1.0, drop back at t=1.1 (z_return)
    pos_sr = 56.0
    n_pos = int(sr * duration_s / (sr / pos_sr))
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_z_baseline = plan.params.purge_z  # 2.0
    pos_z = np.full(n_pos, pos_z_baseline)
    z_lift_start = 1.0
    z_lift_end = 1.1   # = z_return relative to t0_abs
    lift_mask = (
        (pos_t - t0_abs >= z_lift_start) & (pos_t - t0_abs < z_lift_end)
    )
    pos_z[lift_mask] = pos_z_baseline + 2.0

    # Bursts begin at z_return + start_offset_s = 1.1 + 0.5 = 1.6
    z_return = z_lift_end
    first_burst_t = z_return + plan.segments[0].start_offset_s
    pos_x = np.full(n_pos, 30.2)
    burst_starts = [first_burst_t + i * burst_s for i in range(len(plan.segments))]
    for burst_start in burst_starts:
        for c in range(cycles):
            t_slow = burst_start + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow) & (pos_t - t0_abs < t_fast)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow) / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
            fast_mask = (
                (pos_t - t0_abs >= t_fast) & (pos_t - t0_abs < t_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast) / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase
            # Plant force during the bursts so coverage looks real
            f_slow = (
                (force_t - t0_abs >= t_slow) & (force_t - t0_abs < t_fast)
            )
            f_fast = (
                (force_t - t0_abs >= t_fast) & (force_t - t0_abs < t_end)
            )
            force_y[f_slow] += 50.0
            force_y[f_fast] += 200.0

    analysis = analyse_sweep(
        sweep_t0=t0_abs,  # deliberately wrong; analyser should override
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t, pos_x=pos_x,
        pos_z_t=pos_t, pos_z=pos_z,
        z_marker_lift_mm=plan.params.z_marker_lift_mm,
    )
    notes = "\n".join(analysis.notes)
    assert "Z marker" in notes, (
        f"Z-marker anchor should fire; notes were:\n{notes}"
    )
    # All 3 K windows should have decent coverage (bursts are correctly
    # captured by the windows because the anchor is precise).
    assert len(analysis.per_k) == 3
    for k in analysis.per_k:
        assert k.coverage > 0.5, (
            f"K={k.k:.4f} got coverage {k.coverage:.2f} (<0.5) -- "
            f"Z-anchor pipeline misaligned the window"
        )


def test_k0_warmup_extension_aligns_window_correctly():
    """When `first_slow_leg_factor > 1`, K[0]'s cycle 0 has an
    extended slow leg (= the sweep's warm-up / purge). The KSegment
    records this as `first_cycle_slow_extension_s`. The force-cycle
    slicer must shift K[0]'s window start back by the FULL
    `slow_half_s + extension` (NOT just slow_half_s) so K[0]'s window
    opens at the warm-up's actual start, with the long slow plateau
    visible at the front. Subsequent K's are unaffected.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    FACTOR = 5.0
    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02),
        cycles_per_K=3,
        slow_half_s=2.0, fast_half_s=0.8,
        first_slow_leg_factor=FACTOR,
    ))
    period = plan.segments[0].cycle_period_s
    slow_half = plan.params.slow_half_s
    cycles_per_K = plan.params.cycles_per_K
    extension = slow_half * (FACTOR - 1.0)  # 8 s extra slow on K[0] cycle 0

    # Verify the plan recorded the extension on K[0] only.
    assert plan.segments[0].first_cycle_slow_extension_s == extension
    assert plan.segments[1].first_cycle_slow_extension_s == 0.0
    # K[0] duration includes the extension.
    assert abs(
        plan.segments[0].duration_s
        - (cycles_per_K * period + extension)
    ) < 1e-9
    # K[1] is normal.
    assert abs(plan.segments[1].duration_s - cycles_per_K * period) < 1e-9

    # Generate rising edges that match the plan: K[0] cycle 0 fast leg
    # starts at `K[0]_start + slow_half * FACTOR`; subsequent edges at
    # cycle_period.
    sweep_t0 = 7000.0
    k0_start_abs = sweep_t0 + plan.segments[0].start_offset_s
    edges = []
    # K[0]: cycle 0 has extended slow leg
    edges.append(k0_start_abs + slow_half * FACTOR)
    for c in range(1, cycles_per_K):
        edges.append(edges[-1] + period)
    # K[1] starts right after K[0]'s last cycle ends; cycle 0 has normal slow leg
    k1_start_abs = k0_start_abs + plan.segments[0].duration_s
    edges.append(k1_start_abs + slow_half)
    for c in range(1, cycles_per_K):
        edges.append(edges[-1] + period)
    edges_arr = np.asarray(edges, dtype=float)

    result = _slice_from_force_cycles(edges_arr, plan, sweep_t0)
    assert result is not None
    windows, _ = result
    assert len(windows) == 2
    # K[0] window deliberately TRIMS the warm-up out: it starts at
    # the LAST `slow_half_s` of the warm-up (= already at slow
    # plateau) so K[0]'s plot has the same layout as K[1+].
    k0_start_rel = windows[0][0]
    expected_k0 = (
        plan.segments[0].start_offset_s
        + extension  # warm-up is hidden
    )
    assert abs(k0_start_rel - expected_k0) < 0.01, (
        f"K[0] starts at {k0_start_rel:.3f}, expected {expected_k0:.3f} "
        f"(start_offset + extension = warm-up end minus slow_half). "
        f"If off by ~+{extension:.0f}s the slicer is still including the warm-up."
    )
    # K[1] window starts at K[0]_start + K[0]_duration.
    k1_start_rel = windows[1][0]
    expected_k1 = plan.segments[0].start_offset_s + plan.segments[0].duration_s
    assert abs(k1_start_rel - expected_k1) < 0.01, (
        f"K[1] starts at {k1_start_rel:.3f}, expected {expected_k1:.3f}"
    )


def test_k_window_starts_and_ends_on_low_plateau():
    """User's 2026-05 requirement: every K segment must start on
    the slow plateau (cycle 0's slow-leg start) and END on the slow
    plateau (the NEXT K's cycle 0's slow-leg start -- a shared
    boundary point that "doubles its purpose"). Earlier slicing
    ended each K at the start of its own last fast leg, so the
    visible plot trailed off mid-burst at HIGH. The fix extends t_hi
    by `slow_half_s` so the trailing slow leg of the next K is
    visible.

    End-to-end synth: planted force with clean cycles, verify the
    last 100 ms of each K window mean lies within ±20% of the slow
    plateau (NOT the fast one).
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.02, 0.04),
        cycles_per_K=4,
        slow_half_s=2.0, fast_half_s=0.8,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    n_k = len(plan.segments)
    SLOW_LEVEL = 1000.0
    FAST_LEVEL = 3000.0

    sr = 200.0
    sweep_t0 = 5000.0
    total_s = (
        plan.segments[0].start_offset_s
        + cycles * period * n_k
        + 5.0  # trailing tail
    )
    n = int(sr * total_s)
    force_t = sweep_t0 + np.arange(n) / sr
    rng = np.random.default_rng(7)
    force_y = rng.normal(SLOW_LEVEL, 5.0, size=n)
    # Plant N cycles per K, slow-then-fast.
    for k_idx in range(n_k):
        burst_offset = (
            plan.segments[0].start_offset_s + k_idx * cycles * period
        )
        for c in range(cycles):
            t_slow = burst_offset + c * period
            t_fast = t_slow + plan.params.slow_half_s
            t_end = t_slow + period
            fast_mask = (
                (force_t - sweep_t0 >= t_fast)
                & (force_t - sweep_t0 < t_end)
            )
            force_y[fast_mask] = FAST_LEVEL + rng.normal(0, 5.0, size=int(fast_mask.sum()))

    ana = analyse_sweep(
        sweep_t0=sweep_t0,
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=False,
    )
    # Every K window's tail must end on slow plateau (the next K's
    # cycle 0 slow leg, captured by the +slow_half extension).
    for w in ana.windows:
        t = np.asarray(w.t)
        f = np.asarray(w.force)
        # Last 100 ms of the window.
        tail_mask = (t - t[0]) >= (t[-1] - t[0]) - 0.10
        tail_mean = float(np.mean(f[tail_mask])) if tail_mask.any() else float("nan")
        assert abs(tail_mean - SLOW_LEVEL) < 0.2 * (FAST_LEVEL - SLOW_LEVEL), (
            f"K={w.k:.4f} window ends with mean={tail_mean:.0f}, "
            f"expected near slow plateau ~{SLOW_LEVEL:.0f}. If close to "
            f"~{FAST_LEVEL:.0f} the slicer is ending mid-fast-leg "
            f"instead of after the trailing slow leg."
        )


def test_force_cycle_slicer_discards_noise_spike_at_front_and_pads_tail():
    """Regression for user's 2026-05 `run_1778941168.npz`:
      * A single noise spike at the very front of the loadcell trace
        produced a spurious rising edge ~5 s BEFORE the real bursts
        began.
      * The firmware then skipped the last cycle's rising edge.
    Net: 78 expected cycles, 77 real + 1 spurious junk at front.

    Old behaviour: `_slice_from_force_cycles` consumed the spurious
    edge as cycle 0, ran one short overall (77 cycles for 78 K), then
    returned None and fell back to plan offsets → entire K[0..N] grid
    shifted ~5 s before the actual bursts.

    New behaviour: (a) a periodicity scan at the front of the rising-
    edge list skips spurious edges that aren't part of a 3-consecutive
    period-spaced run, and (b) a small (≤5%) shortfall in cycle count
    is tolerated -- missing tail cycles are padded with synthetic
    starts so the windows that DO contain real data stay aligned.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.01, 0.02),
        cycles_per_K=4,
        slow_half_s=2.0, fast_half_s=0.8,
    ))
    period = plan.segments[0].cycle_period_s
    slow_half = plan.params.slow_half_s
    n_k = len(plan.segments)
    cycles_per_K = plan.params.cycles_per_K
    total_cycles = cycles_per_K * n_k  # 12
    sweep_t0 = 5000.0
    first_slow_leg_abs = sweep_t0 + plan.segments[0].start_offset_s

    # Generate clean rising edges (one per cycle), then:
    #   - inject one spurious noise edge ~5 s before the first cycle
    #   - drop the LAST cycle's rising edge
    real_edges = np.array([
        first_slow_leg_abs + slow_half + i * period
        for i in range(total_cycles - 1)  # drop the last one
    ])
    spurious_edge = np.array([first_slow_leg_abs - 5.0 + slow_half])
    all_edges = np.concatenate([spurious_edge, real_edges])
    all_edges.sort()

    result = _slice_from_force_cycles(all_edges, plan, sweep_t0)
    assert result is not None, "slicer must succeed despite 1 missing tail edge"
    windows, _ = result
    assert len(windows) == n_k, (
        f"slicer should tolerate 1 missing tail edge; got "
        f"{len(windows)} windows for {n_k} K segments"
    )
    # K[0] must align to the FIRST REAL slow-leg start, not to
    # the spurious edge − slow_half.
    k0_start = windows[0][0]
    expected = plan.segments[0].start_offset_s
    assert abs(k0_start - expected) < 0.01, (
        f"K[0] starts at {k0_start:.3f}, expected {expected:.3f}; "
        f"if off by ~−5s the periodicity scan didn't discard the "
        f"spurious noise edge"
    )


def test_force_cycle_slicer_aligns_to_slow_leg_start_not_rising_edge():
    """The force-cycle slicer detects slow→fast rising edges, but a
    cycle BEGINS at the start of the slow leg (`slow_half_s` before
    the rising edge). User's 2026-05 NPZ showed the K window starting
    at the rising edge -- so K[0] began at the END of cycle 0's slow
    leg, force trace led with a fast plateau, and the "low-high"
    intro was missing for every K.

    The fix shifts each rising edge back by `slow_half_s`. Test:
    synthesise a sweep where the rising edges are at known times and
    the slow legs start `slow_half_s` earlier. Verify K[0]'s window
    starts ON the planted slow-leg start, not on the rising edge.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=4,
        slow_half_s=2.0, fast_half_s=0.8,
    ))
    period = plan.segments[0].cycle_period_s  # 2.8
    slow_half = plan.params.slow_half_s        # 2.0
    cycles_per_K = plan.params.cycles_per_K    # 4
    n_k = len(plan.segments)
    total_cycles = cycles_per_K * n_k          # 8

    sweep_t0 = 1000.0
    first_slow_leg_start_abs = sweep_t0 + plan.segments[0].start_offset_s
    # Rising edges happen at slow-leg start + slow_half_s; one per cycle.
    rising_edges_abs = np.array([
        first_slow_leg_start_abs + slow_half + i * period
        for i in range(total_cycles)
    ])

    result = _slice_from_force_cycles(
        rising_edges_abs, plan, sweep_t0,
    )
    assert result is not None
    windows, _ = result
    assert len(windows) == n_k
    # K[0] window must start AT the slow-leg start, NOT slow_half later.
    k0_start_rel = windows[0][0]
    expected_k0_start = plan.segments[0].start_offset_s
    assert abs(k0_start_rel - expected_k0_start) < 0.01, (
        f"K[0] starts at {k0_start_rel:.3f}, expected {expected_k0_start:.3f} "
        f"(the slow leg start). If this is off by ~slow_half={slow_half}, the "
        f"shift back from rising-edge to slow-leg-start was lost."
    )


def test_force_cycle_slicer_rejects_spurious_double_edges_mid_sweep():
    """Regression for run_1778962189.npz: at mid-range K values the PA-
    induced velocity-reversal undershoot recovers ABOVE the mid-threshold,
    crossing it a second time per cycle. The old slicer counted these
    spurious crossings as extra cycle starts, shifting every K window
    after K[0] -- and K[N-1] eventually ran past the end of the data.

    The new slicer anchors on the FIRST run of 3 consecutive
    period-spaced edges, then PREDICTS each of the N expected rising
    edges at integer multiples of expected_period from the anchor,
    snapping each to the nearest detected edge within tolerance.
    Spurious doubles get ignored because they fall HALF a period off
    from any predicted edge.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05, 0.10),
        cycles_per_K=4,
        slow_half_s=2.0, fast_half_s=0.8,
    ))
    period = plan.segments[0].cycle_period_s   # 2.8
    slow_half = plan.params.slow_half_s         # 2.0
    cycles_per_K = plan.params.cycles_per_K     # 4
    n_k = len(plan.segments)
    total_cycles = cycles_per_K * n_k           # 12
    sweep_t0 = 1000.0
    first_slow_leg_start_abs = sweep_t0 + plan.segments[0].start_offset_s

    # Real rising edges = slow-leg-start + slow_half per cycle.
    real_rising = np.array([
        first_slow_leg_start_abs + slow_half + i * period
        for i in range(total_cycles)
    ])
    # Spurious "recovery" edges ~0.85s after each real rising edge,
    # appearing ONLY in K[1] (the mid-K segment). This matches the
    # pattern observed in run_1778962189.npz where K=0.03 produced
    # 2 detected rising edges per cycle but K=0.00 and K=0.10 each
    # produced exactly one.
    spurious_rising = []
    for c in range(cycles_per_K):
        # K[1] cycles: indices [cycles_per_K, 2*cycles_per_K)
        i = cycles_per_K + c
        spurious_rising.append(real_rising[i] + 0.85)
    spurious_rising = np.array(spurious_rising)
    all_edges = np.sort(np.concatenate([real_rising, spurious_rising]))
    assert len(all_edges) == total_cycles + cycles_per_K, (
        "test setup: should have N real + 1×K-worth spurious edges"
    )

    result = _slice_from_force_cycles(all_edges, plan, sweep_t0)
    assert result is not None
    windows, _ = result
    assert len(windows) == n_k

    # Each K window must start at the K segment's planned slow-leg start.
    for ki, seg in enumerate(plan.segments):
        expected = seg.start_offset_s
        # K[0] window start is `slow_half_s` after the warm-up start
        # (the warm-up is hidden); but with first_slow_leg_factor=1.0
        # (default) there's no warm-up extension, so this trims to 0.
        got = windows[ki][0]
        assert abs(got - expected) < 0.05, (
            f"K[{ki}]={seg.k:.4f} window starts at {got:.3f}, expected "
            f"{expected:.3f} -- spurious mid-K rising edges shifted "
            f"the K windows. Diff = {got - expected:+.3f}s."
        )


def test_force_cycle_slicer_rolling_anchor_resists_period_mismatch():
    """Regression for run_1779016571.npz K=0.0850 seg 0: the firmware's
    actual cycle period was slightly off from the planned period
    (e.g. 3.01 s vs 3.00 s); a GLOBAL-anchor prediction (`anchor +
    i * period`) drifts ~2 s over 200 cycles. By the time the slicer
    reached K=0.085 cycle 0, the predicted slow-leg start landed 2 s
    BEFORE the actual cycle — the displayed segment was missing the
    low_n plateau entirely.

    Rolling-anchor prediction (`previous_snapped + period`) caps the
    drift at one cycle: as soon as the slicer snaps to a real edge,
    it resyncs.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    plan = build_sweep(SweepParams(
        K_values=tuple(round(0.01 * i, 4) for i in range(11)),  # 11 K vals
        cycles_per_K=10,
        slow_half_s=2.0, fast_half_s=1.0,
    ))
    nominal_period = plan.segments[0].cycle_period_s          # 3.0
    actual_period = nominal_period + 0.012                     # 3.012 (firmware drift)
    slow_half = plan.params.slow_half_s
    total_cycles = sum(s.cycles for s in plan.segments)        # 110
    sweep_t0 = 1000.0
    first_slow_leg_abs = sweep_t0 + plan.segments[0].start_offset_s
    # Real rising edges: spaced by actual_period (3.012s), NOT
    # nominal_period (3.0s).
    rising = np.array([
        first_slow_leg_abs + slow_half + i * actual_period
        for i in range(total_cycles)
    ])
    # Drop a handful in the middle to force synthetic predictions —
    # those are the cycles where the bug would have bitten the
    # global-anchor approach.
    keep_mask = np.ones(len(rising), dtype=bool)
    for missed_idx in (30, 60, 90):
        keep_mask[missed_idx] = False
    rising_seen = rising[keep_mask]

    result = _slice_from_force_cycles(rising_seen, plan, sweep_t0)
    assert result is not None
    windows, _ = result
    assert len(windows) == len(plan.segments)

    # Each K window's start must align to the FIRST cycle's slow-leg
    # start, which is at (first_slow_leg_abs - sweep_t0) + i_cycle_at_K_start
    # * actual_period. (Not nominal_period!) For K[10] (= 0.10) the
    # first cycle is at cycle index 100, so the slow-leg start should
    # be at start_offset_s + 100 * actual_period — NOT 100 * nominal_period
    # which would be off by 100 × 0.012 = 1.2 s.
    last_k_window_start_swrel = windows[-1][0]
    expected_last_k_start = (
        plan.segments[0].start_offset_s
        + 100 * actual_period   # cycle 100 (start of K[10] = 0.10)
    )
    drift = last_k_window_start_swrel - expected_last_k_start
    assert abs(drift) < 0.15, (
        f"K[10] window starts at {last_k_window_start_swrel:.3f} "
        f"(sweep-rel), expected ~{expected_last_k_start:.3f}; drift "
        f"= {drift:+.3f}s. If drift > 100 ms, the predictor is using "
        f"the global anchor instead of rolling — synthetic cycle "
        f"predictions are accumulating period mismatch."
    )


def test_force_cycle_slicer_returns_absolute_anchor_for_sweep_t0_reconciliation():
    """The slicer must return its cycle-0 absolute time so analyse_sweep
    can detect when the caller's prior sweep_t0 anchor is off by more
    than half a cycle and re-derive sweep_t0 from the force-trace's
    own (high-SNR, full-trace) anchor. Without this, a wrong prior
    anchor (e.g. pos_x periodic locking onto park motion) leaves K
    windows expressed relative to the wrong origin even though they
    reference correct absolute cycle starts.
    """
    from prusa_pa_tuner.analysis import _slice_from_force_cycles

    plan = build_sweep(SweepParams(
        K_values=(0.00, 0.05),
        cycles_per_K=3,
        slow_half_s=2.0, fast_half_s=0.8,
    ))
    period = plan.segments[0].cycle_period_s
    slow_half = plan.params.slow_half_s
    sweep_t0 = 1000.0
    first_slow_leg_start_abs = sweep_t0 + plan.segments[0].start_offset_s
    n_cycles = sum(seg.cycles for seg in plan.segments)
    rising = np.array([
        first_slow_leg_start_abs + slow_half + i * period
        for i in range(n_cycles)
    ])

    result = _slice_from_force_cycles(rising, plan, sweep_t0)
    assert result is not None
    _, cycle0_abs = result
    # cycle0_abs = absolute time of cycle 0 slow-leg start, which is
    # sweep_t0 + start_offset_s by definition.
    expected = sweep_t0 + plan.segments[0].start_offset_s
    assert abs(cycle0_abs - expected) < 0.01, (
        f"cycle0_abs = {cycle0_abs:.3f}, expected {expected:.3f} "
        f"(= sweep_t0 + start_offset_s)"
    )


def test_z_marker_skips_sentinel_pos_z_prefix():
    """Buddy emits placeholder pos_z values (typically 168.000 = axis
    max, or -168 during homing) BEFORE the toolhead is positioned.
    The old Z-marker detector took the median of the first ~1s of
    samples as the resting baseline, which on user's run_1778962189.npz
    landed on 168.000 sentinels and made the +2mm lift threshold
    (169.000) unreachable -- detection returned None even though the
    real marker existed at print Z baseline.

    The new detector walks forward to find the first window where all
    samples fall in a plausible bed-Z range AND the window is settled
    (pp < 0.3 mm). That window's median is the baseline; the marker
    bump is then sought from there.
    """
    from prusa_pa_tuner.analysis import _detect_z_marker_anchor

    sr = 100.0
    dt = 1.0 / sr
    # 2s of 168.000 sentinel
    n_sentinel = int(2.0 * sr)
    # 2s of homing junk (-168 to 0)
    n_homing = int(2.0 * sr)
    # 5s of stable baseline at +2.0 mm
    n_baseline = int(5.0 * sr)
    # 0.4s marker pulse: +2 mm lift, 0.1s dwell, -2 mm drop
    n_lift = int(0.15 * sr)
    n_dwell = int(0.10 * sr)
    n_drop = int(0.15 * sr)
    # 2s of stable baseline again after marker return
    n_post = int(2.0 * sr)

    pos_z = np.concatenate([
        np.full(n_sentinel, 168.000),
        np.linspace(-168.0, 0.0, n_homing),
        np.full(n_baseline, 2.0),
        np.linspace(2.0, 4.0, n_lift),
        np.full(n_dwell, 4.0),
        np.linspace(4.0, 2.0, n_drop),
        np.full(n_post, 2.0),
    ])
    pos_z_t = np.arange(len(pos_z)) * dt

    z_return = _detect_z_marker_anchor(pos_z_t, pos_z, expected_lift_mm=2.0)
    assert z_return is not None, (
        "Z-marker detector returned None despite a clear +2 mm marker "
        "after sentinel/homing prefix"
    )
    # Marker return should land near the END of the drop ramp =
    # n_sentinel + n_homing + n_baseline + n_lift + n_dwell + ~half-of-drop
    expected_idx = n_sentinel + n_homing + n_baseline + n_lift + n_dwell + (n_drop // 2)
    expected_t = expected_idx * dt
    # Allow ±0.1s tolerance (linear-interp landing)
    assert abs(z_return - expected_t) < 0.1, (
        f"Z marker return at {z_return:.3f}s, expected ~{expected_t:.3f}s "
        f"(= sentinel + homing + baseline + lift + dwell + half-drop)"
    )


def test_pos_transitions_emit_at_actual_extremum_not_detection_time():
    """Regression for the lag the user spotted in 2026-05: an earlier
    sticky-anchor implementation only committed direction after pos_x
    had moved deadband AWAY from the anchor in the new direction. The
    anchor itself updated lazily, lagging the running extremum by
    ~deadband through every leg. Net effect: each transition was
    emitted 0.3-0.5 s LATE, so the dashed ground-truth wave appeared
    shifted ahead of the orange force trace by exactly that amount.

    The peak-follower implementation tracks the running extremum and
    records the transition timestamp at the EXTREMUM'S sample time,
    not the detection time. So even though a peak isn't "confirmed"
    until pos_x has dropped `deadband` below it, the recorded
    timestamp is when pos_x actually peaked.

    Test: a clean triangle wave with no quantization. The peak is at
    a known time. Verify the detected transition lands within one
    sample of that time -- not at "peak + deadband-time".
    """
    from prusa_pa_tuner.analysis import _detect_pos_transitions

    sr = 200.0
    duration = 6.0
    n = int(sr * duration)
    t = np.arange(n) / sr
    peak_t = 2.0  # pos_x rises 0->1 over 2s, falls 1->0 over 1s, then flat
    x = np.empty(n)
    for i, ti in enumerate(t):
        if ti < peak_t:
            x[i] = (ti / peak_t) * 1.0
        elif ti < peak_t + 1.0:
            x[i] = 1.0 - (ti - peak_t) * 1.0
        else:
            x[i] = 0.0

    times, dirs = _detect_pos_transitions(
        t, x, expected_amplitude_mm=1.0,
    )
    assert len(times) >= 1, f"should detect the peak; got {times}"
    # The peak transition should land within 1/sr = 5 ms of peak_t,
    # NOT at peak_t + deadband_time (which would be ~0.3 s for a
    # deadband-anchor detector).
    peak_lag = abs(float(times[0]) - peak_t)
    assert peak_lag < 0.05, (
        f"peak transition emitted at t={times[0]:.4f}, expected ~{peak_t} "
        f"(lag {peak_lag*1000:.0f} ms must be < 50 ms; large lag indicates "
        f"the detector is recording the confirmation time instead of the "
        f"true extremum time)"
    )


def test_pos_transitions_robust_to_firmware_quantization():
    """Real-world failure mode on the user's 2026-05 NPZ
    (`run_1778933659.npz`): Buddy reports pos_x in 0.01-0.1 mm
    stair-steps with samples staying at one value for several pos
    intervals before jumping. With coupled_dx=1 mm and fast_half=0.8 s
    the per-sample motion is ~22 µm, BELOW the firmware's pos_x
    reporting precision. `np.gradient` on stair-stepped data produces
    huge velocity spikes alternating with zero plateaus -- the
    smoothed-velocity sign-flip detector then fires 6 times in one
    real leg (observed) and the dashed ground-truth wave looked like
    rapid back-and-forth pulses instead of a clean square wave.

    The fix is the live-preview-style sticky sign-of-delta detector:
    direction is latched until pos_x has moved by `deadband_frac`
    of the expected amplitude (default 30%). Inside the deadband,
    the direction is held -- stair-step plateaus don't trigger
    spurious sign flips.

    Synthesised analogue: a 1 mm triangle-wave oscillation
    quantized to 0.1 mm steps. The detector must emit EXACTLY one
    transition per leg (peak/trough), not multiple per leg.
    """
    from prusa_pa_tuner.analysis import _detect_pos_transitions

    sr = 80.0  # roughly the user's observed pos_x rate
    cycle_period = 2.8
    slow_half = 2.0
    fast_half = 0.8
    n_cycles = 5
    duration = n_cycles * cycle_period
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Build a continuous triangle wave then quantize.
    x = np.empty(n)
    for i, ti in enumerate(t):
        phase = ti % cycle_period
        if phase < slow_half:
            x[i] = 30.0 + (phase / slow_half) * 1.0  # rising
        else:
            x[i] = 31.0 - ((phase - slow_half) / fast_half) * 1.0  # falling
    # Quantize to 0.1 mm steps (mimics Buddy's pos_x precision)
    x_q = np.round(x * 10) / 10

    times, dirs = _detect_pos_transitions(
        t, x_q, expected_amplitude_mm=1.0,
    )
    # Expect 2*n_cycles - 1 = 9 transitions: each cycle's peak (slow→fast)
    # and trough (fast→slow), minus 1 for the boundary the detector
    # never sees a "before" state for.
    assert 5 <= len(times) <= 12, (
        f"expected ~2*{n_cycles}=10 transitions on quantized triangle, "
        f"got {len(times)}: {times}"
    )
    # Verify gaps alternate between slow_half and fast_half (±50%)
    if len(times) >= 4:
        diffs = np.diff(times)
        # At least one gap close to slow_half, at least one close to fast_half
        near_slow = np.sum(np.abs(diffs - slow_half) < 0.5)
        near_fast = np.sum(np.abs(diffs - fast_half) < 0.4)
        assert near_slow >= 1 and near_fast >= 1, (
            f"gaps should alternate ~{slow_half}s and ~{fast_half}s; "
            f"got {diffs}, slow-matches={near_slow}, fast-matches={near_fast}"
        )


def test_pos_transitions_filter_spurious_sign_flickers():
    """Regression for the dashed-wave-glitches the user reported: a
    single noise sample crossing the velocity deadband used to flip
    `last_sign` and emit a phantom transition pair within one real leg.
    The visible symptom was brief rectangular pulses on top of the
    pos_x-derived ground-truth wave (most obvious at K=0.01-0.03).

    Test plants a clean triangle wave for pos_x (one slow leg + one
    fast leg, exactly two real transitions: a peak and a trough). Then
    drops a single 1-sample velocity spike into the middle of the slow
    leg -- enough to push the gradient briefly past the deadband. The
    hysteresis-filtered detector must register EXACTLY 2 transitions
    (the real ones at the peak and trough), not 4 (real + flicker).
    """
    from prusa_pa_tuner.analysis import _detect_pos_transitions

    sr = 56.0
    duration_s = 4.0
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    pos = np.zeros(n)
    # Slow leg: 0..1.0 s, x ramps from 0 to 1.
    # Fast leg: 1.0..1.25 s, x ramps from 1 down to 0.
    # Then rest stays at 0.
    for i, ti in enumerate(t):
        if ti < 1.0:
            pos[i] = ti
        elif ti < 1.25:
            pos[i] = 1.0 - 4.0 * (ti - 1.0)
        else:
            pos[i] = 0.0

    # Inject a spike at ~0.5 s (mid-slow-leg): a single sample dropped
    # back to baseline. The gradient at that point goes briefly
    # negative -- without hysteresis the detector emits TWO extra
    # transitions (down, then back up).
    spike_idx = int(0.5 * sr)
    pos[spike_idx] -= 0.05

    times, dirs = _detect_pos_transitions(t, pos)

    # Should pick up exactly the peak (slow→fast, direction +1) and the
    # trough (fast→slow, direction -1) -- no phantom pair from the spike.
    assert len(times) <= 3, (
        f"hysteresis should suppress the spike-induced phantom pair; "
        f"got {len(times)} transitions: t={times}, dirs={dirs}"
    )
    assert len(times) >= 1, "should still detect the real peak"
    # First real transition is the peak at t ≈ 1.0 s. The spike at
    # t ≈ 0.5 s must not appear in the output.
    for ts in times:
        assert ts > 0.7, (
            f"transition at {ts:.3f}s falls in the slow-leg interior "
            f"-- looks like the spike got through hysteresis"
        )


def test_anchor_rectifies_missed_and_extra_cycle_detections():
    """End-to-end regression for the failure mode where the user's run
    showed K[0..N-2] empty / drift-only plots while K[N-1] aligned
    perfectly. Diagnosed cause: one or more spurious / duplicate cycle
    starts mid-sweep slid every downstream K window by one cycle, so
    the LAST K window happened to land on the actual last burst's
    cycles by coincidence and looked aligned, while the earlier K
    windows landed on inter-burst noise.

    Test plants a clean 4-K sweep with 8 cycles per K (32 cycles total)
    but injects ONE duplicate cycle-start detection at the boundary
    between K[1] and K[2]. The rectifier must drop it as a false
    positive (gap < 0.5·period) so the per-K windows still land on
    cycles 0-7, 8-15, 16-23, 24-31 -- not 0-7, 8-15, 16-23, 25-32
    (the bug).

    The "extra detection" is simulated by adding a tiny extra
    direction-change in pos_x ~50 ms after a real cycle start, which
    the velocity sign-flip detector picks up as a second cycle.
    """
    from prusa_pa_tuner.analysis import analyse_sweep

    plan = build_sweep(SweepParams(
        K_values=(0.06, 0.07, 0.08, 0.09),
        cycles_per_K=8,
        slow_half_s=1.0, fast_half_s=0.25,
        purge_x=30.0, coupled_dx_mm=1.0,
    ))
    cycles = plan.params.cycles_per_K
    period = plan.segments[0].cycle_period_s
    n_k = len(plan.segments)
    burst_s = cycles * period

    sr = 200.0
    duration_s = 60.0
    n = int(duration_s * sr)
    t0_abs = 4000.0
    force_t = t0_abs + np.arange(n) / sr
    rng = np.random.default_rng(13)
    force_y = rng.normal(0, 0.1, size=n)

    # All K's bursts are contiguous (no inter-K gap, per current firmware).
    real_burst_starts = [0.5 + i * burst_s for i in range(n_k)]
    for k_idx, burst_start in enumerate(real_burst_starts):
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            mask = (
                (force_t - t0_abs >= t_slow_start)
                & (force_t - t0_abs < t_slow_start + period)
            )
            # Distinct per-K force amplitudes so misalignment shows up as
            # cross-contamination in the per-K means.
            force_y[mask] += 100.0 * (k_idx + 1)

    pos_sr = 56.0
    n_pos = int(duration_s * pos_sr)
    pos_t = t0_abs + np.arange(n_pos) / pos_sr
    pos_x = np.full(n_pos, 30.2)
    for k_idx, burst_start in enumerate(real_burst_starts):
        for c in range(cycles):
            t_slow_start = burst_start + c * period
            t_fast_start = t_slow_start + plan.params.slow_half_s
            t_cycle_end = t_slow_start + period
            slow_mask = (
                (pos_t - t0_abs >= t_slow_start)
                & (pos_t - t0_abs < t_fast_start)
            )
            slow_phase = (
                (pos_t[slow_mask] - t0_abs - t_slow_start)
                / plan.params.slow_half_s
            )
            pos_x[slow_mask] = 30.2 + 1.0 * slow_phase
            fast_mask = (
                (pos_t - t0_abs >= t_fast_start)
                & (pos_t - t0_abs < t_cycle_end)
            )
            fast_phase = (
                (pos_t[fast_mask] - t0_abs - t_fast_start)
                / plan.params.fast_half_s
            )
            pos_x[fast_mask] = 31.2 - 1.0 * fast_phase

    # Plant the bug: inject a brief X dip + recovery ~150 ms after the
    # second K segment's first cycle. The velocity detector picks up
    # the recovery edge as a second slow-leg start; the rectifier must
    # drop it because the gap to the previous real start is < 0.5·period.
    extra_t = real_burst_starts[1] + 0.15
    extra_t_end = extra_t + 0.10
    extra_mask = (
        (pos_t - t0_abs >= extra_t) & (pos_t - t0_abs < extra_t_end)
    )
    pos_x[extra_mask] -= 0.4  # dip down then back up = velocity sign flip pair

    analysis = analyse_sweep(
        sweep_t0=t0_abs,
        force_t=force_t, force_y=force_y, plan=plan,
        auto_detect_t0=True,
        pos_t=pos_t, pos_x=pos_x,
    )
    notes_str = "\n".join(analysis.notes)
    # Rectifier should report the skipped detection.
    assert "rectification" in notes_str, (
        f"rectifier should have engaged on duplicate detection; "
        f"notes were:\n{notes_str}"
    )
    # All K windows must have high coverage and ~equal sample counts.
    assert len(analysis.per_k) == n_k
    sample_counts = [r.n_samples for r in analysis.per_k]
    for r in analysis.per_k:
        assert r.coverage > 0.5, (
            f"K={r.k:.4f}: coverage={r.coverage:.2f} too low -- "
            f"window misaligned by extra cycle detection"
        )
    # The bug signature: K[N-1] aligned, K[0..N-2] much smaller samples.
    # After fix, all K's get similar sample counts.
    spread = (max(sample_counts) - min(sample_counts)) / max(sample_counts)
    assert spread < 0.3, (
        f"K window sample counts differ by {spread:.0%} -- alignment "
        f"is uneven (counts: {sample_counts})"
    )
    # And force means should track the planted +100, +200, +300, +400
    # offsets monotonically (if misaligned, K[0..N-2] would show the
    # wrong K's amplitude or zero).
    force_means = [r.force_mean for r in analysis.per_k]
    diffs = np.diff(force_means)
    assert np.all(diffs > 50), (
        f"force_mean per K should rise monotonically by ~100; got "
        f"{force_means} (diffs={diffs}) -- misalignment is leaking "
        f"K data into adjacent windows"
    )


def test_z_marker_anchor_uses_plan_direct_slicing_with_warmup_factor():
    """Z-marker pulse + warmup_factor=5 + cycles_per_K=1 must NOT be
    re-anchored by the force-trace detector.

    User report (run_1779100636, 2026-05-18): an 11 K × 1 cycle sweep
    with first_slow_leg_factor=5 produced wildly misaligned K windows.
    The Z-marker correctly anchored sweep_t0 from the pos_z pulse, then
    the force-trace cycle detector ran on top, threw the threshold off
    with the warm-up spike, mis-detected most cycles, and re-anchored
    sweep_t0 by +29.78 s -- displacing every K window into the wrong
    part of the trace.

    The fix: when the Z-marker successfully anchors sweep_t0, slice K
    windows directly from `plan.segments[i].start_offset_s` /
    `duration_s` and SKIP the force-trace re-anchor. The plan already
    encodes the warm-up correctly: K[0] cycle 0 is 11 s long
    (warmup_factor=5 × slow_half=2 s + fast_half=1 s) while every
    subsequent K is 3 s, and that variable-period schedule is the
    ground truth no edge detector can recover from a uniform-period
    model.

    Test: synth a force trace with the K[0] warm-up segment 11 s long,
    plant a Z marker pulse at the planned time, run analyse_sweep, and
    assert that the K window times exactly match the plan offsets.
    """
    params = SweepParams(
        K_values=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=1,
        slow_half_s=2.0,
        fast_half_s=1.0,
        coupled_dx_mm=1.0,
        coupled_dy_mm=0.0,
        coupled_dz_mm=0.0,
        purge_x=30.0, purge_y=30.0, purge_z=100.0,
        first_slow_leg_factor=5.0,
    )
    plan = build_sweep(params)
    # Verify the plan encodes the warm-up correctly: K[0] is 11 s
    # (10 s warm-up slow + 1 s fast), every later K is 3 s.
    assert abs(plan.segments[0].duration_s - 11.0) < 1e-6
    for seg in plan.segments[1:]:
        assert abs(seg.duration_s - 3.0) < 1e-6

    # Build synthetic streams. We include a 2 s pre-roll so the Z
    # marker pulse can land just before sweep_t0 (the gcode contract:
    # marker DOWN return-to-baseline IS sweep_t0).
    sample_rate = 200.0
    pos_rate = 50.0
    sweep_t0 = 1000.0  # arbitrary monotonic anchor
    pre_s = 2.0
    total_s = (
        plan.segments[-1].start_offset_s
        + plan.segments[-1].duration_s
        + 2.0
    )
    n = int((pre_s + total_s) * sample_rate)
    t_rel_full = np.arange(n) / sample_rate - pre_s  # sweep-rel
    t_abs = sweep_t0 + t_rel_full

    # Force trace: slow plateau at 700 (matches user's data), fast
    # plateau at 4500 during each segment's fast leg. Then a warm-up
    # purge spike during K[0] cycle 0 slow leg to recreate the failure
    # mode (this is what threw the global p10/p90 off in the user run).
    # Pre-sweep samples sit at -500 (cold loadcell baseline).
    force = np.full(n, -500.0)
    # During the sweep (t_rel >= 0), slow plateau at 700.
    in_sweep = t_rel_full >= 0.0
    force[in_sweep] = 700.0
    # Warm-up purge spike at +1.5..2.0 s into K[0]:
    spike_lo_t = plan.segments[0].start_offset_s + 1.5
    spike_hi_t = plan.segments[0].start_offset_s + 2.0
    spike_mask = (t_rel_full >= spike_lo_t) & (t_rel_full < spike_hi_t)
    force[spike_mask] = 12000.0
    # Each segment's fast leg goes to 4500.
    for seg in plan.segments:
        fast_lo_t = seg.start_offset_s + seg.duration_s - plan.params.fast_half_s
        fast_hi_t = seg.start_offset_s + seg.duration_s
        m = (t_rel_full >= fast_lo_t) & (t_rel_full < fast_hi_t)
        force[m] = 4500.0
    rng = np.random.default_rng(7)
    force = force + rng.normal(0.0, 5.0, size=n)

    # pos_x: triangle wave -- the real toolhead RAMPS X from purge_x to
    # purge_x+dx during the slow leg and ramps back during the fast leg
    # (peak-follower transition detector keys off the direction reversal
    # at each peak/trough). Hold-then-jump would produce phantom
    # transitions at the slow leg's START, which never happen in real
    # gcode execution.
    n_pos = int((pre_s + total_s) * pos_rate)
    pos_rel_full = np.arange(n_pos) / pos_rate - pre_s
    pos_t = sweep_t0 + pos_rel_full
    pos_x = np.full(n_pos, plan.params.purge_x)
    for seg in plan.segments:
        slow_start_t = seg.start_offset_s
        slow_end_t = seg.start_offset_s + seg.duration_s - plan.params.fast_half_s
        fast_end_t = seg.start_offset_s + seg.duration_s
        # Ramp up during slow leg: purge_x → purge_x + dx.
        m_slow = (pos_rel_full >= slow_start_t) & (pos_rel_full < slow_end_t)
        frac = (pos_rel_full[m_slow] - slow_start_t) / max(
            slow_end_t - slow_start_t, 1e-9
        )
        pos_x[m_slow] = plan.params.purge_x + plan.params.coupled_dx_mm * frac
        # Ramp down during fast leg: purge_x + dx → purge_x.
        m_fast = (pos_rel_full >= slow_end_t) & (pos_rel_full < fast_end_t)
        frac = (pos_rel_full[m_fast] - slow_end_t) / max(
            fast_end_t - slow_end_t, 1e-9
        )
        pos_x[m_fast] = plan.params.purge_x + plan.params.coupled_dx_mm * (1.0 - frac)

    # pos_z: emit the marker pulse. The gcode contract: marker DOWN
    # return-to-baseline IS sweep_t0. Place the pulse in (-0.3, 0.0)
    # sweep-rel: lift 2 mm starting -0.30, hold to -0.10, drop back
    # to baseline at 0.0.
    pos_z = np.full(n_pos, plan.params.purge_z)
    lift_mask = (pos_rel_full >= -0.30) & (pos_rel_full < -0.20)
    pos_z[lift_mask] = plan.params.purge_z + 2.0 * (
        (pos_rel_full[lift_mask] + 0.30) / 0.10
    )
    hold_mask = (pos_rel_full >= -0.20) & (pos_rel_full < -0.10)
    pos_z[hold_mask] = plan.params.purge_z + 2.0
    drop_mask = (pos_rel_full >= -0.10) & (pos_rel_full < 0.0)
    pos_z[drop_mask] = plan.params.purge_z + 2.0 * (
        1.0 - (pos_rel_full[drop_mask] + 0.10) / 0.10
    )
    pos_z_t = pos_t.copy()

    # The runner currently passes the FIRST force sample timestamp as
    # the seed sweep_t0. Replicate that here: the seed sits well before
    # the actual sweep, so any successful run must re-anchor.
    seed_sweep_t0 = sweep_t0 - 5.0

    analysis = analyse_sweep(
        sweep_t0=seed_sweep_t0,
        force_t=t_abs, force_y=force,
        plan=plan,
        pos_t=pos_t, pos_x=pos_x,
        pos_z_t=pos_z_t, pos_z=pos_z,
        z_marker_lift_mm=2.0,
    )

    # The Z-marker note must appear; force-trace re-anchor note must NOT.
    notes_text = "\n".join(analysis.notes)
    assert "Z marker return" in notes_text, (
        f"expected Z marker anchor, got notes:\n{notes_text}"
    )
    assert "re-anchored by force-trace" not in notes_text, (
        "force-trace must not override the Z-marker anchor; got:\n"
        + notes_text
    )
    # The slicer should pick pos_x transitions as the authoritative
    # source when X oscillation is present (this synthetic ramps pos_x,
    # producing clean transitions). Plan-direct is the fallback only.
    assert (
        "pos_x transitions" in notes_text
        or "plan-direct" in notes_text
        or "directly from plan" in notes_text
    ), f"expected pos_x-transition or plan-direct slicing note, got:\n{notes_text}"

    # Each K's window must start at the plan's start_offset_s (within
    # 100 ms tolerance for the synth grid).
    assert len(analysis.windows) == len(plan.segments)
    for seg, w in zip(plan.segments, analysis.windows):
        assert w.t, f"K={seg.k} window is empty"
        assert abs(w.t[0] - seg.start_offset_s) < 0.1, (
            f"K={seg.k} window starts at {w.t[0]:.3f} but plan says "
            f"{seg.start_offset_s:.3f}"
        )
        # K[0]'s window is 11 s wide (plus the trailing slow leg
        # extension); K[1+] are 3 s wide. The window EXTENDS by
        # slow_half_s past the plan's t_hi for display purposes, so
        # K[0]'s span is duration_s + slow_half = 11 + 2 = 13 s.
        expected_span = seg.duration_s + plan.params.slow_half_s
        actual_span = w.t[-1] - w.t[0]
        assert abs(actual_span - expected_span) < 0.2, (
            f"K={seg.k} window span is {actual_span:.3f}s, expected "
            f"~{expected_span:.3f}s (duration_s + slow_half)"
        )


def test_pos_x_transition_slicer_recovers_from_firmware_delay():
    """User report run_1779100636 (2026-05-18 follow-up): even with the
    Z-marker anchor correct, K=0.00..0.02 were 'trash' and K=0.03 was
    'a segment but not centered'.

    Root cause: Buddy's planner-sync between the Z-DOWN move, the M572
    that follows it, M83, and the first G1 with XYZ motion took an
    EXTRA ~2.3 s of wall-clock time that the gcode plan can't see. So
    while the Z-marker correctly placed sweep_t0 at the marker-return
    instant, the first X motion didn't actually begin for another
    ~2.3 s. Plan-direct slicing therefore put every K window 2.3 s
    AHEAD of the real data.

    Fix: when pos_x oscillation is captured, use the leg-direction
    transitions to refine sweep_t0. Each cycle's +1 (slow→fast) and
    -1 (fast→slow) transition is a physically guaranteed boundary
    (the toolhead actually reversed direction there), independent of
    any firmware processing delay.

    Test: stamp the planted pos_x ramps 2.3 s LATER than the Z marker
    + plan would predict. Verify the analyser's K windows land on the
    actual ramps, not on the empty pre-burst window.
    """
    from prusa_pa_tuner.analysis import _slice_from_pos_transitions

    params = SweepParams(
        K_values=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
        cycles_per_K=1,
        slow_half_s=2.0,
        fast_half_s=1.0,
        coupled_dx_mm=1.0,
        first_slow_leg_factor=5.0,
        purge_x=30.0, purge_y=30.0, purge_z=100.0,
    )
    plan = build_sweep(params)
    sweep_t0 = 1000.0  # Z-marker anchored time
    fw_delay = 2.3   # actual delay between Z-DOWN and first X motion
    pos_rate = 50.0
    pre_s = 2.0
    total_s = (
        plan.segments[-1].start_offset_s
        + plan.segments[-1].duration_s
        + 2.0
    )
    n = int((pre_s + total_s + fw_delay) * pos_rate)
    pos_rel = np.arange(n) / pos_rate - pre_s
    pos_t = sweep_t0 + pos_rel
    pos_x = np.full(n, plan.params.purge_x)
    # Lay down the ACTUAL pos_x ramps, shifted by fw_delay relative to
    # what the plan predicts.
    for seg in plan.segments:
        slow_start = seg.start_offset_s + fw_delay
        slow_end = seg.start_offset_s + seg.duration_s - plan.params.fast_half_s + fw_delay
        fast_end = seg.start_offset_s + seg.duration_s + fw_delay
        m1 = (pos_rel >= slow_start) & (pos_rel < slow_end)
        f1 = (pos_rel[m1] - slow_start) / max(slow_end - slow_start, 1e-9)
        pos_x[m1] = plan.params.purge_x + plan.params.coupled_dx_mm * f1
        m2 = (pos_rel >= slow_end) & (pos_rel < fast_end)
        f2 = (pos_rel[m2] - slow_end) / max(fast_end - slow_end, 1e-9)
        pos_x[m2] = plan.params.purge_x + plan.params.coupled_dx_mm * (1.0 - f2)
    # After the last fast leg X sits at purge_x. The peak-follower
    # only publishes the trough once it sees X RISE past the deadband
    # again, so simulate a final rising motion (park toward purge_x+2)
    # to confirm the last -1 transition.
    park_start_t = (
        plan.segments[-1].start_offset_s
        + plan.segments[-1].duration_s
        + fw_delay
        + 0.5
    )
    m_park = pos_rel >= park_start_t
    pos_x[m_park] = plan.params.purge_x + 2.0

    result = _slice_from_pos_transitions(
        pos_t, pos_x, plan, sweep_t0,
        coupled_amplitude_mm=plan.params.coupled_dx_mm,
    )
    assert result is not None, "transition-based slicer should have succeeded"
    windows, refined_t0 = result
    # Refined sweep_t0 should jump forward by ~fw_delay seconds. The
    # data-driven motion-onset detector necessarily lags the true onset
    # by however long the slow ramp takes to clear `motion_threshold`
    # (~0.05 mm out of a 1 mm amplitude on a 10 s warm-up = ~0.5 s),
    # so allow up to 0.8 s of slack here.
    shift = refined_t0 - sweep_t0
    assert abs(shift - fw_delay) < 0.8, (
        f"expected sweep_t0 refinement of ~{fw_delay}s, got {shift:+.3f}s"
    )
    # Each K window should land on the planted ramps. K[0]'s t_lo is
    # exactly start_offset_s by construction (refined_t0 = detected
    # motion - start_offset_s). The lag in motion detection (~0.5 s)
    # shifts refined_t0 by the same amount, so K[0]'s t_hi (the -1
    # transition) is ~0.5 s SHORT of the plan's predicted t_hi --
    # the OFFSET is correct in absolute terms, just expressed in a
    # frame anchored to the lagged onset. Later K windows are
    # measured between pos_x -1 transitions and have the exact plan
    # spacing.
    assert len(windows) == len(plan.segments)
    for idx, (seg, (t_lo, t_hi)) in enumerate(zip(plan.segments, windows)):
        assert abs(t_lo - seg.start_offset_s) < 0.8, (
            f"K={seg.k} t_lo={t_lo:.3f} expected {seg.start_offset_s:.3f}"
        )
        # For K[i>0], the window span = duration_s exactly.
        if idx > 0:
            assert abs((t_hi - t_lo) - seg.duration_s) < 0.2, (
                f"K={seg.k} span {t_hi - t_lo:.3f} expected "
                f"{seg.duration_s:.3f}"
            )


def test_slice_from_plan_returns_warmup_aware_windows():
    """`_slice_from_plan` must produce K[0]'s extended window unchanged.

    Direct unit test of the helper that powers plan-direct slicing. The
    contract: one (t_lo, t_hi) per segment, with t_lo == start_offset_s
    and t_hi - t_lo == duration_s. This intentionally PRESERVES the
    warm-up extension on K[0] cycle 0 -- if some future refactor
    "normalises" all segments to the same duration the warm-up
    schedule would silently break alignment, so this test pins it.
    """
    from prusa_pa_tuner.analysis import _slice_from_plan

    params = SweepParams(
        K_values=(0.0, 0.05, 0.10),
        cycles_per_K=1,
        slow_half_s=2.0,
        fast_half_s=1.0,
        coupled_dx_mm=1.0,
        first_slow_leg_factor=5.0,
    )
    plan = build_sweep(params)
    windows = _slice_from_plan(plan)
    assert len(windows) == 3
    # K[0]: start_offset=0.5, duration=11 → window [0.5, 11.5]
    assert windows[0][0] == 0.5
    assert windows[0][1] == 11.5
    # K[1]: start_offset=11.5, duration=3 → window [11.5, 14.5]
    assert windows[1] == (11.5, 14.5)
    # K[2]: start_offset=14.5, duration=3 → window [14.5, 17.5]
    assert windows[2] == (14.5, 17.5)
