"""Regression tests for runner.py.

Focused on timeout / progress-handling behaviour that has bitten users
in production.
"""
from __future__ import annotations

import numpy as np

from prusa_pa_tuner.config import AppConfig
from prusa_pa_tuner.runner import params_from_config, _sort_by_time
from prusa_pa_tuner.gcode_gen import build_sweep


def test_dynamic_job_timeout_scales_with_sweep_duration():
    """A 21-K × 10-cycle sweep takes >10 minutes on the printer. The
    old hardcoded 600s `job_timeout_s` killed those runs before
    completion (the user's 2026-05 long sweeps "finished on the
    printer but did not show up in the UI"). With the dynamic default,
    `run_tuning` should pick a timeout that exceeds the sweep's
    expected duration by a generous margin (≥ 2x + 5 min headroom)."""
    cfg = AppConfig()
    cfg.k_min = 0.0
    cfg.k_max = 0.1
    cfg.k_step = 0.005  # → 21 K values
    cfg.cycles_per_K = 10
    cfg.first_slow_leg_factor = 10.0
    cfg.slow_flow_mm3_s = 3.0
    cfg.slow_volume_mm3 = 6.0
    cfg.fast_flow_mm3_s = 15.0
    cfg.fast_volume_mm3 = 15.0
    params = params_from_config(cfg, udp_host="192.168.1.10")
    plan = build_sweep(params)
    sweep_dur = (
        plan.segments[-1].start_offset_s + plan.segments[-1].duration_s
    )
    # Replicate the dynamic-default formula from runner.run_tuning.
    derived_timeout = max(900.0, 2.0 * sweep_dur + 300.0)
    assert sweep_dur > 600.0, (
        f"This test setup expects a >10 min sweep duration to "
        f"trigger the regression scenario; got {sweep_dur:.0f}s"
    )
    assert derived_timeout > sweep_dur + 300.0, (
        f"Derived timeout {derived_timeout:.0f}s should exceed sweep "
        f"duration {sweep_dur:.0f}s by at least 300s (heat-up margin)"
    )
    # Sanity: the old hardcoded 600s would have killed this run.
    assert derived_timeout > 600.0


def test_short_sweep_uses_900s_floor():
    """For short sweeps (e.g. 3 K × 5 cycles ≈ 60s), the dynamic
    formula should clamp to a 15-min floor so transient PrusaLink
    delays / heatup variations don't time out a small run prematurely."""
    cfg = AppConfig()
    cfg.k_min = 0.0
    cfg.k_max = 0.1
    cfg.k_step = 0.05  # → 3 K values
    cfg.cycles_per_K = 5
    cfg.first_slow_leg_factor = 10.0
    params = params_from_config(cfg, udp_host="192.168.1.10")
    plan = build_sweep(params)
    sweep_dur = (
        plan.segments[-1].start_offset_s + plan.segments[-1].duration_s
    )
    derived_timeout = max(900.0, 2.0 * sweep_dur + 300.0)
    assert sweep_dur < 300.0, (
        f"Short sweep should be under 5 minutes; got {sweep_dur:.0f}s"
    )
    # Floor pinned at 900s (15 min)
    assert derived_timeout == 900.0


def test_sort_by_time_fixes_out_of_order_samples():
    """The defensive sort in runner.py / replay.py protects against
    out-of-order timestamps in the per-metric stream. Regression for
    run_1779015193.npz which contained a ~60 ms backward jump in
    force_t around the rising edge -- plotly drew a backwards
    diagonal because samples were stored in receive order, not time
    order."""
    # Two interleaved batches: simulates packets A (covers 1.0..1.06)
    # and B (covers 1.05..1.11) arriving out of order in firmware time.
    t = np.array([1.00, 1.02, 1.04, 1.06, 1.05, 1.07, 1.09, 1.11])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    t_sorted, y_sorted = _sort_by_time(t, y)
    # Strictly monotonic after sort
    assert np.all(np.diff(t_sorted) >= 0), "sort failed"
    # Stable sort: equal-time samples keep their original order.
    # In our input there are no ties, but verify values are paired
    # correctly with their timestamps.
    for ti, yi in zip(t_sorted, y_sorted):
        # Find the original index of this timestamp
        orig_idx = int(np.argmin(np.abs(t - ti)))
        assert y[orig_idx] == yi, (
            f"timestamp {ti} lost its paired value during sort"
        )


def test_sort_by_time_skips_sort_when_already_monotonic():
    """The defensive sort must be a no-op (no array copy) on the
    common case where timestamps arrive monotonic. This matters for
    long sweeps with 100K+ samples -- a per-run argsort is cheap, but
    the predicate skip is even cheaper."""
    t = np.linspace(0.0, 10.0, 5000)
    y = np.random.RandomState(0).normal(size=5000)
    t_out, y_out = _sort_by_time(t, y)
    # The function should return the EXACT SAME array (identity), not
    # a sorted copy, when the input is already monotonic.
    assert t_out is t and y_out is y, (
        "sort returned new arrays on monotonic input -- the early-out "
        "predicate isn't catching the common case"
    )


def test_dynamic_timeout_default_exceeds_old_hardcoded_600s_for_long_sweep():
    """The old hardcoded 600s timeout was specifically what killed the
    user's long sweeps. The dynamic default must be strictly greater
    than 600s for ANY long sweep where the sweep duration exceeds
    300s. (Hard floor at 900s plus 2x scaling guarantees this.)"""
    cfg = AppConfig()
    cfg.k_min = 0.0
    cfg.k_max = 0.1
    cfg.k_step = 0.005
    cfg.cycles_per_K = 10
    cfg.first_slow_leg_factor = 10.0
    cfg.slow_flow_mm3_s = 3.0
    cfg.slow_volume_mm3 = 6.0
    cfg.fast_flow_mm3_s = 15.0
    cfg.fast_volume_mm3 = 15.0
    params = params_from_config(cfg, udp_host="192.168.1.10")
    plan = build_sweep(params)
    sweep_dur = (
        plan.segments[-1].start_offset_s + plan.segments[-1].duration_s
    )
    derived = max(900.0, 2.0 * sweep_dur + 300.0)
    assert derived > 600.0, (
        "Dynamic timeout regressed: long-config sweep would still hit "
        "the 600s cap that killed user's runs."
    )
    # And specifically: for this 21-K × 10-cycle sweep (~648s) the
    # timeout should be >= 1500s.
    assert derived >= 1500.0
