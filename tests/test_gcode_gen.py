import re

from prusa_pa_tuner.gcode_gen import SweepParams, build_sweep


# Burst lines may be pure-E ("G1 E...") OR coupled XE ("G1 X... E...") --
# both forms count as one extrusion move. Use a single matcher.
_G1_E_LINE = re.compile(r"^G1\b.*\bE-?\d")


def _is_g1_e_line(line: str) -> bool:
    return bool(_G1_E_LINE.match(line))


def test_basic_sweep_contains_per_K_M572():
    params = SweepParams(
        K_values=(0.0, 0.04, 0.08),
        cycles_per_K=3,
        udp_host="10.0.0.5",
        udp_port=8500,
    )
    plan = build_sweep(params)
    assert len(plan.segments) == 3
    assert plan.segments[0].k == 0.0
    assert plan.segments[-1].k == 0.08

    g = plan.gcode
    assert "M334 10.0.0.5 8500" in g
    # We enable ONE stream -- loadcell_value, the raw force. loadcell_hp,
    # gcode, and tmc_sg_e are silent on this firmware build (verified via
    # /api/diagnostics) so we don't bother M331-ing them.
    assert "M331 loadcell_value" in g
    assert "M331 loadcell_hp" not in g
    assert "M331 gcode" not in g
    assert "M331 tmc_sg_e" not in g
    # Disable list uses the actually-observed noisy metric names.
    assert "M332 cmdcnt" in g
    assert "M332 stp_stall" in g
    assert "M332 chamber_temp" in g
    assert "M332 heater_enabled" in g
    # Never M332 loadcell_value -- it's the one stream we depend on.
    assert "M332 loadcell_value" not in g
    # Prusa Buddy/Core One uses M572 S<value> for pressure advance, NOT
    # Marlin's M900 K -- which Buddy silently ignores. See
    # https://help.prusa3d.com/article/prusa-firmware-specific-g-code-commands_112173
    assert "M572 S0.0000" in g
    assert "M572 S0.0400" in g
    assert "M572 S0.0800" in g
    assert "M900" not in g
    assert ";PA_TUNER SWEEP_START" in g
    assert ";PA_TUNER SWEEP_END" in g
    assert g.endswith("\n")


def test_segments_monotonic_offsets():
    plan = build_sweep(SweepParams(K_values=(0.0, 0.02, 0.04, 0.06)))
    offsets = [s.start_offset_s for s in plan.segments]
    assert offsets == sorted(offsets)
    # later segments don't overlap previous
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert b.start_offset_s >= a.start_offset_s + a.duration_s


def test_cycle_count_matches_param():
    params = SweepParams(K_values=(0.05,), cycles_per_K=5,
                         slow_half_s=0.4, fast_half_s=0.4)
    plan = build_sweep(params)
    # 5 cycles = 5 * (slow + fast) = 5 * 0.8 = 4.0 s
    assert plan.segments[0].duration_s == 4.0
    assert plan.segments[0].cycle_period_s == 0.8
    # gcode should contain 5*2 = 10 burst lines for this K (slow + fast
    # each), plus 1 trailing slow leg for the closing low plateau
    # display. Burst lines may be pure-E or coupled XE (see _G1_E_LINE).
    # No explicit prime any more -- the first slow leg's warm-up
    # extension (when factor > 1) serves as the prime; here factor=1
    # (the SweepParams back-compat default).
    g_lines = [l for l in plan.gcode.splitlines() if _is_g1_e_line(l)]
    assert len(g_lines) == 10 + 1


def test_trailing_slow_leg_emitted_at_end_of_sweep():
    """The slicer extends each K's display window by `slow_half_s` past
    its last cycle so the trailing low plateau is visible. For K[N-1]
    (the very last K), there's no K[N] to fill that extension, so
    without an explicit trailing slow leg the loadcell drops to zero
    in that window tail and the user sees the last K's plot end on
    nothing instead of the slow plateau.

    Verify the gcode emits exactly one extra slow leg AFTER the last
    K segment's last fast leg, with matching slow-leg geometry
    (E increment, X coupling, feedrate).
    """
    p = SweepParams(
        K_values=(0.0, 0.05), cycles_per_K=2,
        slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, coupled_dx_mm=0.05,
    )
    plan = build_sweep(p)
    lines = plan.gcode.splitlines()
    # Find the "; --- trailing slow leg" marker.
    tail_idx = next(
        (i for i, l in enumerate(lines) if "trailing slow leg" in l), None
    )
    assert tail_idx is not None, (
        f"gcode must contain '; --- trailing slow leg' marker\n"
        f"got:\n" + "\n".join(lines[-15:])
    )
    # Next non-empty line must be a G1 to (purge_x + dx, ...) at the
    # slow-leg feedrate (= slow trajectory speed mm/min).
    next_g = next(
        l for l in lines[tail_idx + 1:] if l.strip().startswith("G1 ")
    )
    assert "X30.0500" in next_g, (
        f"trailing slow leg should go to X={p.purge_x + p.coupled_dx_mm} "
        f"(matches slow-leg geometry); got {next_g}"
    )


def test_inter_k_retract_emitted_when_enabled():
    params = SweepParams(K_values=(0.05, 0.06), cycles_per_K=2,
                         slow_half_s=0.4, fast_half_s=0.4,
                         inter_k_retract_mm=1.5)
    plan = build_sweep(params)
    g_lines = [l for l in plan.gcode.splitlines() if _is_g1_e_line(l)]
    # No prime any more. Each K emits 4 burst lines (2 cycles × slow+fast)
    # + retract + unretract = 6 lines. 2 K values × 6 = 12 G1 E lines.
    # Plus 1 trailing slow leg for the closing display tail.
    assert len(g_lines) == 2 * (4 + 2) + 1


def test_coupled_x_motion_alternates_around_purge_x():
    """Bursts must include a tiny XY component so Buddy/Marlin classifies
    each move as a print move and actually applies M572 Pressure Advance.
    The X coord must alternate slow→(x+dx), fast→x within each cycle, so
    there is zero net X drift over the sweep.
    """
    params = SweepParams(
        K_values=(0.05,), cycles_per_K=3,
        slow_half_s=0.4, fast_half_s=0.4,
        purge_x=30.0, coupled_dx_mm=0.05,
    )
    plan = build_sweep(params)
    # Pull out every burst line with an X coordinate.
    burst_xs: list[float] = []
    for line in plan.gcode.splitlines():
        m = re.match(r"^G1 X(-?\d+\.\d+) E", line)
        if m:
            burst_xs.append(float(m.group(1)))
    # 3 cycles × 2 bursts = 6 X-bearing burst lines + 1 trailing slow
    # leg for the closing low-plateau display = 7.
    assert len(burst_xs) == 7
    # Pattern must be slow→30.05, fast→30.0, repeating. The trailing
    # slow leg is also slow→30.05.
    expected_pattern = [30.05, 30.0, 30.05, 30.0, 30.05, 30.0, 30.05]
    for i, x in enumerate(burst_xs):
        assert abs(x - expected_pattern[i]) < 1e-6, (
            f"burst {i}: X={x}, expected {expected_pattern[i]}"
        )


def test_all_axes_zero_emits_pure_e_bursts():
    """Setting all coupled axes to 0 disables the coupling for diagnostic
    A/B runs -- burst lines must then be pure G1 E... with no XYZ coord.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.05,), cycles_per_K=2,
        coupled_dx_mm=0.0, coupled_dy_mm=0.0, coupled_dz_mm=0.0,
    ))
    burst_lines = [
        l for l in plan.gcode.splitlines()
        if _is_g1_e_line(l) and "F1800" not in l  # skip retracts
    ]
    for line in burst_lines:
        assert line.startswith("G1 E"), (
            f"all axes zero must emit pure-E bursts, but found: {line!r}"
        )


def test_independent_dy_dz_amplitudes_emit_only_requested_axes():
    """When only dy is set, burst lines must include Y but not X or Z.
    When only dz is set, burst lines must include Z but not X or Y.
    """
    plan_y = build_sweep(SweepParams(
        K_values=(0.05,), cycles_per_K=1, purge_x=30, purge_y=30, purge_z=50,
        coupled_dx_mm=0.0, coupled_dy_mm=0.1, coupled_dz_mm=0.0,
    ))
    y_bursts = [
        l for l in plan_y.gcode.splitlines()
        if l.startswith("G1 Y") and "E" in l
    ]
    assert y_bursts, "dy>0 must emit G1 Y... bursts"
    for line in y_bursts:
        assert " X" not in line and " Z" not in line, (
            f"dy-only burst must omit X and Z, got: {line!r}"
        )

    plan_z = build_sweep(SweepParams(
        K_values=(0.05,), cycles_per_K=1, purge_x=30, purge_y=30, purge_z=50,
        coupled_dx_mm=0.0, coupled_dy_mm=0.0, coupled_dz_mm=0.1,
    ))
    z_bursts = [
        l for l in plan_z.gcode.splitlines()
        if l.startswith("G1 Z") and "E" in l
    ]
    assert z_bursts, "dz>0 must emit G1 Z... bursts"
    for line in z_bursts:
        assert " X" not in line and " Y" not in line, (
            f"dz-only burst must omit X and Y, got: {line!r}"
        )


def test_composite_f_matches_xyz_trajectory_velocity():
    """For composite XYZ+E moves Marlin interprets F as the XYZ trajectory
    velocity. The emitted F must be 60 * axis_len / leg_duration so the
    move actually lasts the configured `*_half_s` (rather than completing
    in a fraction of that and starving the loadcell of cycles).

    With dx=1.0, slow_half_s=0.5: axis_len=1.0, F = 60*1.0/0.5 = 120 mm/min.
    """
    params = SweepParams(
        K_values=(0.05,), cycles_per_K=1,
        slow_half_s=0.5, fast_half_s=0.1,
        slow_feed_mm_s=0.8, fast_feed_mm_s=8.0,
        purge_x=30.0,
        coupled_dx_mm=1.0, coupled_dy_mm=0.0, coupled_dz_mm=0.0,
    )
    plan = build_sweep(params)
    slow_line = next(
        l for l in plan.gcode.splitlines()
        if re.match(r"^G1 X31\.0000 E", l)
    )
    fast_line = next(
        l for l in plan.gcode.splitlines()
        if re.match(r"^G1 X30\.0000 E", l)
    )
    f_slow = float(re.search(r"F([\d.]+)", slow_line).group(1))
    f_fast = float(re.search(r"F([\d.]+)", fast_line).group(1))
    # slow: 60 * 1.0 / 0.5 = 120; fast: 60 * 1.0 / 0.1 = 600
    assert abs(f_slow - 120.0) < 0.01, f"expected 120, got {f_slow}"
    assert abs(f_fast - 600.0) < 0.01, f"expected 600, got {f_fast}"


def test_accel_pinned_via_M201_and_M204():
    """Our sweep emits pure-E moves (G1 E... with no XYZ change), which
    Buddy/Marlin classifies as retract-class. That means:
      - the per-axis cap M201 E must be raised, otherwise the planner
        clamps to the stock cap regardless of M204
      - the active accel M204 R must be set (P/T don't apply to pure-E)

    The earlier emission `M204 P{accel} T{accel}` was a no-op for our
    motion: it set print/travel modes we don't use, and the actual E
    moves ran at the stock R+E cap (~1250-1500 mm/s² on Buddy defaults).
    """
    plan = build_sweep(SweepParams(K_values=(0.0,), accel_mm_s2=250.0))
    assert "M201 E250" in plan.gcode, (
        "must raise the E-axis max accel cap; without it the planner clamps "
        "every E move to the firmware default regardless of M204"
    )
    assert "M204 P250 R250 T250" in plan.gcode, (
        "must set R (retract-class accel) -- that's what pure-E moves use; "
        "P and T are belt-and-suspenders for prime/travel moves"
    )


def test_baseline_dwell_brackets_are_emitted_before_prime():
    """The pre-prime baseline window: hot nozzle, no extrusion yet, brackets
    a static G4 dwell with M117 markers so the runner can slice the
    loadcell trace as a tare reference. It MUST precede any G1 E line
    (priming) -- otherwise the baseline reads pressure from the prime, not
    a clean zero.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.05,), cycles_per_K=2, baseline_dwell_s=1.5,
    ))
    g_lines = plan.gcode.splitlines()
    bs = next(i for i, l in enumerate(g_lines) if "PA_BASELINE_START" in l)
    be = next(i for i, l in enumerate(g_lines) if "PA_BASELINE_END" in l)
    first_extrude = next(
        i for i, l in enumerate(g_lines) if _is_g1_e_line(l)
    )
    assert bs < be < first_extrude, (
        f"baseline must bracket a static dwell before priming "
        f"(start={bs}, end={be}, first G1 E={first_extrude})"
    )
    # The dwell between the markers is the actual hold; nothing motion-
    # provoking in between.
    between = g_lines[bs + 1 : be]
    assert any(l.startswith("G4 P") for l in between)
    assert not any(l.startswith("G1 ") for l in between)


def test_baseline_dwell_skipped_when_zero():
    plan = build_sweep(SweepParams(K_values=(0.05,), baseline_dwell_s=0.0))
    assert "PA_BASELINE_START" not in plan.gcode
    assert "PA_BASELINE_END" not in plan.gcode


def test_cumulative_absolute_extrusion_is_monotonic():
    """G1 E values must be a strictly monotonically increasing absolute
    sequence. The bug we hit on real hardware was that the firmware
    interpreted relative-looking values (`E0.8` then `E2.0`) as absolute
    targets and cycled the extruder between two positions -- which is exactly
    what this test rules out by construction.
    """
    plan = build_sweep(SweepParams(
        K_values=(0.0, 0.04, 0.08),
        cycles_per_K=4,
        slow_half_s=1.0,
        fast_half_s=0.25,
    ))
    g = plan.gcode
    # Absolute mode must be set BEFORE any extrusion.
    first_g1e_idx = next(
        i for i, l in enumerate(g.splitlines()) if _is_g1_e_line(l)
    )
    m82_line_idx = next(
        i for i, l in enumerate(g.splitlines()) if "M82" in l
    )
    assert m82_line_idx < first_g1e_idx, (
        "M82 absolute-E must precede the first G1 E line"
    )
    # And M83 (relative) must NOT appear -- if anyone re-introduces it, the
    # cumulative-absolute contract breaks.
    assert "M83" not in g, "M83 (relative E) must not be emitted -- we use absolute mode"

    # Extract every G1 E value (whether the line is pure-E or coupled XE)
    # and check it's strictly increasing.
    e_re = re.compile(r"\bE(-?\d+\.\d+)")
    e_values: list[float] = []
    for line in g.splitlines():
        if not _is_g1_e_line(line):
            continue
        m = e_re.search(line)
        if m:
            e_values.append(float(m.group(1)))
    assert len(e_values) > 5
    for a, b in zip(e_values, e_values[1:]):
        assert b > a, (
            f"E sequence not monotonic: {a} followed by {b}. "
            f"This is exactly the bug that made the printer cycle between "
            f"two extruder positions instead of extruding continuously."
        )
