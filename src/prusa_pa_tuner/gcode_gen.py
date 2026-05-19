"""Generate the K-sweep test G-code for the PA tuning run.

The job:
  1. Configure UDP target (M334) and enable loadcell metric (M331 probe_load_line).
  2. Home, heat, park at a user-defined purge position above the bed.
  3. For each K in the sweep:
        - M572 S<value>             (set Pressure Advance -- Prusa-specific;
                                     Buddy/Core One does NOT honour Marlin's
                                     M900 K, it silently no-ops)
        - M117 PA_K=<value>          (visible marker, also useful to spot in serial log)
        - emit a uniquely-tagged comment that the runner uses to slice the timeseries
        - extrude a square wave of cycles_per_K cycles, each cycle = slow burst + fast burst
        - small retract + dwell to settle
  4. Stop streaming and cool down.

Notes:
  - Prusa Buddy / Core One uses M572 S<value> for Pressure Advance, where
    the value is in seconds (advance time, equivalent to Klipper's PA).
    Typical PLA values fall in 0.00..0.10. We sweep this range by default.
    Note: Marlin's M900 K is NOT supported on Prusa Buddy -- see
    https://help.prusa3d.com/article/prusa-firmware-specific-g-code-commands_112173
  - Extrusion velocities are in mm/s of *filament feed* (not toolhead motion).
    Snapmaker U1 defaults: slow=0.8 mm/s, fast=8 mm/s. We mirror those.
  - The runner relies on the comment markers `;PA_TUNER K_START=<k>` and
    `;PA_TUNER K_END=<k>` plus the matching wall-clock arrival of the next packets
    after the printer parses them. Those comments don't generate motion so they're
    effectively free, but Buddy doesn't echo comments to UDP — sync via M117 echo
    or the `sdpos` metric is more robust. The runner times segments using the
    plan's `K_segments` schedule (start_time, duration) returned alongside the gcode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class SweepParams:
    nozzle_temp: float = 215.0  # the test (sweep) temperature
    # Preheat target held during homing/park/baseline-dwell. Switches to
    # `nozzle_temp` at the start of the first purge. Defaults 10 °C above
    # `nozzle_temp`. Set equal to `nozzle_temp` to disable the two-stage
    # warmup.
    preheat_temp: float = 225.0
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    filament_label: str = "PLA"

    # extrusion velocities in mm/s of filament feed (Snapmaker U1 defaults)
    slow_feed_mm_s: float = 0.8
    fast_feed_mm_s: float = 8.0

    # duration of each half of the square wave (seconds).
    # Asymmetric, matching U1: slow leg establishes steady-state pressure
    # (0.8 mm / 0.8 mm/s = 1.0 s); fast leg is the transient under test
    # (2.0 mm / 8.0 mm/s = 0.25 s).
    slow_half_s: float = 1.0
    fast_half_s: float = 0.25
    cycles_per_K: int = 14

    # Extrusion-axis acceleration emitted via M204. Snapmaker U1 pinned this
    # to 200 mm/s² on the assumption that low accel keeps the velocity
    # transient identical across K and isolates PA. In practice that left
    # PA's effect on the loadcell below the noise floor (R²≈0 across K).
    # bd_pressure runs in a similar effective-E-accel regime but pulls
    # signal from XY-coupled print-line motion that we can't replicate in
    # a pure-E sweep. The lever we *do* have is raising E accel directly:
    # 5000 mm/s² shrinks the velocity transition from ~36 ms to ~1.4 ms,
    # boosting dp/dt by ~25× and giving the bd_pressure-style amplitude
    # metric enough signal to localize K_opt.
    accel_mm_s2: float = 5000.0

    # bd_pressure-style fine sweep: 51 values from 0.000 to 0.100 in 0.002
    # steps. Replaces the previous coarse 9-step 0..0.40 grid that fed the
    # linear-regression zero-crossing extractor. The new analyser uses
    # argmin over `amplitude + |asymmetry|`, which only localizes well on a
    # fine grid through the true K_opt (typical PLA ~0.02..0.06).
    K_values: tuple[float, ...] = tuple(round(i * 0.002, 4) for i in range(51))

    # safe purge position — well above the bed so dripping filament doesn't stick
    purge_x: float = 30.0
    purge_y: float = 30.0
    purge_z: float = 50.0

    # housekeeping
    udp_host: str = "192.168.1.10"
    udp_port: int = 8514
    # Single loadcell metric the analyser consumes. `loadcell_value` is the
    # raw tared force stream (~180 Hz on Core One). On this firmware build
    # loadcell_hp / gcode / tmc_sg_e are silent so we no longer subscribe
    # to them; see project_buddy_metric_mechanics for the audit.
    loadcell_metric: str = "loadcell_value"

    # Rest between K segments. Default zero so K segments are back-to-back
    # -- the bd_pressure cost depends on cycle-to-cycle pressure continuity;
    # an idle dwell would let melt pressure relax and bias the first cycle
    # of each new K. If the user opts into a non-zero retract via
    # `inter_k_retract_mm`, the dwell is the post-retract settle and is
    # emitted only in that branch.
    inter_k_dwell_s: float = 0.0
    inter_k_retract_mm: float = 0.0

    # Pre-sweep baseline dwell. After heat-up + park, BEFORE priming, the
    # nozzle is hot but no filament has been pushed -- so the loadcell is
    # reading the static head load only. Recording this window gives us a
    # clean "zero" reference for diagnostics (drift, noise floor, whether
    # taring even happened) and an absolute reference line in the per-K
    # plots. Set to 0 to disable.
    baseline_dwell_s: float = 2.0

    # Alternating axis motion coupled into every burst so the planner
    # classifies each burst as a "print move" and actually applies M572
    # Pressure Advance. Marlin/Buddy skip PA on pure-E moves (planner.cpp
    # gates `block->use_advance_lead` on `block->steps.a || block->steps.b`,
    # i.e. an X or Y stepper event). Pure Z+E does NOT trigger PA -- Z is
    # on its own stepper, decoupled from A/B -- so at least one of dx/dy
    # must be non-zero or PA stays inactive. dz is exposed anyway so the
    # user can experiment with what couples least into the loadcell trace.
    #
    # The slow leg moves the toolhead from (x_base, y_base, z_base) by
    # +(dx, dy, dz); the fast leg returns it. Zero net drift per cycle,
    # zero net drift across the whole sweep.
    coupled_dx_mm: float = 0.05
    coupled_dy_mm: float = 0.0
    coupled_dz_mm: float = 0.0

    # Warm-up factor for K[0]'s VERY FIRST slow leg. The first slow
    # extrusion of the whole sweep is multiplied by this factor so the
    # nozzle reaches its steady-state melt pressure and any old
    # filament gets fully purged before the first measurable cycle.
    # Replaces the legacy fixed-2mm prime + 500 ms dwell -- with
    # factor=10 and slow_half=2 s this gives 20 s of slow flow
    # (~25 mm at 1.25 mm/s) which fully establishes pressure before
    # the first slow→fast transition. Set to 1 to disable extension
    # (back-compat). All subsequent slow legs (including K[0]'s
    # cycles 1..N) are unaffected.
    first_slow_leg_factor: float = 1.0

    # Pre-burst Z-marker pulse magnitude. The gcode generator lifts the
    # toolhead by this amount, briefly holds, then drops it back to
    # `purge_z`. The analyser detects the unique pos_z signature and
    # uses the return-to-baseline timestamp as sweep_t0 -- the most
    # robust anchor we have, since no other motion in the run looks
    # like a single ~2 mm Z excursion. 2 mm is well above pos_z noise
    # / planner jitter and small enough to stay within machine limits
    # at any reasonable purge_z. Set to 0 to disable the marker (the
    # analyser will fall back to pos_x periodicity detection).
    z_marker_lift_mm: float = 2.0

    label: str = "PA Tuner sweep"


@dataclass(slots=True)
class KSegment:
    """Planned timing for one K value, used by the analyzer to slice the timeseries."""

    k: float
    start_offset_s: float  # seconds from "sweep start" marker
    duration_s: float
    cycle_period_s: float  # slow_half_s + fast_half_s
    cycles: int
    # Extra slow-leg time prepended to cycle 0 (= warm-up extension).
    # Only K[0] uses a non-zero value (= `slow_half_s · (factor − 1)`)
    # so the first slow extrusion is long enough to fully establish
    # nozzle pressure / purge old filament. Subsequent K's keep this
    # at 0. The analyser shifts K[0]'s cycle-0 slow-leg-start
    # detection back by `slow_half_s + first_cycle_slow_extension_s`
    # instead of just `slow_half_s`.
    first_cycle_slow_extension_s: float = 0.0


@dataclass(slots=True)
class SweepPlan:
    gcode: str
    segments: list[KSegment]
    params: SweepParams


def _feed_to_mm_min(feed_mm_s: float) -> float:
    return feed_mm_s * 60.0


def _e_amount(feed_mm_s: float, duration_s: float) -> float:
    return feed_mm_s * duration_s


def _g1_xyze(
    x: float, y: float, z: float, e: float, f_mm_min: float,
    dx_active: float, dy_active: float, dz_active: float,
) -> str:
    """Emit a `G1` line that only includes the axes the sweep is actually
    moving. Axes with zero amplitude are omitted so the gcode is concise
    and readable, but at least one of X/Y/Z must be present (the caller
    guarantees has_xyz=True before calling this).
    """
    parts = ["G1"]
    if abs(dx_active) > 1e-9:
        parts.append(f"X{x:.4f}")
    if abs(dy_active) > 1e-9:
        parts.append(f"Y{y:.4f}")
    if abs(dz_active) > 1e-9:
        parts.append(f"Z{z:.4f}")
    parts.append(f"E{e:.4f}")
    parts.append(f"F{f_mm_min:.2f}")
    return " ".join(parts)


def build_sweep(params: SweepParams) -> SweepPlan:
    """Build the sweep gcode and the timing plan for analysis."""
    p = params
    lines: list[str] = []
    segments: list[KSegment] = []

    cycle_period = p.slow_half_s + p.fast_half_s
    burst_s = cycle_period * p.cycles_per_K

    # Slicer-compatibility header. Buddy/Core One firmware shows a "gcode is
    # not fully compatible" warning unless it recognises a slicer signature
    # plus a minimal config block at the top. We forge a PrusaSlicer-style
    # block — the values are advisory; what matters is presence of the keys.
    lines.append("; generated by PrusaSlicer 2.8.1+win64")
    lines.append(f"; {p.label}")
    lines.append(f"; nozzle_diameter = {p.nozzle_diameter}")
    lines.append(f"; filament_diameter = {p.filament_diameter}")
    lines.append(f"; filament_type = {p.filament_label}")
    lines.append(f"; first_layer_temperature = {p.nozzle_temp:.0f}")
    lines.append(f"; temperature = {p.nozzle_temp:.0f}")
    lines.append(f"; first_layer_bed_temperature = 0")
    lines.append(f"; bed_temperature = 0")
    lines.append("; layer_height = 0.2")
    lines.append("; max_print_height = 280")
    lines.append("; printer_model = COREONE")
    lines.append("; printer_notes = PrusaPATuner -- free-air PA calibration")
    lines.append(f"; K_values = {list(p.K_values)}")
    lines.append("")

    # Prusa firmware feature assertions. M862.6 P"Input shaper" is the
    # specific check that turns OFF the "not sliced for input shaping"
    # warning on Buddy 6.5.3+. The others match what PrusaSlicer emits so
    # the firmware accepts the gcode without prompting.
    lines.append("M17 ; enable steppers")
    lines.append(f"M862.1 P{p.nozzle_diameter} A0 F1 ; nozzle check (HF)")
    lines.append('M862.3 P "COREONE" ; printer model check')
    lines.append("M862.5 P2 ; g-code level check")
    lines.append('M862.6 P"Input shaper" ; FW feature check (kills "not sliced for input shaping" prompt)')
    lines.append("M115 U6.5.3+12780 ; require Buddy firmware >= 6.5.3")
    lines.append("")

    # configure UDP + enable streaming
    lines.append(f"M334 {p.udp_host} {p.udp_port} ; stream metrics to host")

    # Silence everything we don't consume before turning on what we need.
    # IMPORTANT: these names match what THIS firmware build emits (verified
    # via /api/diagnostics on the user's printer). Buddy's `M331 <name>` is
    # a strict case-sensitive strcmp -- it silently no-ops on miss and the
    # "Metric not found" reply goes only to serial, which PrusaLink throws
    # away. So this list is curated against observed live names; do NOT
    # add speculative names (they'd just be filler).
    metrics_to_silence = (
        # Default-on noisy stuff we observed at runtime.
        "cmdcnt", "stp_stall",
        "runtime", "stack",
        "fan", "input_current", "heater_current",
        "oc_inp", "oc_nozz",
        "bed_voltage", "heater_voltage",
        "door_sensor", "chamber_temp",
        "points_dropped", "cur_mmu_imp",
        "esp_in", "esp_out", "eth_out",
        "heater_enabled",
        # Loadcell variants we don't use (loadcell_value is the primary).
        "loadcell_hysteresis",
        "loadcell_threshold", "loadcell_threshold_cont",
        # Gcode-queue telemetry -- not text echoes, just queue state.
        "gcd_que_sz", "ftch_cmds", "ftch_dur", "ftch_occ",
        "ftch_status", "ftch_tstatus",
        # Printer state we don't need.
        "is_printing", "home", "home_diff",
        "phxy_home", "phxy_meas", "phxy_probe",
        "fw_version", "buddy_bom", "filament",
        "classified", "detection",
        "metrics", "hit:", "opened:",
    )
    lines.append("; --- silence non-essential metrics to lower UDP load ---")
    for m in metrics_to_silence:
        lines.append(f"M332 {m}")
    lines.append("")
    # Enable the streams we consume:
    #   * loadcell_value -- raw ~180 Hz force, the primary signal.
    #   * pos_x / pos_y / pos_z -- toolhead position. We use pos_x as the
    #     sweep_t0 anchor: the first time X moves off purge_x is the start
    #     of the first burst, accurate to one position-sample period. This
    #     beats the loadcell auto-detect, which can mis-anchor by ±1 s
    #     when K is low (no PA = small loadcell response = ambiguous start).
    #     pos_y and pos_z are enabled too so the user can experiment with
    #     coupled dy/dz amplitudes.
    # gcode / loadcell_hp / tmc_sg_e are not enabled: verified silent on
    # this firmware via /api/diagnostics. M331 on a non-existent metric
    # silently no-ops; if pos_x doesn't exist on a future firmware build,
    # the analyser falls back to the loadcell auto-detect.
    lines.append(f"M331 {p.loadcell_metric} ; primary loadcell stream")
    lines.append("M331 pos_x ; toolhead X -- sweep_t0 anchor")
    lines.append("M331 pos_y ; toolhead Y -- optional anchor")
    lines.append("M331 pos_z ; toolhead Z -- optional anchor")
    lines.append("")

    # Disable filament-stuck detection. Our slow leg is 0.8 mm/s for 1 s --
    # that's the exact "looks like the filament has stopped" pattern that
    # trips Buddy's stuck-filament sensor, sending the printer to ATTENTION
    # mid-test. PrusaSlicer's gcode also disables this for the same reason
    # during the slow purge moves.
    lines.append("M591 S0 ; disable filament stuck detection")

    lines.append("G90 ; absolute XYZ")

    # Heat the nozzle only -- we extrude in free air at purge_z, so the bed
    # never needs to heat. Saves ~3 minutes per run on a cold start.
    #
    # Two-stage warmup: heat to `preheat_temp` (typically nozzle_temp + 10)
    # during homing + park, then drop the setpoint to `nozzle_temp` right
    # before the first purge. The preheat overshoot guarantees any residual
    # filament from the previous run is fully molten before we measure,
    # and the small temperature drop happens during the prime / pre-roll
    # so the nozzle is at or very near `nozzle_temp` when the first burst
    # fires.
    preheat = max(p.preheat_temp, p.nozzle_temp)
    lines.append(f"M104 S{preheat:.0f} ; preheat (test temp + headroom)")
    lines.append("G28 ; home")
    lines.append(f"M109 S{preheat:.0f} ; wait for preheat")

    # CRITICAL: use ABSOLUTE extrusion mode (M82) with a running cumulative E
    # counter, NOT M83 relative. Buddy/Core One's G28 + heat sequence can
    # silently flip the extruder back to absolute even after we asked for
    # relative, and if we emit relative-style numbers like `G1 E0.8` then
    # `G1 E2.0`, the firmware reads them as absolute targets that cycle the
    # extruder between positions 0.8 and 2.0 -- net zero feed, looking
    # exactly like the push/pull with no PA oscillation that we observed.
    # Cumulative-absolute is foolproof: every G1 E uses a monotonic running
    # total, so it works regardless of whether M82 or M83 is active.
    lines.append("G90 ; absolute XYZ (reassert after home)")
    lines.append("M82 ; ABSOLUTE E -- cumulative E targets follow")
    lines.append("G92 E0 ; reset E counter to zero")
    # Set acceleration for the pure-E moves the sweep uses.
    #
    # Marlin/Buddy classify moves by which axes change:
    #   - G1 with X/Y/Z + E   → "print" move    (M204 P, capped by M201 X/Y/Z/E)
    #   - G1 with E only      → "retract" move  (M204 R, capped by M201 E)
    #   - G1 with X/Y/Z only  → "travel" move   (M204 T, capped by M201 X/Y/Z)
    #
    # Our sweep emits pure-E moves (`G1 E{target} F{rate}` with no XYZ
    # change), so the active accel is **M204 R**, capped by **M201 E**.
    # The earlier `M204 P/T` emission did nothing for our motion -- it was
    # setting print/travel modes we don't use, and the actual E moves ran
    # at whatever the stock M204 R + M201 E happened to be (typically
    # ~1250-1500 mm/s² on Buddy defaults).
    #
    # We set both M201 E (the per-axis cap) and M204 R (the active value):
    # without M201 E the planner clamps to the stock cap regardless of
    # M204; without M204 R the planner uses the previous R value even
    # though the cap is raised.
    #
    # P and T are still emitted at the same value as belt-and-suspenders
    # for any move class we don't expect (e.g. the prime / park travel).
    lines.append(
        f"M201 E{p.accel_mm_s2:.0f} ; raise E-axis max accel cap"
    )
    lines.append(
        f"M204 P{p.accel_mm_s2:.0f} R{p.accel_mm_s2:.0f} T{p.accel_mm_s2:.0f} "
        f"; pin accel (R applies to our pure-E sweep)"
    )

    # Running E target. Starts at 0 (just G92'd). Every G1 E below uses an
    # absolute target derived from this counter, then increments it.
    cum_e = 0.0

    # park at purge position
    lines.append(f"G1 X{p.purge_x:.2f} Y{p.purge_y:.2f} Z{p.purge_z:.2f} F6000")
    lines.append("G4 P500 ; settle")

    # Pre-prime loadcell baseline: hot nozzle, no extrusion yet -- this is
    # the static head load only. The runner watches for the M117 markers
    # below in the gcode-event stream and slices `baseline_t_start`/`_end`
    # to pass to the analyser as a tare reference / drift diagnostic.
    # Anything that's not a G4 here would corrupt the zero (M-codes that
    # provoke motion, fan toggles, etc.) -- keep it strictly static.
    if p.baseline_dwell_s > 0:
        lines.append("M117 PA_BASELINE_START")
        lines.append(";PA_TUNER BASELINE_START")
        lines.append(f"G4 P{int(p.baseline_dwell_s * 1000)} ; hold for tare/baseline")
        lines.append("M117 PA_BASELINE_END")
        lines.append(";PA_TUNER BASELINE_END")

    # Switch the nozzle setpoint to the test temperature at the start of
    # the first purge. M104 (no wait) lets the prime move overlap the
    # ~10 °C drop -- by the time the first burst fires the nozzle has
    # settled at `nozzle_temp`. We deliberately do NOT M109 here, so a
    # well-tuned preheat introduces no extra delay.
    if preheat > p.nozzle_temp:
        lines.append(
            f"M104 S{p.nozzle_temp:.0f} ; drop to test temp at start of purge"
        )

    # NOTE: the legacy "G1 E2 ; prime + G4 P500" is intentionally
    # absent. The first slow leg of K[0]'s cycle 0 is extended by
    # `first_slow_leg_factor` (default 10×) so the warm-up slow flow
    # IS the prime -- no separate static prime is needed. A short
    # explicit prime followed by a dwell let melt pressure relax
    # between the prime and the first measurable cycle, which was
    # exactly the "first samples end up a bit low" symptom the user
    # reported. The extended slow leg keeps flow continuous from
    # purge through the first transition.

    # Sweep-start Z marker: lift the toolhead by `z_marker_lift_mm`,
    # brief hold, drop back to z_base. This produces a one-shot, unique
    # signature in the pos_z stream that the analyser can lock onto as
    # `sweep_t0` -- far more robust than pos_x periodicity detection
    # (which gets confused by park motion, planner-lookahead jitter, and
    # the bursts themselves). The Z motion happens AFTER the priming
    # extrude (so it doesn't affect the prime flow) and BEFORE the first
    # burst. By the gcode-side contract, the analyser treats the
    # pos_z return-to-baseline as exact sweep_t0, so the pre-roll dwell
    # below IS the post-Z settle window (`start_offset_s` seconds long).
    # When `z_marker_lift_mm <= 0` the marker is disabled and the
    # analyser falls back to pos_x periodicity detection.
    if p.z_marker_lift_mm > 0:
        lines.append("; --- sweep_t0 marker: Z lift then drop ---")
        lines.append(
            f"G1 Z{p.purge_z + p.z_marker_lift_mm:.3f} F1800 ; sweep marker UP"
        )
        lines.append("G4 P100 ; brief hold at top")
        lines.append(
            f"G1 Z{p.purge_z:.3f} F1800 ; sweep marker DOWN -- pos_z return = sweep_t0"
        )

    lines.append("M117 PA_SWEEP_START")
    lines.append(";PA_TUNER SWEEP_START")
    lines.append("")

    elapsed_s = 0.0
    # tiny pre-roll dwell to make sure SWEEP_START marker arrives before first K
    pre_roll = 0.5
    lines.append(f"G4 P{int(pre_roll * 1000)}")
    elapsed_s += pre_roll

    # Precompute per-cycle E increments and pure-E feedrates (used as a
    # fallback when no axis is coupled in).
    slow_e_inc = _e_amount(p.slow_feed_mm_s, p.slow_half_s)
    fast_e_inc = _e_amount(p.fast_feed_mm_s, p.fast_half_s)
    slow_f_mm_min = _feed_to_mm_min(p.slow_feed_mm_s)
    fast_f_mm_min = _feed_to_mm_min(p.fast_feed_mm_s)

    # Coupled XYZ oscillation. See SweepParams.coupled_d{x,y,z}_mm for the
    # reasoning -- Buddy/Marlin only applies M572 PA to moves with X or Y
    # stepper motion. Each slow leg drives the toolhead from base to
    # (base + dx, base + dy, base + dz); each fast leg returns it.
    #
    # F (mm/min) is what the firmware interprets:
    #   * pure-E move (axis_len == 0): F is the E feedrate directly
    #   * composite move (any XYZ delta): F is the XYZ trajectory velocity;
    #     E is slaved to complete during the same time. So to make the leg
    #     last exactly `slow_half_s` (or `fast_half_s`), we set
    #         F = 60 * axis_len / leg_duration_s
    #     and E completes its `*_e_inc` mm in that same duration. This
    #     keeps the configured volumetric flow rate physically correct
    #     regardless of how large the XYZ delta is.
    dx = float(p.coupled_dx_mm)
    dy = float(p.coupled_dy_mm)
    dz = float(p.coupled_dz_mm)
    axis_len = math.sqrt(dx * dx + dy * dy + dz * dz)
    has_xyz = axis_len > 1e-9
    if has_xyz:
        slow_f_xyz = 60.0 * axis_len / max(p.slow_half_s, 1e-6)  # mm/min
        fast_f_xyz = 60.0 * axis_len / max(p.fast_half_s, 1e-6)
    x_base = p.purge_x
    y_base = p.purge_y
    z_base = p.purge_z

    # K[0] cycle 0 warm-up extension. Only the first slow leg of the
    # whole sweep is extended; all other slow legs use the normal
    # slow_half_s duration.
    warmup_factor = max(1.0, float(p.first_slow_leg_factor))
    warmup_extension_s = p.slow_half_s * (warmup_factor - 1.0)
    warmup_extra_e = _e_amount(p.slow_feed_mm_s, warmup_extension_s)
    # Composite-move feedrate when the slow leg is extended: same
    # geometric XY/Z distance covered, just over a longer time -- so F
    # gets divided by `warmup_factor` (since F is XYZ trajectory mm/min
    # and the trajectory length is unchanged).
    if has_xyz and warmup_factor > 1.0:
        warmup_slow_f_xyz = 60.0 * axis_len / max(
            p.slow_half_s * warmup_factor, 1e-6
        )

    for k_idx, k in enumerate(p.K_values):
        # K marker + advance setting
        lines.append("")
        lines.append(f"; --- K={k:.4f} ---")
        # Reassert absolute-E mode every K segment for safety. Cheap insurance
        # against firmware silently flipping E mode mid-print. Doesn't touch
        # cum_e -- we keep accumulating the running E target.
        lines.append("M82 ; ABSOLUTE E (defensive per-K reassert)")
        lines.append(f"M572 S{k:.4f} ; pressure advance (Prusa-specific)")
        lines.append(f"M117 PA_K={k:.4f}")
        lines.append(f";PA_TUNER K_START k={k:.4f}")

        # Only K[0] gets the warm-up extension (= the sweep's first
        # slow leg). All other K[i] have first_cycle_slow_extension_s = 0.
        seg_extension_s = warmup_extension_s if k_idx == 0 else 0.0
        seg_duration = burst_s + seg_extension_s

        segments.append(
            KSegment(
                k=k,
                start_offset_s=elapsed_s,
                duration_s=seg_duration,
                cycle_period_s=cycle_period,
                cycles=p.cycles_per_K,
                first_cycle_slow_extension_s=seg_extension_s,
            )
        )

        # Square-wave bursts as cumulative-absolute E targets. Each line
        # advances the running counter -- the firmware sees a strictly
        # monotonic E coordinate, which extrudes the correct distance whether
        # M82 or M83 happens to be the active mode at the moment.
        #
        # When axis coupling is active, the slow leg drives the toolhead to
        # the offset corner and the fast leg returns it -- zero net drift
        # per cycle. The composite XYZ+E move is what makes Buddy actually
        # apply M572.
        for c_idx in range(p.cycles_per_K):
            # K[0] cycle 0 only: extend the slow leg by warmup_factor.
            is_warmup_cycle = (k_idx == 0 and c_idx == 0 and warmup_factor > 1.0)
            if is_warmup_cycle:
                cum_e += slow_e_inc + warmup_extra_e
                slow_f_this_cycle = (
                    warmup_slow_f_xyz if has_xyz else _feed_to_mm_min(p.slow_feed_mm_s)
                )
                # F for pure-E is just slow_feed_mm_s (since the EXTRA
                # E is also at slow_feed); for composite XYZ+E with
                # extended duration, F is the divided XY trajectory
                # speed so the leg lasts slow_half * factor.
            else:
                cum_e += slow_e_inc
                slow_f_this_cycle = (
                    slow_f_xyz if has_xyz else slow_f_mm_min
                )
            if has_xyz:
                lines.append(
                    _g1_xyze(
                        x_base + dx, y_base + dy, z_base + dz,
                        cum_e, slow_f_this_cycle, dx, dy, dz,
                    )
                )
            else:
                lines.append(f"G1 E{cum_e:.4f} F{slow_f_this_cycle:.1f}")
            # fast leg: extrude fast_e_inc + (optionally) return XYZ
            cum_e += fast_e_inc
            if has_xyz:
                lines.append(
                    _g1_xyze(
                        x_base, y_base, z_base,
                        cum_e, fast_f_xyz, dx, dy, dz,
                    )
                )
            else:
                lines.append(f"G1 E{cum_e:.4f} F{fast_f_mm_min:.1f}")
        elapsed_s += seg_duration

        lines.append(f";PA_TUNER K_END k={k:.4f}")

        # Inter-K transition. No retract by default (it causes a force
        # transient that contaminates the next K window) AND no dwell by
        # default (a pause would let melt pressure relax and bias the
        # first cycle of the next K, which the bd_pressure cost
        # penalises). Skip the G4 entirely when the dwell is zero.
        if p.inter_k_retract_mm > 0:
            cum_e -= p.inter_k_retract_mm
            lines.append(f"G1 E{cum_e:.4f} F1800 ; retract")
            if p.inter_k_dwell_s > 0:
                lines.append(f"G4 P{int(p.inter_k_dwell_s * 1000)}")
            cum_e += p.inter_k_retract_mm
            lines.append(f"G1 E{cum_e:.4f} F1800 ; un-retract")
        elif p.inter_k_dwell_s > 0:
            lines.append(f"G4 P{int(p.inter_k_dwell_s * 1000)}")
        elapsed_s += p.inter_k_dwell_s

    # Trailing slow leg: the last K's display window extends
    # `slow_half_s` past its last cycle for the "ends on low"
    # boundary visibility. Without an actual slow extrusion to fill
    # that extension, the loadcell drops to zero (no flow) and the
    # final low plateau isn't visible. We emit one more slow leg
    # matching the cycle's slow-leg geometry so the last K's plot
    # closes on a clean slow plateau, mirroring how every other K
    # closes via the shared boundary with K[i+1]'s warm-up start.
    lines.append("")
    lines.append("; --- trailing slow leg (display tail) ---")
    cum_e += slow_e_inc
    if has_xyz:
        lines.append(
            _g1_xyze(
                x_base + dx, y_base + dy, z_base + dz,
                cum_e, slow_f_xyz, dx, dy, dz,
            )
        )
    else:
        lines.append(f"G1 E{cum_e:.4f} F{slow_f_mm_min:.1f}")

    lines.append("")
    lines.append("M117 PA_SWEEP_END")
    lines.append(";PA_TUNER SWEEP_END")

    # cleanup
    # The M332 disable list at the top of this sweep already brought the
    # printer to a known minimal-stream state, and our M331 enables here are
    # session-scoped (they fall away when the job ends). We don't need a
    # trailing M332 cleanup -- the streams we enabled stop on their own, and
    # M334 with no args is a no-op on current firmware per metrics.md.
    lines.append("M572 S0 ; reset pressure advance")
    lines.append("M591 R ; restore stuck detection (matches PrusaSlicer's pattern)")
    lines.append("M104 S0 ; nozzle off")
    lines.append("M84 ; disable motors")

    return SweepPlan(gcode="\n".join(lines) + "\n", segments=segments, params=p)
