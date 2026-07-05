"""Generate the stepped free-air flow-ramp G-code for the Max Flow test.

Idea: extrude filament in air at a sequence of increasing volumetric flow
rates (mm³/s) and watch the nozzle loadcell back-pressure. In the
well-behaved regime the steady-state force follows a sub-linear power law
`F ≈ a·Qᵇ + c` (b < 1, shear-thinning melt). When the hotend can no
longer melt fast enough the force breaks *upward* off that curve, and the
per-level force variance rises sharply just before the extruder starts
skipping (force then collapses). flow_analysis.py extracts those points.

Each flow level is a single constant-feed pure-E move held for `dwell_s`;
the analyser discards the first `settle_frac` of each level (melt-pressure
+ acceleration transient) and measures the steady-state mean + variance on
the remainder.

The job mirrors the PA sweep's scaffolding so the same runner / metric
plumbing / Z-marker anchor work unchanged:
  1. Configure UDP (M334), silence non-essential metrics (M332), enable
     loadcell_value + pos_z (M331). pos_x/pos_y are NOT needed -- unlike
     the PA sweep there is no Pressure Advance to trigger, so no XY
     coupling is required; pure-E extrusion in free air still transmits
     back-pressure to the decoupled nozzle as a Z-force.
  2. Home, two-stage preheat, park above the bed, hold a baseline window.
  3. Warm-up extrude at min flow to purge old filament + reach steady
     melt pressure.
  4. Z-marker pulse (lift/hold/drop) -> unambiguous sweep_t0 anchor.
  5. For each flow level: one G1 E move at that level's feedrate for
     `dwell_s`, tagged with a comment + M117 for the gcode log.
  6. Reset + cool down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .gcode_preamble import (
    FLOW_MARKER_PREFIX,
    baseline_dwell,
    firmware_asserts,
    heat_home_setup,
    metric_setup,
    slicer_header,
    sweep_end_marker,
    sweep_start_marker,
    z_marker,
)


@dataclass(slots=True)
class FlowRampParams:
    nozzle_temp: float = 215.0
    preheat_temp: float = 225.0
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    filament_label: str = "PLA"

    # Sweep range (volumetric flow, mm³/s). Inclusive of max when it lands
    # on the grid.
    min_flow_mm3_s: float = 5.0
    max_flow_mm3_s: float = 30.0
    flow_step_mm3_s: float = 1.0

    # Hold per level (s) and the fraction of it discarded as settle before
    # measuring. dwell_s · (1 − settle_frac) is the measured window.
    dwell_s: float = 2.0
    settle_frac: float = 0.5
    # Pre-sweep warm-up extrusion at min flow (s).
    warmup_s: float = 3.0
    # No-flow hold after the start marker (extruder idle) -> static-load
    # reference the analyser tares off.
    tare_dwell_s: float = 1.5

    # Pure-E moves run under M204 R / M201 E (see build_sweep notes). High
    # accel keeps each level's velocity transient short so the settle
    # window is dominated by melt-pressure equilibration, not ramp-up.
    accel_mm_s2: float = 5000.0

    purge_x: float = 30.0
    purge_y: float = 30.0
    purge_z: float = 50.0
    baseline_dwell_s: float = 2.0
    z_marker_lift_mm: float = 2.0

    udp_host: str = "192.168.1.10"
    udp_port: int = 8514
    loadcell_metric: str = "loadcell_value"
    label: str = "Max flow test"


@dataclass(slots=True)
class FlowSegment:
    """Planned timing for one flow level, used to slice the timeseries."""
    flow_mm3_s: float
    feed_mm_s: float       # filament feed velocity for this level
    start_offset_s: float  # seconds from sweep_t0 (Z-marker return)
    duration_s: float      # = dwell_s


@dataclass(slots=True)
class FlowPlan:
    gcode: str
    segments: list[FlowSegment]
    params: FlowRampParams


def _filament_area_mm2(filament_diameter: float) -> float:
    return math.pi * (filament_diameter / 2.0) ** 2


def flow_levels(p: FlowRampParams) -> tuple[float, ...]:
    """Inclusive min..max flow grid in `flow_step` increments.

    Mirrors runner._k_range: integer-index loop with round(...,4) so
    summed steps don't drift (e.g. 5.0 + 25*1.0 reads 30.0 exactly).
    """
    lo, hi, step = p.min_flow_mm3_s, p.max_flow_mm3_s, p.flow_step_mm3_s
    if step <= 0 or hi < lo:
        return (round(lo, 4),)
    n = int(round((hi - lo) / step)) + 1
    return tuple(round(lo + i * step, 4) for i in range(n))


def build_flow_ramp(params: FlowRampParams) -> FlowPlan:
    """Build the max-flow test gcode and the per-level timing plan."""
    p = params
    lines: list[str] = []
    segments: list[FlowSegment] = []
    area = _filament_area_mm2(p.filament_diameter)
    levels = flow_levels(p)

    # ---- slicer-compatibility header (kills the "not fully compatible"
    # warning on Buddy/Core One). Mirrors build_sweep. ----
    slicer_header(
        lines,
        label=p.label,
        nozzle_diameter=p.nozzle_diameter,
        filament_diameter=p.filament_diameter,
        filament_label=p.filament_label,
        nozzle_temp=p.nozzle_temp,
        printer_notes="PrusaPATuner -- free-air max-flow test",
        extra_comment_lines=(f"; flow_levels_mm3_s = {list(levels)}",),
    )

    firmware_asserts(lines, nozzle_diameter=p.nozzle_diameter)

    # ---- metrics: stream to host, silence noise, enable what we consume ----
    # loadcell_value = primary back-pressure signal; pos_z = Z-marker
    # anchor for sweep_t0. No pos_x/pos_y: no PA, so no XY coupling.
    metric_setup(
        lines,
        udp_host=p.udp_host,
        udp_port=p.udp_port,
        enables=[
            (p.loadcell_metric, "primary loadcell stream"),
            ("pos_z", "toolhead Z -- sweep_t0 anchor"),
        ],
    )

    # Stuck-detection off (our slow warm-up flow looks like a stalled feed),
    # two-stage preheat + home, cumulative-absolute E, accel pinning, park.
    # See gcode_preamble.heat_home_setup / build_sweep for the rationale.
    preheat = heat_home_setup(
        lines,
        preheat_temp=p.preheat_temp,
        nozzle_temp=p.nozzle_temp,
        accel_mm_s2=p.accel_mm_s2,
        accel_comment="pin accel (R applies to our pure-E moves)",
        purge_x=p.purge_x,
        purge_y=p.purge_y,
        purge_z=p.purge_z,
    )

    cum_e = 0.0

    # Pre-prime baseline: hot nozzle, no extrusion -- static head load only.
    if p.baseline_dwell_s > 0:
        baseline_dwell(
            lines,
            m117_prefix="FLOW",
            marker_prefix=FLOW_MARKER_PREFIX,
            dwell_s=p.baseline_dwell_s,
        )

    # Drop to the test temperature; the warm-up extrude overlaps the small
    # setpoint drop so no extra M109 wait is needed.
    if preheat > p.nozzle_temp:
        lines.append(f"M104 S{p.nozzle_temp:.0f} ; drop to test temp")

    # Warm-up: extrude at min flow to purge old filament + reach steady
    # melt pressure BEFORE the marker, so level 0 (== min flow) is already
    # settled when the measured sweep begins.
    min_feed_mm_s = levels[0] / max(area, 1e-9)
    if p.warmup_s > 0 and min_feed_mm_s > 0:
        warmup_e = min_feed_mm_s * p.warmup_s
        cum_e += warmup_e
        lines.append("; --- warm-up extrude at min flow (purge + settle) ---")
        lines.append(f"G1 E{cum_e:.4f} F{min_feed_mm_s * 60.0:.1f}")

    # Sweep-start Z marker (lift / hold / drop). pos_z return-to-baseline is
    # the exact sweep_t0; the pre-roll dwell below is the post-Z settle.
    if p.z_marker_lift_mm > 0:
        z_marker(lines, purge_z=p.purge_z, lift_mm=p.z_marker_lift_mm)

    sweep_start_marker(lines, m117_prefix="FLOW", marker_prefix=FLOW_MARKER_PREFIX)

    elapsed_s = 0.0
    pre_roll = 0.5
    lines.append(f"G4 P{int(pre_roll * 1000)}")
    elapsed_s += pre_roll

    # No-flow tare hold: extruder idle, so the loadcell reads the static
    # head load only. The analyser averages this window (everything from
    # sweep_t0 up to the first level) and subtracts it, so the plotted
    # force starts at ~0 and reads as back-pressure above baseline.
    if p.tare_dwell_s > 0:
        lines.append(f"{FLOW_MARKER_PREFIX} TARE")
        lines.append(f"G4 P{int(p.tare_dwell_s * 1000)} ; no-flow tare hold")
        elapsed_s += p.tare_dwell_s

    # Each level: one constant-feed pure-E move held for dwell_s. Levels are
    # back-to-back (no idle dwell) so melt pressure doesn't relax between
    # them; the analyser's per-level settle window absorbs the step
    # transient.
    for q in levels:
        feed_mm_s = q / max(area, 1e-9)
        e_inc = feed_mm_s * p.dwell_s
        cum_e += e_inc
        lines.append("")
        lines.append(f"; --- flow={q:.3f} mm3/s (feed {feed_mm_s:.3f} mm/s) ---")
        lines.append(f"M117 FLOW={q:.2f}")
        lines.append(f"{FLOW_MARKER_PREFIX} LEVEL_START q={q:.4f}")
        lines.append(f"G1 E{cum_e:.4f} F{feed_mm_s * 60.0:.1f}")
        lines.append(f"{FLOW_MARKER_PREFIX} LEVEL_END q={q:.4f}")
        segments.append(
            FlowSegment(
                flow_mm3_s=q,
                feed_mm_s=feed_mm_s,
                start_offset_s=elapsed_s,
                duration_s=p.dwell_s,
            )
        )
        elapsed_s += p.dwell_s

    lines.append("")
    sweep_end_marker(lines, m117_prefix="FLOW", marker_prefix=FLOW_MARKER_PREFIX)

    # cleanup
    lines.append("M591 R ; restore stuck detection")
    lines.append("M104 S0 ; nozzle off")
    lines.append("M84 ; disable motors")

    return FlowPlan(gcode="\n".join(lines) + "\n", segments=segments, params=p)
