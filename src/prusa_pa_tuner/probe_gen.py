"""Generate the lateral touch-probe characterisation G-code.

This is step 1 of the touch-probe project: a *sensor-characterisation* test,
not a finished probe. It answers one question that decides whether a lateral
touch probe is even possible on this machine:

    When the nozzle TIP is pushed sideways (±X / ±Y) into a rigid object,
    does the Z-axis loadcell register a clean, monotonic, repeatable force?

The Nextruder loadcell is an *axial* (Z) sensor; a lateral tip force reaches
it only as a bending moment about the mount. Whether that yields a usable
signal is unknown and untestable from the desk -- so we measure it.

Why this is safe to run open-loop (the printer prints a whole file; the host
cannot halt motion on contact over the laggy UDP/PrusaLink path):

  * The nozzle creeps only a small, BOUNDED distance (`creep_mm`, default
    1 mm) past the standoff, and does so SLOWLY (`slow_feed`, default
    0.5 mm/s). Even in the worst case -- contact right at the standoff and
    nothing halting the move -- the machine pushes at most `creep_mm` into a
    rigid object at crawl speed. That is a gentle lean, not a crash. The
    gantry's own compliance absorbs it; XY may skip a step at worst.
  * Default probe temperature is COLD (0 = no heat). A cold brass tip won't
    melt/mar a plastic target, and the loadcell is mechanical so it reads
    fine cold.

Geometry: the user positions a rigid target so the nozzle just contacts it
partway through the creep, then tells us the approach axis + direction and a
standoff point. Each "touch" is:

    1. fast approach to the standoff      (predetermined, stops short -> safe)
    2. brief settle
    3. SLOW creep `creep_mm` toward target (the measured probe)
    4. brief settle (hold at the far end so the contact plateau is captured)
    5. fast retract `backoff_mm` away

We repeat `n_touches` times from the SAME standoff so the analyser can
overlay the force-vs-position curves and see whether the contact "knee"
lands at the same place every time (repeatability).

The scaffolding mirrors flow_gen / gcode_gen so the same runner / metric
plumbing works unchanged: slicer-compat header, M334 UDP target, M332
silence, M331 for loadcell_value + pos_x/pos_y/pos_z. No extrusion at all
(pure motion), so there is no E axis, no purge, and no filament handling.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gcode_preamble import (
    PROBE_MARKER_PREFIX,
    firmware_asserts,
    metric_setup,
    slicer_header,
    sweep_end_marker,
    sweep_start_marker,
)


@dataclass(slots=True)
class ProbeParams:
    # --- direction the head moves to find the part ---
    probe_axis: str = "X"          # "X" or "Y"
    probe_dir: str = "+"           # "+" or "-"

    # --- standoff: where the slow creep begins (mm, machine coords) ---
    start_x: float = 125.0
    start_y: float = 110.0
    probe_z: float = 5.0           # height to probe at; also the travel Z to
    #                                the standoff -- keep the path to start_xy
    #                                clear of obstacles at this Z.

    # --- the probe move itself ---
    creep_mm: float = 1.0          # SLOW creep distance toward the target;
    #                                this is the hard cap on overtravel.
    slow_feed_mm_min: float = 30.0  # creep speed (0.5 mm/s) -- gentle.
    travel_feed_mm_min: float = 3000.0  # rapid repositioning between touches.
    n_touches: int = 5
    backoff_mm: float = 1.0        # retract past the standoff between touches
    #                                so each touch fully clears contact first.
    settle_ms: int = 300           # hold at the standoff and at the far end.

    # --- thermal ---
    probe_temp: float = 0.0        # 0 => COLD probe (no heat); >0 heats first.

    # --- capture scaffolding ---
    baseline_dwell_s: float = 1.0  # idle hold at the standoff before touch 0
    #                                (static loadcell reading / tare reference).

    udp_host: str = "192.168.1.10"
    udp_port: int = 8514
    loadcell_metric: str = "loadcell_value"
    label: str = "Touch-probe lateral characterisation"

    def dir_sign(self) -> float:
        return -1.0 if str(self.probe_dir).strip().startswith("-") else 1.0

    def axis(self) -> str:
        return "Y" if str(self.probe_axis).strip().upper().startswith("Y") else "X"


@dataclass(slots=True)
class ProbeTouch:
    """Planned timing + geometry for one touch's SLOW creep window.

    Positions are in machine coords along the probe axis. The analyser
    segments touches from the position stream directly (robust), but these
    offsets give a timing cross-check / fallback.
    """
    idx: int
    start_offset_s: float   # creep start, seconds from sweep_t0
    duration_s: float       # creep duration = creep_mm / slow_feed (s)
    axis_start: float       # standoff coordinate (creep start)
    axis_end: float         # far coordinate (creep end = start + creep*dir)


@dataclass(slots=True)
class ProbePlan:
    gcode: str
    touches: list[ProbeTouch]
    params: ProbeParams
    axis: str               # "X" or "Y" -- resolved probe axis
    dir_sign: float         # +1.0 / -1.0


def build_probe_test(params: ProbeParams) -> ProbePlan:
    """Build the lateral touch-probe characterisation gcode + touch plan."""
    p = params
    axis = p.axis()
    sign = p.dir_sign()
    other_axis = "Y" if axis == "X" else "X"
    other_val = p.start_y if axis == "X" else p.start_x

    start = p.start_x if axis == "X" else p.start_y
    far = start + sign * p.creep_mm
    pre = start - sign * p.backoff_mm           # retract / pre-approach point

    slow_mm_s = max(p.slow_feed_mm_min / 60.0, 1e-6)
    creep_dur_s = p.creep_mm / slow_mm_s

    def axis_line(ax_val: float, feed_mm_min: float, comment: str) -> str:
        """One probe-axis move; the other axis is pinned to its standoff."""
        if axis == "X":
            return f"G1 X{ax_val:.3f} Y{other_val:.3f} F{feed_mm_min:.0f} ; {comment}"
        return f"G1 X{other_val:.3f} Y{ax_val:.3f} F{feed_mm_min:.0f} ; {comment}"

    lines: list[str] = []
    touches: list[ProbeTouch] = []

    # ---- slicer-compatibility header (mirrors build_flow_ramp; kills the
    # "not fully compatible" warning on Buddy/Core One). The probe never
    # extrudes, so the filament/temperature keys are advisory placeholders
    # (temp 0 = cold). ----
    slicer_header(
        lines,
        label=p.label,
        nozzle_diameter=0.4,
        filament_diameter=1.75,
        filament_label="PLA",
        nozzle_temp=0.0,
        printer_notes="PrusaPATuner -- lateral touch-probe test",
        extra_comment_lines=(
            f"; probe axis={axis}{'+' if sign > 0 else '-'} "
            f"creep={p.creep_mm} mm slow={p.slow_feed_mm_min} mm/min "
            f"touches={p.n_touches}",
        ),
    )

    # TODO: pass the probe params' nozzle diameter here once ProbeParams
    # grows one -- 0.4 is hardcoded today because the probe is contact-only
    # (no extrusion) and the check is advisory.
    firmware_asserts(lines, nozzle_diameter=0.4)

    # ---- metrics: stream to host, silence noise, enable what we consume ----
    # loadcell_value = the contact-force signal under test; pos_x/pos_y give
    # the toolhead position the analyser plots force against; pos_z for
    # completeness (the probe never moves Z during touches).
    metric_setup(
        lines,
        udp_host=p.udp_host,
        udp_port=p.udp_port,
        enables=[
            (p.loadcell_metric, "primary loadcell stream"),
            ("pos_x", "toolhead X"),
            ("pos_y", "toolhead Y"),
            ("pos_z", "toolhead Z"),
        ],
    )

    lines.append("G90 ; absolute XYZ")

    # Optional heat. Default COLD (probe_temp == 0): a cold brass tip won't
    # mar/melt a plastic target and the loadcell reads fine cold.
    if p.probe_temp > 0:
        lines.append(f"M104 S{p.probe_temp:.0f} ; preheat nozzle")
    lines.append("G28 ; home")
    if p.probe_temp > 0:
        lines.append(f"M109 S{p.probe_temp:.0f} ; wait for probe temp")

    lines.append("G90 ; absolute XYZ (reassert after home)")

    # Move to the probe Z first, then to the pre-approach point (one backoff
    # short of the standoff). Keep the path to start_xy clear at this Z.
    lines.append(f"G1 Z{p.probe_z:.3f} F{p.travel_feed_mm_min:.0f} ; to probe Z")
    lines.append(axis_line(pre, p.travel_feed_mm_min, "to pre-approach point"))
    lines.append("G4 P300 ; settle")

    # Idle baseline: nozzle parked at the standoff, no contact -> static
    # loadcell reading the analyser can tare against. Deliberately NOT the
    # shared gcode_preamble.baseline_dwell block -- the probe baseline is
    # bracketed by axis moves and uses a single M117.
    if p.baseline_dwell_s > 0:
        lines.append(axis_line(start, p.travel_feed_mm_min, "to standoff for baseline"))
        lines.append("M117 PROBE_BASELINE")
        lines.append(f"{PROBE_MARKER_PREFIX} BASELINE_START")
        lines.append(f"G4 P{int(p.baseline_dwell_s * 1000)} ; idle baseline hold")
        lines.append(f"{PROBE_MARKER_PREFIX} BASELINE_END")
        lines.append(axis_line(pre, p.travel_feed_mm_min, "back off before touch 0"))

    sweep_start_marker(lines, m117_prefix="PROBE", marker_prefix=PROBE_MARKER_PREFIX)

    # sweep_t0 = the instant the SWEEP_START marker is parsed; the analyser
    # works in position-space so the exact value is informational. We track
    # elapsed_s only to fill in each touch's planned offset (cross-check).
    elapsed_s = 0.0
    settle_s = p.settle_ms / 1000.0

    for i in range(p.n_touches):
        lines.append(f"; --- touch {i} ---")
        lines.append(f"M117 PROBE={i}")
        # 1. fast approach to the standoff (stops short of contact -> safe)
        lines.append(axis_line(start, p.travel_feed_mm_min, f"touch {i}: fast approach"))
        # 2. settle at the standoff
        lines.append(f"G4 P{p.settle_ms} ; settle at standoff")
        elapsed_s += settle_s
        # 3. SLOW creep into the target (the measured window)
        lines.append(f"{PROBE_MARKER_PREFIX} TOUCH_START i={i}")
        lines.append(axis_line(far, p.slow_feed_mm_min, f"touch {i}: SLOW creep"))
        touches.append(
            ProbeTouch(
                idx=i,
                start_offset_s=elapsed_s,
                duration_s=creep_dur_s,
                axis_start=start,
                axis_end=far,
            )
        )
        elapsed_s += creep_dur_s
        # 4. hold at the far end so the contact plateau is captured
        lines.append(f"G4 P{p.settle_ms} ; hold at far end")
        lines.append(f"{PROBE_MARKER_PREFIX} TOUCH_END i={i}")
        elapsed_s += settle_s
        # 5. fast retract past the standoff to fully clear contact
        lines.append(axis_line(pre, p.travel_feed_mm_min, f"touch {i}: retract"))
        lines.append("")

    sweep_end_marker(lines, m117_prefix="PROBE", marker_prefix=PROBE_MARKER_PREFIX)

    # cleanup
    if p.probe_temp > 0:
        lines.append("M104 S0 ; nozzle off")
    lines.append("M84 ; disable motors")

    return ProbePlan(
        gcode="\n".join(lines) + "\n",
        touches=touches,
        params=p,
        axis=axis,
        dir_sign=sign,
    )
