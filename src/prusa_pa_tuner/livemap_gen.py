"""Wrap an uploaded ASCII print with the Live Map metric-enable preamble.

The user uploads a normal sliced .gcode; we prepend a small block that:
  * points the printer's UDP metric stream at this host (M334),
  * silences the non-essential metrics (M332) to keep UDP load down, and
  * enables the four streams Live Map consumes: loadcell_value + pos_x/y/z
    (M331).

Crucially we DO NOT touch the print geometry -- only M-codes are added at the
very top, so the printed part is byte-identical to what the slicer produced.
Any pre-existing M334 in the upload is stripped so the user's (or a stale)
UDP target can't override ours. See project_live_map_module memory.
"""
from __future__ import annotations

from .gcode_preamble import metric_setup


def build_livemap_gcode(
    user_gcode: str,
    *,
    udp_host: str,
    udp_port: int,
    loadcell_metric: str = "loadcell_value",
) -> str:
    """Return upload-ready gcode: metric preamble + the user's gcode verbatim.

    `user_gcode` must be plain ASCII (we do not decode binary .bgcode).
    """
    pre: list[str] = []
    pre.append("; --- PrusaPATuner Live Map preamble (injected) ---")
    pre.append("; geometry below is the user's sliced gcode, untouched.")
    # The four streams Live Map maps onto the gcode preview. loadcell_value is
    # the nozzle force; pos_x/y/z place each force sample in the print (and on
    # the same recv_monotonic host clock as the force, so they're inherently
    # time-aligned -- the whole point of the streamed-position approach).
    metric_setup(
        pre,
        udp_host=udp_host,
        udp_port=udp_port,
        enables=[
            (loadcell_metric, "nozzle force"),
            ("pos_x", "toolhead X"),
            ("pos_y", "toolhead Y"),
            ("pos_z", "toolhead Z (layer)"),
        ],
        silence_header="; silence non-essential metrics to lower UDP load",
        blank_after_silence=False,
        blank_after_enables=False,
    )
    pre.append("; --- end Live Map preamble ---")
    pre.append("")

    cleaned = _strip_m334(user_gcode)
    body = cleaned if cleaned.endswith("\n") else cleaned + "\n"
    return "\n".join(pre) + body


def _strip_m334(gcode: str) -> str:
    """Remove any standalone M334 line so it can't override our UDP target."""
    out: list[str] = []
    for line in gcode.splitlines():
        stripped = line.lstrip()
        # tolerate inline comments / leading whitespace; match the command token
        first = stripped.split(maxsplit=1)[0].upper() if stripped else ""
        if first == "M334":
            # Don't echo the original command -- the preamble already set the
            # UDP target, and echoing it would just re-introduce a stale host.
            out.append("; [Live Map] removed a slicer M334 (UDP target set above)")
            continue
        out.append(line)
    return "\n".join(out)
