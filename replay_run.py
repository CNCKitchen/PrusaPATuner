#!/usr/bin/env python3
"""CLI to re-analyse a saved npz dump from `runs/`.

Usage:
    python replay_run.py runs/run_<ts>.npz
    python replay_run.py runs/run_<ts>.npz --json out.json
    python replay_run.py runs/run_<ts>.npz --weights overshoot=3,undershoot=1

Prints a per-K summary table with the bd_pressure composite cost and
each per-metric K_opt. Use this for the fast inner-loop (tweak
`analysis.py`, re-replay, see new numbers) without booting the FastAPI
server or kicking off a real printer run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running this script from the repo root without installing the
# package — append `src/` to sys.path so `prusa_pa_tuner` resolves.
SRC = Path(__file__).parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prusa_pa_tuner.analysis import (  # noqa: E402
    BD_DEFAULT_WEIGHTS, BD_METRIC_NAMES, _argmin_with_parabolic,
    _bd_compute_cost,
)
from prusa_pa_tuner.replay import replay  # noqa: E402
from prusa_pa_tuner.runner import _analysis_to_dict  # noqa: E402


def _parse_weights(s: str | None) -> dict[str, float]:
    """Parse `--weights overshoot=3,undershoot=1` into a weight dict.

    Unspecified weights fall back to `BD_DEFAULT_WEIGHTS`. Unknown
    metric names raise — typo protection beats silent-wrong-result.
    """
    weights = dict(BD_DEFAULT_WEIGHTS)
    if not s:
        return weights
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--weights: bad entry {chunk!r} (need name=value)")
        name, val = chunk.split("=", 1)
        name = name.strip()
        if name not in BD_DEFAULT_WEIGHTS:
            raise SystemExit(
                f"--weights: unknown metric {name!r}; "
                f"valid: {', '.join(BD_DEFAULT_WEIGHTS.keys())}"
            )
        weights[name] = float(val)
    return weights


def _per_metric_k_opts(bd_per_k) -> dict[str, float | None]:
    """For each composite-cost-contributing metric, find K minimising
    the SIGNED normalised value (or absolute, for asymmetric
    overshoot/undershoot)."""
    ks = np.asarray([r.k for r in bd_per_k], dtype=float)
    out: dict[str, float | None] = {}
    for name in BD_DEFAULT_WEIGHTS:
        vals = np.asarray(
            [r.normalised.get(name, float("nan")) for r in bd_per_k],
            dtype=float,
        )
        if name in ("overshoot", "undershoot"):
            vals = np.maximum(vals, 0.0)
        out[name] = _argmin_with_parabolic(ks, vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to runs/run_<ts>.npz")
    ap.add_argument(
        "--weights",
        help="override composite cost weights, "
        "e.g. 'overshoot=3,undershoot=1.5'",
    )
    ap.add_argument(
        "--json",
        help="dump full _analysis_to_dict payload to this path "
        "(matches the /api/runs/<f>/analyse response shape)",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    plan, analysis = replay(path)
    weights = _parse_weights(args.weights)
    cost_custom = _bd_compute_cost(analysis.bd_per_k, weights)
    quality_ks = np.asarray(
        [r.k for r in analysis.bd_per_k], dtype=float
    )
    k_opt_custom = _argmin_with_parabolic(quality_ks, cost_custom)

    per_metric = _per_metric_k_opts(analysis.bd_per_k)

    print(f"=== {path.name} ===")
    print(f"K values:           {len(plan.segments)}")
    print(f"cycles/K:           {plan.params.cycles_per_K}")
    print(f"slow/fast halves:   {plan.params.slow_half_s}s / {plan.params.fast_half_s}s")
    print(f"coupled_dx_mm:      {plan.params.coupled_dx_mm:.3f} (derived from pos_x swing)")
    print(f"sample rate:        {analysis.sample_rate_hz:.1f} Hz incoming")
    # Shared-exclusion summary: both integral_area and bd_pressure
    # consume the same per-cycle BdSegment list and honour its
    # `excluded` flag. Tot/included rolled up across K's.
    tot_inc = sum(r.integral_n_included for r in analysis.per_k)
    tot_total = sum(r.integral_n_total for r in analysis.per_k)
    print(
        f"cycles included:    {tot_inc}/{tot_total} "
        f"(shared bd-exclusion gate: dropouts in critical zone, "
        f"sample rate <40 Hz, signal-below-noise)"
    )
    if analysis.integral_fit is not None:
        ig = analysis.integral_fit
        print(
            f"integral_fit:       K_opt={ig.k_opt:.4f}  slope={ig.slope:.0f}  "
            f"R²={ig.r_squared:.3f}"
        )
    else:
        print("integral_fit:       — (not enough cycles passed the shared gate)")
    print(f"bd_k_opt (default): {analysis.bd_k_opt}")
    print(f"bd_k_opt (custom):  {k_opt_custom}")
    print()
    print("per-metric K_opt:")
    for name, k in per_metric.items():
        ks_str = f"{k:.4f}" if k is not None else "-"
        print(f"  {name:>20s}: {ks_str}")
    print()
    print(
        f"{'K':>8} {'segs':>8} "
        + " ".join(f"{n[:8]:>10s}" for n in BD_METRIC_NAMES)
        + f" {'cost':>10s}"
    )
    for i, r in enumerate(analysis.bd_per_k):
        cells = []
        for n in BD_METRIC_NAMES:
            v = r.medians.get(n, float("nan"))
            cells.append(
                f"{v:>10.3g}" if np.isfinite(v) else f"{'nan':>10s}"
            )
        c = cost_custom[i]
        cost_cell = f"{c:>10.3g}" if np.isfinite(c) else f"{'nan':>10s}"
        print(
            f"{r.k:>8.4f} "
            f"{r.n_segments_included:>4d}/{r.n_segments_total:<3d} "
            + " ".join(cells) + " "
            + cost_cell
        )

    if analysis.notes:
        print()
        print("notes:")
        for n in analysis.notes:
            # On Windows consoles, fall back to ASCII for unicode chars.
            try:
                print(f"  - {n}")
            except UnicodeEncodeError:
                print(f"  - {n.encode('ascii', errors='replace').decode('ascii')}")

    if args.json:
        payload = {
            "filename": path.name,
            "k_values": [seg.k for seg in plan.segments],
            "analysis": _analysis_to_dict(analysis),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote full payload to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
