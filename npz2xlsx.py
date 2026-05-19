"""Convert a captured run NPZ into an xlsx for inspection in Excel.

Each stream (loadcell, pos_x, pos_y, pos_z) becomes its own sheet with:
  * t_monotonic   -- raw monotonic-clock seconds (the analyser's reference)
  * t_rel_s       -- zero-based seconds since the first sample (Excel-chart-friendly)
  * t_wall_utc    -- ISO-8601 wall-clock UTC (when the NPZ carries an anchor)
  * <stream name> -- the value column

A `meta` sheet lists the scalar plan parameters (k_values, cycle_period_s, ...).
"""
import argparse
import numpy as np
import pandas as pd


# Map of (timestamp-array-name, value-array-name, sheet-name).
STREAMS = [
    ("force_t", "force_y", "loadcell"),
    ("pos_t",   "pos_x",   "pos_x"),
    ("pos_y_t", "pos_y",   "pos_y"),
    ("pos_z_t", "pos_z",   "pos_z"),
]
SCALAR_KEYS = [
    "sweep_t0", "k_values", "cycle_period_s", "cycles_per_K",
    "slow_half_s", "fast_half_s", "slow_feed_mm_s", "fast_feed_mm_s",
    "mono_anchor_mono", "mono_anchor_unix", "started_at_unix",
]


def _wall_clock_anchor(d) -> tuple[float, float] | None:
    """Return (mono_anchor_mono, mono_anchor_unix) if both are in the NPZ.

    With these two scalars, any monotonic timestamp `m` converts to a
    unix timestamp via `m - mono_anchor_mono + mono_anchor_unix`.
    """
    if "mono_anchor_mono" in d.files and "mono_anchor_unix" in d.files:
        try:
            return float(d["mono_anchor_mono"][0]), float(d["mono_anchor_unix"][0])
        except (IndexError, ValueError, TypeError):
            return None
    return None


def npz2xlsx(src: str, dst: str) -> None:
    d = np.load(src)
    anchor = _wall_clock_anchor(d)

    with pd.ExcelWriter(dst, engine="openpyxl") as xw:
        for tk, vk, sheet in STREAMS:
            if tk not in d.files or vk not in d.files:
                continue
            t = d[tk]
            v = d[vk]
            if t.size == 0 or v.size == 0:
                continue
            cols = {
                "t_monotonic": t,
                "t_rel_s": t - t[0],
            }
            if anchor is not None:
                mono_a, unix_a = anchor
                unix_ts = t - mono_a + unix_a
                cols["t_wall_utc"] = pd.to_datetime(unix_ts, unit="s", utc=True)
            cols[sheet] = v
            pd.DataFrame(cols).to_excel(xw, sheet_name=sheet, index=False)

        # meta sheet: one column per scalar
        meta_cols = {}
        for k in SCALAR_KEYS:
            if k in d.files:
                arr = d[k]
                meta_cols[k] = arr if arr.size else [None]
        if meta_cols:
            # Pad to the longest column for a tidy single-sheet view.
            max_len = max(len(v) for v in meta_cols.values())
            for k, v in list(meta_cols.items()):
                if len(v) < max_len:
                    meta_cols[k] = list(v) + [None] * (max_len - len(v))
            pd.DataFrame(meta_cols).to_excel(xw, sheet_name="meta", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ run dump to XLSX")
    parser.add_argument("src", help="source .npz file")
    parser.add_argument("dst", nargs="?", help="destination .xlsx file")
    args = parser.parse_args()
    dst = args.dst or args.src.rsplit(".", 1)[0] + ".xlsx"
    npz2xlsx(args.src, dst)
    print(f"wrote {dst}")
