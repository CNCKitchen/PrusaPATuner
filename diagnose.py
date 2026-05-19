"""Automated diagnostic: upload+print a test gcode while sniffing UDP.

PrusaLink on Core One doesn't expose an arbitrary-gcode injection endpoint, so we
upload a tiny job file and let the printer run it.

Sequence (default):
  1. Start UDP listener
  2. Upload+print diagnostic .gcode containing:
       M334 <ourip> 8514
       M331 probe_load_line / tmc_sg_e / loadcell_* / ...
       G28
       M104 + M109 (skipped if already hot)
       M83 + G1 E<extrude_mm> F<feed>
       M104 S0
       M332 ... cleanup
       M334
  3. Poll until IDLE
  4. Print per-window rates: baseline / G28 / extrude
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm

CANDIDATE_METRICS = [
    "probe_load_line",
    "tmc_sg_e", "tmc_sg_x", "tmc_sg_y", "tmc_sg_z",
    "nozzle_pwm",
    "loadcell_scale", "loadcell_threshold", "loadcell_threshold_cont", "loadcell_hysteresis",
]

UDP_TARGET_MARKER_BASE = "DIAG_T"


def local_ip_toward(host: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 80))
        return s.getsockname()[0]
    finally:
        s.close()


class PrusaLink:
    def __init__(self, host: str, user: str, password: str):
        self.host = host
        mgr = HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, f"http://{host}", user, password)
        self.opener = urllib.request.build_opener(HTTPDigestAuthHandler(mgr))

    def status(self) -> dict | None:
        try:
            req = urllib.request.Request(f"http://{self.host}/api/v1/status",
                                         headers={"Accept": "application/json"})
            with self.opener.open(req, timeout=5) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            return None

    def upload_and_print(self, filename: str, gcode: bytes) -> tuple[bool, str]:
        url = f"http://{self.host}/api/v1/files/usb/{filename}"
        req = urllib.request.Request(
            url, data=gcode, method="PUT",
            headers={
                "Content-Type": "text/x.gcode",
                "Overwrite": "?1",
                "Print-After-Upload": "?1",
            },
        )
        try:
            with self.opener.open(req, timeout=30) as r:
                return True, f"HTTP {r.status} {r.read().decode('utf-8','replace')[:300]}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code} {e.read().decode('utf-8','replace')[:300]}"
        except Exception as e:
            return False, str(e)


def state_text(st: dict | None) -> str:
    if not st:
        return "?"
    pr = st.get("printer") or {}
    return str(pr.get("state") or st.get("state") or "?").upper()


def wait_state(pl: PrusaLink, want: str, timeout: float, settle: float = 1.0) -> bool:
    t_end = time.monotonic() + timeout
    last_match = 0.0
    while time.monotonic() < t_end:
        s = state_text(pl.status())
        if s == want:
            if last_match == 0:
                last_match = time.monotonic()
            elif time.monotonic() - last_match >= settle:
                return True
        else:
            last_match = 0.0
        time.sleep(0.5)
    return False


def build_gcode(our_ip: str, port: int, metrics: list[str],
                temp: float, extrude_mm: float, feedrate_mm_min: float,
                do_home: bool, do_extrude: bool, current_nozzle_temp: float) -> bytes:
    L: list[str] = []
    L += [
        "; diagnose -- probe_load_line / tmc_sg_e capture",
        f"M334 {our_ip} {port}",
    ]
    for m in metrics:
        L.append(f"M331 {m}")
    L += [
        "M117 DIAG_START",
        "G4 P500",
    ]
    if do_home:
        L += [
            "M117 DIAG_G28_START",
            "G28",
            "M117 DIAG_G28_END",
            "G4 P500",
        ]
    if do_extrude:
        if current_nozzle_temp < temp - 5:
            L += [
                f"M104 S{temp:.0f}",
                f"M109 S{temp:.0f}",
            ]
        L += [
            "M83 ; relative E",
            "M117 DIAG_EXTRUDE_START",
            f"G1 E{extrude_mm:.2f} F{feedrate_mm_min:.0f}",
            "M117 DIAG_EXTRUDE_END",
            "G4 P500",
            "M104 S0 ; cool",
        ]
    L += ["M117 DIAG_END"]
    for m in metrics:
        L.append(f"M332 {m}")
    L += ["M334", "M84"]
    return ("\n".join(L) + "\n").encode("utf-8")


class Sniffer:
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.stop_event = threading.Event()
        self.counts: Counter[str] = Counter()
        self.events: list[tuple[float, str, str]] = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.t0 = 0.0

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            now = time.monotonic()
            for line in data.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                head = line.split(" ", 1)[0]
                name = head.split(",", 1)[0]
                with self.lock:
                    self.counts[name] += 1
                    self.events.append((now, name, line))

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.sock.close()

    def rates_between(self, t_lo: float, t_hi: float) -> list[tuple[str, float, int]]:
        with self.lock:
            wins = [e for e in self.events if t_lo <= e[0] <= t_hi]
        names = Counter(e[1] for e in wins)
        dur = max(0.001, t_hi - t_lo)
        return sorted(((n, c / dur, c) for n, c in names.items()), key=lambda x: -x[1])

    def find_marker(self, label: str) -> float | None:
        """Find the wall-clock time the printer's gcode metric stream first echoed
        the M117 marker. We rely on the `gcode` metric (DIS by default) or, failing
        that, on `print_filename` / `cmdcnt` cadence. Simplest robust hack: just look
        for the substring in the raw line of ANY metric (some printers echo M117 in
        their event logs)."""
        with self.lock:
            for t, _, raw in self.events:
                if label in raw:
                    return t
        return None

    def sample_lines(self, name: str, n: int = 3) -> list[str]:
        out: list[str] = []
        with self.lock:
            for _, na, raw in self.events:
                if na == name:
                    out.append(raw)
                    if len(out) >= n:
                        break
        return out


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Printer IP or hostname")
    ap.add_argument("--user", default="maker")
    ap.add_argument("--password", required=True)
    ap.add_argument("--port", type=int, default=8514)
    ap.add_argument("--temp", type=float, default=215.0)
    ap.add_argument("--extrude-mm", type=float, default=30.0)
    ap.add_argument("--feedrate-mm-min", type=float, default=60.0)
    ap.add_argument("--no-extrude", action="store_true")
    ap.add_argument("--no-home", action="store_true")
    args = ap.parse_args()

    my_ip = local_ip_toward(args.host)
    print(f"PrusaLink:  http://{args.host}  user={args.user}")
    print(f"UDP target: {my_ip}:{args.port}")

    pl = PrusaLink(args.host, args.user, args.password)
    st = pl.status()
    if not st:
        print("Cannot reach PrusaLink. Is Enabled = ON?")
        return 2
    pr = st.get("printer") or {}
    cur_temp = float(pr.get("temp_nozzle", 0.0))
    print(f"Printer state: {state_text(st)}   nozzle={cur_temp:.1f} C")

    if state_text(st) != "IDLE":
        print("Printer is not IDLE. Aborting so we don't fight a running job.")
        return 2

    # start sniffer
    banner("starting UDP sniffer")
    try:
        sniff = Sniffer(args.port)
    except PermissionError as e:
        print(f"bind failed: {e}")
        return 2
    sniff.start()

    # build + upload + print
    gcode = build_gcode(
        our_ip=my_ip, port=args.port, metrics=CANDIDATE_METRICS,
        temp=args.temp, extrude_mm=args.extrude_mm,
        feedrate_mm_min=args.feedrate_mm_min,
        do_home=not args.no_home, do_extrude=not args.no_extrude,
        current_nozzle_temp=cur_temp,
    )
    banner("diagnostic gcode")
    print(gcode.decode("utf-8"))

    banner("upload & start")
    ok, info = pl.upload_and_print("_pa_diag.gcode", gcode)
    print(f"  {'OK' if ok else 'FAIL'}  {info[:300]}")
    if not ok:
        sniff.stop()
        return 3

    # wait for the job to actually start
    print("waiting for job to begin...")
    t_started = time.monotonic()
    for _ in range(60):
        s = state_text(pl.status())
        if s != "IDLE":
            print(f"  state -> {s}  ({time.monotonic() - t_started:.1f}s)")
            break
        time.sleep(0.5)

    # poll until back to IDLE (job finished)
    print("waiting for job to finish (poll every 1s)...")
    t_end_deadline = time.monotonic() + 360
    while time.monotonic() < t_end_deadline:
        s = state_text(pl.status())
        if s == "IDLE" or s == "FINISHED":
            break
        time.sleep(1.0)
    t_done = time.monotonic()
    print(f"  done after {t_done - t_started:.1f}s")

    # let tail flush
    time.sleep(1.5)
    sniff.stop()

    # --- analyse ---
    banner("rate summary over full capture")
    full_lo = sniff.t0
    full_hi = time.monotonic()
    for n, hz, c in sniff.rates_between(full_lo, full_hi)[:40]:
        flag = "  STREAMING" if hz > 5 else ""
        print(f"  {n:30s}  {hz:7.1f} Hz   n={c:5d}{flag}")

    banner("force-relevant metrics: did any of these arrive at all?")
    for m in ("probe_load_line", "tmc_sg_e", "tmc_sg_x", "tmc_sg_y", "tmc_sg_z",
              "loadcell_scale", "loadcell_threshold", "loadcell_threshold_cont",
              "loadcell_hysteresis", "nozzle_pwm"):
        n = sniff.counts.get(m, 0)
        if n == 0:
            print(f"  {m:30s}  (none)")
        else:
            samples = sniff.sample_lines(m, n=2)
            print(f"  {m:30s}  n={n}")
            for s in samples:
                print(f"    {s}")

    banner("timeline -- when did probe_load_line / tmc_sg_e fire?")
    for m in ("probe_load_line", "tmc_sg_e", "loadcell_threshold_cont"):
        ts = [t - sniff.t0 for t, name, _ in sniff.events if name == m]
        if not ts:
            print(f"  {m}: never")
            continue
        # bin into 1-second windows
        bins: Counter[int] = Counter()
        for t in ts:
            bins[int(t)] += 1
        burst_summary = ",".join(f"{sec}s:{cnt}" for sec, cnt in sorted(bins.items()) if cnt)
        print(f"  {m}: total={len(ts)}  per-second-bursts: {burst_summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
