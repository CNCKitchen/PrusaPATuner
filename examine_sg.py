"""Re-run a short capture focused on tmc_sg_e while the motor is moving.

We do two things while sniffing:
  1. A G28 (motors at speed -> SG should report real values if it works at all)
  2. A slow extrude (the PA-relevant condition)

Then dump the value histogram for tmc_sg_e separately for each window so we can
tell whether SG is a usable signal.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from collections import Counter
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm
import urllib.request, urllib.error, json


def opener_for(host: str, user: str, pw: str):
    mgr = HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, f"http://{host}", user, pw)
    return urllib.request.build_opener(HTTPDigestAuthHandler(mgr))


def my_ip(host: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 80)); return s.getsockname()[0]
    finally:
        s.close()


def upload_print(opener, host: str, name: str, gcode: bytes) -> str:
    req = urllib.request.Request(
        f"http://{host}/api/v1/files/usb/{name}", data=gcode, method="PUT",
        headers={
            "Content-Type": "text/x.gcode",
            "Overwrite": "?1",
            "Print-After-Upload": "?1",
        },
    )
    with opener.open(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")[:120]


def state(opener, host: str) -> str:
    try:
        with opener.open(f"http://{host}/api/v1/status", timeout=5) as r:
            d = json.loads(r.read().decode())
            return (d.get("printer") or {}).get("state", "?").upper()
    except Exception:
        return "?"


def parse_int_val(line: str) -> int | None:
    # "tmc_sg_e v=123i 4567"  ->  123
    try:
        fields = line.split(" ", 2)
        for tok in fields[1].split(","):
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k == "v":
                    if v.endswith(("i", "u")): v = v[:-1]
                    return int(v)
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Printer IP or hostname")
    ap.add_argument("--user", default="maker")
    ap.add_argument("--password", required=True)
    ap.add_argument("--port", type=int, default=8514)
    ap.add_argument("--temp", type=float, default=215.0)
    ap.add_argument("--extrude-mm", type=float, default=20.0)
    ap.add_argument("--feedrate-mm-min", type=float, default=120.0)  # 2 mm/s
    args = ap.parse_args()

    op = opener_for(args.host, args.user, args.password)
    ip = my_ip(args.host)
    print(f"local IP: {ip}    state: {state(op, args.host)}")

    # UDP listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.2)
    samples_sg: list[tuple[float, int]] = []
    samples_other: list[tuple[float, str, str]] = []
    stop = threading.Event()

    def loop():
        t0 = time.monotonic()
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            now = time.monotonic() - t0
            for line in data.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line: continue
                name = line.split(" ", 1)[0].split(",", 1)[0]
                if name == "tmc_sg_e":
                    v = parse_int_val(line)
                    if v is not None:
                        samples_sg.append((now, v))
                samples_other.append((now, name, line))

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    t0 = time.monotonic()

    gcode = (
        f"M334 {ip} {args.port}\n"
        "M331 tmc_sg_e\n"
        "M331 tmc_sg_x\n"
        "M331 tmc_sg_y\n"
        "M331 tmc_sg_z\n"
        "M117 SG_START\n"
        "G4 P1500\n"
        "M117 SG_G28\n"
        "G28\n"
        "G4 P1000\n"
        f"M104 S{args.temp:.0f}\nM109 S{args.temp:.0f}\n"
        "G91\n"
        "G1 Z5 F600 ; lift\n"
        "G90\n"
        "M83\n"
        "M117 SG_EXTRUDE\n"
        f"G1 E{args.extrude_mm:.1f} F{args.feedrate_mm_min:.0f}\n"
        "M117 SG_END\n"
        "M104 S0\n"
        "M332 tmc_sg_e\nM332 tmc_sg_x\nM332 tmc_sg_y\nM332 tmc_sg_z\nM334\nM84\n"
    ).encode("utf-8")

    print("uploading + starting...")
    print(upload_print(op, args.host, "_pa_sg.gcode", gcode))

    # rough wait
    deadline = time.monotonic() + 300
    print("waiting for finish...")
    while time.monotonic() < deadline:
        s = state(op, args.host)
        if s in ("IDLE", "FINISHED"):
            break
        time.sleep(1.0)
    time.sleep(1.5)
    stop.set()
    t.join(timeout=2)
    sock.close()

    print(f"\ntmc_sg_e total samples: {len(samples_sg)}")
    if samples_sg:
        vals = [v for _, v in samples_sg]
        c = Counter(vals)
        print(f"  unique values: {len(c)}")
        print(f"  min/max/mean : {min(vals)} / {max(vals)} / {sum(vals)/len(vals):.2f}")
        print(f"  top 20 values (value: count):")
        for v, n in c.most_common(20):
            print(f"    {v:5d}  x{n}")
        print(f"  first 30 timestamped values (s_after_start, v):")
        for (ts, v) in samples_sg[:30]:
            print(f"    {ts:6.2f}s  v={v}")
        print(f"  ...")
        print(f"  last 30:")
        for (ts, v) in samples_sg[-30:]:
            print(f"    {ts:6.2f}s  v={v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
