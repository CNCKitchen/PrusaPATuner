"""Quick check: does your printer's firmware actually emit loadcell_value / loadcell_hp?

Uploads a tiny gcode that enables every known loadcell metric, then sniffs for ~12 s
while the printer is idle (no motion required -- loadcell_value should stream even
when idle, per Export.md).
"""
from __future__ import annotations

import argparse, json, socket, threading, time, urllib.request, urllib.error
from collections import Counter
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm


METRICS = [
    "loadcell_value",
    "loadcell_hp",
    "loadcell_xy",
    "loadcell_age",
    "loadcell",  # CUSTOM diagnose triple
    "loadcell_scale",  # known to exist in master too; control
    "probe_load_line",
]


def opener_for(host, user, pw):
    mgr = HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, f"http://{host}", user, pw)
    return urllib.request.build_opener(HTTPDigestAuthHandler(mgr))


def my_ip(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect((host, 80)); return s.getsockname()[0]
    finally: s.close()


def upload_print(opener, host, name, gcode):
    req = urllib.request.Request(
        f"http://{host}/api/v1/files/usb/{name}", data=gcode, method="PUT",
        headers={"Content-Type":"text/x.gcode","Overwrite":"?1","Print-After-Upload":"?1"},
    )
    with opener.open(req, timeout=30) as r:
        return r.read().decode("utf-8","replace")[:200]


def status(opener, host):
    try:
        with opener.open(f"http://{host}/api/v1/status", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Printer IP or hostname")
    ap.add_argument("--user", default="maker")
    ap.add_argument("--password", required=True)
    ap.add_argument("--port", type=int, default=8514)
    ap.add_argument("--listen-s", type=float, default=12.0)
    args = ap.parse_args()

    op = opener_for(args.host, args.user, args.password)
    ip = my_ip(args.host)
    print(f"local IP: {ip}    printer: {args.host}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.2)
    counts = Counter()
    samples = {}  # name -> first 3 raw lines
    stop = threading.Event()
    t_first = {}
    t0 = time.monotonic()

    def loop():
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            now = time.monotonic() - t0
            for line in data.decode("utf-8","replace").splitlines():
                line = line.strip()
                if not line: continue
                name = line.split(" ",1)[0].split(",",1)[0]
                counts[name] += 1
                t_first.setdefault(name, now)
                if name in METRICS and len(samples.setdefault(name, [])) < 3:
                    samples[name].append(line)

    t = threading.Thread(target=loop, daemon=True); t.start()

    # build gcode -- no motion, just enable metrics.
    # NOTE: we deliberately do NOT emit M332 cleanup here. M332 persists across
    # runs and would silently undo a user's manually-enabled metric.
    L = [f"M334 {ip} {args.port}"]
    for m in METRICS:
        L.append(f"M331 {m}")
    L += ["M117 LC_CHECK", "G4 P3000"]  # 3s idle pause
    gcode = ("\n".join(L) + "\n").encode("utf-8")
    print("uploading + starting (no motion, just metric activation)...")
    print(upload_print(op, args.host, "_lc_check.gcode", gcode))

    print(f"listening for {args.listen_s} s ...")
    time.sleep(args.listen_s)
    stop.set(); t.join(timeout=2); sock.close()

    print("\n=== which loadcell-* metrics streamed? ===")
    for m in METRICS:
        n = counts.get(m, 0)
        first = t_first.get(m)
        if n == 0:
            print(f"  {m:24s}  ABSENT")
        else:
            print(f"  {m:24s}  n={n:5d}   first@{first:.2f}s")
            for s in samples.get(m, [])[:2]:
                print(f"      {s}")

    print("\n=== top 15 metrics by volume ===")
    for n, c in counts.most_common(15):
        print(f"  {n:24s}  {c}")


if __name__ == "__main__":
    main()
