"""Standalone UDP sniffer with optional PrusaLink M331/M332 control.

The go/no-go gate for the rest of the project. Stdlib only — no `pip install` needed.

Quick start (uses PrusaLink to enable metrics for you):

    python sniff.py --prusalink-host 192.168.2.X --api-key <KEY> --log capture.log

What that does:
    1. POSTs `M334 <this-ip> 8514` so the printer streams to your machine.
    2. POSTs `M331 <metric>` for each metric in --enable (defaults below).
    3. Listens on UDP 8514, prints per-metric Hz rates every few seconds.
    4. On Ctrl-C: POSTs `M332 <metric>` to disable each, and (optionally) `M334`
       with no args to stop the stream.

Default metrics enabled:
    probe_load_line, tmc_sg_e, tmc_sg_x, tmc_sg_y, tmc_sg_z, nozzle_pwm,
    loadcell_scale, loadcell_threshold, loadcell_threshold_cont, loadcell_hysteresis

Manual mode (no PrusaLink — toggle the metrics on the printer touchscreen yourself):

    python sniff.py --log capture.log
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict


DEFAULT_METRICS = [
    # openprinttag-branch continuous loadcell metrics -- these are the ones we
    # actually want for PA tuning. They're DISABLED by default in the firmware.
    "loadcell_value",
    "loadcell_hp",
    "loadcell_xy",
    "loadcell_age",
    "loadcell",
    # probe-event-only force metric (useful as a control / probe trigger check)
    "probe_load_line",
    # stepper diagnostics (mostly streams during motion; on Core One the extruder
    # SG returns 0 -- kept here for diagnostic purposes only)
    "tmc_sg_e",
    "nozzle_pwm",
]


def _local_ip_toward(host: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _make_opener(host: str, user: str | None, password: str | None,
                 api_key: str | None) -> urllib.request.OpenerDirector:
    """Build a urllib opener with HTTP Digest auth (Core One PrusaLink) or none.
    The api_key, if supplied, is added as X-Api-Key on each request via headers
    (not via this opener)."""
    if user and password:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, f"http://{host}", user, password)
        digest = urllib.request.HTTPDigestAuthHandler(mgr)
        basic = urllib.request.HTTPBasicAuthHandler(mgr)
        return urllib.request.build_opener(digest, basic)
    return urllib.request.build_opener()


def _send_gcode(opener: urllib.request.OpenerDirector, host: str,
                api_key: str | None, command: str,
                timeout: float = 5.0) -> tuple[bool, str]:
    """POST a G-code to PrusaLink. Tries v1 first then legacy. Supports either
    Digest auth (via the opener) or X-Api-Key header."""
    body = json.dumps({"command": command}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    last_err = ""
    for path in ("/api/v1/printer/command", "/api/printer/command"):
        url = f"http://{host}{path}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with opener.open(req, timeout=timeout) as resp:
                return True, f"{path} HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                last_err = f"{path} HTTP {e.code}"
                continue
            return False, f"{path} HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = f"{path} {e}"
            continue
    return False, last_err or "no PrusaLink command endpoint responded"


class TeeWriter:
    """Mirror prints to stdout AND a UTF-8 file (works around PowerShell's Tee-Object
    writing UTF-16 by default)."""

    def __init__(self, path: str | None):
        self.fh = open(path, "w", encoding="utf-8", newline="\n") if path else None

    def write(self, s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()
        if self.fh:
            self.fh.write(s)
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sniff Prusa UDP metrics.")
    parser.add_argument("--port", type=int, default=8514,
                        help="UDP port to bind (default 8514 -- Prusa stock metrics port)")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--filter", default="",
                        help="Only show metric names containing this substring")
    parser.add_argument("--summary-every", type=float, default=2.0,
                        help="Print a rate summary every N seconds (default 2.0)")
    parser.add_argument("--show-raw", action="store_true",
                        help="Print each raw line (very chatty)")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Mirror output to this file (UTF-8). Use this instead of "
                             "PowerShell `Tee-Object` (which writes UTF-16).")
    parser.add_argument("--prusalink-host", default=None, metavar="IP",
                        help="If set, send M334 + M331 over PrusaLink to enable metric streaming. "
                             "Requires --user/--password (Core One Digest auth) or --api-key.")
    parser.add_argument("--user", default="maker",
                        help="PrusaLink Digest-auth user (default 'maker' -- Core One).")
    parser.add_argument("--password", default=None,
                        help="PrusaLink Digest-auth password (shown on Settings -> Network -> "
                             "PrusaLink on the printer).")
    parser.add_argument("--api-key", default=None,
                        help="Legacy PrusaLink API key (X-Api-Key). Use --password on Core One.")
    parser.add_argument("--enable", action="append", default=None, metavar="METRIC",
                        help="Metric to enable via M331 (repeatable). "
                             f"Default: {','.join(DEFAULT_METRICS)}")
    parser.add_argument("--no-disable-on-exit", action="store_true",
                        help="Don't send M332/M334 cleanup commands on Ctrl-C.")
    args = parser.parse_args()

    out = TeeWriter(args.log)

    def say(s: str = "") -> None:
        out.write(s + "\n")

    # --- bind UDP ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.bind, args.port))
    except PermissionError as e:
        say(
            f"[sniff] cannot bind UDP {args.bind}:{args.port}  ({e})\n"
            "[sniff] On Windows this usually means:\n"
            "        1. Another process has the port (most likely a leftover python). Check:\n"
            f"             Get-NetUDPEndpoint -LocalPort {args.port}\n"
            "        2. The port is in a reserved range:\n"
            "             netsh int ipv4 show excludedportrange protocol=udp\n"
            f"        3. Firewall blocks inbound UDP on {args.port}.\n"
            "        Workaround: pass --port 9514 (or any free port) AND set Metrics Port to\n"
            "        the same value on the printer."
        )
        out.close()
        return 2
    except OSError as e:
        say(f"[sniff] bind failed: {e}")
        out.close()
        return 2
    sock.settimeout(0.5)

    say(f"[sniff] listening on {args.bind}:{args.port}  filter={args.filter!r}")
    if args.log:
        say(f"[sniff] mirroring to {args.log}")

    # --- optional PrusaLink control ---
    enabled_metrics: list[str] = []
    opener = _make_opener(
        args.prusalink_host or "",
        args.user if args.password else None,
        args.password,
        args.api_key,
    )
    if args.prusalink_host:
        if not (args.password or args.api_key):
            say("[sniff] --prusalink-host requires either --password (Core One Digest) or --api-key.")
            out.close()
            return 2
        my_ip = _local_ip_toward(args.prusalink_host)
        auth_kind = "Digest user=" + args.user if args.password else "X-Api-Key"
        say(f"[sniff] PrusaLink mode -- host={args.prusalink_host}  local_ip={my_ip}  auth={auth_kind}")

        m334 = f"M334 {my_ip} {args.port}"
        ok, info = _send_gcode(opener, args.prusalink_host, args.api_key, m334)
        say(f"[sniff]   -> {m334}    {'OK' if ok else 'FAIL'}  ({info})")
        if not ok:
            say("[sniff] PrusaLink unreachable or rejecting auth.")
            say("[sniff]   - Is 'Enabled' = ON in Settings -> Network -> PrusaLink ?")
            say("[sniff]   - Did you copy the password exactly?  (regen with 'Generate Password')")
            sock.close()
            out.close()
            return 3

        metrics = args.enable if args.enable else DEFAULT_METRICS
        for m in metrics:
            ok, info = _send_gcode(opener, args.prusalink_host, args.api_key, f"M331 {m}")
            say(f"[sniff]   -> M331 {m:<24}  {'OK' if ok else 'FAIL'}  ({info})")
            if ok:
                enabled_metrics.append(m)
    else:
        say("[sniff] manual mode — enable metrics yourself on the printer's touchscreen:")
        say(f"[sniff]   Settings -> Network -> Metrics & Log:  Host=<this-ip>  Port={args.port}  ON")
        say("[sniff]   Metrics List -> toggle the metrics you want to see.")

    say("[sniff] Ctrl-C to stop.")
    say()

    # --- listen ---
    counts: Counter[str] = Counter()
    last_summary = time.monotonic()
    last_seen: dict[str, float] = {}
    samples_window: dict[str, int] = defaultdict(int)

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                pass
            else:
                now = time.monotonic()
                for line in data.decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    head = line.split(" ", 1)[0]
                    name = head.split(",", 1)[0]
                    if args.filter and args.filter not in name:
                        continue
                    counts[name] += 1
                    samples_window[name] += 1
                    last_seen[name] = now
                    if args.show_raw:
                        say(f"[{addr[0]}] {line}")

            now = time.monotonic()
            if now - last_summary >= args.summary_every:
                elapsed = now - last_summary
                say(f"--- {now - last_summary:5.1f}s rates (Hz) ---")
                for name, n in sorted(samples_window.items(), key=lambda kv: -kv[1]):
                    age = now - last_seen[name]
                    rate = n / elapsed if elapsed > 0 else 0.0
                    flag = "  STREAMING" if rate > 5 else ""
                    say(f"  {name:30s}  {rate:7.1f} Hz   total={counts[name]:6d}   age={age:5.2f}s{flag}")
                if not samples_window:
                    say("  (no packets in window)")
                samples_window.clear()
                last_summary = now

    except KeyboardInterrupt:
        say()
        say("[sniff] stopped.")
        say("[sniff] Total counts:")
        for name, n in counts.most_common():
            say(f"  {name:30s}  {n}")
    finally:
        sock.close()
        if (args.prusalink_host and (args.password or args.api_key)
                and not args.no_disable_on_exit and enabled_metrics):
            say()
            say("[sniff] cleanup -- disabling metrics:")
            for m in enabled_metrics:
                ok, info = _send_gcode(opener, args.prusalink_host, args.api_key, f"M332 {m}")
                say(f"[sniff]   -> M332 {m:<24}  {'OK' if ok else 'FAIL'}  ({info})")
            ok, info = _send_gcode(opener, args.prusalink_host, args.api_key, "M334")
            say(f"[sniff]   -> M334 (stop stream)     {'OK' if ok else 'FAIL'}  ({info})")
        out.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
