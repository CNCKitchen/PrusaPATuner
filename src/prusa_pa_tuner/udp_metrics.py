"""Async UDP listener and InfluxDB line-protocol parser for Prusa metrics.

Prusa Buddy firmware emits one or more lines per UDP packet in (a customised flavour of)
InfluxDB line protocol:

    <name>[,tag=v[,tag=v...]] <field=v[,field=v...]> [timestamp_ns_or_us]

Custom-typed metrics put multiple fields after the space. Numeric fields may be int (suffix `i`),
float, bool (`t`/`f`), or string (quoted). Tags are always strings.

We intentionally keep this parser tolerant: malformed lines are logged and skipped, not raised,
because the firmware occasionally truncates packets and we don't want a single bad byte to kill
a tuning run.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MetricSample:
    name: str
    tags: dict[str, str]
    fields: dict[str, float | int | bool | str]
    # Timestamp from the printer if present (nanoseconds), else None.
    printer_ts_ns: int | None
    # Wall-clock receive time on this machine (monotonic seconds).
    recv_monotonic: float

    @property
    def value(self) -> float | int | bool | str | None:
        """Return the single 'v' field if present, else the first field's value."""
        if "v" in self.fields:
            return self.fields["v"]
        if self.fields:
            return next(iter(self.fields.values()))
        return None


def _parse_value(raw: str) -> float | int | bool | str:
    if not raw:
        return ""
    if raw[0] == '"' and raw[-1] == '"':
        # quoted string — InfluxDB-style escaping (\\\" inside)
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if raw in ("t", "T", "true", "True", "TRUE"):
        return True
    if raw in ("f", "F", "false", "False", "FALSE"):
        return False
    if raw.endswith("i") or raw.endswith("u"):
        # integer/unsigned suffix
        try:
            return int(raw[:-1])
        except ValueError:
            pass
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _split_unescaped(s: str, sep: str) -> list[str]:
    """Split on `sep`, respecting backslash escapes and quoted strings."""
    out: list[str] = []
    buf: list[str] = []
    i = 0
    in_quote = False
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(s[i : i + 2])
            i += 2
            continue
        if c == '"':
            in_quote = not in_quote
            buf.append(c)
            i += 1
            continue
        if c == sep and not in_quote:
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


def _unwrap_syslog(line: str) -> str:
    """Strip RFC 5424 syslog framing if present, return the inner payload.

    Buddy/Core One emits some metrics over the syslog UDP path rather than
    the raw metric path. When Settings -> Network -> Metrics & Log routes
    both to the same port (the common setup), our UDP listener receives
    lines shaped like:

        <14>1 - 10:9c:70:2b:7a:6b buddy - - - msg=51135,tm=2259318650,v=4 loadcell_value v=12.3 -3693

    The 7th space-delimited token starts the actual InfluxDB-line-protocol
    payload (here `loadcell_value v=12.3 -3693`).

    Returns:
      * the original line if there is no `<PRI>` prefix at all (raw metric
        line that needs no unwrap)
      * the inner payload if a complete 8-header-token wrapper is detected
      * an EMPTY string if the line LOOKS like syslog (starts with `<PRI>`)
        but the wrapper is truncated/malformed -- previously we returned
        the raw line in this case and `parse_line` then registered the
        priority prefix (e.g. `<14>1`) as a fake metric name, which polluted
        the diagnostics table and `/api/metrics_seen` output. Returning ""
        signals "skip this line entirely" via parse_line's empty-line guard.
    """
    if not line or line[0] != "<":
        return line
    close = line.find(">")
    if close < 0 or close > 5:
        return line
    # 5424 header: <PRI>VER SP TIMESTAMP SP HOSTNAME SP APPNAME SP PROCID
    # SP MSGID SP STRUCTURED-DATA SP MSG. Buddy emits:
    #   <14>1 - <mac> buddy - - - msg=N,tm=T,v=V <inner-influx-line>
    # That's 8 header tokens then the MSG. split(" ", 8) -> 9 parts, parts[8]
    # is the inner InfluxDB-line-protocol payload.
    parts = line.split(" ", 8)
    if len(parts) < 9:
        # Looked like a syslog wrapper but the header is incomplete.
        # Do NOT pass through -- the `<PRI>VER` prefix would otherwise be
        # parsed as the metric name.
        return ""
    return parts[8]


def parse_line(line: str, recv_monotonic: float | None = None) -> MetricSample | None:
    """Parse a single InfluxDB-line-protocol line. Returns None on malformed input."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if recv_monotonic is None:
        recv_monotonic = time.monotonic()

    # If this is a syslog-framed copy (Metrics Port == Syslog Port), strip
    # the syslog header so we see the actual metric payload.
    line = _unwrap_syslog(line)
    if not line:
        return None

    # First space separates "name+tags" from "fields[ ts]". But fields can contain quoted strings
    # with spaces, so we split on the FIRST unescaped/unquoted space.
    parts = _split_unescaped(line, " ")
    if len(parts) < 2:
        return None
    name_part = parts[0]
    fields_part = parts[1]
    ts_part = parts[2] if len(parts) > 2 else None

    # name + tags
    name_tokens = _split_unescaped(name_part, ",")
    name = name_tokens[0]
    tags: dict[str, str] = {}
    for tok in name_tokens[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        tags[k] = v

    # fields
    fields: dict[str, float | int | bool | str] = {}
    for tok in _split_unescaped(fields_part, ","):
        if "=" not in tok:
            # bare field — treat as "v=<value>"
            if tok:
                fields["v"] = _parse_value(tok)
            continue
        k, v = tok.split("=", 1)
        fields[k] = _parse_value(v)

    ts_ns: int | None = None
    if ts_part:
        try:
            ts_ns = int(ts_part)
        except ValueError:
            ts_ns = None

    return MetricSample(
        name=name,
        tags=tags,
        fields=fields,
        printer_ts_ns=ts_ns,
        recv_monotonic=recv_monotonic,
    )


class MetricStream:
    """Async UDP listener with per-metric fan-out queues."""

    def __init__(self, bind: str = "0.0.0.0", port: int = 8500, ring_size: int = 65536):
        self.bind = bind
        self.port = port
        self.ring_size = ring_size

        # per-metric ring buffers (history) — useful for grabbing recent N samples
        self._rings: dict[str, deque[MetricSample]] = defaultdict(
            lambda: deque(maxlen=self.ring_size)
        )
        # per-metric live subscribers (asyncio.Queue), fanned out by the receive loop
        self._subscribers: dict[str, list[asyncio.Queue[MetricSample]]] = defaultdict(list)
        # global subscribers (every sample)
        self._global_subs: list[asyncio.Queue[MetricSample]] = []

        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _UdpProtocol | None = None
        self._packets_received = 0
        self._malformed = 0
        # samples dropped because a subscriber's queue was full -- this is the
        # signal that the WebSocket / UI can't keep up with the incoming rate.
        # If this number is climbing during a run, the user will see gaps in
        # the live plot.
        self._dropped_backpressure = 0
        # Last UDP-receive time per metric name. The firmware buffers up to
        # ~1 KB per packet and may emit dozens of samples for the same
        # metric at once (we see ~26 samples/packet for loadcell_value at
        # 184 Hz throughput but only ~7 packets/sec). If every sample in a
        # batch were stamped with the packet's recv_monotonic, the live
        # plot would render as vertical clusters with horizontal gaps. We
        # use this dict to spread each batch uniformly back across the
        # interval since the previous packet of the same metric -- see
        # `_on_packet` for the assignment.
        self._last_metric_recv: dict[str, float] = {}
        # Last DISPATCHED sample timestamp per metric. Distinct from
        # `_last_metric_recv` (the previous PACKET's host arrival time)
        # because two consecutive packets can overlap in firmware time:
        # each anchors its newest sample at its own `recv`, but the
        # earlier samples spread back ~50 ms while the inter-packet gap
        # is ~30 ms -- so the second packet's earliest samples land
        # BEFORE the first packet's latest, producing out-of-order
        # timestamps in the per-metric stream. Observed on user's
        # run_1779015193.npz K=0.05 seg 1: a ~60 ms backward jump
        # right after the rising edge made plotly draw a "jumpback"
        # diagonal in the force trace. We enforce monotonicity here
        # by clipping each new sample's recv_monotonic to be strictly
        # greater than the previous dispatched sample for this metric.
        self._last_metric_sample_t: dict[str, float] = {}

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _UdpProtocol(self._on_packet),
            local_addr=(self.bind, self.port),
            allow_broadcast=True,
        )
        log.info("UDP listener bound on %s:%d", self.bind, self.port)

    def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        # close all subscriber queues
        for qs in self._subscribers.values():
            for q in qs:
                q.put_nowait(None)  # type: ignore[arg-type]
        for q in self._global_subs:
            q.put_nowait(None)  # type: ignore[arg-type]

    def _on_packet(self, data: bytes, _addr: tuple[str, int]) -> None:
        self._packets_received += 1
        recv = time.monotonic()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            self._malformed += 1
            return
        # Parse first, group by metric, THEN dispatch with spread timestamps.
        # This is what makes the live plot render as a continuous line
        # instead of vertical clusters every ~140 ms.
        by_name: dict[str, list[MetricSample]] = {}
        for line in text.splitlines():
            sample = parse_line(line, recv)
            if sample is None:
                if line.strip():
                    self._malformed += 1
                continue
            by_name.setdefault(sample.name, []).append(sample)
        for name, batch in by_name.items():
            last = self._last_metric_recv.get(name)
            self._last_metric_recv[name] = recv
            n = len(batch)
            if n == 1 or last is None or recv <= last:
                # Trivial case or first packet for this metric -- leave the
                # timestamp at recv.
                for s in batch:
                    self._dispatch_monotonic(name, s, s.recv_monotonic)
                continue
            # PREFERRED: use the firmware-emitted per-sample timestamps if
            # this batch carries them. Buddy puts a relative offset (signed
            # int, in microseconds back from "now-on-printer") at the end
            # of each InfluxDB line: `loadcell_value v=12.3 -3693` means
            # "this sample is 3.693 ms before the packet's emit instant".
            # When the offsets look sane (monotonically ordered, within
            # the packet's age) we anchor the newest sample to recv and
            # apply the per-sample deltas. This preserves the actual
            # firmware-side sampling cadence -- which is BURSTY at the
            # ADC-batch level, not uniform -- and stops the live plot
            # from looking stretched/compressed when the firmware emits a
            # batch unevenly.
            offsets_us = [s.printer_ts_ns for s in batch]
            if all(o is not None for o in offsets_us):
                arr = np.asarray(offsets_us, dtype=float) / 1e6  # us → s
                # All negative-or-zero, the most-recent (largest = closest
                # to 0) is the packet's "now" sample. Use that as the
                # anchor at recv; subtract its value to get per-sample
                # offsets back from the anchor.
                anchor_offset = float(arr.max())
                deltas = arr - anchor_offset  # all <= 0, in seconds
                # Two sanity gates before trusting the firmware-offset
                # spread:
                #   (a) total span < 2× inter-packet gap, OR < 500 ms
                #       (otherwise the timestamp unit is probably wrong
                #       and the spread would corrupt the timeline);
                #   (b) the resulting batch's EARLIEST sample lands
                #       AFTER the previous packet's host arrival time --
                #       otherwise the firmware-offset placement would
                #       overlap the previous packet's coverage region
                #       and we'd dispatch duplicate-time samples for
                #       different ADC readings. Observed on the user's
                #       run_1779015193.npz at K=0.05 seg 1: two
                #       consecutive packets each spread ~60 ms back,
                #       host inter-packet gap was ~15 ms, so packet B's
                #       earliest samples landed ~45 ms BEFORE packet A's
                #       latest -- plotly drew a backward diagonal on
                #       the rising-edge line. Falling back to uniform
                #       spread for the overlapping batch eliminates the
                #       overlap (uniform spans only (last, recv]).
                span_ok = (-float(deltas.min())) < max(2.0 * (recv - last), 0.5)
                earliest_assigned = recv + float(deltas.min())
                no_overlap = earliest_assigned >= last - 1e-6
                if span_ok and no_overlap:
                    for i, s in enumerate(batch):
                        self._dispatch_monotonic(name, s, recv + float(deltas[i]))
                    continue
            # FALLBACK: uniform spread across (last, recv]. Used when the
            # firmware didn't include timestamps (older builds, some
            # metric configs) or the timestamp span was unreasonable.
            span = recv - last
            step = span / n
            for i, s in enumerate(batch):
                self._dispatch_monotonic(name, s, recv - (n - 1 - i) * step)

    def _dispatch_monotonic(
        self, name: str, sample: MetricSample, assigned_t: float,
    ) -> None:
        """Stamp `sample.recv_monotonic` while enforcing strict monotonic
        order within this metric's stream.

        If `assigned_t` would be ≤ the previously dispatched sample's
        time (which happens when two consecutive packets cover
        overlapping firmware-time spans), bump it forward by a tiny
        epsilon so the per-metric stream stays strictly monotonic. This
        keeps the live plot and the analyser-side seg-windows from
        seeing a back-jump after the firmware-offset spread overlaps a
        prior packet's tail.
        """
        prev = self._last_metric_sample_t.get(name)
        if prev is not None and assigned_t <= prev:
            assigned_t = prev + 1e-6
        sample.recv_monotonic = assigned_t
        self._last_metric_sample_t[name] = assigned_t
        self._dispatch(sample)

    def _dispatch(self, sample: MetricSample) -> None:
        self._rings[sample.name].append(sample)
        for q in self._subscribers.get(sample.name, ()):
            # drop on overflow rather than block; this is telemetry
            if q.full():
                self._dropped_backpressure += 1
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(sample)
        for q in self._global_subs:
            if q.full():
                self._dropped_backpressure += 1
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(sample)

    async def subscribe(self, name: str, maxsize: int = 4096) -> AsyncIterator[MetricSample]:
        q: asyncio.Queue[MetricSample] = asyncio.Queue(maxsize=maxsize)
        self._subscribers[name].append(q)
        try:
            while True:
                item = await q.get()
                if item is None:  # sentinel on stop
                    return
                yield item
        finally:
            self._subscribers[name].remove(q)

    async def subscribe_all(self, maxsize: int = 4096) -> AsyncIterator[MetricSample]:
        q: asyncio.Queue[MetricSample] = asyncio.Queue(maxsize=maxsize)
        self._global_subs.append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            self._global_subs.remove(q)

    def snapshot(self, name: str) -> list[MetricSample]:
        """Return a snapshot copy of the ring buffer for a metric."""
        return list(self._rings.get(name, ()))

    def clear(self, name: str | None = None) -> None:
        if name is None:
            self._rings.clear()
        else:
            self._rings.pop(name, None)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "packets": self._packets_received,
            "malformed_lines": self._malformed,
            "metrics_seen": len(self._rings),
            "samples_total": sum(len(r) for r in self._rings.values()),
            "dropped_backpressure": self._dropped_backpressure,
        }

    def metric_rates(self, window_s: float = 5.0) -> dict[str, float]:
        """Per-metric samples/sec over the last `window_s` seconds.

        Computed by walking each ring buffer from newest backwards until a
        sample older than `window_s` is hit. Counts the samples in that
        slice and divides by the actual elapsed time between oldest-in-
        window and now -- so a metric that just started streaming reports
        its true instantaneous rate, not an under-estimate dragged down by
        the empty earlier window.

        Use this in the UI to verify the printer is actually emitting at
        the rates you expect: e.g. loadcell_value should be ~100 Hz, and
        if it's reading 10 Hz the firmware throttle is dominating.
        """
        now = time.monotonic()
        cutoff = now - window_s
        out: dict[str, float] = {}
        for name, ring in self._rings.items():
            if not ring:
                continue
            count = 0
            oldest_t: float | None = None
            # deque supports reversed() in O(1) per step
            for s in reversed(ring):
                if s.recv_monotonic < cutoff:
                    break
                count += 1
                oldest_t = s.recv_monotonic
            if count < 2 or oldest_t is None:
                continue
            span = max(now - oldest_t, 1e-9)
            out[name] = count / span
        return out


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_packet):
        self.on_packet = on_packet

    def datagram_received(self, data, addr):  # noqa: D401
        self.on_packet(data, addr)

    def error_received(self, exc):
        log.warning("UDP error: %s", exc)
