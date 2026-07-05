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
    # Syslog-header metadata, when the line arrived RFC 5424-wrapped
    # (Metrics Port == Syslog Port setup). `fw_seq` is the firmware's
    # per-packet message counter (`msg=N`) -- gaps mean lost packets.
    # `fw_tm` is the firmware clock at packet emit (`tm=T`, unit
    # auto-detected by FirmwareClockMapper). Both None on raw lines.
    fw_seq: int | None = None
    fw_tm: int | None = None

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


def _parse_syslog_meta(token: str) -> tuple[int | None, int | None]:
    """Parse `msg=51135,tm=2259318650,v=4` (syslog header token 7) into
    (seq, tm). Either may be None when missing/unparseable."""
    seq: int | None = None
    tm: int | None = None
    for part in token.split(","):
        if part.startswith("msg="):
            try:
                seq = int(part[4:])
            except ValueError:
                pass
        elif part.startswith("tm="):
            try:
                tm = int(part[3:])
            except ValueError:
                pass
    return seq, tm


def _unwrap_syslog(line: str) -> tuple[str, int | None, int | None]:
    """Strip RFC 5424 syslog framing if present, return the inner payload
    plus the header's (msg=seq, tm=firmware-time) metadata.

    Buddy/Core One emits some metrics over the syslog UDP path rather than
    the raw metric path. When Settings -> Network -> Metrics & Log routes
    both to the same port (the common setup), our UDP listener receives
    lines shaped like:

        <14>1 - 10:9c:70:2b:7a:6b buddy - - - msg=51135,tm=2259318650,v=4 loadcell_value v=12.3 -3693

    The 7th space-delimited token starts the actual InfluxDB-line-protocol
    payload (here `loadcell_value v=12.3 -3693`). The token before it
    (`msg=51135,tm=2259318650,v=4`) carries the firmware's packet
    sequence number and emit-time clock -- both load-bearing for the
    firmware-timebase reconstruction, so they are parsed and returned
    instead of discarded.

    Returns `(payload, seq, tm)`:
      * `(line, None, None)` if there is no `<PRI>` prefix at all (raw
        metric line that needs no unwrap)
      * `(inner_payload, seq, tm)` if a complete 8-header-token wrapper
        is detected
      * `("", None, None)` if the line LOOKS like syslog (starts with
        `<PRI>`) but the wrapper is truncated/malformed -- previously we
        returned the raw line in this case and `parse_line` then
        registered the priority prefix (e.g. `<14>1`) as a fake metric
        name, which polluted the diagnostics table and
        `/api/metrics_seen` output. Returning "" signals "skip this line
        entirely" via parse_line's empty-line guard.
    """
    if not line or line[0] != "<":
        return line, None, None
    close = line.find(">")
    if close < 0 or close > 5:
        return line, None, None
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
        return "", None, None
    seq, tm = _parse_syslog_meta(parts[7])
    return parts[8], seq, tm


def parse_line(line: str, recv_monotonic: float | None = None) -> MetricSample | None:
    """Parse a single InfluxDB-line-protocol line. Returns None on malformed input."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if recv_monotonic is None:
        recv_monotonic = time.monotonic()

    # If this is a syslog-framed copy (Metrics Port == Syslog Port), strip
    # the syslog header so we see the actual metric payload. The header's
    # msg=/tm= metadata rides along on the sample.
    line, fw_seq, fw_tm = _unwrap_syslog(line)
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
        fw_seq=fw_seq,
        fw_tm=fw_tm,
    )


class FirmwareClockMapper:
    """Map the firmware's `tm=` clock to host monotonic time.

    Why: every downstream alignment problem (window slicing, pos_x vs
    loadcell cross-stream sync) stems from timestamps reconstructed
    from host UDP arrival times, which carry Wi-Fi jitter and packet
    batching. The firmware's own clock has none of that -- two samples
    1 ms apart in firmware time really were 1 ms apart. This mapper
    turns raw `tm` ticks into host-monotonic seconds so all metric
    streams share one consistent, jitter-free timebase.

    Method:
      * The tick unit is auto-detected (s / ms / µs / ns) by comparing
        the tm span against the host-clock span over a calibration
        window. If no candidate unit matches, the mapper disables
        itself and callers fall back to host-arrival timestamps.
      * The offset is the sliding-window MINIMUM of
        `recv - tm * scale`: network delay is always >= the minimum
        observed delay, so the min tracks the true offset from below.
        The window (default 512 packets, ~1 min at Buddy's ~7 pkt/s)
        also absorbs crystal drift between printer and host
        (~1e-5 -> well under 1 ms over the window).
      * 32-bit wraparound (a µs counter wraps every ~71.6 min --
        within one long sweep) is unwrapped when tm jumps backwards by
        more than half the 32-bit range; a smaller backward jump means
        the printer rebooted and the mapper resets.
    """

    SCALE_CANDIDATES = (1.0, 1e-3, 1e-6, 1e-9)  # seconds per tick
    SCALE_TOLERANCE = 0.25  # accept ratio within ±25% of a candidate
    MIN_PAIRS = 20
    MIN_SPAN_S = 2.0
    GIVE_UP_SPAN_S = 15.0  # no candidate matched after this long → disable
    WRAP = 2 ** 32

    def __init__(self, window: int = 512):
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)
        self._scale: float | None = None  # None=calibrating, 0.0=disabled
        self._last_raw_tm: int | None = None
        self._wrap_offset: int = 0

    @property
    def state(self) -> str:
        if self._scale is None:
            return "calibrating"
        if self._scale == 0.0:
            return "disabled"
        return f"calibrated (1 tick = {self._scale:g} s)"

    def reset(self) -> None:
        self._pairs.clear()
        self._scale = None
        self._last_raw_tm = None
        self._wrap_offset = 0

    def add_anchor(self, tm: int, recv: float) -> float | None:
        """Feed one (firmware tm, host recv) pair; returns the mapped
        host-monotonic emit time, or None while uncalibrated/disabled."""
        if self._last_raw_tm is not None and tm < self._last_raw_tm:
            back = self._last_raw_tm - tm
            if back > self.WRAP // 2:
                # Counter wrapped (uint32 µs wraps every ~71.6 min).
                self._wrap_offset += self.WRAP
            else:
                # Firmware clock went backwards without wrapping --
                # printer reboot. Start over.
                log.warning(
                    "firmware clock went backwards (%d -> %d); "
                    "printer reboot assumed, resetting clock mapper",
                    self._last_raw_tm, tm,
                )
                self.reset()
        self._last_raw_tm = tm
        tmu = float(tm + self._wrap_offset)
        self._pairs.append((tmu, recv))
        if self._scale is None:
            self._try_calibrate()
        return self.map(tmu)

    def _try_calibrate(self) -> None:
        if len(self._pairs) < self.MIN_PAIRS:
            return
        tm0, r0 = self._pairs[0]
        tm1, r1 = self._pairs[-1]
        host_span = r1 - r0
        if host_span < self.MIN_SPAN_S or tm1 <= tm0:
            return
        ratio = host_span / (tm1 - tm0)
        for cand in self.SCALE_CANDIDATES:
            if abs(ratio / cand - 1.0) <= self.SCALE_TOLERANCE:
                self._scale = cand
                log.info(
                    "firmware clock calibrated: 1 tick = %g s "
                    "(ratio %.3g over %.1f s / %d packets)",
                    cand, ratio, host_span, len(self._pairs),
                )
                return
        if host_span > self.GIVE_UP_SPAN_S:
            self._scale = 0.0
            log.warning(
                "firmware clock unit unrecognised (ratio %.3g s/tick); "
                "falling back to host-arrival timestamps", ratio,
            )

    def map(self, tm_unwrapped: float) -> float | None:
        """Unwrapped firmware ticks → host monotonic seconds."""
        if not self._scale:
            return None
        s = self._scale
        offset = min(r - t * s for t, r in self._pairs)
        return tm_unwrapped * s + offset


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
        # Firmware timebase reconstruction. When lines arrive syslog-
        # wrapped (Metrics Port == Syslog Port), each packet carries
        # msg=<seq> and tm=<firmware clock>. The mapper converts tm to
        # host-monotonic time so sample timestamps come from the
        # firmware's jitter-free clock instead of UDP arrival times.
        self._fw_clock = FirmwareClockMapper()
        self._last_fw_seq: int | None = None
        # Packet-loss accounting from msg= sequence gaps. Deterministic,
        # unlike the analyser's time-gap heuristics. `_loss_events`
        # holds (host_monotonic, n_lost) so runners can dump them
        # alongside the sample streams.
        self._packets_lost = 0
        self._loss_events: deque[tuple[float, int]] = deque(maxlen=4096)

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
        packet_seq: int | None = None
        packet_tm: int | None = None
        for line in text.splitlines():
            sample = parse_line(line, recv)
            if sample is None:
                if line.strip():
                    self._malformed += 1
                continue
            if packet_seq is None and sample.fw_seq is not None:
                packet_seq = sample.fw_seq
            if sample.fw_tm is not None:
                # If each line carries its own header, keep the newest tm
                # (= the packet's emit instant); if only the first line is
                # wrapped, this is simply that line's tm.
                packet_tm = (
                    sample.fw_tm if packet_tm is None
                    else max(packet_tm, sample.fw_tm)
                )
            by_name.setdefault(sample.name, []).append(sample)

        # Sequence-gap accounting: msg= increments once per packet, so a
        # jump of N+1 means exactly N packets were lost on the network.
        if packet_seq is not None:
            if self._last_fw_seq is not None:
                gap = packet_seq - self._last_fw_seq - 1
                if 0 < gap < 10_000:
                    self._packets_lost += gap
                    self._loss_events.append((recv, gap))
                elif packet_seq < self._last_fw_seq:
                    # Counter went backwards -- printer reboot / counter
                    # reset. Don't count it as loss.
                    log.info(
                        "metric seq counter reset (%d -> %d)",
                        self._last_fw_seq, packet_seq,
                    )
            self._last_fw_seq = packet_seq

        # Firmware-clock emit time for this packet. Once the mapper is
        # calibrated this is the preferred timestamp anchor: it removes
        # host-arrival jitter entirely and keeps ALL metric streams on
        # one mutually-consistent timebase.
        fw_emit_t: float | None = None
        if packet_tm is not None:
            fw_emit_t = self._fw_clock.add_anchor(packet_tm, recv)
            # Sanity: the mapped emit time must not sit in the future
            # of the packet's arrival (delay is non-negative) nor
            # implausibly far in the past. Outside that band the
            # calibration is off -- fall back to arrival-anchored paths.
            if fw_emit_t is not None and not (
                recv - 5.0 <= fw_emit_t <= recv + 0.05
            ):
                fw_emit_t = None

        for name, batch in by_name.items():
            last = self._last_metric_recv.get(name)
            self._last_metric_recv[name] = recv
            n = len(batch)
            # PREFERRED: firmware-clock timestamps. Anchor at the mapped
            # emit instant and apply each line's µs offset. Unlike the
            # recv-anchored paths below this is jitter-free AND
            # consistent across metrics: a loadcell sample and a pos_x
            # sample with the same firmware time get the same host
            # timestamp, which is what the analyser's cross-stream
            # alignment ultimately relies on.
            if fw_emit_t is not None:
                for s in batch:
                    off_us = s.printer_ts_ns  # µs offset despite the name
                    t = (
                        fw_emit_t + float(off_us) / 1e6
                        if off_us is not None
                        else fw_emit_t
                    )
                    self._dispatch_monotonic(name, s, t)
                continue
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
    def stats(self) -> dict[str, int | str]:
        return {
            "packets": self._packets_received,
            "malformed_lines": self._malformed,
            "metrics_seen": len(self._rings),
            "samples_total": sum(len(r) for r in self._rings.values()),
            "dropped_backpressure": self._dropped_backpressure,
            # Deterministic packet loss from msg= sequence gaps -- the
            # ground truth the analyser's time-gap heuristics only
            # approximate. Non-zero during a run means real UDP loss.
            "packets_lost": self._packets_lost,
            "fw_clock": self._fw_clock.state,
        }

    def loss_events(self) -> list[tuple[float, int]]:
        """(host_monotonic, n_packets_lost) for every msg= sequence gap
        seen so far. Runners dump these next to the sample streams so
        offline analysis can exclude windows around real packet loss."""
        return list(self._loss_events)

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
