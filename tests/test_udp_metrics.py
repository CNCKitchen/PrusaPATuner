import asyncio

import pytest

from prusa_pa_tuner.udp_metrics import (
    FirmwareClockMapper,
    MetricStream,
    parse_line,
)


def test_simple_value():
    s = parse_line("temp_bed v=58.7 1700000000000")
    assert s is not None
    assert s.name == "temp_bed"
    assert s.fields == {"v": 58.7}
    assert s.printer_ts_ns == 1700000000000
    assert s.value == 58.7


def test_integer_suffix():
    s = parse_line("cpu_usage v=42i")
    assert s.fields == {"v": 42}
    assert isinstance(s.fields["v"], int)


def test_tags_and_multi_field():
    s = parse_line('temp_noz,n=0,a=1 t=210.3,target=215.0 1700')
    assert s.name == "temp_noz"
    assert s.tags == {"n": "0", "a": "1"}
    assert s.fields == {"t": 210.3, "target": 215.0}


def test_quoted_string_with_space():
    s = parse_line('print_filename v="my file.gcode"')
    assert s.fields["v"] == "my file.gcode"


def test_bool_field():
    s = parse_line("is_printing v=t")
    assert s.fields["v"] is True


def test_no_timestamp():
    s = parse_line("fan,id=0 rpm=2400i")
    assert s.printer_ts_ns is None
    assert s.fields["rpm"] == 2400


def test_empty_returns_none():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("# comment") is None


def test_malformed_returns_none():
    assert parse_line("not_a_metric_no_fields") is None


def test_syslog_wrapped_metric_is_unwrapped():
    """When Settings -> Network -> Metrics Port == Syslog Port (common
    Core One setup), STRING-type metrics (gcode, fw_version, ...) arrive
    wrapped in RFC 5424 syslog framing. We must strip the header so the
    inner InfluxDB-line-protocol payload becomes the real metric."""
    # Numeric metric arriving via syslog priority 14 (informational).
    line = (
        "<14>1 - 10:9c:70:2b:7a:6b buddy - - - "
        "msg=51136,tm=2259369059,v=4 loadcell_value v=-17933.166016 -4191"
    )
    s = parse_line(line)
    assert s is not None
    assert s.name == "loadcell_value"
    assert s.fields["v"] == -17933.166016

    # String-type gcode metric arriving via syslog priority 12.
    line = (
        '<12>1 - 10:9c:70:2b:7a:6b buddy - - - '
        'msg=99,tm=12345,v=4 gcode v="G1 E2.8000 F48.0" 1234567'
    )
    s = parse_line(line)
    assert s is not None
    assert s.name == "gcode"
    assert s.fields["v"] == "G1 E2.8000 F48.0"


def test_non_syslog_lines_unchanged():
    """Plain InfluxDB lines (no <PRI> prefix) must still parse normally."""
    s = parse_line("loadcell_value v=-17930.996094 -3693")
    assert s is not None
    assert s.name == "loadcell_value"
    assert s.fields["v"] == -17930.996094


def test_packet_batch_uses_firmware_timestamps_when_present():
    """Buddy emits per-sample relative timestamps at the end of each
    InfluxDB line (signed µs back from "now-on-printer"). When the
    whole batch in a packet carries these, the dispatcher must apply
    the per-sample deltas instead of uniform-spreading the batch
    across (last_recv, recv]. This preserves the firmware's actual
    sampling cadence -- which is BURSTY at the ADC-batch level, not
    uniform -- so the live plot doesn't look stretched/compressed
    when the firmware emits a batch unevenly.

    Packet contains 3 loadcell samples at firmware-relative offsets
    -10000, -5000, 0 µs (i.e. 10 ms, 5 ms, 0 ms ago). After dispatch
    the spacing between consecutive samples should be exactly 5 ms
    apart -- not (recv - last) / 3 as the legacy uniform-spread would
    produce.
    """
    stream = MetricStream(port=0)
    captured: list[float] = []

    # Fake the time function so recv is deterministic.
    import prusa_pa_tuner.udp_metrics as udp_mod

    original_monotonic = udp_mod.time.monotonic
    times = iter([100.0, 100.5])  # first packet at 100.0, second at 100.5
    udp_mod.time.monotonic = lambda: next(times)
    try:
        # First packet: seed `_last_metric_recv` for the metric so the
        # n>1 branch is exercised on the second packet.
        stream._on_packet(b"loadcell_value v=1.0 0", ("test", 0))
        # Second packet: 3 samples with firmware-relative offsets -10000,
        # -5000, 0 µs (in batch-emit order: oldest first, newest last).
        packet = b"\n".join([
            b"loadcell_value v=10.0 -10000",
            b"loadcell_value v=20.0 -5000",
            b"loadcell_value v=30.0 0",
        ])
        # Hook the dispatcher to capture recv_monotonic per sample.
        original_dispatch = stream._dispatch
        def capture(sample):
            captured.append(sample.recv_monotonic)
            return original_dispatch(sample)
        stream._dispatch = capture
        stream._on_packet(packet, ("test", 0))
    finally:
        udp_mod.time.monotonic = original_monotonic

    assert len(captured) == 3, f"expected 3 dispatched samples, got {len(captured)}"
    # The newest sample (last in batch, offset 0) anchors at recv = 100.5
    assert captured[-1] == pytest.approx(100.5, abs=1e-6)
    # The middle sample (offset -5000 µs from anchor) lands 5 ms earlier
    assert captured[-2] == pytest.approx(100.495, abs=1e-6)
    # The oldest (offset -10000 µs) lands 10 ms earlier
    assert captured[-3] == pytest.approx(100.490, abs=1e-6)


def test_packet_batch_falls_back_to_uniform_when_no_timestamps():
    """When the firmware doesn't include per-sample timestamps the
    dispatcher must fall back to uniform spread across (last, recv].
    Verifies the back-compat path still works on builds that don't
    emit timestamps.
    """
    stream = MetricStream(port=0)
    captured: list[float] = []

    import prusa_pa_tuner.udp_metrics as udp_mod
    original_monotonic = udp_mod.time.monotonic
    times = iter([100.0, 100.6])
    udp_mod.time.monotonic = lambda: next(times)
    try:
        stream._on_packet(b"fan rpm=1000i", ("t", 0))
        # No timestamp suffix on any line:
        packet = b"\n".join([
            b"fan rpm=2000i",
            b"fan rpm=2100i",
            b"fan rpm=2200i",
        ])
        original_dispatch = stream._dispatch
        def capture(sample):
            captured.append(sample.recv_monotonic)
            return original_dispatch(sample)
        stream._dispatch = capture
        stream._on_packet(packet, ("t", 0))
    finally:
        udp_mod.time.monotonic = original_monotonic

    assert len(captured) == 3
    # Uniform spread: step = (100.6 - 100.0) / 3 = 0.2; samples at
    # 100.2, 100.4, 100.6
    assert captured[-1] == pytest.approx(100.6, abs=1e-6)
    assert captured[-2] == pytest.approx(100.4, abs=1e-6)
    assert captured[-3] == pytest.approx(100.2, abs=1e-6)


def test_packet_overlap_falls_back_to_uniform_to_keep_monotonic():
    """When the firmware-offset spread would place the new packet's
    earliest sample BEFORE the previous packet's host arrival time
    (because host inter-packet gap < firmware batch span), the
    dispatcher must fall back to uniform spread to avoid out-of-order
    timestamps in the per-metric stream.

    Regression for run_1779015193.npz K=0.05 seg 1: two consecutive
    packets each carried ~60ms-span batches but arrived only ~15ms
    apart, so the second packet's firmware-offset-anchored samples
    landed BEFORE the first packet's latest sample. Plotly drew a
    backwards diagonal on the rising-edge line because samples were
    not monotonic.
    """
    stream = MetricStream(port=0)
    captured: list[float] = []

    import prusa_pa_tuner.udp_metrics as udp_mod
    original_monotonic = udp_mod.time.monotonic
    # Packet A at 100.0, packet B at 100.015 (15ms gap) but each
    # batch's firmware-offset span is 60ms. Without the overlap gate,
    # packet B's earliest sample would land at 100.015 − 0.060 = 99.955,
    # WAY before packet A's earliest at 100.000 − 0.060 = 99.940. Wait,
    # actually packet A is the first packet for this metric so its
    # samples sit at 100.0 (the trivial-case branch). We need a prior
    # packet to seed _last_metric_recv. Use 3 packets: A (seed at 99.95),
    # B at 100.0 with 60ms span, C at 100.015 with 60ms span.
    times = iter([99.95, 100.0, 100.015])
    udp_mod.time.monotonic = lambda: next(times)
    try:
        # Seed: first packet, single sample. Sets _last_metric_recv to 99.95.
        stream._on_packet(b"loadcell_value v=1.0 0", ("t", 0))
        original_dispatch = stream._dispatch
        def capture(sample):
            captured.append(sample.recv_monotonic)
            return original_dispatch(sample)
        stream._dispatch = capture
        # Packet B at recv=100.0, 3 samples spanning 60ms (offsets
        # −60000, −30000, 0 µs). Firmware-offset assignment places
        # them at 99.940, 99.970, 100.000.
        packet_b = b"\n".join([
            b"loadcell_value v=10.0 -60000",
            b"loadcell_value v=20.0 -30000",
            b"loadcell_value v=30.0 0",
        ])
        stream._on_packet(packet_b, ("t", 0))
        # Packet C at recv=100.015, 3 samples spanning 60ms. The
        # firmware-offset assignment would place earliest at 99.955
        # which is BEFORE packet B's latest at 100.000 -- this is the
        # bug. With the overlap gate, the dispatcher falls back to
        # uniform spread across (100.0, 100.015].
        packet_c = b"\n".join([
            b"loadcell_value v=40.0 -60000",
            b"loadcell_value v=50.0 -30000",
            b"loadcell_value v=60.0 0",
        ])
        stream._on_packet(packet_c, ("t", 0))
    finally:
        udp_mod.time.monotonic = original_monotonic

    # 3 samples for packet B + 3 for packet C = 6 captured
    assert len(captured) == 6
    # All captured timestamps must be strictly monotonic.
    for i in range(1, len(captured)):
        assert captured[i] > captured[i - 1], (
            f"sample[{i}]={captured[i]} not strictly > "
            f"sample[{i-1}]={captured[i-1]} -- the overlap gate "
            f"did not catch the back-jump"
        )
    # Packet B samples: still get firmware-offset spread (no overlap
    # with the seed packet 99.95, since 100.0 - 60ms = 99.94 ≥ 99.95
    # is false! Actually 99.94 < 99.95 — so packet B ALSO overlaps the
    # seed. Uniform spread it is for B: (99.95, 100.0] / 3 → 99.967, 99.983, 100.0.
    # Packet C samples: also uniform across (100.0, 100.015] / 3 →
    # 100.005, 100.010, 100.015.
    # The exact values depend on the spread formula; the strict
    # monotonicity test above is what actually matters.


def test_syslog_header_meta_is_captured():
    """msg= (packet sequence) and tm= (firmware clock) from the syslog
    header must ride along on the parsed sample -- they are the raw
    material for firmware-timebase reconstruction and deterministic
    packet-loss detection."""
    line = (
        "<14>1 - 10:9c:70:2b:7a:6b buddy - - - "
        "msg=51136,tm=2259369059,v=4 loadcell_value v=-17933.166016 -4191"
    )
    s = parse_line(line)
    assert s is not None
    assert s.fw_seq == 51136
    assert s.fw_tm == 2259369059
    # Raw (unwrapped) lines carry no header meta.
    s2 = parse_line("loadcell_value v=1.0 -100")
    assert s2.fw_seq is None
    assert s2.fw_tm is None


def test_fw_clock_mapper_calibrates_microseconds():
    """Feed anchors whose tm advances 1e6 ticks per host second -- the
    mapper must detect the µs unit and map tm to host time within the
    jitter envelope (offset = sliding min of recv - tm*scale)."""
    m = FirmwareClockMapper()
    base_tm = 2_000_000_000
    # 30 packets over 3 s, with +0..5 ms of simulated network jitter.
    jitter = [0.005, 0.001, 0.003, 0.0, 0.002] * 6
    mapped = None
    for i in range(30):
        tm = base_tm + i * 100_000  # 100 ms of firmware time per packet
        recv = 500.0 + i * 0.1 + jitter[i]
        mapped = m.add_anchor(tm, recv)
    assert m.state.startswith("calibrated")
    assert "1e-06" in m.state
    # The zero-jitter anchor (recv delay 0.0) pins the offset, so the
    # final mapped time equals the jitter-free arrival time.
    assert mapped == pytest.approx(500.0 + 29 * 0.1, abs=1e-6)


def test_fw_clock_mapper_rejects_unknown_unit():
    """A tm that advances at a rate matching no known unit (here 3x
    faster than µs) must disable the mapper after the give-up span,
    so dispatch falls back to host-arrival timestamps."""
    m = FirmwareClockMapper()
    for i in range(200):
        m.add_anchor(1000 + i * 300_000, 500.0 + i * 0.1)
    assert m.state == "disabled"
    assert m.map(999_999.0) is None


def test_fw_clock_mapper_unwraps_32bit_rollover():
    """A µs counter wraps every ~71.6 min. When tm jumps backwards by
    more than half the 32-bit range, the mapper must unwrap instead of
    resetting, and mapped time must stay continuous."""
    m = FirmwareClockMapper()
    wrap = 2 ** 32
    start = wrap - 15 * 100_000  # 15 packets before rollover
    recv0 = 500.0
    t_mid = None
    t_last = None
    for i in range(30):
        tm = (start + i * 100_000) % wrap  # wraps at i == 15
        t = m.add_anchor(tm, recv0 + i * 0.1)
        if i == 25:
            t_mid = t
        if i == 29:
            t_last = t
    # Calibration completes at i>=20 (MIN_PAIRS + 2 s span), which is
    # AFTER the rollover at i==15 -- so a successful calibration here
    # proves the unwrap kept the tm sequence linear across the wrap.
    assert m.state.startswith("calibrated")
    assert t_mid is not None and t_last is not None
    # Mapped time keeps advancing at exactly 100 ms per packet across
    # the post-wrap region.
    assert t_last - t_mid == pytest.approx(0.4, abs=1e-6)


def test_packet_with_fw_clock_uses_firmware_timebase():
    """Once the mapper is calibrated, sample timestamps must come from
    tm= + per-line offsets: two packets whose HOST arrival jitters but
    whose firmware clock is perfectly regular must produce regularly
    spaced samples (the whole point of the firmware timebase)."""
    stream = MetricStream(port=0)
    captured: list[float] = []

    import prusa_pa_tuner.udp_metrics as udp_mod
    original_monotonic = udp_mod.time.monotonic

    def syslog(seq: int, tm: int, name: str, val: float, off: int) -> bytes:
        return (
            f"<14>1 - mac buddy - - - msg={seq},tm={tm},v=4 "
            f"{name} v={val} {off}"
        ).encode()

    try:
        # Calibration phase: 30 packets 100 ms apart with 0..5 ms of
        # network delay (at least one zero-delay packet pins the
        # offset's sliding minimum at the true clock offset). The two
        # measurement packets then arrive with +20 ms and +30 ms delay
        # -- delays are always >= 0, matching physical reality.
        jitter = [0.005, 0.001, 0.003, 0.0, 0.002] * 6
        t_host = iter(
            [100.0 + i * 0.1 + jitter[i] for i in range(30)]
            + [103.02, 103.13]
        )
        udp_mod.time.monotonic = lambda: next(t_host)
        base_tm = 50_000_000
        for i in range(30):
            stream._on_packet(
                syslog(i, base_tm + i * 100_000, "loadcell_value", 1.0, 0),
                ("t", 0),
            )
        assert stream._fw_clock.state.startswith("calibrated")

        original_dispatch = stream._dispatch
        def capture(sample):
            captured.append(sample.recv_monotonic)
            return original_dispatch(sample)
        stream._dispatch = capture

        # Packet A (arrives +20 ms) and B (arrives +30 ms): the host
        # sees them 110 ms apart, but the firmware clock says exactly
        # 100 ms apart, each with 2 samples 5 ms apart.
        tm_a = base_tm + 30 * 100_000
        tm_b = tm_a + 100_000
        stream._on_packet(
            syslog(30, tm_a, "loadcell_value", 2.0, -5000)
            + b"\nloadcell_value v=3.0 0",
            ("t", 0),
        )
        stream._on_packet(
            syslog(31, tm_b, "loadcell_value", 4.0, -5000)
            + b"\nloadcell_value v=5.0 0",
            ("t", 0),
        )
    finally:
        udp_mod.time.monotonic = original_monotonic

    assert len(captured) == 4
    # Intra-packet spacing: exactly 5 ms.
    assert captured[1] - captured[0] == pytest.approx(0.005, abs=1e-6)
    assert captured[3] - captured[2] == pytest.approx(0.005, abs=1e-6)
    # Inter-packet spacing follows the FIRMWARE clock (100 ms), not the
    # jittered host arrivals (60 ms apart).
    assert captured[2] - captured[0] == pytest.approx(0.100, abs=1e-6)


def test_seq_gap_counts_lost_packets():
    """msg= jumping from 10 to 14 means exactly 3 packets were lost."""
    stream = MetricStream(port=0)
    import prusa_pa_tuner.udp_metrics as udp_mod
    original_monotonic = udp_mod.time.monotonic
    t_host = iter([100.0, 100.1, 100.2])
    udp_mod.time.monotonic = lambda: next(t_host)
    try:
        for seq in (10, 14, 15):
            stream._on_packet(
                f"<14>1 - mac buddy - - - msg={seq},tm={seq * 1000},v=4 "
                f"fan rpm=100i".encode(),
                ("t", 0),
            )
    finally:
        udp_mod.time.monotonic = original_monotonic
    assert stream.stats["packets_lost"] == 3
    events = stream.loss_events()
    assert len(events) == 1
    assert events[0] == (100.1, 3)


def test_malformed_syslog_does_not_leak_priority_as_metric_name():
    """The earlier _unwrap_syslog returned the raw line on incomplete
    wrappers; parse_line then registered `<14>1` as a fake metric name and
    polluted the diagnostics table / metrics_seen output. The fix is to
    drop these lines entirely. Several real-world malformations to cover:
    truncated headers, missing structured-data, header-only with no MSG.
    """
    # Header only, no MSG payload at all (8 tokens instead of 9):
    assert parse_line("<14>1 - host buddy - - - msg=1,tm=2,v=4") is None
    # Truncated mid-header:
    assert parse_line("<14>1 - host buddy") is None
    # Just the priority prefix and nothing else:
    assert parse_line("<14>1") is None
    # Whitespace-only after priority:
    assert parse_line("<14>1   ") is None
