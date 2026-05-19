# Prusa PA Tuner

Automatic **Pressure Advance (`M572 S`)** calibration for **Prusa printers with a nozzle
loadcell** (Core One, MK4 / MK4S, MK3.9, XL), using the loadcell as a back-pressure
sensor — no extra hardware, no printed test patches, no eyeballing of corner artefacts.

> **Status:** experimental. Works on current stock Buddy firmware — see
> [Step 0](#step-0--activate-the-metrics-on-the-printer) for how to enable the metrics
> stream the tool relies on.

![Results — composite step-response cost across K, with live re-weightable metrics and per-metric K_opt agreement](docs/bd_pressure_results.png)

## Why this exists

Pressure Advance is one of the biggest knobs for clean fast prints — it's what stops
corners blobbing and seams over-extruding when the toolhead decelerates into a turn. On
the Core One's direct-drive hotend (and on the MK4-family heads), getting K right means
you can push print speeds without paying for it in corner quality. The conventional
workflow — print a pattern, squint at corners, pick a number — is slow and subjective.

Three projects showed PA can be tuned objectively from a single sensor, without printing
anything:

- **Bambu Lab A1 / A1 mini** — first consumer printer to ship automatic PA calibration
  using its nozzle force sensor: extrude in air, measure back-pressure, sweep K, pick
  the cleanest step response. The whole idea below is downstream of theirs. Bambu hasn't
  open-sourced the algorithm, but the *approach* is what we're reproducing.
- **Snapmaker U1** ([`u1-klipper/flow_calibrator.py`](https://github.com/Snapmaker/u1-klipper/blob/main/klippy/extras/flow_calibrator.py)) —
  open-source implementation using an inductance coil to read filament-diameter changes
  as a back-pressure proxy, then sweeps K with a slow/fast square-wave extrusion in air.
- **markniu's `bd_pressure`** ([github.com/markniu/bd_pressure](https://github.com/markniu/bd_pressure)) —
  open-source strain-gauge module that reads extruder back-pressure directly during a
  planned accel/decel sweep and picks the K that minimises the step-response transient.

Every modern Prusa with a nozzle loadcell already has the sensor needed to do the same
trick. The nozzle assembly is mechanically decoupled from the extruder, so back-pressure
transmitted to it shows up as a Z-force on the cell even with the nozzle in air. This
project combines U1's motion geometry with bd_pressure's step-response analysis, running
on the printer's stock loadcell over UDP.

## Supported printers

- **Prusa Core One** — primary development target.
- **Prusa MK4 / MK4S / MK3.9** — same loadcell, same Buddy firmware family. Should work
  with no changes; community testing very welcome.
- **Prusa XL** — has the loadcell. Multi-tool sequencing is not handled yet, so you'll
  have to manually pick a tool and sweep PA per material.

If your Prusa runs current Buddy firmware and has loadcell-based first-layer
calibration, it has what this tool needs.

## How it works

1. The tool generates a single G-code job that sweeps K (`M572 S<k>`) across a
   configurable grid. At each K it extrudes an asymmetric slow/fast/slow square-wave in
   air at a safe parking position, with a small XY wiggle so Buddy actually applies PA
   (firmware only honours `M572` on print moves with XY motion — pure E moves are
   ignored).
2. A pre-burst Z-pulse marker writes an unambiguous timestamp into the loadcell stream
   so the analyser can align planned segments with measured force.
3. Buddy streams `loadcell_hp` (highpass-filtered force) and `loadcell_value` (raw tared
   grams) over UDP as InfluxDB line-protocol metrics while the job runs. See
   [`Prusa-Firmware-Buddy/doc/metrics.md`](https://github.com/prusa3d/Prusa-Firmware-Buddy/blob/master/doc/metrics.md)
   for the protocol.
4. Three analyses run side-by-side on the captured timeseries:
   - **bd_pressure-style step response** — per-segment overshoot, undershoot, settling,
     plateau slope; composite cost minimised across K.
   - **Phase-lag (cross-correlation)** — time-domain shift between commanded velocity
     and measured force. Optimal K is where the lag crosses zero (Ellis 3DP framing:
     under → force lags, over → force leads).
   - **Integral-area (U1 style)** — integrated centred force across each velocity
     transition; linear fit area-vs-K, solve for zero.
5. The web UI plots every intermediate signal (raw force, segments, per-K metrics,
   composite cost) so you can sanity-check before trusting the number.

## Requirements

- A Prusa printer with a nozzle loadcell and current Buddy firmware (see
  [Supported printers](#supported-printers)).
- **PrusaLink** enabled with Digest credentials (user `maker` plus the auto-generated
  password under Settings → Network → PrusaLink on the touchscreen).
- **Wired Ethernet strongly recommended.** Prusa's onboard WiFi is fragile — packets get
  dropped under load and runs come back corrupted. If you absolutely must use WiFi,
  **point the printer's antenna directly at your access point** and keep them close.
  A LAN cable solves it outright.
- Host machine on the **same subnet** as the printer, with a firewall rule that permits
  **inbound UDP/8514**. Buddy refuses to stream if it can't reach you.
- **Python 3.11+** on the host.

## Install

```bash
git clone https://github.com/CNCKitchen/PrusaPATuner
cd PrusaPATuner
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -e .
```

## Step 0 — activate the metrics on the printer

Buddy's metrics subsystem needs two things to stream:

1. **A host to stream to.** On the printer touchscreen, open the **Metrics & Log**
   settings (Settings → System → Metrics & Log on most Buddy builds), enter the **IP of
   the computer running this tool**, and port `8514`. `prusa-pa-tuner` prints the LAN
   IP it thinks the printer should target on startup — use that.

2. **The specific metrics enabled.** In the same menu, enable **only** the four metrics
   this tool needs:
   - `loadcell_hp`
   - `loadcell_value`
   - `loadcell_xy`
   - `loadcell_age`

   Enable nothing else. The metrics throttle on Buddy is compile-time and every extra
   active stream eats into the ~100 Hz loadcell budget — extra metrics directly cause
   UDP drops and corrupted runs. Fewer is faster.

> **Use the touchscreen menu, not `M331` over PrusaLink.** The g-code path silently
> no-ops on unknown metric names and the success/fail status isn't returned to
> PrusaLink, so you can't tell what actually got enabled. The touchscreen menu is the
> only confirmed-reliable path.

> **Metric activation is not persistent.** It resets on every power cycle. Re-enable the
> four metrics at the start of each tuning session.

After enabling, run `sniff.py` from the host to confirm packets are actually arriving:

```powershell
python sniff.py --prusalink-host <printer-IP> --password <PrusaLink-password> --log capture.log
```

You should see `loadcell_hp`, `loadcell_value`, `loadcell_xy`, `loadcell_age` ticking
at ~100 Hz each. If it's silent, check (in order):

- the printer's Metrics & Log host IP matches your host's LAN address,
- inbound UDP/8514 is allowed by the host firewall,
- you're on the same subnet,
- the four metrics are still enabled (they reset on power cycle — see above).

Full protocol reference: [`Prusa-Firmware-Buddy/doc/metrics.md`](https://github.com/prusa3d/Prusa-Firmware-Buddy/blob/master/doc/metrics.md).

## Run the app

```bash
prusa-pa-tuner
# or
python -m prusa_pa_tuner
```

A browser opens at <http://127.0.0.1:8765/>.

1. **Printer** — enter the printer's IP and PrusaLink password.
2. **Filament & temps** — material label, nozzle temp, bed temp.
3. **Sweep parameters** — defaults match bd_pressure's granularity:
   K from 0.00 to 0.10 in 0.002 steps, slow 1.92 mm³/s × 1.0 s, fast 19.24 mm³/s ×
   0.25 s, 14 cycles per K, accel 5000 mm/s². Tighten the K range once you know roughly
   where your filament lands.
4. **Start tuning run.** The app will:
   - Detect this host's LAN IP toward the printer.
   - Generate the sweep G-code (Preview button shows it).
   - Upload via PrusaLink, start the print.
   - Stream loadcell timeseries live to the chart.
   - Analyse on job-end and surface K_opt from all three algorithms.
5. **Copy `M572 S…`** to paste into your slicer's filament profile.

Raw run data is saved to `runs/run_<timestamp>.npz` (gitignored by default). Use
`replay_run.py` to re-analyse old runs without re-printing.

## Algorithm notes

The "tune PA from the nozzle force sensor" idea originates with **Bambu Lab's A1
series** — their stock auto-calibration was the first consumer demonstration. The
algorithm isn't open, but the geometry and analysis pieces below are.

The square-wave-in-air motion is taken from
[Snapmaker U1](https://github.com/Snapmaker/u1-klipper/blob/main/klippy/extras/flow_calibrator.py):
their flow calibrator uses an inductance coil to read filament-diameter changes, which
are physically modulated by extruder back-pressure. On a Prusa loadcell, the same
back-pressure is transmitted through the decoupled nozzle assembly and shows up as a
Z-force — the same dynamic signature in a different sensor domain (force vs diameter).

The step-response decomposition (rising-edge overshoot, plateau slope, falling-edge
undershoot, settling, etc.) is adapted from
[markniu's `bd_pressure`](https://github.com/markniu/bd_pressure). bd_pressure ships its
own strain-gauge sensor and runs the analysis in real time on a Klipper host; we run the
same conceptual analysis offline on the captured UDP timeseries.

![BD_PRESSURE segment browser — step through every low-high-low burst and see the 8 region metrics extracted from it (baseline, rising edge, overshoot, plateau, plateau creep, falling edge, undershoot, recovery tail)](docs/bd_pressure_segments.png)

The segment browser above is a debug view of one cycle at K=0.05: you can scrub through
every K and every segment, and see exactly which numbers the analyser pulled out of each
burst. This is the "show your work" surface that makes K_opt verifiable instead of a
black box — if a K looks wrong, click into its segments and find out why.

We use `loadcell_hp` as the primary input because its on-firmware highpass strips the
~1500–2200 g static baseline (tool weight + sensor drift), leaving just the grams-scale
dynamic component PA actually modulates. If `loadcell_hp` is silent the runner falls
back to `loadcell_value` with software detrending.

Physics intuition (under-comp K → force lags command, optimal K → in phase, over-comp K →
force leads) follows the
[Ellis 3DP Pressure/Linear Advance guide](https://ellis3dp.com/Print-Tuning-Guide/articles/pressure_linear_advance/introduction.html).

We run all three algorithms because it isn't yet established which one gives the most
trustworthy answer on real Prusa loadcell data. Open issues with `runs/*.npz` attached
welcomed.

## Project layout

```
src/prusa_pa_tuner/
  app.py            # FastAPI app + websocket runner
  config.py         # persistent user config (lives outside the repo)
  gcode_gen.py      # generates the K-sweep .gcode
  prusalink.py      # PrusaLink REST client (Digest + legacy X-Api-Key)
  udp_metrics.py    # async UDP listener + line-protocol parser
  analysis.py       # bd_pressure + phase-lag + U1 integral-area
  runner.py         # end-to-end orchestration
  netutil.py        # detect local IP toward the printer
  replay.py         # offline replay of saved runs/*.npz
  static/           # web UI (vanilla HTML + Plotly + JS)
sniff.py            # standalone UDP sniffer (go/no-go gate)
replay_run.py       # CLI: replay a saved run through the analyser
diagnose.py         # one-shot extrude + UDP capture for triage
check_loadcell.py   # PrusaLink metric-availability probe
examine_sg.py       # stationary-extrude force capture
tests/              # pytest
```

User-specific config (printer IP, PrusaLink password) is stored *outside the repo* at
`%APPDATA%/PrusaPATuner/config.json` on Windows or `~/.prusa_pa_tuner/config.json`
elsewhere, and is loaded/saved by the web UI. Nothing printer-identifying is committed
back into the working tree.

## License

Released under the **GNU Affero General Public License v3.0 or later** — see
[`LICENSE`](LICENSE).

The AGPL choice is deliberate: if you run a modified version of this tool as part of a
hosted service (e.g. a cloud calibration service for someone else's printer), you must
make your modifications available to that service's users.

## Acknowledgements

- **Bambu Lab A1 / A1 mini** — original consumer-printer demonstration of nozzle-
  force-based automatic PA calibration; the conceptual starting point for everything
  below.
- [Snapmaker U1 Klipper](https://github.com/Snapmaker/u1-klipper) — flow calibrator
  motion geometry and integral-area math.
- [markniu/bd_pressure](https://github.com/markniu/bd_pressure) — step-response
  decomposition and composite-cost K selection.
- [Ellis 3DP Print Tuning Guide](https://ellis3dp.com/Print-Tuning-Guide/articles/pressure_linear_advance/introduction.html)
  — physics framing and PA intuition.
- [Prusa firmware-specific G-code reference](https://help.prusa3d.com/article/prusa-firmware-specific-g-code-commands_112173)
  — canonical source for what Buddy actually accepts.
- [Prusa-Firmware-Buddy `doc/metrics.md`](https://github.com/prusa3d/Prusa-Firmware-Buddy/blob/master/doc/metrics.md)
  — UDP metrics protocol and the canonical list of available metric names.
