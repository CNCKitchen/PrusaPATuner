// Plain JS — no build step. Talks to the FastAPI backend.

const FIELDS = [
  "printer_host", "printer_user", "printer_password", "printer_api_key", "udp_port",
  "filament_label", "nozzle_temp", "preheat_temp", "nozzle_diameter", "filament_diameter",
  "slow_flow_mm3_s", "slow_volume_mm3", "fast_flow_mm3_s", "fast_volume_mm3",
  "cycles_per_K", "accel_mm_s2",
  "k_min", "k_max", "k_step",
  "purge_x", "purge_y", "purge_z",
  "coupled_dx_mm", "coupled_dy_mm", "coupled_dz_mm",
  "first_slow_leg_factor",
];

function $(id) { return document.getElementById(id); }

function readForm() {
  const cfg = {};
  for (const f of FIELDS) {
    const el = $(f);
    if (!el) continue;
    const v = el.value;
    if (el.type === "number") cfg[f] = parseFloat(v);
    else cfg[f] = v;
  }
  return cfg;
}

function writeForm(cfg) {
  for (const f of FIELDS) {
    if (cfg[f] === undefined) continue;
    const el = $(f);
    if (el) el.value = cfg[f];
  }
}

async function loadConfig() {
  const r = await fetch("/api/config");
  if (r.ok) writeForm(await r.json());
}

async function saveConfig() {
  const cfg = readForm();
  const r = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (!r.ok) {
    alert("Save failed: " + (await r.text()));
    return;
  }
  flash($("btn_save"), "Saved");
}

async function previewGcode() {
  await saveConfig();
  window.open("/api/preview", "_blank");
}

async function startRun() {
  await saveConfig();
  const r = await fetch("/api/run", { method: "POST" });
  if (!r.ok) alert("Start failed: " + (await r.text()));
}

async function cancelRun() {
  await fetch("/api/cancel", { method: "POST" });
}

// Track the previous packet count + timestamp so we can compute a true
// packets/sec rate (the server only exposes the running total).
const diagState = { lastPkts: null, lastT: null };

async function refreshDiagnostics() {
  let data;
  try {
    const r = await fetch("/api/diagnostics?window_s=5");
    if (!r.ok) return;
    data = await r.json();
  } catch (_) { return; }
  const s = data.stats || {};
  const rates = data.rates_hz || {};
  const now = performance.now() / 1000;
  let pktRate = 0;
  if (diagState.lastPkts !== null && diagState.lastT !== null) {
    const dp = (s.packets || 0) - diagState.lastPkts;
    const dt = now - diagState.lastT;
    if (dt > 0.1) pktRate = dp / dt;
  }
  diagState.lastPkts = s.packets || 0;
  diagState.lastT = now;

  $("diag_pkt_rate").textContent = pktRate ? pktRate.toFixed(1) : "—";
  $("diag_pkts").textContent = (s.packets ?? "—");
  $("diag_dropped").textContent = (s.dropped_backpressure ?? "—");
  $("diag_malformed").textContent = (s.malformed_lines ?? "—");
  $("diag_n_metrics").textContent = (s.metrics_seen ?? "—");

  // Sort metrics by rate descending; ties broken by total samples.
  const names = Object.keys(rates);
  // The /api/metrics_seen endpoint also surfaces names without rate; pull
  // them too so the table shows "0 Hz" rows for metrics that ARE arriving
  // but were emitted fewer than two times in the window.
  try {
    const r2 = await fetch("/api/metrics_seen");
    if (r2.ok) {
      const seen = await r2.json();
      for (const n of Object.keys(seen.names || {})) {
        if (!(n in rates)) rates[n] = 0;
      }
    }
  } catch (_) {}

  const tbody = $("diag_rates_body");
  if (!tbody) return;
  const all = Object.keys(rates).map((name) => [name, rates[name]]);
  all.sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
  tbody.innerHTML = all
    .map(([name, hz]) => {
      const color = hz > 50 ? "#2ea043" : hz > 5 ? "#d29922" : "#7d8590";
      return `<tr>
        <td style="padding:3px 8px;color:#e6edf3;">${name}</td>
        <td style="padding:3px 8px;text-align:right;color:${color};">${hz.toFixed(1)}</td>
        <td style="padding:3px 8px;text-align:right;color:#7d8590;">—</td>
      </tr>`;
    })
    .join("");
}

let diagTimer = null;
function startDiagnosticsPoll() {
  if (diagTimer !== null) return;
  refreshDiagnostics();
  diagTimer = setInterval(refreshDiagnostics, 1000);
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

// ---- live plot ----
const live = {
  // Loadcell trace (left y-axis).
  t: [],
  y: [],
  // pos_x DIRECTION trace (right y-axis): ±1 latched on the sign of
  // (pos_x - prev_pos_x). +1 = X currently moving toward (purge_x + dx),
  // -1 = moving back toward purge_x. The user uses this as a clean
  // square-wave phase reference -- comparing the rising/falling edges of
  // this square wave against loadcell peaks tells them the PA-induced
  // phase shift at a glance. Plotting raw pos_x instead would show ramps
  // and is harder to read off cycle-by-cycle.
  posT: [],
  posY: [],
  // Time-windowed buffer (not sample-count-windowed). Both traces drop
  // samples older than (latest_t - windowSeconds). Without this, the
  // slower pos_x stream covers far more time than the loadcell stream
  // at the same sample budget and the x-axis stretches unevenly.
  windowSeconds: 20.0,
  initialized: false,
  // X-axis epoch. recv_monotonic is the Python process's monotonic clock
  // (seconds-since-boot, can be ~10⁵ s); subtract this so the axis starts
  // at 0 on page open and resets to 0 every time a run starts.
  t0: null,
  // Last accepted loadcell timestamp; used to drop out-of-order samples.
  lastT: -Infinity,
  // Direction-latch state for the pos_x → square-wave derivation.
  posLastValue: null,
  posLastDir: 0,
  // Significant-step threshold (mm). Smaller deltas are treated as noise
  // and the latch holds its prior direction.
  posDirEps: 0.005,
};

function resetLive() {
  live.t = [];
  live.y = [];
  live.posT = [];
  live.posY = [];
  live.t0 = null;
  live.lastT = -Infinity;
  live.posLastValue = null;
  live.posLastDir = 0;
}

function pruneLive() {
  // Time-window both traces to live.windowSeconds. Cutoff is anchored to
  // whichever trace has the more recent sample so a slow stream doesn't
  // anchor the window in the distant past.
  const lastLoad = live.t.length ? live.t[live.t.length - 1] : -Infinity;
  const lastPos = live.posT.length ? live.posT[live.posT.length - 1] : -Infinity;
  const last = Math.max(lastLoad, lastPos);
  if (!Number.isFinite(last)) return;
  const cutoff = last - live.windowSeconds;
  let i = 0;
  while (i < live.t.length && live.t[i] < cutoff) i++;
  if (i > 0) { live.t = live.t.slice(i); live.y = live.y.slice(i); }
  let j = 0;
  while (j < live.posT.length && live.posT[j] < cutoff) j++;
  if (j > 0) { live.posT = live.posT.slice(j); live.posY = live.posY.slice(j); }
}

function pushLive(t, v) {
  if (v === null || v === undefined) return;
  if (live.t0 === null) live.t0 = t;
  const rel = t - live.t0;
  // Drop samples that arrive earlier than the latest one (UDP re-ordering).
  if (rel < live.lastT) return;
  live.t.push(rel);
  live.y.push(v);
  live.lastT = rel;
  pruneLive();
}

function pushPos(t, v) {
  if (v === null || v === undefined) return;
  if (live.t0 === null) live.t0 = t;
  const rel = t - live.t0;
  // Latch a ±1 direction from the sign of (v - prev). Hold the prior
  // value when the step is below the noise threshold, so brief samples
  // at intermediate values during a transition don't toggle the latch.
  let dir = live.posLastDir;
  if (live.posLastValue !== null) {
    const delta = v - live.posLastValue;
    if (delta > live.posDirEps) dir = 1;
    else if (delta < -live.posDirEps) dir = -1;
  }
  live.posLastValue = v;
  live.posLastDir = dir;
  live.posT.push(rel);
  live.posY.push(dir);
  pruneLive();
}

function renderLive() {
  if (!live.t.length && !live.posT.length) return;
  const data = [
    {
      x: live.t, y: live.y,
      type: "scattergl", mode: "lines",
      line: { color: "#f7931e" },
      connectgaps: true,
      name: "loadcell",
      yaxis: "y",
    },
    {
      x: live.posT, y: live.posY,
      type: "scattergl", mode: "lines",
      line: { color: "#58a6ff", width: 1, shape: "hv" },
      connectgaps: true,
      name: "dir(pos_x)",
      yaxis: "y2",
    },
  ];
  const layout = {
    margin: { l: 50, r: 55, t: 10, b: 30 },
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "t (s, from open / last run start)" },
    yaxis: {
      gridcolor: "#2a2f37",
      title: { text: "loadcell", font: { color: "#f7931e" } },
      tickfont: { color: "#f7931e" },
    },
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      title: { text: "dir(pos_x)", font: { color: "#58a6ff" } },
      tickfont: { color: "#58a6ff" },
      range: [-1.5, 1.5],
      tickvals: [-1, 0, 1],
      ticktext: ["−X", "0", "+X"],
    },
    showlegend: false,
  };
  if (!live.initialized) {
    Plotly.newPlot("live_plot", data, layout, { displayModeBar: false, responsive: true });
    live.initialized = true;
  } else {
    Plotly.react("live_plot", data, layout);
  }
}

let lastRender = 0;
function maybeRender() {
  const now = performance.now();
  if (now - lastRender > 250) {
    renderLive();
    lastRender = now;
  }
}

// ---- bd_pressure segment browser ----
// State for the two-pane viewer. `bdState.analysis` is whatever was last
// loaded (live sweep OR a replayed npz). `selectedK` is the K row in the
// left navigator; `selectedSegIdx` is the segment index inside that K.
// The right pane re-renders any time either of those changes.
const bdState = {
  analysis: null,        // SweepAnalysis dict from the API
  windowsByK: {},        // {k: KWindow} keyed by k for quick lookup
  segmentsByK: {},       // {k: [BdSegment, ...]}
  selectedK: null,
  selectedSegIdx: 0,
  weights: {},           // current cost weights (mirror of slider state)
};

// Region color scheme — matches the bd_pressure reference image so the
// stats sidebar swatches and the on-plot shading make instant mental
// sense to anyone who's read the writeup.
const REGION_COLORS = {
  R1: "#58a6ff",  // baseline (blue)
  R2: "#3fb950",  // rising edge (green)
  R3: "#a371f7",  // overshoot (purple)
  R4: "#d29922",  // high plateau (yellow)
  R5: "#f7931e",  // creep (orange)
  R6: "#d9534f",  // falling edge (red)
  R7: "#db61a2",  // undershoot (pink)
  R8: "#39c5cf",  // recovery (cyan)
};

// Map region → which metrics belong to it (mirrors BD_METRIC_NAMES on
// the Python side). Used by the stats sidebar to group numbers.
const REGION_METRICS = {
  R1: ["baseline_median", "baseline_noise_std"],
  R2: ["rise_delay", "rise_error_area", "rise_slope"],
  R3: ["overshoot"],
  R4: ["high_level"],
  R5: ["plateau_slope", "plateau_creep"],
  R6: ["fall_delay", "fall_error_area"],
  R7: ["undershoot"],
  R8: ["tail_area", "settling_time"],
};

const REGION_TITLES = {
  R1: "1. low baseline",
  R2: "2. rising edge",
  R3: "3. overshoot",
  R4: "4. high plateau",
  R5: "5. plateau creep",
  R6: "6. falling edge",
  R7: "7. undershoot",
  R8: "8. recovery tail",
};

function fmt(v, digits = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) {
    return v.toExponential(2);
  }
  return v.toFixed(digits);
}

function downsampleXY(xs, ys, maxPoints) {
  const n = Math.min(xs.length, ys.length);
  if (n <= maxPoints) return [xs.slice(0, n), ys.slice(0, n)];
  const stride = Math.ceil(n / maxPoints);
  const outX = [];
  const outY = [];
  for (let i = 0; i < n; i += stride) {
    outX.push(xs[i]);
    outY.push(ys[i]);
  }
  return [outX, outY];
}

// ---- segment browser: K navigator (left pane) ----
function renderBdKList() {
  const list = $("bd_k_list");
  if (!list) return;
  list.innerHTML = "";
  if (!bdState.analysis || !bdState.analysis.bd_per_k) {
    list.innerHTML = '<div class="meta" style="padding:10px;">no run loaded</div>';
    return;
  }
  const kOpt = bdState.analysis.bd_k_opt;
  for (const r of bdState.analysis.bd_per_k) {
    const row = document.createElement("div");
    row.className = "bd-k-row";
    if (bdState.selectedK !== null && Math.abs(r.k - bdState.selectedK) < 1e-6) {
      row.classList.add("active");
    }
    if (kOpt !== null && kOpt !== undefined
        && Number.isFinite(kOpt)
        && Math.abs(r.k - kOpt) < 0.0026) {
      row.classList.add("k-opt");
    }
    // Color-code seg count: green ≥ 11, yellow 4..10, red < 4
    let segClass = "red";
    if (r.n_segments_included >= 11) segClass = "green";
    else if (r.n_segments_included >= 4) segClass = "yellow";
    row.innerHTML = `
      <span class="k">${r.k.toFixed(4)}</span>
      <span class="segs ${segClass}">${r.n_segments_included}/${r.n_segments_total}</span>
    `;
    row.onclick = () => {
      bdState.selectedK = r.k;
      bdState.selectedSegIdx = 0;
      renderBdKList();
      renderBdSegment();
    };
    list.appendChild(row);
  }
}

function _segmentsForSelectedK() {
  if (bdState.selectedK === null) return [];
  return (bdState.segmentsByK[bdState.selectedK] || []);
}

// ---- segment browser: single segment plot + stats (right pane) ----
function renderBdSegment() {
  const label = $("bd_segment_label");
  const plot = $("bd_segment_plot");
  const stats = $("bd_segment_stats");
  const banner = $("bd_excluded_banner");
  if (!label || !plot || !stats || !banner) return;
  if (bdState.selectedK === null) {
    label.textContent = "select a K on the left";
    Plotly.purge(plot);
    stats.innerHTML = "";
    banner.style.display = "none";
    return;
  }
  const segs = _segmentsForSelectedK();
  if (!segs.length) {
    label.textContent = `K=${bdState.selectedK.toFixed(4)}: no segments produced`;
    Plotly.purge(plot);
    stats.innerHTML = "";
    banner.style.display = "none";
    return;
  }
  const idx = Math.max(0, Math.min(bdState.selectedSegIdx, segs.length - 1));
  bdState.selectedSegIdx = idx;
  const seg = segs[idx];
  const window_ = bdState.windowsByK[bdState.selectedK];
  label.textContent = `K=${seg.k.toFixed(4)} · segment ${seg.seg_idx + 1}/${segs.length}`;
  if (seg.excluded) {
    banner.textContent = "EXCLUDED: " + (seg.exclusion_reasons || []).join("; ");
    banner.style.display = "";
  } else {
    banner.style.display = "none";
  }
  _drawBdSegmentPlot(seg, window_);
  _renderBdSegmentStats(seg);
}

function _drawBdSegmentPlot(seg, window_) {
  const plot = $("bd_segment_plot");
  if (!plot) return;
  // Pull the force trace from the per-K KWindow and slice to this
  // segment's [t_start, t_end] (sweep-rel times).
  let tFull = window_ && Array.isArray(window_.t) ? window_.t : [];
  let yFull = window_ && Array.isArray(window_.force) ? window_.force : [];
  let dropoutFull =
    window_ && Array.isArray(window_.dropout_t) ? window_.dropout_t : [];

  // Slice using binary-search. Use the server-computed display crop
  // [t_lo_display, t_hi_display] (inset from [t_start, t_end] by ~10%
  // of slow_half on each side) so the plot doesn't show neighbour-
  // cycle artifacts at the shared boundary. Falls back to t_start /
  // t_end when the segment was returned by an early-exit path that
  // didn't fill the display fields.
  const tDispLo = Number.isFinite(seg.t_lo_display) && seg.t_lo_display > 0
    ? seg.t_lo_display
    : seg.t_start;
  const tDispHi = Number.isFinite(seg.t_hi_display) && seg.t_hi_display > 0
    ? seg.t_hi_display
    : seg.t_end;
  const lo = _lower_bound(tFull, tDispLo);
  const hi = _lower_bound(tFull, tDispHi);
  // tFull[hi] is the first sample >= tDispHi; exclude it (slice end
  // exclusive) so a wide firmware-throttle gap can't drag a
  // next-cycle fast-leg sample into this plot.
  const tSeg = tFull.slice(lo, hi);
  const ySeg = yFull.slice(lo, hi);
  // segment-relative time so prev/next stepping keeps the x-axis aligned
  const t0 = seg.t_start;
  const t = tSeg.map((x) => x - t0);
  const yRaw = ySeg.slice();
  const [tF, yF] = downsampleXY(t, yRaw, 1500);

  const baseline = seg.metrics.baseline_median;
  const high = seg.metrics.high_level;
  const tRiseR = seg.t_rise - t0;
  const tFallR = seg.t_fall - t0;
  // Region boundaries: prefer the threshold-based t_rise_end / t_fall_start
  // / t_fall_end which mark where force crosses 90% / 10% of the leg
  // delta. Fall back to the argmax-based t_peak / t_trough only when
  // threshold detection failed (very low SNR or no fast-leg plateau).
  const tRiseEndR = seg.t_rise_end !== null && seg.t_rise_end !== undefined
    ? seg.t_rise_end - t0
    : (seg.t_peak !== null ? seg.t_peak - t0 : null);
  // R4 plateau / R6 fall boundary -- use the DETECTED fall start when
  // available (the actual force often begins falling slightly before
  // the commanded t_fall due to PA lag). Falls back to the commanded
  // t_fall when not detected.
  const tFallStartR = seg.t_fall_start !== null && seg.t_fall_start !== undefined
    ? seg.t_fall_start - t0
    : (seg.t_fall - t0);
  const tFallEndR = seg.t_fall_end !== null && seg.t_fall_end !== undefined
    ? seg.t_fall_end - t0
    : (seg.t_trough !== null ? seg.t_trough - t0 : null);
  // Peak / trough markers (R3, R7) keep their own location for the
  // overshoot/undershoot annotation pins.
  const tPeakR = seg.t_peak !== null ? seg.t_peak - t0 : null;
  const tTroughR = seg.t_trough !== null ? seg.t_trough - t0 : null;

  const traces = [
    {
      x: tF, y: yF,
      type: "scatter", mode: "lines+markers",
      line: { color: "#f7931e", width: 1.6 },
      marker: { color: "#f7931e", size: 3 },
      name: "force",
      hovertemplate: "t=%{x:.3f}s<br>y=%{y:.1f}<extra></extra>",
    },
  ];

  const showTransitions = $("bd_overlay_transitions").checked;
  const showLevels = $("bd_overlay_levels").checked;
  const showPeaks = $("bd_overlay_peaks").checked;
  const showRegions = $("bd_overlay_regions").checked;
  const showAreas = $("bd_overlay_areas").checked;
  const showSlope = $("bd_overlay_slope").checked;
  const showLabels = $("bd_overlay_labels").checked;

  const shapes = [];
  const annotations = [];

  if (showRegions) {
    // Shade R1..R8. Region boundaries follow the threshold-based
    // t_rise_end (90% of delta) and t_fall_end (10% of delta) — these
    // mark the END of the actual rising/falling transition, NOT the
    // argmax which can land deep into the creeping plateau and make
    // R2 swallow the whole plateau or R6 swallow the recovery.
    const fast = tFallR - tRiseR;
    const lowNext = (t[t.length - 1] || 0) - tFallR;
    const r2End = tRiseEndR !== null ? tRiseEndR : tRiseR + 0.1 * fast;
    // R4/R6 boundary: prefer the threshold-detected fall start; the
    // commanded t_fall is the fallback. This shifts the boundary by
    // a few tens of ms to match the actual force fall (PA lag), so
    // the early fall transient lives in R6 (where it belongs) instead
    // of corrupting R4 (plateau) and the rise_error_area.
    const r4End = tFallStartR;
    const r6End = tFallEndR !== null ? tFallEndR : tFallR + 0.1 * lowNext;
    const r3X = tPeakR !== null ? tPeakR : r2End;
    const r7X = tTroughR !== null ? tTroughR : r6End;
    const regionBands = [
      { id: "R1", x0: 0, x1: tRiseR, alpha: 0.10 },
      { id: "R2", x0: tRiseR, x1: r2End, alpha: 0.12 },
      { id: "R3", x0: r3X - 0.005, x1: r3X + 0.005, alpha: 0.30 },
      { id: "R4", x0: r2End, x1: r4End, alpha: 0.12 },
      { id: "R5", x0: r2End, x1: r4End, alpha: 0.06 },
      { id: "R6", x0: r4End, x1: r6End, alpha: 0.12 },
      { id: "R7", x0: r7X - 0.005, x1: r7X + 0.005, alpha: 0.30 },
      { id: "R8", x0: r6End, x1: t[t.length - 1] || (tFallR + lowNext), alpha: 0.10 },
    ];
    for (const b of regionBands) {
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: b.x0, x1: b.x1, y0: 0, y1: 1,
        fillcolor: REGION_COLORS[b.id], opacity: b.alpha,
        line: { width: 0 }, layer: "below",
      });
    }
  }

  if (showTransitions) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: tRiseR, x1: tRiseR, y0: 0, y1: 1,
      line: { color: "#7d8590", width: 1, dash: "dash" }, layer: "below",
    });
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: tFallR, x1: tFallR, y0: 0, y1: 1,
      line: { color: "#7d8590", width: 1, dash: "dash" }, layer: "below",
    });
  }
  if (showLevels && Number.isFinite(baseline)) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: baseline, y1: baseline,
      line: { color: "#58a6ff", width: 1, dash: "dot" }, layer: "below",
    });
  }
  if (showLevels && Number.isFinite(high)) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: high, y1: high,
      line: { color: "#d29922", width: 1, dash: "dot" }, layer: "below",
    });
  }
  if (showPeaks && tPeakR !== null && Number.isFinite(baseline)) {
    const peakY = baseline + (seg.metrics.high_level - baseline) + seg.metrics.overshoot;
    traces.push({
      x: [tPeakR], y: [peakY],
      type: "scatter", mode: "markers",
      marker: { color: REGION_COLORS.R3, size: 11, symbol: "diamond", line: { width: 1, color: "#fff" } },
      name: "peak",
      hovertemplate: "peak<br>t=%{x:.3f}s<br>overshoot=" + fmt(seg.metrics.overshoot, 1) + "<extra></extra>",
    });
  }
  if (showPeaks && tTroughR !== null && Number.isFinite(baseline)) {
    const troughY = baseline - seg.metrics.undershoot;
    traces.push({
      x: [tTroughR], y: [troughY],
      type: "scatter", mode: "markers",
      marker: { color: REGION_COLORS.R7, size: 11, symbol: "diamond", line: { width: 1, color: "#fff" } },
      name: "trough",
      hovertemplate: "trough<br>t=%{x:.3f}s<br>undershoot=" + fmt(seg.metrics.undershoot, 1) + "<extra></extra>",
    });
  }
  if (showSlope && Number.isFinite(seg.metrics.plateau_slope) && Number.isFinite(high)) {
    // Draw the plateau-slope linear fit as a green segment across R4.
    const tPlatLo = (tPeakR !== null ? tPeakR : tRiseR + 0.5 * (tFallR - tRiseR)) + 0.05;
    const tPlatHi = tFallR - 0.02;
    const slope = seg.metrics.plateau_slope;
    // The fit passes through high_level at the plateau midpoint
    const mid = 0.5 * (tPlatLo + tPlatHi);
    traces.push({
      x: [tPlatLo, tPlatHi],
      y: [high + slope * (tPlatLo + t0 - (mid + t0)), high + slope * (tPlatHi + t0 - (mid + t0))],
      type: "scatter", mode: "lines",
      line: { color: REGION_COLORS.R5, width: 2 },
      name: "slope fit",
      hovertemplate: "slope=" + fmt(slope) + "<extra></extra>",
    });
  }
  if (showAreas && Number.isFinite(baseline) && Number.isFinite(high)) {
    // Crude area-fill traces: shade between trace and reference level.
    // R2 (rising): force vs high_level over [t_rise, t_fall].
    const r2Mask = t.map((x) => x >= tRiseR && x <= tFallR);
    const r2X = [], r2Y = [];
    for (let i = 0; i < t.length; i++) {
      if (r2Mask[i]) { r2X.push(t[i]); r2Y.push(yRaw[i]); }
    }
    if (r2X.length >= 2) {
      traces.push({
        x: [...r2X, ...r2X.slice().reverse()],
        y: [...r2Y, ...r2X.map(() => high).reverse()],
        type: "scatter", fill: "toself",
        fillcolor: "rgba(63, 185, 80, 0.20)",
        line: { width: 0 },
        name: "rise area",
        hoverinfo: "skip",
      });
    }
    // R6+R8 (falling + recovery): force vs baseline over [t_fall, t_end].
    const r6X = [], r6Y = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] >= tFallR) { r6X.push(t[i]); r6Y.push(yRaw[i]); }
    }
    if (r6X.length >= 2) {
      traces.push({
        x: [...r6X, ...r6X.slice().reverse()],
        y: [...r6Y, ...r6X.map(() => baseline).reverse()],
        type: "scatter", fill: "toself",
        fillcolor: "rgba(217, 83, 79, 0.20)",
        line: { width: 0 },
        name: "fall+recovery area",
        hoverinfo: "skip",
      });
    }
  }
  if (showLabels) {
    if (Number.isFinite(seg.metrics.overshoot) && tPeakR !== null) {
      annotations.push({
        x: tPeakR, y: baseline + (high - baseline) + seg.metrics.overshoot,
        text: `Δ=${fmt(seg.metrics.overshoot, 1)}`,
        showarrow: true, arrowhead: 0, arrowcolor: REGION_COLORS.R3,
        font: { color: REGION_COLORS.R3, size: 11 },
        xanchor: "left", yanchor: "bottom",
      });
    }
    if (Number.isFinite(seg.metrics.undershoot) && tTroughR !== null) {
      annotations.push({
        x: tTroughR, y: baseline - seg.metrics.undershoot,
        text: `Δ=${fmt(seg.metrics.undershoot, 1)}`,
        showarrow: true, arrowhead: 0, arrowcolor: REGION_COLORS.R7,
        font: { color: REGION_COLORS.R7, size: 11 },
        xanchor: "left", yanchor: "top",
      });
    }
  }
  // Mark dropouts inside this segment with red Xs.
  const segDrop = dropoutFull
    .filter((dt) => dt >= seg.t_start && dt <= seg.t_end)
    .map((dt) => dt - t0);
  if (segDrop.length) {
    const dropY = segDrop.map((dt) => {
      const i = _lower_bound(t, dt);
      return yRaw[Math.min(i, yRaw.length - 1)] ?? 0;
    });
    traces.push({
      x: segDrop, y: dropY,
      type: "scatter", mode: "markers",
      marker: { color: "#ff4040", size: 11, symbol: "x", line: { width: 2 } },
      name: "dropout",
      hovertemplate: "dropout<br>t=%{x:.3f}s<extra></extra>",
    });
  }
  const layout = {
    margin: { l: 60, r: 20, t: 8, b: 36 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "t (s, segment-relative)" },
    yaxis: { gridcolor: "#2a2f37", title: "force (raw)", zeroline: false },
    showlegend: false,
    shapes: shapes,
    annotations: annotations,
  };
  Plotly.react(plot, traces, layout, { displayModeBar: false, responsive: true });
}

function _renderBdSegmentStats(seg) {
  const stats = $("bd_segment_stats");
  if (!stats) return;
  const blocks = Object.keys(REGION_METRICS).map((rid) => {
    const metricRows = REGION_METRICS[rid]
      .map((m) => `<div class="metric"><span>${m}</span><span class="value">${fmt(seg.metrics[m])}</span></div>`)
      .join("");
    return `
      <div class="bd-region" style="border-left-color:${REGION_COLORS[rid]};">
        <div class="name" style="color:${REGION_COLORS[rid]};">${REGION_TITLES[rid]}</div>
        ${metricRows}
      </div>
    `;
  });
  stats.innerHTML = blocks.join("");
}

// Binary search: index of first element ≥ target.
function _lower_bound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// ---- weight sliders + per-K metric plots ----
// Metrics that contribute to the composite cost. Each gets a weight
// slider in the Results panel and a row in the "per-metric K_opt"
// table. Must mirror the keys in BD_DEFAULT_WEIGHTS on the Python side
// (analysis.py). Order here = display order in the slider panel.
//   - area metrics ({rise/fall}_error_area, overshoot, undershoot,
//     tail_area, plateau_slope) bound the RIGHT side of the cost
//     valley.
//   - timing metrics (rise_delay, fall_delay, settling_time) bound
//     the LEFT side: at low K the response is slow, so these are
//     large and the cost rises again -- pushing the minimum to the
//     actual elbow.
const COST_METRICS = [
  "rise_error_area", "overshoot", "undershoot", "tail_area", "plateau_slope",
  "rise_delay", "fall_delay", "settling_time",
];

// Every per-K metric the analyser exposes — drives the metric grid.
const ALL_DISPLAY_METRICS = [
  "overshoot", "undershoot", "rise_error_area", "fall_error_area",
  "tail_area", "plateau_slope", "plateau_creep", "high_level",
  "baseline_noise_std", "rise_delay", "fall_delay", "settling_time",
];

function _activeCostMetrics() {
  // Cost metrics in display order. Starts from the static COST_METRICS
  // list (which dictates ordering) but ALSO picks up any extras the
  // server shipped in bd_default_weights that we forgot to add here --
  // prevents the "Python ships a new weighted metric but the UI has
  // no slider for it" footgun the user hit when rise_delay /
  // fall_delay / settling_time were added.
  const seen = new Set();
  const out = [];
  for (const n of COST_METRICS) {
    if (!seen.has(n)) { seen.add(n); out.push(n); }
  }
  const serverWeights = (bdState.analysis && bdState.analysis.bd_default_weights) || {};
  for (const n of Object.keys(serverWeights)) {
    if (!seen.has(n)) { seen.add(n); out.push(n); }
  }
  return out;
}

function renderBdWeightSliders() {
  const panel = $("bd_weight_sliders");
  if (!panel) return;
  panel.innerHTML = "";
  for (const name of _activeCostMetrics()) {
    const w = bdState.weights[name] ?? 1.0;
    const row = document.createElement("label");
    row.innerHTML = `
      <div class="slider-row"><span class="name">${name}</span><span class="value" data-name="${name}">${w.toFixed(2)}</span></div>
      <input type="range" min="0" max="5" step="0.05" value="${w}" data-name="${name}">
    `;
    const slider = row.querySelector("input");
    const display = row.querySelector(".value");
    slider.oninput = () => {
      const v = parseFloat(slider.value);
      bdState.weights[name] = v;
      display.textContent = v.toFixed(2);
      renderBdCostAndKOpt();
    };
    panel.appendChild(row);
  }
}

function _ksAndCost() {
  // Compute composite cost per K from the analyser's normalised metrics
  // and the current slider weights. Clip overshoot/undershoot at 0 on
  // the negative side (matches the Python implementation).
  const ks = [];
  const cost = [];
  if (!bdState.analysis || !bdState.analysis.bd_per_k) return { ks, cost };
  for (const r of bdState.analysis.bd_per_k) {
    ks.push(r.k);
    let total = 0;
    let nan = false;
    for (const [name, w] of Object.entries(bdState.weights)) {
      let v = r.normalised[name];
      if (v === null || v === undefined || !Number.isFinite(v)) { nan = true; break; }
      if (name === "overshoot" || name === "undershoot") v = Math.max(0, v);
      total += w * v;
    }
    cost.push(nan ? NaN : total);
  }
  return { ks, cost };
}

// JS port of _argmin_with_parabolic. Same clamping behaviour.
function jsArgminParabolic(ks, ys) {
  const finiteKs = [], finiteYs = [];
  for (let i = 0; i < ks.length; i++) {
    if (Number.isFinite(ys[i])) { finiteKs.push(ks[i]); finiteYs.push(ys[i]); }
  }
  if (!finiteKs.length) return null;
  let iMin = 0;
  for (let i = 1; i < finiteYs.length; i++) {
    if (finiteYs[i] < finiteYs[iMin]) iMin = i;
  }
  if (finiteYs.length < 3 || iMin === 0 || iMin === finiteYs.length - 1) {
    return finiteKs[iMin];
  }
  const x0 = finiteKs[iMin - 1], x1 = finiteKs[iMin], x2 = finiteKs[iMin + 1];
  const y0 = finiteYs[iMin - 1], y1 = finiteYs[iMin], y2 = finiteYs[iMin + 1];
  const denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
  if (denom === 0) return x1;
  const a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
  const b = (x2*x2*(y0 - y1) + x1*x1*(y2 - y0) + x0*x0*(y1 - y2)) / denom;
  if (a <= 0) return x1;
  const vx = -b / (2 * a);
  return Math.max(Math.min(vx, x2), x0);
}

function renderBdCostAndKOpt() {
  const { ks, cost } = _ksAndCost();
  const kOpt = jsArgminParabolic(ks, cost);
  $("bd_k_opt_display").textContent = kOpt !== null ? kOpt.toFixed(4) : "—";
  // K-color the navigator if we re-picked a different K opt.
  if (bdState.analysis) {
    bdState.analysis.bd_k_opt = kOpt;
    renderBdKList();
  }
  _drawBdCostPlot(ks, cost, kOpt);
  _renderPerMetricTable();
}

function _drawBdCostPlot(ks, cost, kOpt) {
  const plot = $("bd_cost_plot");
  if (!plot) return;
  const finite = cost.map((v) => Number.isFinite(v) ? v : null);
  const traces = [{
    x: ks, y: finite,
    type: "scatter", mode: "lines+markers",
    line: { color: "#f7931e", width: 1.4 },
    marker: { color: "#f7931e", size: 6 },
    name: "cost",
  }];
  const shapes = [];
  if (kOpt !== null && Number.isFinite(kOpt)) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: kOpt, x1: kOpt, y0: 0, y1: 1,
      line: { color: "#2ea043", dash: "dash", width: 1.5 },
    });
  }
  const layout = {
    margin: { l: 50, r: 20, t: 8, b: 36 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: { gridcolor: "#2a2f37", title: "composite cost" },
    showlegend: false,
    shapes: shapes,
  };
  Plotly.react(plot, traces, layout, { displayModeBar: false, responsive: true });
}

function _renderPerMetricTable() {
  const table = $("bd_per_metric_table");
  if (!table || !bdState.analysis) return;
  const ks = bdState.analysis.bd_per_k.map((r) => r.k);
  const rows = [`<tr><th>metric</th><th>K_opt</th></tr>`];
  for (const name of _activeCostMetrics()) {
    let ys = bdState.analysis.bd_per_k.map((r) => r.normalised[name]);
    if (name === "overshoot" || name === "undershoot") {
      ys = ys.map((v) => Number.isFinite(v) ? Math.max(0, v) : v);
    }
    const k = jsArgminParabolic(ks, ys);
    rows.push(`<tr><td>${name}</td><td class="kopt">${k !== null ? k.toFixed(4) : "—"}</td></tr>`);
  }
  table.innerHTML = rows.join("");
}

function renderBdMetricGrid() {
  const grid = $("bd_metric_grid");
  if (!grid || !bdState.analysis) return;
  grid.innerHTML = "";
  const ks = bdState.analysis.bd_per_k.map((r) => r.k);
  for (const name of ALL_DISPLAY_METRICS) {
    const cell = document.createElement("div");
    cell.className = "bd-metric-cell";
    const plotId = `bd_metric_${name}`;
    const vals = bdState.analysis.bd_per_k.map((r) => r.medians[name]);
    // Error bars: median absolute deviation of this metric across the
    // included segments at each K (×1.4826 ≈ σ-equivalent). MADs are
    // computed server-side in `_bd_aggregate_per_k`. A tight bar means
    // the segment-to-segment variance is small relative to the median
    // -- the metric is reliable at this K. A huge bar means the median
    // is dominated by noise / outliers.
    const mads = bdState.analysis.bd_per_k.map(
      (r) => (r.mads && Number.isFinite(r.mads[name])) ? r.mads[name] : null,
    );
    let valsForKopt = vals;
    if (name === "overshoot" || name === "undershoot") {
      valsForKopt = vals.map((v) => Number.isFinite(v) ? Math.max(0, v) : v);
    }
    const kOpt = jsArgminParabolic(ks, valsForKopt);
    cell.innerHTML = `
      <div class="label"><span>${name}</span><span class="kopt">${kOpt !== null ? "K_opt=" + kOpt.toFixed(4) : ""}</span></div>
      <div id="${plotId}" style="width:100%;height:160px;"></div>
    `;
    grid.appendChild(cell);
    const traces = [{
      x: ks, y: vals.map((v) => Number.isFinite(v) ? v : null),
      type: "scatter", mode: "lines+markers",
      line: { color: "#f7931e", width: 1 },
      marker: { color: "#f7931e", size: 4 },
      error_y: {
        type: "data",
        array: mads.map((v) => v !== null ? v : 0),
        visible: true,
        color: "#f7931e",
        thickness: 1,
        width: 3,
      },
      hovertemplate: "K=%{x:.4f}<br>median=%{y:.3g}<br>MAD=%{error_y.array:.3g}<extra></extra>",
    }];
    const shapes = [];
    if (kOpt !== null && Number.isFinite(kOpt)) {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: kOpt, x1: kOpt, y0: 0, y1: 1,
        line: { color: "#2ea043", dash: "dash", width: 1.2 },
      });
    }
    const layout = {
      margin: { l: 36, r: 10, t: 4, b: 24 },
      paper_bgcolor: "#161b22", plot_bgcolor: "#161b22",
      font: { color: "#e6edf3", size: 10 },
      xaxis: { gridcolor: "#2a2f37" },
      yaxis: { gridcolor: "#2a2f37" },
      showlegend: false,
      shapes: shapes,
    };
    Plotly.newPlot(plotId, traces, layout, { displayModeBar: false, responsive: true });
  }
}

// Take a full analysis dict from /api/status or /api/runs and prime the
// segment browser + sliders + metric grid. Called from renderRun() when
// a sweep finishes, and from the replay handlers when a saved run is
// selected.
function loadBdAnalysis(analysis) {
  bdState.analysis = analysis;
  // Build O(1) lookups.
  bdState.windowsByK = {};
  for (const w of analysis.windows || []) {
    bdState.windowsByK[w.k] = w;
  }
  bdState.segmentsByK = {};
  for (const s of analysis.bd_segments || []) {
    if (!bdState.segmentsByK[s.k]) bdState.segmentsByK[s.k] = [];
    bdState.segmentsByK[s.k].push(s);
  }
  for (const k of Object.keys(bdState.segmentsByK)) {
    bdState.segmentsByK[k].sort((a, b) => a.seg_idx - b.seg_idx);
  }
  // Initialise weights from server defaults (only on first load /
  // when keys are missing -- preserve user slider state across re-renders).
  for (const [name, w] of Object.entries(analysis.bd_default_weights || {})) {
    if (!(name in bdState.weights)) bdState.weights[name] = w;
  }
  // Pick a K to focus by default: bd_k_opt if available, else the
  // first one with included segments, else just the first K.
  if (bdState.selectedK === null
      || !bdState.segmentsByK[bdState.selectedK]) {
    let chosen = null;
    if (analysis.bd_k_opt !== null && analysis.bd_k_opt !== undefined) {
      // Find the closest K in the sweep to bd_k_opt
      let bestDelta = Infinity;
      for (const r of analysis.bd_per_k) {
        const d = Math.abs(r.k - analysis.bd_k_opt);
        if (d < bestDelta) { bestDelta = d; chosen = r.k; }
      }
    }
    if (chosen === null) {
      const ok = (analysis.bd_per_k || []).find((r) => r.n_segments_included > 0);
      chosen = ok ? ok.k : (analysis.bd_per_k[0] || {}).k ?? null;
    }
    bdState.selectedK = chosen;
    bdState.selectedSegIdx = 0;
  }
  renderBdWeightSliders();
  renderBdKList();
  renderBdSegment();
  renderBdCostAndKOpt();
  renderBdMetricGrid();
}

// ---- result plots ----
function plotFit(divId, k, y, fit, yTitle, customdata) {
  // `customdata` (optional): array of strings shown on hover as
  // "<value>\n<extra>". Used to surface per-K bd cycle inclusion
  // (e.g. "9/10 cycles") on the integral plot so users can see what
  // the shared-exclusion gate dropped.
  const trace = {
    x: k, y: y, mode: "markers", type: "scatter",
    marker: { color: "#f7931e", size: 9 },
    name: "measured",
  };
  if (customdata && customdata.length === k.length) {
    trace.customdata = customdata;
    trace.hovertemplate = "K=%{x:.4f}<br>y=%{y:.3g}<br>%{customdata}<extra></extra>";
  }
  const traces = [trace];
  if (fit && fit.k_opt !== null) {
    const xs = [Math.min(...k), Math.max(...k)];
    const ys = xs.map((x) => fit.slope * x + fit.intercept);
    traces.push({
      x: xs, y: ys, mode: "lines", type: "scatter",
      line: { color: "#2ea043", dash: "dash" },
      name: "fit",
    });
    traces.push({
      x: [fit.k_opt], y: [0], mode: "markers", type: "scatter",
      marker: { color: "#2ea043", size: 12, symbol: "x" },
      name: "K_opt",
    });
  }
  const layout = {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: { gridcolor: "#2a2f37", title: yTitle, zeroline: true, zerolinecolor: "#444" },
    showlegend: false,
  };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
}

// bd_pressure-style argmin plots: scatter of metric vs K with a vertical
// line at k_opt_argmin (the recommended K). No fitted slope -- argmin is
// the direct estimator, the curve shape is the diagnostic.
function plotArgmin(divId, k, y, kOpt, yTitle, zeroLine) {
  const finiteK = [];
  const finiteY = [];
  for (let i = 0; i < k.length; i++) {
    if (Number.isFinite(y[i])) {
      finiteK.push(k[i]);
      finiteY.push(y[i]);
    }
  }
  const traces = [{
    x: finiteK, y: finiteY, mode: "lines+markers", type: "scatter",
    marker: { color: "#f7931e", size: 7 },
    line: { color: "#f7931e", width: 1.2 },
    name: "measured",
  }];
  if (kOpt !== null && kOpt !== undefined && Number.isFinite(kOpt)) {
    const yMin = finiteY.length ? Math.min(...finiteY) : 0;
    const yMax = finiteY.length ? Math.max(...finiteY) : 1;
    const pad = (yMax - yMin) * 0.1 || 1;
    traces.push({
      x: [kOpt, kOpt], y: [yMin - pad, yMax + pad],
      mode: "lines", type: "scatter",
      line: { color: "#2ea043", dash: "dash", width: 1.5 },
      name: "K_opt",
    });
  }
  const layout = {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: {
      gridcolor: "#2a2f37",
      title: yTitle,
      zeroline: !!zeroLine,
      zerolinecolor: "#444",
    },
    showlegend: false,
  };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
}

let lastRunData = null;
let lastRunState = null;

function renderRun(run) {
  // Reset the live trace at the START of a run (idle/done/error → preparing
  // /running). The next incoming sample will set t0 fresh so the X axis
  // reads from 0 for the new test.
  const wasIdle = lastRunState === null
    || lastRunState === "idle"
    || lastRunState === "done"
    || lastRunState === "error";
  const nowActive = run.state === "preparing" || run.state === "running";
  if (wasIdle && nowActive) {
    resetLive();
    if (live.initialized) renderLive();
  }
  lastRunState = run.state;
  lastRunData = run;
  $("status_state").textContent = run.state;
  $("status_msg").textContent = run.message || "";
  // current_k is a float advance value (e.g. 0.04); show with 4 decimals so
  // both small (0.0000) and larger (0.1000) values render the same width.
  $("status_k").textContent =
    typeof run.current_k === "number" ? run.current_k.toFixed(4) : "—";
  $("status_pct").textContent = (run.progress_pct ?? 0).toFixed(0);

  const a = run.analysis;
  if (!a) return;
  const k = a.per_k.map((r) => r.k);
  const lag = a.per_k.map((r) => r.phase_lag_ms);
  const area = a.per_k.map((r) => r.integral_area);
  const areaLegacy = a.per_k.map((r) => r.integral_area_legacy);
  // Per-K bd-cycle inclusion for the integral plot's hover tooltip.
  // The bd_pressure analysis flags each cycle as included/excluded
  // (dropouts, low sample rate, signal-below-noise, ...); the integral
  // metric now consumes the SAME flags so both methods agree on which
  // cycles are good. Showing "9/10 cycles" tells the user how many
  // survived for each K.
  const integralCounts = a.per_k.map((r) => {
    const inc = r.integral_n_included, tot = r.integral_n_total;
    return tot > 0 ? `${inc}/${tot} cycles included` : "no bd cycles built";
  });

  plotFit("phase_plot", k, lag, a.phase_fit, "lag (ms)");
  plotFit("integral_plot", k, area, a.integral_fit, "area", integralCounts);
  plotFit("integral_legacy_plot", k, areaLegacy, a.integral_legacy_fit, "area (legacy)");

  // bd_pressure: hand the whole analysis to the segment browser / weight
  // sliders / metric grid. They keep their own state across renders so
  // user slider tweaks and segment selection survive live updates.
  loadBdAnalysis(a);

  const pf = a.phase_fit, ig = a.integral_fit, igL = a.integral_legacy_fit;
  $("phase_k_opt").textContent = pf && pf.k_opt !== null ? pf.k_opt.toFixed(4) : "—";
  $("phase_slope").textContent = pf ? pf.slope.toExponential(2) : "—";
  $("phase_r2").textContent = pf ? pf.r_squared.toFixed(3) : "—";
  $("integral_k_opt").textContent = ig && ig.k_opt !== null ? ig.k_opt.toFixed(4) : "—";
  $("integral_slope").textContent = ig ? ig.slope.toExponential(2) : "—";
  $("integral_r2").textContent = ig ? ig.r_squared.toFixed(3) : "—";
  $("integral_legacy_k_opt").textContent = igL && igL.k_opt !== null ? igL.k_opt.toFixed(4) : "—";
  $("integral_legacy_slope").textContent = igL ? igL.slope.toExponential(2) : "—";
  $("integral_legacy_r2").textContent = igL ? igL.r_squared.toFixed(3) : "—";

  const b = a.baseline;
  $("baseline_mean").textContent = b && b.mean !== null ? b.mean.toFixed(2) : "—";
  $("baseline_std").textContent = b && b.std !== null ? b.std.toFixed(3) : "—";
  $("baseline_drift").textContent = b && b.drift !== null
    ? (b.drift >= 0 ? "+" : "") + b.drift.toFixed(3) : "—";
  $("baseline_n").textContent = b && b.n_samples !== null ? b.n_samples : "—";

  if (a.notes && a.notes.length) {
    $("notes").innerHTML = "<strong>Notes:</strong><br/>" + a.notes.map((n) => `• ${n}`).join("<br/>");
  } else {
    $("notes").textContent = "";
  }
}

async function copyPressureAdvance() {
  if (!lastRunData || !lastRunData.analysis) {
    flash($("btn_copy"), "No result yet");
    return;
  }
  const a = lastRunData.analysis;
  // Prefer the bd_pressure step-response K_opt (composite cost argmin
  // with the user's current slider weights). The argmin/parabolic-interp
  // K is in `bdState` once loadBdAnalysis ran. Fall back to phase-lag /
  // integral fits when bd has no usable K (e.g. all segments excluded).
  let k = null;
  let source = "";
  const bdK = bdState.analysis ? bdState.analysis.bd_k_opt : null;
  if (bdK !== null && bdK !== undefined && Number.isFinite(bdK)) {
    k = bdK;
    source = "bd_pressure";
  } else if (a.phase_fit && a.phase_fit.k_opt !== null) {
    k = a.phase_fit.k_opt;
    source = "phase";
  } else if (a.integral_fit && a.integral_fit.k_opt !== null) {
    k = a.integral_fit.k_opt;
    source = "integral";
  }
  if (k === null) {
    flash($("btn_copy"), "No K_opt");
    return;
  }
  // Prusa Buddy/Core One uses M572 S<value>, NOT Marlin's M900 K.
  const txt = `M572 S${k.toFixed(4)} ; PA tuner (${source})`;
  await navigator.clipboard.writeText(txt);
  $("copy_status").textContent = `Copied: ${txt}`;
  setTimeout(() => ($("copy_status").textContent = ""), 3000);
}

// ---- websocket ----
function openWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "force") {
      // legacy single-sample format -- kept for compatibility but the
      // server now batches.
      pushLive(msg.t, msg.v);
      maybeRender();
    } else if (msg.type === "force_batch") {
      const ts = msg.t, vs = msg.v;
      if (Array.isArray(ts) && Array.isArray(vs)) {
        const n = Math.min(ts.length, vs.length);
        for (let i = 0; i < n; i++) pushLive(ts[i], vs[i]);
        maybeRender();
      }
    } else if (msg.type === "pos_batch") {
      // Secondary axis: toolhead X position. Overlay so the user can see
      // every X transition that triggers a burst (sweep_t0 anchor).
      const ts = msg.t, vs = msg.v;
      if (Array.isArray(ts) && Array.isArray(vs)) {
        const n = Math.min(ts.length, vs.length);
        for (let i = 0; i < n; i++) pushPos(ts[i], vs[i]);
        maybeRender();
      }
    } else if (msg.type === "run") {
      renderRun(msg.data);
    }
  };
  ws.onclose = () => setTimeout(openWs, 1500);
  ws.onerror = () => ws.close();
}

// ---- replay dropdown (saved runs/ npz) ----
// Per-run metadata keyed by filename, populated each refresh so the
// `change` handler can look up filament + temp without re-fetching.
const runsByFilename = {};

function _formatLocalTimestamp(unixSec) {
  // toISOString() always emits UTC, which is 1-2 h off from CET/CEST on
  // the user's machine and was the cause of the "2 or 3 h difference"
  // complaint. Build the timestamp in local time instead.
  const d = new Date(unixSec * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

function _renderRunMeta(filename) {
  const el = $("run_meta");
  if (!el) return;
  const run = runsByFilename[filename];
  if (!run) {
    el.textContent = "";
    return;
  }
  const parts = [];
  if (run.filament_label) parts.push(run.filament_label);
  if (run.nozzle_temp > 0) parts.push(`${run.nozzle_temp.toFixed(0)} °C`);
  el.textContent = parts.join("  @  ");
}

async function refreshRunsList() {
  const sel = $("runs_select");
  if (!sel) return;
  try {
    const r = await fetch("/api/runs");
    if (!r.ok) return;
    const data = await r.json();
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    // Reset and re-fill the lookup table.
    for (const key of Object.keys(runsByFilename)) delete runsByFilename[key];
    for (const run of (data.runs || [])) {
      runsByFilename[run.filename] = run;
      const opt = document.createElement("option");
      opt.value = run.filename;
      const date = _formatLocalTimestamp(run.mtime_unix);
      opt.textContent = `${date}  ${run.n_K}K × ${run.cycles_per_K}cyc  (${run.filename})`;
      sel.appendChild(opt);
    }
    sel.value = prev;
    _renderRunMeta(sel.value);
  } catch (_) { /* network hiccup -- next refresh handles it */ }
}

async function loadReplay(filename) {
  if (!filename) return;
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(filename)}/analyse`, { method: "POST" });
    if (!r.ok) {
      alert("Replay failed: " + (await r.text()));
      return;
    }
    const data = await r.json();
    // Synthesize a minimal `run` shape the existing renderRun expects.
    const fakeRun = {
      state: "done",
      message: `replay: ${filename}`,
      progress_pct: 100,
      current_k: null,
      analysis: data.analysis,
    };
    renderRun(fakeRun);
  } catch (e) {
    alert("Replay error: " + e);
  }
}

// ---- wire up ----
document.addEventListener("DOMContentLoaded", () => {
  $("btn_save").onclick = saveConfig;
  $("btn_preview").onclick = previewGcode;
  $("btn_run").onclick = startRun;
  $("btn_cancel").onclick = cancelRun;
  $("btn_copy").onclick = copyPressureAdvance;

  // Replay picker
  $("btn_replay_refresh").onclick = refreshRunsList;
  $("runs_select").onchange = (ev) => {
    _renderRunMeta(ev.target.value);
    loadReplay(ev.target.value);
  };
  refreshRunsList();

  // bd_pressure browser: prev/next + arrow keys, overlay toggles.
  // Stepping wraps over K boundaries: at the last segment of K[i],
  // "next" advances to K[i+1] segment 0; at segment 0 of K[i], "prev"
  // jumps to K[i-1]'s last segment. Stops at the absolute first/last
  // segment in the sweep (no circular wrap -- the user said "go to the
  // next or previous one", not "loop forever").
  $("bd_prev").onclick = () => {
    if (bdState.selectedK === null) return;
    if (bdState.selectedSegIdx > 0) {
      bdState.selectedSegIdx -= 1;
    } else {
      // Find the previous K that has at least one segment.
      const perK = (bdState.analysis && bdState.analysis.bd_per_k) || [];
      const curIdx = perK.findIndex(
        (r) => Math.abs(r.k - bdState.selectedK) < 1e-6,
      );
      for (let j = curIdx - 1; j >= 0; j--) {
        const prevSegs = bdState.segmentsByK[perK[j].k] || [];
        if (prevSegs.length > 0) {
          bdState.selectedK = perK[j].k;
          bdState.selectedSegIdx = prevSegs.length - 1;
          break;
        }
      }
    }
    renderBdKList();
    renderBdSegment();
  };
  $("bd_next").onclick = () => {
    if (bdState.selectedK === null) return;
    const segs = _segmentsForSelectedK();
    if (bdState.selectedSegIdx < segs.length - 1) {
      bdState.selectedSegIdx += 1;
    } else {
      // Find the next K that has at least one segment.
      const perK = (bdState.analysis && bdState.analysis.bd_per_k) || [];
      const curIdx = perK.findIndex(
        (r) => Math.abs(r.k - bdState.selectedK) < 1e-6,
      );
      for (let j = curIdx + 1; j < perK.length; j++) {
        const nextSegs = bdState.segmentsByK[perK[j].k] || [];
        if (nextSegs.length > 0) {
          bdState.selectedK = perK[j].k;
          bdState.selectedSegIdx = 0;
          break;
        }
      }
    }
    renderBdKList();
    renderBdSegment();
  };
  document.addEventListener("keydown", (ev) => {
    // Don't hijack arrows while focused in an input/select.
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    if (ev.key === "ArrowLeft") { $("bd_prev").click(); ev.preventDefault(); }
    else if (ev.key === "ArrowRight") { $("bd_next").click(); ev.preventDefault(); }
  });
  for (const id of [
    "bd_overlay_transitions", "bd_overlay_levels", "bd_overlay_peaks",
    "bd_overlay_regions", "bd_overlay_areas", "bd_overlay_slope",
    "bd_overlay_labels",
  ]) {
    const el = $(id);
    if (el) el.onchange = () => renderBdSegment();
  }

  loadConfig();
  openWs();
  startDiagnosticsPoll();
});
