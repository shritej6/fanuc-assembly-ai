"""Self-contained, dependency-free dashboard document."""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FANUC Cell Operations</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111310;
      --surface: #191c18;
      --line: #373c34;
      --ink: #f3f0e7;
      --muted: #aaa99f;
      --green: #65d386;
      --amber: #f0b34d;
      --red: #ee675f;
      --blue: #7da9ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      min-height: 100vh;
    }
    header, main { width: min(1440px, calc(100% - 40px)); margin-inline: auto; }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: end;
      padding: 30px 0 20px;
      border-bottom: 1px solid var(--line);
    }
    .eyebrow, dt { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; }
    h1 { margin: 5px 0 0; font: 500 clamp(1.6rem, 4vw, 2.6rem)/1.1 ui-sans-serif, system-ui, sans-serif; }
    .source { text-align: right; }
    .source strong { display: block; margin-top: 7px; font-size: 1rem; }
    main { padding: 20px 0 42px; }
    .status-strip {
      display: grid;
      grid-template-columns: auto minmax(160px, .55fr) 1fr auto;
      align-items: center;
      gap: 18px;
      padding: 18px 0;
      border-bottom: 1px solid var(--line);
    }
    .beacon { width: 18px; height: 18px; border-radius: 50%; background: var(--red); box-shadow: 0 0 18px currentColor; }
    .beacon.green { background: var(--green); color: var(--green); }
    .beacon.amber { background: var(--amber); color: var(--amber); }
    .beacon.red { background: var(--red); color: var(--red); }
    .machine-state { font: 600 1.1rem ui-sans-serif, system-ui, sans-serif; }
    .fault { color: var(--muted); overflow-wrap: anywhere; }
    .freshness { font-size: .78rem; color: var(--muted); text-align: right; }
    .freshness.stale { color: var(--red); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      border-bottom: 1px solid var(--line);
    }
    .metric { min-height: 142px; padding: 22px 18px; border-right: 1px solid var(--line); }
    .metric:nth-child(4n) { border-right: 0; }
    .metric:nth-child(-n+4) { border-bottom: 1px solid var(--line); }
    .metric .value { display: block; margin-top: 14px; font: 500 2rem/1 ui-sans-serif, system-ui, sans-serif; }
    .metric .reason { display: block; margin-top: 10px; color: var(--amber); font-size: .72rem; line-height: 1.35; }
    .workspace { display: grid; grid-template-columns: 1fr 1.3fr; gap: 0; }
    section { padding: 24px 0; }
    section + section { border-left: 1px solid var(--line); padding-left: 24px; }
    h2 { margin: 0 0 18px; font: 600 .82rem ui-sans-serif, system-ui, sans-serif; text-transform: uppercase; letter-spacing: .09em; }
    .counts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; background: var(--line); }
    .counts div { background: var(--surface); padding: 15px; }
    dd { margin: 7px 0 0; font-size: 1.5rem; }
    table { width: 100%; border-collapse: collapse; font-size: .78rem; }
    th, td { padding: 11px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: .06em; }
    .empty { color: var(--muted); padding: 12px 0; }
    .error-banner { display: none; color: var(--red); border-bottom: 1px solid var(--red); padding: 14px 0; }
    .error-banner.visible { display: block; }
    .sr-only {
      position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
    }
    @media (max-width: 800px) {
      header, main { width: min(100% - 24px, 720px); }
      header, .status-strip, .workspace { grid-template-columns: 1fr; }
      .source, .freshness { text-align: left; }
      .metric-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
      .metric:nth-child(2n) { border-right: 0; }
      .metric:nth-child(n) { border-bottom: 1px solid var(--line); }
      section + section { border-left: 0; border-top: 1px solid var(--line); padding-left: 0; }
    }
    @media (prefers-reduced-motion: no-preference) {
      .beacon { transition: background-color 180ms ease, color 180ms ease; }
      .value { transition: color 180ms ease; }
    }
  </style>
</head>
<body>
  <header>
    <div><div class="eyebrow">Team V · read-only operations</div><h1>Assembly cell status</h1></div>
    <div class="source"><span class="eyebrow">Data source</span><strong id="source">WAITING</strong></div>
  </header>
  <main>
    <div id="error" class="error-banner" role="alert"></div>
    <div class="status-strip">
      <span id="beacon" class="beacon red" aria-hidden="true"></span>
      <span id="state" class="machine-state">Unknown</span>
      <span id="fault" class="fault">No PackML state available</span>
      <span id="freshness" class="freshness stale">No events</span>
      <span id="status-announcement" class="sr-only" role="status" aria-live="polite" aria-atomic="true"></span>
    </div>
    <div id="metrics" class="metric-grid"></div>
    <div class="workspace">
      <section aria-labelledby="counts-heading">
        <h2 id="counts-heading">Run counts</h2>
        <dl id="counts" class="counts"></dl>
      </section>
      <section aria-labelledby="timeline-heading">
        <h2 id="timeline-heading">Recent PackML transitions</h2>
        <table>
          <thead><tr><th>Time</th><th>State</th><th>Product</th><th>Reason</th></tr></thead>
          <tbody id="timeline"></tbody>
        </table>
        <p id="timeline-empty" class="empty">No state transitions recorded.</p>
      </section>
    </div>
  </main>
  <script>
    const metricLabels = {
      availability: "Availability", performance: "Performance", quality: "Quality", oee: "OEE",
      first_pass_yield: "First-pass yield", final_yield: "Final yield",
      builds_per_assist: "Builds per assist", average_cycle_time_s: "Average cycle time"
    };
    const countLabels = {
      completed_products: "Completed", good_products: "Good", rejected_products: "Rejected",
      assists: "Assists", escapes: "Escapes"
    };
    const percentageMetrics = new Set(["availability", "performance", "quality", "oee", "first_pass_yield", "final_yield"]);
    let lastAnnouncement = "";
    const formatMetric = (name, metric) => {
      if (metric.value === null) return "—";
      if (percentageMetrics.has(name)) return `${(metric.value * 100).toFixed(1)}%`;
      if (name === "average_cycle_time_s") return `${metric.value.toFixed(2)} s`;
      return metric.value.toFixed(2);
    };
    const setText = (id, text) => { document.getElementById(id).textContent = text; };
    function renderMetrics(metrics) {
      const root = document.getElementById("metrics"); root.replaceChildren();
      Object.entries(metricLabels).forEach(([name, label]) => {
        const metric = metrics[name]; const box = document.createElement("div"); box.className = "metric";
        const key = document.createElement("span"); key.className = "eyebrow"; key.textContent = label;
        const value = document.createElement("span"); value.className = "value"; value.textContent = formatMetric(name, metric);
        box.append(key, value);
        if (metric.unavailable_reason) { const reason = document.createElement("span"); reason.className = "reason"; reason.textContent = metric.unavailable_reason; box.append(reason); }
        root.append(box);
      });
    }
    function renderCounts(counts) {
      const root = document.getElementById("counts"); root.replaceChildren();
      Object.entries(countLabels).forEach(([name, label]) => {
        const box = document.createElement("div"); const key = document.createElement("dt"); const value = document.createElement("dd");
        key.textContent = label; value.textContent = String(counts[name]); box.append(key, value); root.append(box);
      });
    }
    function renderTimeline(events) {
      const body = document.getElementById("timeline"); body.replaceChildren();
      document.getElementById("timeline-empty").hidden = events.length > 0;
      events.slice().reverse().forEach((event) => {
        const row = document.createElement("tr");
        [new Date(event.timestamp).toLocaleTimeString(), event.state, event.product ?? "—", event.reason ?? "—"].forEach((text) => {
          const cell = document.createElement("td"); cell.textContent = String(text); row.append(cell);
        }); body.append(row);
      });
    }
    function render(data) {
      setText("source", data.source.toUpperCase()); setText("state", data.packml_state);
      setText("fault", data.active_fault ?? "No active fault");
      setText("freshness", `${data.stale ? "STALE · " : "LIVE · "}${new Date(data.last_event_at).toLocaleTimeString()}`);
      document.getElementById("freshness").classList.toggle("stale", data.stale);
      document.getElementById("beacon").className = `beacon ${data.rag_status}`;
      const announcement = `${data.packml_state}. ${data.active_fault ?? "No active fault"}. Data ${data.stale ? "stale" : "live"}.`;
      if (announcement !== lastAnnouncement) { setText("status-announcement", announcement); lastAnnouncement = announcement; }
      renderMetrics(data.metrics); renderCounts(data.counts); renderTimeline(data.recent_transitions);
      document.getElementById("error").classList.remove("visible");
    }
    async function refresh() {
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" }); const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); render(data);
      } catch (error) {
        const banner = document.getElementById("error"); banner.textContent = `Dashboard data unavailable: ${error.message}`; banner.classList.add("visible");
        document.getElementById("beacon").className = "beacon red"; setText("state", "Data unavailable");
      }
    }
    refresh(); window.setInterval(refresh, 2000);
  </script>
</body>
</html>
"""
