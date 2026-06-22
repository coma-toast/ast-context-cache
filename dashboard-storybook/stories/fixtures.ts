function workerPills(active: number, total: number): string {
  return Array.from({ length: total }, (_, i) =>
    `<span class="worker-pill ${i < active ? "busy" : "idle"}"></span>`,
  ).join("");
}

/** Healthy header bar (matches live dashboard screenshot). */
export const healthBarHealthyFixture = `
<div class="health-bar" role="status" aria-label="Server health">
  <span class="health-item health-embed-item" title="openai · vn/nomic-embed-text-v1.5.Q4_K_M.gguf · openai (768 dims)">
    <span class="health-summary">openai</span>
    <span class="health-status-badge ok">ok</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item" title="Workers: 1 of 4 busy">
    <span class="health-label">Workers</span>
    <div class="worker-strip worker-strip-split" role="img" aria-label="1 of 4 workers busy">${workerPills(1, 4)}</div>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-stack-item" title="Queue 0 / 2176 · Pending 2">
    <span class="health-stack-row">
      <span class="health-label">Queue</span>
      <span class="health-mini-track level-ok"><span class="health-mini-fill" style="width:0%"></span></span>
      <span class="health-value health-stack-value">0</span>
    </span>
    <span class="health-stack-row">
      <span class="health-label">Pending</span>
      <span class="health-mini-track level-ok"><span class="health-mini-fill health-pending-fill" style="width:8%"></span></span>
      <span class="health-value health-stack-value">2</span>
    </span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-item-secondary" title="Embedding throughput (avg/s, 5s window)">
    <span class="health-label">Emb</span>
    <span class="health-value">0/s</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-item-secondary" title="Query cache hit ratio">
    <span class="health-label">Query</span>
    <span class="health-value health-value-muted">—</span>
  </span>
  <span class="health-divider health-divider-secondary"></span>
  <span class="health-item health-item-secondary" title="Uptime 2m">
    <span class="health-label">Up</span>
    <span class="health-value">2m</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item" title="Version"><span class="health-value">v2.6.16</span></span>
</div>`;

/** Degraded / error header (EmbedPanel companion story). */
export const healthBarDegradedFixture = `
<div class="health-bar health-embed-degraded" role="status" aria-label="Server health">
  <span class="health-item health-embed-item" title="openai · text-embedding-3-small · openai (768 dims)">
    <span class="health-summary">openai</span>
    <span class="health-status-badge degraded">degraded</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item" title="Workers: 10 of 15 busy">
    <span class="health-label">Workers</span>
    <div class="worker-strip worker-strip-split" role="img" aria-label="10 of 15 workers busy">${workerPills(10, 15)}</div>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-stack-item" title="Queue 0 / 2176 · Pending 987">
    <span class="health-stack-row">
      <span class="health-label">Queue</span>
      <span class="health-mini-track level-ok"><span class="health-mini-fill" style="width:0%"></span></span>
      <span class="health-value health-stack-value">0</span>
    </span>
    <span class="health-stack-row">
      <span class="health-label">Pending</span>
      <span class="health-mini-track level-critical"><span class="health-mini-fill health-pending-fill" style="width:72%"></span></span>
      <span class="health-value health-stack-value">987</span>
    </span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-item-secondary" title="Embedding throughput (avg/s, 5s window)">
    <span class="health-label">Emb</span>
    <span class="health-value">52/s</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-item-secondary" title="Query cache hit ratio">
    <span class="health-label">Query</span>
    <span class="health-mini-track level-warn"><span class="health-mini-fill" style="width:64%"></span></span>
    <span class="health-value">64%</span>
  </span>
  <span class="health-divider health-divider-secondary"></span>
  <span class="health-item health-item-secondary" title="Uptime 4h">
    <span class="health-label">Up</span>
    <span class="health-value">4h</span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item" title="Version"><span class="health-value">v2.0.16</span></span>
</div>`;

/** @deprecated use healthBarDegradedFixture */
export const healthBarFixture = healthBarDegradedFixture;

export const overviewTopBarFixture = (healthBar: string) => `
<header class="top-bar">
  <div id="health-bar" class="health-bar-inline">${healthBar}</div>
  <div class="header-controls">
    <select class="project-select" aria-label="Filter by project"><option selected>All Projects</option></select>
  </div>
</header>`;

export const statsCardsFixture = `
<div class="grid grid-4 stats-grid">
  <div class="card metric-stat-card">
    <div class="card-title">Queries</div>
    <div class="stat-value stat-accent">447</div>
    <div class="stat-meter-track"><div class="stat-meter-stack">
      <div class="stat-meter-overlap" style="width:23%"></div>
      <div class="stat-meter-day" style="width:0%"></div>
      <div class="stat-meter-avg" style="width:77%"></div>
    </div></div>
    <div class="stat-label">30d: 38,413 · avg/day: 1,280</div>
  </div>
  <div class="card metric-stat-card">
    <div class="card-title">Tokens saved</div>
    <div class="stat-value stat-green">0</div>
    <div class="stat-meter-track"><div class="stat-meter-stack">
      <div class="stat-meter-overlap" style="width:0%"></div>
      <div class="stat-meter-day" style="width:0%"></div>
      <div class="stat-meter-avg" style="width:100%"></div>
    </div></div>
    <div class="stat-label">30d: 1,193,654 · avg/day: 39,788 · dedup: 0 · vs files: 9,527,989</div>
  </div>
  <div class="card metric-stat-card">
    <div class="card-title">Avg duration</div>
    <div class="stat-value stat-orange">3840.7 ms</div>
    <div class="stat-meter-track"><div class="stat-meter-stack">
      <div class="stat-meter-overlap" style="width:15%"></div>
      <div class="stat-meter-day" style="width:55%"></div>
      <div class="stat-meter-avg" style="width:30%"></div>
    </div></div>
    <div class="stat-label">30d avg: 790.3 ms</div>
  </div>
  <div class="card metric-stat-card">
    <div class="card-title">Sessions</div>
    <div class="stat-value stat-purple">2</div>
    <div class="stat-meter-track"><div class="stat-meter-stack">
      <div class="stat-meter-overlap" style="width:26%"></div>
      <div class="stat-meter-day" style="width:0%"></div>
      <div class="stat-meter-avg" style="width:74%"></div>
    </div></div>
    <div class="stat-label">30d: 229 · avg/day: 7.6 · chars: 6,651,620</div>
  </div>
</div>`;

/** @deprecated use statsCardsFixture */
export const statsRowFixture = statsCardsFixture;

export const resourceUtilFixture = `
<div class="card metric-card resource-util-card">
  <div class="card-title">Server utilization</div>
  <div class="resource-util-rows">
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="Process CPU — can exceed 100% on multi-core">CPU</span><span class="resource-bar-value">333%</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-critical" style="width:100%;background:#f0883e"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="Host load averages (1 / 5 / 15 min)">Load avg</span><span class="resource-bar-value">11.82 · 8.53 · 9.62</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-critical" style="width:100%;background:#d2a8ff"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label">Heap</span><span class="resource-bar-value">10.9 MB</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:2%;background:#bc8cff"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label">Vector cache</span><span class="resource-bar-value">8.08 MB</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:2%;background:#58a6ff"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="SQLite usage.db + WAL journal">Database</span><span class="resource-bar-value">2.02 GB · WAL 666.3 MB</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-critical" style="width:100%;background:#3fb950"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="Host block device read throughput">Disk read</span><span class="resource-bar-value">0 MB/s</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#79c0ff"></div></div></div>
    <div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="Host block device write throughput">Disk write</span><span class="resource-bar-value">0 MB/s</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#ffa657"></div></div></div>
  </div>
  <div class="ssd-health-block">
    <div class="ssd-health-head"><span class="ssd-health-title">SSD health</span><span class="ssd-smart-badge ssd-smart-ok">Verified</span></div>
    <div class="ssd-wear-bar"><div class="resource-bar"><div class="resource-bar-head"><span class="resource-bar-label" title="NVMe PERCENTAGE_USED — manufacturer endurance estimate">Wear</span><span class="resource-bar-value">1% used</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:1%;background:#3fb950"></div></div></div></div>
    <div class="ssd-health-grid">
      <div class="ssd-health-item"><span class="ssd-health-label">Model</span><span class="ssd-health-value" title="APPLE SSD AP0512Z">APPLE SSD AP0512Z</span></div>
      <div class="ssd-health-item"><span class="ssd-health-label">Capacity</span><span class="ssd-health-value">500.28 GB</span></div>
      <div class="ssd-health-item"><span class="ssd-health-label">Protocol</span><span class="ssd-health-value">Apple Fabric</span></div>
      <div class="ssd-health-item"><span class="ssd-health-label">Spare</span><span class="ssd-health-value">100%</span></div>
    </div>
  </div>
  <div class="stat-label">Server-wide · relative load (soft caps for gauges)</div>
</div>`;

export const embedPanelHealthyFixture = `
<div class="card metric-card embed-panel">
  <div class="card-title">Embeddings</div>
  <div class="embed-active-row" title="openai · vn/nomic-embed-text-v1.5.Q4_K_M.gguf">
    <span class="embed-active-label">Active</span>
    <span class="badge badge-blue">openai</span>
    <span class="mono embed-active-model">vn/nomic-embed-text-v1.5.Q4_K_M.gguf</span>
    <span class="badge badge-green embed-sync-badge">in sync</span>
    <span class="embed-active-pill loaded">active</span>
  </div>
  <div class="metric-split">
    <div class="embed-ring-stack">
      <div class="ring-gauge ring-ok" style="--ring-pct:0%;--ring-level:ok">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">0</span><span class="ring-gauge-caption">queued</span></div>
      </div>
      <div class="ring-gauge ring-warn" style="--ring-pct:8%;--ring-level:warn">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">2</span><span class="ring-gauge-caption">pending</span></div>
      </div>
    </div>
    <div class="metric-stack">
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Priority</span><span class="channel-bar-nums">0 / 128</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#58a6ff"></div></div></div>
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Background</span><span class="channel-bar-nums">0 / 2,048</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#bc8cff"></div></div></div>
      <div class="metric-row worker-metric-row">
        <span class="metric-row-label">Workers</span>
        <div class="worker-controls" data-embed-workers="5" data-embed-active="2" data-worker-min="0" data-worker-max="15" data-worker-per-row="5" title="Workers: 2 of 5 busy">
          <div class="worker-step-col"><button type="button" class="worker-step-btn" aria-label="Decrease workers">−</button><button type="button" class="worker-step-btn" aria-label="Increase workers">+</button></div>
          <div class="worker-strip worker-strip-split" role="img" aria-label="2 of 5 workers busy">${workerPills(2, 5)}</div>
          <span class="worker-count-label">5</span>
        </div>
        <span class="metric-row-value">2 active</span>
      </div>
      <div class="metric-row">
        <span class="metric-row-label">Throughput</span>
        <div class="channel-bar-track throughput-track"><div class="channel-bar-fill level-ok" style="width:0%"></div></div>
        <span class="metric-row-value">0/s</span>
      </div>
    </div>
  </div>
  <div class="embed-activity">
    <div class="embed-activity-title">In progress</div>
    <ul class="embed-activity-list">
      <li class="embed-activity-item in-progress">sandbox · globals-runtime.js</li>
    </ul>
  </div>
  <div class="embed-activity">
    <div class="embed-activity-title">Recent</div>
    <ul class="embed-activity-list">
      <li class="embed-activity-item">ast-context-cache · globals-runtime.js</li>
    </ul>
  </div>
  <div class="embed-panel-stats">
    <div class="embed-stat" title="Code vectors stored in the index"><div class="embed-stat-value stat-accent">85,148</div><div class="embed-stat-label">Vectors cached</div></div>
    <div class="embed-stat" title="Files completed by embed queue"><div class="embed-stat-value stat-purple">1</div><div class="embed-stat-label">Total embedded</div></div>
    <div class="embed-stat" title="Pinned projects"><div class="embed-stat-value stat-green">1</div><div class="embed-stat-label">Pinned</div></div>
  </div>
</div>`;

export const embedPanelDegradedFixture = `
<div class="card metric-card embed-panel">
  <div class="card-title">Embeddings</div>
  <div class="embed-active-row" title="openai · text-embedding-3-small">
    <span class="embed-active-label">Active</span>
    <span class="badge badge-blue">openai</span>
    <span class="mono embed-active-model">text-embedding-3-small</span>
    <span class="badge badge-orange embed-sync-badge" title="Configured backend differs">out of sync</span>
    <span class="embed-active-pill degraded">degraded</span>
  </div>
  <div class="embed-error-banner embed-warn-banner">
    <div class="embed-error-banner-body">
      <strong>Recent embed errors</strong>
      <span class="embed-error-detail">openai embed: context deadline exceeded (Client.Timeout exceeded…)</span>
    </div>
    <div class="embed-error-banner-actions">
      <button type="button" class="embed-dismiss-btn">Dismiss</button>
      <button type="button" class="embed-retry-btn">Retry now</button>
    </div>
  </div>
  <div class="metric-split">
    <div class="embed-ring-stack">
      <div class="ring-gauge ring-ok" style="--ring-pct:12%;--ring-level:ok">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">0</span><span class="ring-gauge-caption">queued</span></div>
      </div>
      <div class="ring-gauge ring-critical" style="--ring-pct:48%;--ring-level:critical">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">987</span><span class="ring-gauge-caption">pending</span></div>
      </div>
    </div>
    <div class="metric-stack">
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Priority</span><span class="channel-bar-nums">0 / 128</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#58a6ff"></div></div></div>
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Background</span><span class="channel-bar-nums">0 / 2,048</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#bc8cff"></div></div></div>
      <div class="metric-row worker-metric-row">
        <span class="metric-row-label">Workers</span>
        <div class="worker-controls" data-embed-workers="15" data-embed-active="10" data-worker-min="0" data-worker-max="15" data-worker-per-row="5" title="Workers: 10 of 15 busy">
          <div class="worker-step-col"><button type="button" class="worker-step-btn" aria-label="Decrease workers">−</button><button type="button" class="worker-step-btn" aria-label="Increase workers">+</button></div>
          <div class="worker-strip worker-strip-split" role="img" aria-label="10 of 15 workers busy">${workerPills(10, 15)}</div>
          <span class="worker-count-label">15</span>
        </div>
        <span class="metric-row-value">10 active</span>
      </div>
      <div class="metric-row">
        <span class="metric-row-label">Throughput</span>
        <div class="channel-bar-track throughput-track"><div class="channel-bar-fill level-ok" style="width:65%"></div></div>
        <span class="metric-row-value">52/s</span>
      </div>
    </div>
  </div>
  <div class="embed-activity">
    <div class="embed-activity-title">In progress</div>
    <ul class="embed-activity-list">
      <li class="embed-activity-item in-progress">ast-context-cache · handlers.go</li>
      <li class="embed-activity-item in-progress">demo-api · routes.go</li>
    </ul>
  </div>
  <div class="embed-panel-stats">
    <div class="embed-stat" title="Code vectors stored in the index"><div class="embed-stat-value stat-accent">87,937</div><div class="embed-stat-label">Vectors cached</div></div>
    <div class="embed-stat" title="Files completed by embed queue"><div class="embed-stat-value stat-purple">26,110</div><div class="embed-stat-label">Total embedded</div></div>
    <div class="embed-stat" title="Pinned projects"><div class="embed-stat-value stat-green">1</div><div class="embed-stat-label">Pinned</div></div>
  </div>
</div>`;

/** @deprecated use embedPanelDegradedFixture */
export const embedPanelFixture = embedPanelDegradedFixture;

export const corpusBarsFixture = `
<div class="card metric-card corpus-bars">
  <div class="card-title">Index</div>
  <div class="corpus-bar-list">
    <div class="corpus-bar-row"><span class="corpus-bar-label">Symbols</span><div class="channel-bar-track"><div class="channel-bar-fill level-ok stat-accent" style="width:100%"></div></div><span class="corpus-bar-value stat-accent">187,778</span></div>
    <div class="corpus-bar-row"><span class="corpus-bar-label">Files</span><div class="channel-bar-track"><div class="channel-bar-fill level-ok stat-green" style="width:6%"></div></div><span class="corpus-bar-value stat-green">6,427</span></div>
    <div class="corpus-bar-row"><span class="corpus-bar-label">Edges</span><div class="channel-bar-track"><div class="channel-bar-fill level-ok stat-orange" style="width:29%"></div></div><span class="corpus-bar-value stat-orange">31,435</span></div>
  </div>
</div>`;

export const watcherCardFixture = `
<div class="card watcher-card">
  <div class="card-title">File watchers (4/5 active)</div>
  <div class="watcher-columns">
    <div class="watcher-column">
      <div class="watcher-column-title">Local</div>
      <div class="watcher-items">
        <div class="watcher-item"><span class="watcher-dot active"></span> ast-context-cache · main <span class="watcher-actions"><button type="button" class="watcher-action" title="Re-index project and queue embeddings">↻</button><button type="button" class="watcher-action" title="Pause">⏸</button><button type="button" class="watcher-action" title="Delete">🗑</button></span></div>
        <div class="watcher-item"><span class="watcher-dot active"></span> demo-api · main <span class="watcher-actions"><button type="button" class="watcher-action" title="Re-index project and queue embeddings">↻</button><button type="button" class="watcher-action" title="Pause">⏸</button><button type="button" class="watcher-action" title="Delete">🗑</button></span></div>
        <div class="watcher-item stopped"><span class="watcher-dot inactive"></span> example-app · dev <span class="watcher-actions"><button type="button" class="watcher-action" title="Re-index project and queue embeddings">↻</button><button type="button" class="watcher-action" title="Start">▶</button><button type="button" class="watcher-action" title="Delete">🗑</button></span></div>
      </div>
    </div>
    <div class="watcher-column">
      <div class="watcher-column-title">Spaces</div>
      <div class="watcher-space-group">
        <div class="watcher-space-header">staging</div>
        <div class="watcher-items">
          <div class="watcher-item"><span class="watcher-dot active"></span> example-app · staging <span class="watcher-actions"><button type="button" class="watcher-action" title="Re-index project and queue embeddings">↻</button><button type="button" class="watcher-action" title="Pause">⏸</button><button type="button" class="watcher-action" title="Delete">🗑</button></span></div>
        </div>
      </div>
    </div>
  </div>
</div>`;

export const indexHealthHealthyFixture = `
<div class="index-health-row">
  ${resourceUtilFixture}
  ${embedPanelHealthyFixture}
  ${corpusBarsFixture}
</div>
<div class="index-health-watchers-below">
  ${watcherCardFixture}
</div>`;

export const indexHealthDegradedFixture = `
<div class="index-health-row">
  ${resourceUtilFixture}
  ${embedPanelDegradedFixture}
  ${corpusBarsFixture}
</div>
<div class="index-health-watchers-below">
  ${watcherCardFixture}
</div>`;

/** @deprecated use indexHealthDegradedFixture */
export const indexHealthFixture = indexHealthDegradedFixture;

export const overviewFixture = `
${overviewTopBarFixture(healthBarHealthyFixture)}
<main class="main-content">
  <section class="dashboard-panel">
    <div class="panel-heading">
      <h2>Overview</h2>
      <p class="panel-hint">Query activity and index health at a glance</p>
    </div>
    <div class="section-title">Query activity <span class="section-hint">Today vs 30d daily avg · dark purple = overlap (blue ∩ red), red = day only, blue = avg only</span></div>
    ${statsCardsFixture}
    <div class="section-title">Index &amp; runtime <span class="section-hint">Utilization, embeddings, and corpus</span></div>
    ${indexHealthHealthyFixture}
  </section>
</main>`;
