export const embedPanelFixture = `
<div class="card metric-card embed-panel" style="max-width:720px">
  <div class="card-title">Embeddings</div>
  <div class="embed-active-row">
    <span class="embed-active-label">Active</span>
    <span class="badge badge-blue">openai</span>
    <span class="mono embed-active-model">text-embedding-3-small</span>
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
      <div class="ring-gauge ring-ok" style="--ring-pct:12%">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">0</span><span class="ring-gauge-caption">queued</span></div>
      </div>
      <div class="ring-gauge ring-warn" style="--ring-pct:48%">
        <div class="ring-gauge-inner"><span class="ring-gauge-value">987</span><span class="ring-gauge-caption">pending</span></div>
      </div>
    </div>
    <div class="metric-stack">
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Priority</span><span class="channel-bar-nums">0 / 128</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#58a6ff"></div></div></div>
      <div class="channel-bar"><div class="channel-bar-head"><span class="channel-bar-label">Background</span><span class="channel-bar-nums">0 / 2,048</span></div><div class="channel-bar-track"><div class="channel-bar-fill level-ok" style="width:0%;background:#bc8cff"></div></div></div>
      <div class="metric-row worker-metric-row">
        <span class="metric-row-label">Workers</span>
        <div class="worker-controls" data-embed-workers="15" data-embed-active="10">
          <div class="worker-step-col"><button type="button" class="worker-step-btn">−</button><button type="button" class="worker-step-btn">+</button></div>
          <div class="worker-strip worker-strip-split" role="img" aria-label="10 of 15 workers busy">
            <span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span>
            <span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span>
            <span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span>
          </div>
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
      <li class="embed-activity-item in-progress">configSync · tools.go</li>
    </ul>
  </div>
  <div class="embed-panel-stats">
    <div class="embed-stat"><div class="embed-stat-value stat-accent">87,937</div><div class="embed-stat-label">Vectors cached</div></div>
    <div class="embed-stat"><div class="embed-stat-value stat-purple">26,110</div><div class="embed-stat-label">Total embedded</div></div>
    <div class="embed-stat"><div class="embed-stat-value stat-green">1</div><div class="embed-stat-label">Pinned</div></div>
  </div>
</div>`;

export const healthBarFixture = `
<div class="health-bar health-embed-degraded" role="status" aria-label="Server health">
  <span class="health-item health-embed-item"><span class="health-summary">openai</span><span class="health-status-badge degraded">degraded</span></span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item"><span class="health-label">Workers</span>
    <div class="worker-strip worker-strip-split"><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill busy"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span><span class="worker-pill idle"></span></div>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-stack-item">
    <span class="health-stack-row"><span class="health-label">Queue</span><span class="health-value health-stack-value">0</span></span>
    <span class="health-stack-row"><span class="health-label">Pending</span><span class="health-value health-stack-value">987</span></span>
  </span>
  <span class="health-divider"></span>
  <span class="health-item health-gauge-item health-item-secondary"><span class="health-label">Emb</span><span class="health-value">52/s</span></span>
  <span class="health-divider"></span>
  <span class="health-item"><span class="health-value">v2.0.16</span></span>
</div>`;

export const statsRowFixture = `
<div class="stats-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;max-width:960px">
  <div class="card metric-stat-card"><div class="card-title">Tokens saved</div><div class="stat-value stat-accent">1.2M</div><div class="stat-label">30d · 42k/day avg</div></div>
  <div class="card metric-stat-card"><div class="card-title">Queries</div><div class="stat-value">8,412</div><div class="stat-label">30d · 284 today</div></div>
  <div class="card metric-stat-card"><div class="card-title">Virtual context</div><div class="stat-value stat-purple">12 notes</div><div class="stat-label">18% utilization · 2 orphans</div></div>
</div>`;

export const overviewFixture = `
${healthBarFixture}
<div style="height:20px"></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;max-width:960px">
  ${embedPanelFixture}
  ${statsRowFixture}
</div>`;
