const projectFilteredTargets = new Set([
    '#stats-cards',
    '#index-health',
    '#memory-panel',
    '#recent-queries',
    '#symbol-chart',
    '#lang-chart',
    '#tool-chart',
    '#import-chart',
]);

function projectFilterActive() {
    const select = document.querySelector('.project-select');
    return !!(select && select.value);
}

document.addEventListener('alpine:init', () => {
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${location.host}/ws`);

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);

            if (msg.type === 'toast') {
                const toast = msg.data;
                const container = document.getElementById('toast-container');
                if (container) {
                    const div = document.createElement('div');
                    div.className = 'toast';
                    div.style.borderLeft = `3px solid ${toast.toolColor}`;
                    div.innerHTML = `
                        <div class="toast-header">
                            <span class="toast-title" style="color:${toast.toolColor}">${toast.toolName}</span>
                            <span class="toast-time">${toast.timeStr}</span>
                        </div>
                        <div class="toast-body">
                            <span class="toast-query" title="${toast.query}">${toast.query}</span>
                            <span class="toast-meta">${toast.savedText} ${toast.durationMs}</span>
                        </div>
                    `;
                    container.appendChild(div);
                    // Auto-remove after 4 seconds
                    setTimeout(() => {
                        div.classList.add('removing');
                        setTimeout(() => div.remove(), 200);
                    }, 4000);
                    // Limit to 5 toasts
                    while (container.children.length > 5) {
                        container.firstChild.remove();
                    }
                }
            } else if (msg.type === 'partial') {
                const targetSel = msg.data.target;
                if (projectFilterActive() && projectFilteredTargets.has(targetSel)) {
                    return;
                }
                const target = document.querySelector(targetSel);
                if (target) {
                    if (targetSel === '#recent-queries') {
                        const preservedRecentTab = getRecentSubTab(target);
                        if (patchRecentPanel(target, msg.data.html)) {
                            if (preservedRecentTab === 'logs') {
                                requestAnimationFrame(() => scrollRecentLogsToBottom(target, false));
                            }
                            syncLogsPoll();
                        } else {
                            target.innerHTML = msg.data.html;
                            Alpine.flushSync();
                            mountRecentContent(target, preservedRecentTab);
                            syncLogsPoll();
                        }
                    } else {
                        target.innerHTML = msg.data.html;
                        Alpine.flushSync();
                        mountHTMXContent(target);
                    }
                    if (target.id === 'settings-content') {
                        mountSettingsContent(target);
                    }
                    window.dispatchEvent(new CustomEvent('dashboard-ws-partial'));
                    if (targetSel === '#index-health' || targetSel === '#health-bar') {
                        onWorkerPartialUpdated();
                    }
                }
            }
        } catch (e) {
            console.error('WebSocket message error:', e);
        }
    };

    ws.onclose = () => {
        setTimeout(() => location.reload(), 3000);
    };

    ws.onerror = () => {};

    Alpine.store('chart', {
        interval: 'daily',
        metric: 'queries',
        data: []
    });

    Alpine.data('dashboard', () => ({
        selectedProject: '',
        activeTab: 'overview',
        settingsLoaded: false,

        init() {
            const hash = (location.hash || '').replace('#', '');
            const tabs = ['overview', 'memory', 'activity', 'analytics', 'recent', 'settings'];
            if (tabs.includes(hash)) {
                this.activeTab = hash;
            }
            this.bootstrapPanels();
            refreshWorkerLimitsFromServer().then((gotMax) => {
                if (!gotMax) syncWorkerMaxFromDOM();
                if (syncWorkerStateFromDOM()) renderWorkerUI();
            });
            if (this.activeTab === 'settings') {
                this.loadSettings();
            }
            if (this.activeTab === 'memory') {
                this.loadMemory();
            }
            if (this.activeTab === 'recent') {
                refreshRecentPartial(true);
            }
            syncLogsPoll();
        },

        async bootstrapPanels() {
            const panels = [
                ['#health-bar', '/partials/health'],
                ['#stats-cards', '/partials/stats'],
                ['#index-health', '/partials/index-health'],
            ];
            await Promise.all(panels.map(async ([sel, path]) => {
                const el = document.querySelector(sel);
                if (!el) return;
                const stale = el.textContent.includes('Loading...') ||
                    (sel === '#stats-cards' && el.querySelector('.stat-value')?.textContent === '-');
                if (!stale) return;
                const r = await fetch(path);
                if (r.ok) {
                    el.innerHTML = await r.text();
                    mountHTMXContent(el);
                    if (sel === '#index-health' || sel === '#health-bar') {
                        onWorkerPartialUpdated();
                    }
                }
            }));
        },

        projectLabel() {
            if (!this.selectedProject) return '';
            const sel = document.querySelector('.project-select');
            if (sel) {
                const opt = sel.querySelector(`option[value="${CSS.escape(this.selectedProject)}"]`);
                if (opt) {
                    return opt.dataset.label || opt.textContent || this.selectedProject;
                }
            }
            const parts = this.selectedProject.split('/');
            return parts[parts.length - 1] || this.selectedProject;
        },

        setTab(tab) {
            this.activeTab = tab;
            if (location.hash !== '#' + tab) {
                history.replaceState(null, '', '#' + tab);
            }
            if (tab === 'settings') {
                this.loadSettings();
            }
            if (tab === 'memory') {
                this.loadMemory();
            }
            if (tab === 'activity') {
                window.dispatchEvent(new CustomEvent('chart-update'));
            }
            if (tab === 'recent') {
                refreshRecentPartial(true);
            }
            syncLogsPoll();
        },

        async loadSettings() {
            const el = document.getElementById('settings-content');
            if (!el) return;
            const needsLoad = !this.settingsLoaded && !el.querySelector('.section') && !el.querySelector('.project-list');
            if (needsLoad) {
                const r = await fetch('/partials/settings');
                if (r.ok) {
                    el.innerHTML = await r.text();
                    this.settingsLoaded = true;
                }
            }
            mountSettingsContent(el);
        },

        async loadMemory() {
            const el = document.getElementById('memory-panel');
            if (!el) return;
            const project = this.selectedProject;
            const q = project ? `?project_id=${encodeURIComponent(project)}` : '';
            const r = await fetch('/partials/memory' + q);
            if (r.ok) {
                el.innerHTML = await r.text();
                mountHTMXContent(el);
            }
        },

        async applyProjectFilter() {
            const project = this.selectedProject;
            const q = project ? `?project_id=${encodeURIComponent(project)}` : '';
            const panels = [
                ['#stats-cards', '/partials/stats'],
                ['#index-health', '/partials/index-health'],
                ['#memory-panel', '/partials/memory'],
                ['#recent-queries', '/partials/recent'],
                ['#symbol-chart', '/partials/charts/symbols'],
                ['#lang-chart', '/partials/charts/languages'],
                ['#tool-chart', '/partials/charts/tools'],
                ['#import-chart', '/partials/charts/imports'],
            ];
            await Promise.all(panels.map(async ([sel, path]) => {
                const el = document.querySelector(sel);
                if (!el) return;
                const r = await fetch(path + q);
                if (r.ok) {
                    const preservedRecentTab = sel === '#recent-queries' ? getRecentSubTab(el) : null;
                    el.innerHTML = await r.text();
                    if (sel === '#recent-queries') {
                        mountRecentContent(el, preservedRecentTab);
                    } else {
                        mountHTMXContent(el);
                    }
                }
            }));
            window.dispatchEvent(new CustomEvent('dashboard-project-change'));
        }
    }));

    Alpine.data('timeseriesChart', () => ({
        chartData: null,

        init() {
            this.$watch('$store.chart.interval', () => this.loadAndDraw());
            this.$watch('$store.chart.metric', () => this.loadAndDraw());
            window.addEventListener('resize', () => this.draw());
            this._chartDebounce = null;
            this._onWsPartial = () => {
                if (this._chartDebounce) clearTimeout(this._chartDebounce);
                this._chartDebounce = setTimeout(() => this.loadAndDraw(), 400);
            };
            window.addEventListener('dashboard-ws-partial', this._onWsPartial);
            this._onProjectChange = () => this.loadAndDraw();
            window.addEventListener('dashboard-project-change', this._onProjectChange);
            this.loadAndDraw();
        },

        handleData(event) {
            if (event.detail && event.detail.target && event.detail.target.id === 'activity-chart') {
                try {
                    this.chartData = JSON.parse(event.detail.xhr.responseText);
                    this.draw();
                } catch (e) {}
            }
        },

        async loadAndDraw() {
            try {
                const interval = Alpine.store('chart').interval;
                const metric = Alpine.store('chart').metric;
                const projectSelect = document.querySelector('.project-select');
                const project = projectSelect && projectSelect.value ? projectSelect.value : '';
                const sep = project ? '&' : '';
                const projParam = project ? `project_id=${encodeURIComponent(project)}` : '';
                const r = await fetch(`/api/timeseries?interval=${interval}&days=30${sep}${projParam}`);
                this.chartData = await r.json();
                this.draw();
            } catch (e) {}
        },

        draw() {
            const canvas = this.$refs.canvas;
            if (!canvas || !this.chartData) return;

            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            const W = rect.width, H = rect.height;
            ctx.clearRect(0, 0, W, H);

            const data = this.chartData;
            const metric = Alpine.store('chart').metric;
            const interval = Alpine.store('chart').interval;

            if (!data || data.length === 0) {
                ctx.fillStyle = '#8b949e';
                ctx.font = '13px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No data for this period', W / 2, H / 2);
                return;
            }

            const vals = data.map(d => d[metric] || 0);
            const labels = data.map(d => d.timestamp);
            const max = Math.max(...vals, 1);
            const padL = 50, padR = 16, padT = 16, padB = 28;
            const cW = W - padL - padR;
            const cH = H - padT - padB;

            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padT + cH - (i / 4) * cH;
                ctx.beginPath();
                ctx.moveTo(padL, y);
                ctx.lineTo(W - padR, y);
                ctx.stroke();
                ctx.fillStyle = '#8b949e';
                ctx.font = '10px JetBrains Mono, monospace';
                ctx.textAlign = 'right';
                ctx.fillText(fmtNum(Math.round(max * i / 4)), padL - 6, y + 3);
            }

            if (vals.length > 1) {
                const color = metric === 'tokens_saved' ? '#3fb950' : '#58a6ff';
                const grad = ctx.createLinearGradient(0, padT, 0, padT + cH);
                grad.addColorStop(0, color + '40');
                grad.addColorStop(1, color + '00');

                ctx.beginPath();
                vals.forEach((v, i) => {
                    const x = padL + (i / (vals.length - 1)) * cW;
                    const y = padT + cH - (v / max) * cH;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                });
                ctx.lineTo(padL + cW, padT + cH);
                ctx.lineTo(padL, padT + cH);
                ctx.closePath();
                ctx.fillStyle = grad;
                ctx.fill();

                ctx.beginPath();
                vals.forEach((v, i) => {
                    const x = padL + (i / (vals.length - 1)) * cW;
                    const y = padT + cH - (v / max) * cH;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                });
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();

                vals.forEach((v, i) => {
                    const x = padL + (i / (vals.length - 1)) * cW;
                    const y = padT + cH - (v / max) * cH;
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                });
            }

            ctx.fillStyle = '#8b949e';
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            const step = Math.max(1, Math.floor(labels.length / 6));
            labels.forEach((l, i) => {
                if (i % step === 0 || i === labels.length - 1) {
                    const x = padL + (i / Math.max(1, labels.length - 1)) * cW;
                    const short = interval === 'hourly' ? l.slice(11, 16) : l.slice(5, 10);
                    ctx.fillText(short, x, H - 6);
                }
            });
        }
    }));
});

function fmtNum(n) {
    if (n == null) return '-';
    return n.toLocaleString();
}

// Include project filter in all htmx requests
document.body.addEventListener('htmx:configRequest', (event) => {
    const projectSelect = document.querySelector('.project-select');
    if (projectSelect && projectSelect.value) {
        const separator = event.detail.path.includes('?') ? '&' : '?';
        event.detail.path += separator + 'project_id=' + encodeURIComponent(projectSelect.value);
    }
});

const RECENT_SUBTAB_KEY = 'dashboard-recent-subtab';
const RECENT_LOGS_AUTOSCROLL_KEY = 'dashboard-recent-logs-autoscroll';

function recentSubTabFromRadio(radio) {
    if (!radio) return 'mcp';
    if (radio.id === 'recent-tab-indexing') return 'indexing';
    if (radio.id === 'recent-tab-logs') return 'logs';
    return 'mcp';
}

function recentSubTabRadioId(tab) {
    if (tab === 'indexing') return 'recent-tab-indexing';
    if (tab === 'logs') return 'recent-tab-logs';
    return 'recent-tab-mcp';
}

function storedRecentSubTab() {
    const t = sessionStorage.getItem(RECENT_SUBTAB_KEY);
    if (t === 'indexing' || t === 'logs') return t;
    return 'mcp';
}

function getRecentSubTab(root) {
    const panel = root?.querySelector?.('.recent-panel') || root;
    if (panel?.querySelector) {
        const checked = panel.querySelector('.recent-tab-radio:checked');
        if (checked) return recentSubTabFromRadio(checked);
    }
    return storedRecentSubTab();
}

function setRecentSubTab(tab, root) {
    const panel = root?.querySelector?.('.recent-panel') || root;
    if (!panel?.querySelector) return;
    const radio = panel.querySelector('#' + recentSubTabRadioId(tab));
    if (radio) radio.checked = true;
    sessionStorage.setItem(RECENT_SUBTAB_KEY, tab);
}

function patchRecentPanel(container, html) {
    const panel = container.querySelector('.recent-panel');
    if (!panel) return false;
    const wrap = document.createElement('div');
    wrap.innerHTML = html;
    const srcPanel = wrap.querySelector('.recent-panel');
    if (!srcPanel) return false;
    for (const sel of ['.pane-mcp', '.pane-indexing', '.pane-logs']) {
        const src = srcPanel.querySelector(sel);
        const dst = panel.querySelector(sel);
        if (src && dst) dst.innerHTML = src.innerHTML;
    }
    for (const id of ['recent-tab-mcp', 'recent-tab-indexing', 'recent-tab-logs']) {
        const newCount = srcPanel.querySelector(`label[for="${id}"] .recent-count`);
        const oldCount = panel.querySelector(`label[for="${id}"] .recent-count`);
        if (newCount && oldCount) oldCount.textContent = newCount.textContent;
    }
    mountHTMXContent(panel);
    bindRecentLogsAutoscroll(container);
    return true;
}

function patchRecentLogsCard(container, html) {
    const panel = container.querySelector('.recent-panel');
    const dstLogs = panel?.querySelector('.pane-logs');
    if (!dstLogs) return false;
    const wrap = document.createElement('div');
    wrap.innerHTML = html;
    const card = wrap.querySelector('.recent-logs-card') || wrap.firstElementChild;
    if (!card) return false;
    dstLogs.innerHTML = '';
    dstLogs.appendChild(card);
    mountHTMXContent(dstLogs);
    bindRecentLogsAutoscroll(container);
    const newCount = wrap.querySelector('.recent-count');
    const oldCount = panel.querySelector('label[for="recent-tab-logs"] .recent-count');
    if (newCount && oldCount) oldCount.textContent = newCount.textContent;
    return true;
}

function recentLogsAutoscrollEnabled(root) {
    if (sessionStorage.getItem(RECENT_LOGS_AUTOSCROLL_KEY) === 'paused') return false;
    const panel = root?.querySelector?.('.recent-panel') || root;
    const btn = panel?.querySelector?.('[data-recent-logs-autoscroll]');
    if (btn) return btn.getAttribute('aria-pressed') !== 'true';
    return true;
}

function syncRecentLogsAutoscrollBtn(btn) {
    if (!btn) return;
    const paused = sessionStorage.getItem(RECENT_LOGS_AUTOSCROLL_KEY) === 'paused';
    btn.setAttribute('aria-pressed', paused ? 'true' : 'false');
    btn.textContent = paused ? 'Paused' : 'Auto-scroll';
    btn.title = paused ? 'Resume following new log lines' : 'Pause auto-scroll';
}

function bindRecentLogsAutoscroll(root) {
    const panel = root?.querySelector?.('.recent-panel') || root;
    const btn = panel?.querySelector?.('[data-recent-logs-autoscroll]');
    if (btn) syncRecentLogsAutoscrollBtn(btn);
}

function scrollRecentLogsToBottom(root, force) {
    const panel = root?.querySelector?.('.recent-panel') || root;
    if (!panel) return;
    const logsTab = panel.querySelector('#recent-tab-logs');
    if (!logsTab?.checked) return;
    if (!recentLogsAutoscrollEnabled(root)) return;
    const scroll = panel.querySelector('.recent-logs-scroll');
    if (!scroll) return;
    const gap = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    if (force || gap < 48) {
        scroll.scrollTop = scroll.scrollHeight;
    }
}

let logsPollTimer = null;

function recentTabActive() {
    const root = document.querySelector('[x-data="dashboard"]');
    if (!root || typeof Alpine === 'undefined' || !Alpine.$data) return false;
    try {
        return Alpine.$data(root)?.activeTab === 'recent';
    } catch {
        return false;
    }
}

function syncLogsPoll() {
    if (logsPollTimer) {
        clearInterval(logsPollTimer);
        logsPollTimer = null;
    }
    const el = document.getElementById('recent-queries');
    if (!el || !recentTabActive() || getRecentSubTab(el) !== 'logs') return;
    logsPollTimer = setInterval(() => refreshRecentLogsOnly(false), 3000);
}

async function refreshRecentLogsOnly(forceScroll) {
    const el = document.getElementById('recent-queries');
    if (!el) return;
    const r = await fetch('/partials/recent-logs');
    if (!r.ok) return;
    const html = await r.text();
    if (!patchRecentLogsCard(el, html)) {
        await refreshRecentPartial(forceScroll);
        return;
    }
    requestAnimationFrame(() => scrollRecentLogsToBottom(el, !!forceScroll));
}

async function refreshRecentPartial(forceScroll) {
    const el = document.getElementById('recent-queries');
    if (!el) return;
    const preserved = getRecentSubTab(el);
    if (preserved === 'logs') {
        await refreshRecentLogsOnly(forceScroll);
        return;
    }
    const r = await fetch('/partials/recent');
    if (!r.ok) return;
    const html = await r.text();
    if (!patchRecentPanel(el, html)) {
        el.innerHTML = html;
        mountRecentContent(el, preserved);
    }
}

function mountRecentContent(el, preservedTab) {
    if (!el) return;
    const tab = preservedTab || sessionStorage.getItem(RECENT_SUBTAB_KEY) || 'mcp';
    mountHTMXContent(el);
    setRecentSubTab(tab, el);
    bindRecentLogsAutoscroll(el);
    if (tab === 'logs') {
        requestAnimationFrame(() => scrollRecentLogsToBottom(el, true));
    }
    syncLogsPoll();
    if (el.dataset.recentTabBound) return;
    el.dataset.recentTabBound = '1';
    el.addEventListener('change', (e) => {
        if (e.target?.classList?.contains('recent-tab-radio')) {
            const sub = recentSubTabFromRadio(e.target);
            sessionStorage.setItem(RECENT_SUBTAB_KEY, sub);
            if (sub === 'logs') {
                requestAnimationFrame(() => scrollRecentLogsToBottom(el, true));
            }
            syncLogsPoll();
        }
    });
    el.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-recent-logs-autoscroll]');
        if (!btn) return;
        const willPause = btn.getAttribute('aria-pressed') !== 'true';
        sessionStorage.setItem(RECENT_LOGS_AUTOSCROLL_KEY, willPause ? 'paused' : 'on');
        syncRecentLogsAutoscrollBtn(btn);
        if (!willPause) {
            requestAnimationFrame(() => scrollRecentLogsToBottom(el, true));
        }
    });
}

function mountHTMXContent(el) {
    if (!el || typeof htmx === 'undefined') return;
    htmx.process(el);
}

function mountSettingsContent(el) {
    mountHTMXContent(el);
}

async function watcherApiAction(action, projectPath, targetSel) {
    const endpoints = {
        start: '/api/start-watcher',
        stop: '/api/stop-watcher',
        index: '/api/index-project',
        delete: '/api/delete-watcher',
    };
    const url = endpoints[action];
    if (!url || !projectPath) return;
    const projectSelect = document.querySelector('.project-select');
    let reqUrl = url;
    if (projectSelect && projectSelect.value) {
        reqUrl += '?project_id=' + encodeURIComponent(projectSelect.value);
    }
    const target = document.querySelector(targetSel || '#index-health');
    if (!target) return;
    const r = await fetch(reqUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'HX-Request': 'true',
            'HX-Target': (targetSel || '#index-health').replace(/^#/, ''),
        },
        body: JSON.stringify({ project_path: projectPath }),
    });
    if (r.ok) {
        target.innerHTML = await r.text();
        mountHTMXContent(target);
    } else {
        console.error('watcher action:', action, r.status, await r.text());
    }
}

function docSourcesApiUrl(btn) {
    const targetSel = btn.dataset.refreshTarget || '#memory-panel';
    const projectSelect = document.querySelector('.project-select');
    let url = '/api/doc-sources';
    const params = new URLSearchParams();
    const page = parseInt(btn.dataset.docPage, 10);
    if (page > 1) params.set('doc_sources_page', String(page));
    if (projectSelect && projectSelect.value && targetSel !== '#settings-content' && targetSel !== '#memory-panel') {
        params.set('project_id', projectSelect.value);
    }
    const qs = params.toString();
    if (qs) url += '?' + qs;
    return { url, targetSel };
}

async function applyDocSourcesPartial(r, targetSel) {
    const target = document.querySelector(targetSel);
    if (!target) return;
    if (r.ok) {
        target.innerHTML = await r.text();
        mountHTMXContent(target);
        if (targetSel === '#settings-content') {
            mountSettingsContent(target);
        }
    } else {
        console.error('doc source action:', r.status, await r.text());
    }
}

async function loadDocSourcesPage(page, targetSel) {
    const target = document.querySelector(targetSel);
    if (!target || page < 1) return;
    const projectSelect = document.querySelector('.project-select');
    let url = targetSel === '#settings-content' ? '/partials/settings' : targetSel === '#memory-panel' ? '/partials/memory' : '/partials/index-health';
    const params = new URLSearchParams();
    if (page > 1) params.set('doc_sources_page', String(page));
    if (projectSelect && projectSelect.value && targetSel !== '#settings-content' && targetSel !== '#memory-panel') {
        params.set('project_id', projectSelect.value);
    }
    const qs = params.toString();
    if (qs) url += '?' + qs;
    try {
        const r = await fetch(url);
        if (r.ok) {
            target.innerHTML = await r.text();
            mountHTMXContent(target);
            if (targetSel === '#settings-content') {
                mountSettingsContent(target);
            }
        }
    } catch (err) {
        console.error('doc sources page:', err);
    }
}

async function refreshDocSource(btn) {
    const id = parseInt(btn.dataset.docId, 10);
    if (!id) return;
    const { url, targetSel } = docSourcesApiUrl(btn);
    btn.classList.add('htmx-request');
    btn.disabled = true;
    try {
        const r = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'HX-Request': 'true',
                'HX-Target': targetSel.replace(/^#/, ''),
            },
            body: JSON.stringify({ action: 'refresh', id }),
        });
        await applyDocSourcesPartial(r, targetSel);
    } catch (err) {
        console.error('doc source refresh:', err);
    } finally {
        btn.classList.remove('htmx-request');
        btn.disabled = false;
    }
}

async function deleteDocSource(btn) {
    const id = parseInt(btn.dataset.docId, 10);
    if (!id) return;
    const name = btn.dataset.docName || 'this source';
    if (!confirm(`Delete ${name}? Cached content will be removed.`)) return;
    const { url, targetSel } = docSourcesApiUrl(btn);
    btn.classList.add('htmx-request');
    btn.disabled = true;
    try {
        const r = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'HX-Request': 'true',
                'HX-Target': targetSel.replace(/^#/, ''),
            },
            body: JSON.stringify({ action: 'delete', id }),
        });
        await applyDocSourcesPartial(r, targetSel);
    } catch (err) {
        console.error('doc source delete:', err);
    } finally {
        btn.classList.remove('htmx-request');
        btn.disabled = false;
    }
}

const EMBED_TEST_TIMEOUT_MS = 45000;

function formatEmbedTestElapsed(ms) {
    return (ms / 1000).toFixed(1) + 's';
}

function setEmbedTestProgress(out, elapsedMs) {
    out.className = 'perf-hint embed-test-result embed-test-running';
    out.textContent = `Testing… ${formatEmbedTestElapsed(elapsedMs)}`;
}

const EMBED_MODEL_REFRESH_KEYS = new Set([
    'EMBED_OPENAI_BASE_URL',
    'EMBED_OPENAI_API_KEY',
    'OLLAMA_HOST',
    'EMBED_DOCKER_URL',
]);

async function saveEmbedSettingKey(key, value) {
    const r = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, value }),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
        console.error('save embed setting:', key, data.error || r.status);
        return false;
    }
    return true;
}

document.body.addEventListener('change', async (e) => {
    const el = e.target;
    if (!el.closest('.embed-backend-config')) return;
    const key = el.dataset?.settingKey;
    if (!key) return;
    if (await saveEmbedSettingKey(key, el.value) && EMBED_MODEL_REFRESH_KEYS.has(key)) {
        await refreshEmbedModelsForField(el);
    }
});

let embedTextSaveTimer;
document.body.addEventListener('input', (e) => {
    const el = e.target;
    if (!el.closest('.embed-backend-config') || !el.dataset?.settingKey) return;
    if (el.tagName !== 'INPUT' || el.type === 'hidden') return;
    clearTimeout(embedTextSaveTimer);
    const key = el.dataset.settingKey;
    const value = el.value;
    embedTextSaveTimer = setTimeout(async () => {
        if (await saveEmbedSettingKey(key, value) && EMBED_MODEL_REFRESH_KEYS.has(key)) {
            await refreshEmbedModelsForField(el);
        }
    }, 500);
});

function embedModelValue(key) {
    const el = document.getElementById('embedModel-' + key);
    return el ? el.value : '';
}

function embedSettingsPayload() {
    const val = (id) => {
        const el = document.getElementById(id);
        return el ? el.value : '';
    };
    const backend = val('embedBackend');
    const payload = { EMBED_BACKEND: backend };
    switch (backend) {
    case 'onnx':
        payload.MODEL_DIR = val('embedModelDir');
        break;
    case 'http':
        payload.EMBED_HTTP_URL = val('embedHttpURL');
        payload.EMBED_HTTP_BEARER = val('embedHttpBearer');
        break;
    case 'ollama':
        payload.OLLAMA_HOST = val('embedOllamaHost');
        payload.OLLAMA_EMBED_MODEL = embedModelValue('OLLAMA_EMBED_MODEL');
        break;
    case 'openai':
        payload.EMBED_OPENAI_BASE_URL = val('embedOpenAIBase');
        payload.EMBED_OPENAI_API_KEY = val('embedOpenAIKey');
        payload.EMBED_OPENAI_MODEL = embedModelValue('EMBED_OPENAI_MODEL');
        payload.EMBED_OPENAI_DIMENSIONS = val('embedOpenAIDim');
        break;
    case 'docker':
        payload.EMBED_DOCKER_URL = val('embedDockerURL');
        payload.EMBED_DOCKER_MODEL = embedModelValue('EMBED_DOCKER_MODEL');
        payload.EMBED_DOCKER_DIMENSIONS = val('embedDockerDimensions');
        break;
    }
    return payload;
}

function visibleEmbedModelField() {
    const fields = document.querySelectorAll('.embed-model-field');
    for (const f of fields) {
        if (f.offsetParent !== null) return f;
    }
    return null;
}

async function clearEmbedModelField(field) {
    const sel = field.querySelector('.embed-model-select');
    if (!sel) return;
    sel.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Select a model…';
    opt.selected = true;
    opt.disabled = true;
    sel.appendChild(opt);
    const key = field.dataset.settingKey;
    if (key) {
        await saveEmbedSettingKey(key, '');
    }
}

async function refreshEmbedModelField(field, clearSelection = false) {
    const sel = field.querySelector('.embed-model-select');
    const hint = field.querySelector('.embed-models-hint');
    const btn = field.querySelector('.embed-models-refresh');
    if (!sel) return;
    if (clearSelection) {
        await clearEmbedModelField(field);
    }
    const selected = clearSelection ? '' : sel.value;
    if (btn) btn.disabled = true;
    if (hint) {
        hint.className = 'embed-models-hint perf-hint';
        hint.textContent = 'Loading models…';
    }
    try {
        const r = await fetch('/api/embedder/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(embedSettingsPayload()),
        });
        const data = await r.json();
        sel.innerHTML = '';
        const models = data.models || [];
        if (models.length === 0) {
            const opt = document.createElement('option');
            opt.value = selected || '';
            opt.textContent = selected || 'No models found';
            opt.selected = true;
            if (!selected) opt.disabled = true;
            sel.appendChild(opt);
            if (hint) {
                hint.className = 'embed-models-hint perf-hint embed-test-result err';
                hint.textContent = data.error || 'No models returned';
            }
            return;
        }
        const pick = clearSelection ? '' : (data.selected || selected);
        if (pick && !models.includes(pick)) {
            const key = field.dataset.settingKey;
            if (key) await saveEmbedSettingKey(key, '');
        }
        const validPick = pick && models.includes(pick) ? pick : '';
        if (!validPick) {
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = 'Select a model…';
            placeholder.selected = true;
            placeholder.disabled = true;
            sel.appendChild(placeholder);
        }
        models.forEach((m) => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === validPick) opt.selected = true;
            sel.appendChild(opt);
        });
        if (hint) {
            hint.className = 'embed-models-hint perf-hint';
            hint.textContent = models.length + ' model(s) available';
        }
    } catch (err) {
        if (hint) {
            hint.className = 'embed-models-hint perf-hint embed-test-result err';
            hint.textContent = String(err);
        }
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function saveEmbedSettings(backend) {
    const switching = backend !== undefined && backend !== null;
    const payload = embedSettingsPayload();
    if (switching) {
        payload.EMBED_BACKEND = backend;
        delete payload.OLLAMA_EMBED_MODEL;
        delete payload.EMBED_OPENAI_MODEL;
        delete payload.EMBED_DOCKER_MODEL;
    }
    try {
        const r = await fetch('/api/settings/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await r.json();
        if (!r.ok || data.error) {
            console.error('save embed settings:', data.error || r.status);
            return;
        }
        htmx.ajax('GET', '/partials/settings', { target: '#settings-content', swap: 'innerHTML' });
    } catch (err) {
        console.error('save embed settings:', err);
    }
}

document.body.addEventListener('htmx:afterSwap', (e) => {
    if (e.detail.target?.id === 'settings-content' || e.detail.target?.id === 'index-health' || e.detail.target?.id === 'memory-panel') {
        mountHTMXContent(e.detail.target);
    }
    if (e.detail.target?.id === 'index-health') {
        onWorkerPartialUpdated();
    }
});

document.body.addEventListener('htmx:afterRequest', (e) => {
    const key = e.detail.requestConfig?.parameters?.key;
    if (!e.detail.successful) return;
    if (key === 'dashboard_log_tail_lines' || key === 'dashboard_log_line_chars') {
        refreshRecentLogsOnly(true);
    }
    if (key === 'embed_worker_max') {
        refreshWorkerLimitsFromServer().then(() => refreshIndexHealthPartial());
    }
});

window.addEventListener('dashboard-ws-partial', onWorkerPartialUpdated);

const workerUIState = { target: null, server: { total: 0, live: 0, active: 0 }, min: 0, max: 15, perRow: 5, visibleMax: 20 };

function parseWorkerLimit(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

async function refreshWorkerLimitsFromServer() {
    try {
        const r = await fetch('/api/embed-workers');
        if (!r.ok) return false;
        const data = await r.json();
        if (data.max_workers != null) {
            workerUIState.max = parseWorkerLimit(data.max_workers, workerUIState.max);
        }
        const el = document.querySelector('.worker-controls[data-embed-workers]');
        if (el && data.max_workers != null) {
            el.dataset.workerMax = String(workerUIState.max);
        }
        return data.max_workers != null;
    } catch (_) {
        return false;
    }
}

// Max comes from /api/embed-workers; DOM can be stale after settings change or WS skip.
function syncWorkerMaxFromDOM() {
    const el = document.querySelector('.worker-controls[data-embed-workers]');
    if (!el) return;
    workerUIState.max = parseWorkerLimit(el.dataset.workerMax, workerUIState.max);
}

function syncWorkerStateFromDOM() {
    const el = document.querySelector('.worker-controls[data-embed-workers]');
    if (!el) return false;
    workerUIState.server.total = parseInt(el.dataset.embedWorkers, 10) || 0;
    workerUIState.server.live = parseInt(el.dataset.embedWorkersLive, 10) || 0;
    workerUIState.server.active = parseInt(el.dataset.embedActive, 10) || 0;
    workerUIState.min = parseInt(el.dataset.workerMin, 10) || 0;
    workerUIState.perRow = parseInt(el.dataset.workerPerRow, 10) || 5;
    workerUIState.visibleMax = parseInt(el.dataset.workerVisibleMax, 10) || 20;
    if (workerUIState.target !== null && workerUIState.server.total === workerUIState.target) {
        workerUIState.target = null;
    }
    return true;
}

function getWorkerRenderPlan() {
    const target = workerUIState.target ?? workerUIState.server.total;
    const serverTotal = workerUIState.server.total;
    const serverLive = workerUIState.server.live;
    const active = workerUIState.server.active;
    const rawTotal = Math.max(target, serverLive);
    const hasEllipsis = rawTotal > workerUIState.visibleMax;
    const pillCount = hasEllipsis ? workerUIState.visibleMax - 1 : rawTotal;
    const visible = hasEllipsis ? workerUIState.visibleMax : pillCount;
    const compact = workerUIState.max > 15;
    return { target, serverTotal, serverLive, active, rawTotal, pillCount, visible, hasEllipsis, compact, perRow: workerUIState.perRow };
}

function workerPillClass(index, plan) {
    const busy = index < plan.active;
    if (index >= plan.target && index < plan.serverLive) {
        return busy ? 'worker-pill draining draining-busy' : 'worker-pill draining';
    }
    if (index >= plan.serverTotal && index < plan.target) {
        if (index < plan.serverLive) {
            return busy ? 'worker-pill busy' : 'worker-pill idle';
        }
        return 'worker-pill pending';
    }
    return busy ? 'worker-pill busy' : 'worker-pill idle';
}

function buildWorkerStrip(plan) {
    const strip = document.createElement('div');
    strip.className = 'worker-strip';
    if (plan.compact) {
        strip.classList.add('worker-strip-compact');
    } else if (plan.visible > plan.perRow) {
        strip.classList.add('worker-strip-split');
    }
    const busyLabel = `${Math.min(plan.active, plan.target)} of ${plan.target} workers busy`;
    strip.title = busyLabel;
    strip.setAttribute('role', 'img');
    strip.setAttribute('aria-label', busyLabel);
    for (let i = 0; i < plan.pillCount; i++) {
        const pill = document.createElement('span');
        pill.className = workerPillClass(i, plan);
        strip.appendChild(pill);
    }
    if (plan.hasEllipsis) {
        const ell = document.createElement('span');
        ell.className = 'worker-pill worker-pill-ellipsis';
        ell.setAttribute('aria-hidden', 'true');
        ell.textContent = '…';
        strip.appendChild(ell);
    }
    return strip;
}

function workerControlsTitle(plan) {
    if (plan.target === 0) {
        return 'Workers paused — click + to resume';
    }
    const pending = Math.max(0, plan.target - plan.serverTotal);
    const draining = Math.max(0, plan.serverLive - plan.target);
    if (draining > 0 || pending > 0) {
        return `Workers: ${plan.target} target · ${plan.active} busy` +
            (draining ? ` · ${draining} draining` : '') +
            (pending ? ` · ${pending} starting` : '');
    }
    return `Workers: ${plan.active} of ${plan.target} busy`;
}

function renderWorkerUI() {
    const el = document.querySelector('.worker-controls[data-embed-workers]');
    if (!el) return;
    el.dataset.workerMax = String(workerUIState.max);
    el.dataset.workerVisibleMax = String(workerUIState.visibleMax);
    const plan = getWorkerRenderPlan();
    el.classList.toggle('worker-controls-paused', plan.target === 0);
    el.title = workerControlsTitle(plan);
    const label = el.querySelector('.worker-count-label');
    if (label) label.textContent = String(plan.target);
    const oldStrip = el.querySelector('.worker-strip');
    const newStrip = buildWorkerStrip(plan);
    if (oldStrip) {
        oldStrip.replaceWith(newStrip);
    } else {
        const col = el.querySelector('.worker-step-col');
        if (col) col.insertAdjacentElement('afterend', newStrip);
    }
    const minus = el.querySelector('[data-worker-delta="-1"]');
    const plus = el.querySelector('[data-worker-delta="1"]');
    if (minus) minus.disabled = plan.target <= workerUIState.min;
    if (plus) plus.disabled = plan.target >= workerUIState.max;
    const row = el.closest('.worker-metric-row');
    const status = row?.querySelector('.metric-row-value');
    if (status) {
        status.textContent = plan.target === 0 ? 'paused' : `${plan.active} active`;
    }
    updateNavbarWorkers(plan);
}

function updateNavbarWorkers(plan) {
    if (!plan) plan = getWorkerRenderPlan();
    const items = document.querySelectorAll('.health-item');
    for (const item of items) {
        const lab = item.querySelector('.health-label');
        if (!lab || lab.textContent.trim() !== 'Workers') continue;
        let paused = item.querySelector('.health-value-muted');
        const oldStrip = item.querySelector('.worker-strip');
        if (plan.target === 0) {
            if (oldStrip) oldStrip.remove();
            if (!paused) {
                paused = document.createElement('span');
                paused.className = 'health-value health-value-muted';
                paused.textContent = 'paused';
                item.appendChild(paused);
            }
            item.title = 'Workers paused (0) — use + on embeddings card to resume';
            continue;
        }
        if (paused) paused.remove();
        const newStrip = buildWorkerStrip(plan);
        if (oldStrip) {
            oldStrip.replaceWith(newStrip);
        } else {
            item.appendChild(newStrip);
        }
        item.title = workerControlsTitle(plan);
    }
}

function applyWorkerDelta(delta) {
    if (!syncWorkerStateFromDOM()) return;
    const cur = workerUIState.target ?? workerUIState.server.total;
    const next = cur + delta;
    if (next < workerUIState.min || next > workerUIState.max) return;
    workerUIState.target = next;
    renderWorkerUI();
}

async function onWorkerPartialUpdated() {
    syncWorkerStateFromDOM();
    const gotMax = await refreshWorkerLimitsFromServer();
    if (!gotMax) syncWorkerMaxFromDOM();
    if (syncWorkerStateFromDOM()) renderWorkerUI();
}

async function refreshHealthBarPartial() {
    const target = document.querySelector('#health-bar');
    if (!target) return;
    const r = await fetch('/partials/health');
    if (r.ok) {
        target.innerHTML = await r.text();
        mountHTMXContent(target);
    }
}

async function nudgeEmbedderRetry(btn) {
    if (!btn || btn.disabled) return;
    btn.disabled = true;
    btn.classList.add('htmx-request');
    try {
        const r = await fetch('/api/embedder/retry', { method: 'POST' });
        const data = await r.json().catch(() => ({}));
        if (!r.ok && data.error) {
            console.error('embedder retry:', data.error);
        }
        await refreshEmbedderPanels();
    } catch (err) {
        console.error('embedder retry:', err);
    } finally {
        btn.disabled = false;
        btn.classList.remove('htmx-request');
    }
}

async function dismissEmbedderAlert(btn) {
    if (!btn || btn.disabled) return;
    btn.disabled = true;
    btn.classList.add('htmx-request');
    try {
        const r = await fetch('/api/embedder/dismiss-alert', { method: 'POST' });
        const data = await r.json().catch(() => ({}));
        if (!r.ok && data.error) {
            console.error('embedder dismiss:', data.error);
            return;
        }
        await refreshEmbedderPanels();
    } catch (err) {
        console.error('embedder dismiss:', err);
    } finally {
        btn.disabled = false;
        btn.classList.remove('htmx-request');
    }
}

async function refreshEmbedderPanels() {
    await Promise.all([refreshIndexHealthPartial(), refreshHealthBarPartial()]);
    const settings = document.getElementById('settings-content');
    if (settings?.querySelector('.embed-active-row')) {
        const sr = await fetch('/partials/settings');
        if (sr.ok) {
            settings.innerHTML = await sr.text();
            mountSettingsContent(settings);
        }
    }
}

async function refreshIndexHealthPartial() {
    const target = document.querySelector('#index-health');
    if (!target) return;
    const projectSelect = document.querySelector('.project-select');
    let url = '/partials/index-health';
    if (projectSelect && projectSelect.value) {
        url += '?project_id=' + encodeURIComponent(projectSelect.value);
    }
    const r = await fetch(url);
    if (r.ok) {
        target.innerHTML = await r.text();
        mountHTMXContent(target);
        onWorkerPartialUpdated();
    }
}

async function adjustEmbedWorkers(delta) {
    if (!Number.isFinite(delta) || delta === 0) return null;
    const r = await fetch('/api/embed-workers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error) {
        console.error('embed workers:', data.error || r.status);
        workerUIState.target = null;
        onWorkerPartialUpdated();
        return null;
    }
    if (data.max_workers != null) {
        workerUIState.max = parseWorkerLimit(data.max_workers, workerUIState.max);
    }
    return data.workers;
}

document.body.addEventListener('click', async (e) => {
    const workerBtn = e.target.closest('.worker-step-btn');
    if (workerBtn) {
        e.preventDefault();
        const gotMax = await refreshWorkerLimitsFromServer();
        if (!gotMax) syncWorkerMaxFromDOM();
        if (!syncWorkerStateFromDOM()) return;
        renderWorkerUI();
        const delta = parseInt(workerBtn.dataset.workerDelta, 10);
        if (!Number.isFinite(delta) || delta === 0) return;
        const plan = getWorkerRenderPlan();
        if (delta < 0 && plan.target <= workerUIState.min) return;
        if (delta > 0 && plan.target >= workerUIState.max) return;
        applyWorkerDelta(delta);
        adjustEmbedWorkers(delta);
        return;
    }
    const pagerBtn = e.target.closest('.doc-sources-pager-btn');
    if (pagerBtn && !pagerBtn.disabled) {
        e.preventDefault();
        const page = parseInt(pagerBtn.dataset.docPage, 10);
        loadDocSourcesPage(page, pagerBtn.dataset.refreshTarget || '#memory-panel');
        return;
    }
    const delBtn = e.target.closest('.doc-source-delete');
    if (delBtn) {
        e.preventDefault();
        deleteDocSource(delBtn);
        return;
    }
    const btn = e.target.closest('.doc-source-refresh');
    if (btn) {
        e.preventDefault();
        refreshDocSource(btn);
        return;
    }
    const wbtn = e.target.closest('.watcher-action');
    if (wbtn) {
        e.preventDefault();
        const action = wbtn.dataset.watcherAction;
        const projectPath = wbtn.dataset.projectPath;
        if (action === 'delete') {
            const name = wbtn.dataset.projectName || projectPath;
            if (!confirm(`Delete ${name}? This removes all indexed data.`)) return;
        }
        wbtn.disabled = true;
        watcherApiAction(action, projectPath, '#index-health').finally(() => {
            wbtn.disabled = false;
        });
        return;
    }
    const retryBtn = e.target.closest('[data-embedder-retry]');
    if (retryBtn) {
        e.preventDefault();
        nudgeEmbedderRetry(retryBtn);
        return;
    }
    const dismissBtn = e.target.closest('[data-embedder-dismiss]');
    if (dismissBtn) {
        e.preventDefault();
        dismissEmbedderAlert(dismissBtn);
    }
});

async function refreshEmbedModelsForField(el) {
    const field = el?.closest?.('.embed-model-field') || visibleEmbedModelField();
    if (field) await refreshEmbedModelField(field);
}

document.body.addEventListener('click', async (e) => {
    if (e.target.classList.contains('embed-models-refresh')) {
        const field = e.target.closest('.embed-model-field');
        if (field) await refreshEmbedModelField(field, true);
        return;
    }
    if (e.target.id !== 'embedTestBtn') return;
    const btn = e.target;
    const out = document.getElementById('embedTestResult');
    if (!out) return;
    btn.disabled = true;
    btn.classList.add('embed-test-running');
    out.style.display = '';
    const started = performance.now();
    setEmbedTestProgress(out, 0);
    const timer = setInterval(() => {
        setEmbedTestProgress(out, performance.now() - started);
    }, 100);
    const ac = new AbortController();
    const abortTimer = setTimeout(() => ac.abort(), EMBED_TEST_TIMEOUT_MS);
    try {
        const r = await fetch('/api/embedder/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(embedSettingsPayload()),
            signal: ac.signal,
        });
        const data = await r.json();
        if (data.ok) {
            out.className = 'perf-hint embed-test-result ok';
            let msg = `OK · ${data.backend} · ${data.model} · ${data.dimensions} dims · ${data.latency_ms}ms`;
            if (data.endpoint) msg += ` · ${data.endpoint}`;
            if (data.env_overrides && data.env_overrides.length) {
                msg += ` (env overrides active at runtime: ${data.env_overrides.join(', ')})`;
            }
            out.textContent = msg;
        } else {
            out.className = 'perf-hint embed-test-result err';
            out.textContent = data.error || 'Test failed';
        }
    } catch (err) {
        out.className = 'perf-hint embed-test-result err';
        if (err.name === 'AbortError') {
            out.textContent = `Timed out after ${formatEmbedTestElapsed(EMBED_TEST_TIMEOUT_MS)}`;
        } else {
            out.textContent = String(err);
        }
    } finally {
        clearInterval(timer);
        clearTimeout(abortTimer);
        btn.classList.remove('embed-test-running');
        btn.disabled = false;
    }
});