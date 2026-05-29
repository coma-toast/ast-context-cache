const projectFilteredTargets = new Set([
    '#stats-cards',
    '#index-health',
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
                    target.innerHTML = msg.data.html;
                    Alpine.flushSync();
                    window.dispatchEvent(new CustomEvent('dashboard-ws-partial'));
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
        settingsOpen: false,

        projectLabel() {
            if (!this.selectedProject) return '';
            const parts = this.selectedProject.split('/');
            return parts[parts.length - 1] || this.selectedProject;
        },

        async openSettings() {
            this.settingsOpen = true;
            const el = document.getElementById('settings-content');
            if (!el) return;
            const needsLoad = !el.querySelector('.section') && !el.querySelector('.project-list');
            if (needsLoad) {
                const r = await fetch('/partials/settings');
                if (r.ok) el.innerHTML = await r.text();
            }
        },

        async applyProjectFilter() {
            const project = this.selectedProject;
            const q = project ? `?project_id=${encodeURIComponent(project)}` : '';
            const panels = [
                ['#stats-cards', '/partials/stats'],
                ['#index-health', '/partials/index-health'],
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
                if (r.ok) el.innerHTML = await r.text();
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

function embedModelValue(key) {
    const el = document.getElementById('embedModel-' + key);
    return el ? el.value : '';
}

function embedSettingsPayload() {
    const val = (id) => {
        const el = document.getElementById(id);
        return el ? el.value : '';
    };
    return {
        EMBED_BACKEND: val('embedBackend'),
        MODEL_DIR: val('embedModelDir'),
        EMBED_HTTP_URL: val('embedHttpURL'),
        EMBED_HTTP_BEARER: val('embedHttpBearer'),
        OLLAMA_HOST: val('embedOllamaHost'),
        OLLAMA_EMBED_MODEL: embedModelValue('OLLAMA_EMBED_MODEL'),
        EMBED_OPENAI_BASE_URL: val('embedOpenAIBase'),
        EMBED_OPENAI_API_KEY: val('embedOpenAIKey'),
        EMBED_OPENAI_MODEL: embedModelValue('EMBED_OPENAI_MODEL'),
        EMBED_OPENAI_DIMENSIONS: val('embedOpenAIDim'),
        EMBED_DOCKER_URL: val('embedDockerURL'),
        EMBED_DOCKER_MODEL: embedModelValue('EMBED_DOCKER_MODEL'),
        EMBED_DOCKER_DIMENSIONS: val('embedDockerDimensions'),
    };
}

function visibleEmbedModelField() {
    const fields = document.querySelectorAll('.embed-model-field');
    for (const f of fields) {
        if (f.offsetParent !== null) return f;
    }
    return null;
}

async function refreshEmbedModelField(field) {
    const sel = field.querySelector('.embed-model-select');
    const hint = field.querySelector('.embed-models-hint');
    const btn = field.querySelector('.embed-models-refresh');
    if (!sel) return;
    const selected = sel.value;
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
        const pick = data.selected || selected;
        models.forEach((m) => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === pick) opt.selected = true;
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

async function refreshEmbedModelsForField(el) {
    const field = el?.closest?.('.embed-model-field') || visibleEmbedModelField();
    if (field) await refreshEmbedModelField(field);
}

document.body.addEventListener('click', async (e) => {
    if (e.target.classList.contains('embed-models-refresh')) {
        const field = e.target.closest('.embed-model-field');
        if (field) await refreshEmbedModelField(field);
        return;
    }
    if (e.target.id !== 'embedTestBtn') return;
    const btn = e.target;
    const out = document.getElementById('embedTestResult');
    if (!out) return;
    btn.disabled = true;
    out.className = 'perf-hint embed-test-result';
    out.textContent = 'Testing…';
    try {
        const r = await fetch('/api/embedder/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(embedSettingsPayload()),
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
        out.textContent = String(err);
    } finally {
        btn.disabled = false;
    }
});