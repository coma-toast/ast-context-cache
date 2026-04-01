document.addEventListener('alpine:init', () => {
    Alpine.store('chart', {
        interval: 'daily',
        metric: 'queries',
        data: []
    });

    Alpine.data('dashboard', () => ({
        selectedProject: '',
        settingsOpen: false,
        refreshing: false,
        progressing: true,

        init() {
            this.startProgressBar();
        },

        async refresh() {
            this.refreshing = true;
            htmx.trigger(document.body, 'refresh');
            setTimeout(() => {
                this.refreshing = false;
                this.resetProgressBar();
            }, 1000);
        },

        startProgressBar() {
            this.progressing = false;
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    this.progressing = true;
                });
            });
            this._progressInterval = setInterval(() => {
                this.progressing = false;
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        this.progressing = true;
                    });
                });
            }, 30000);
        },

        resetProgressBar() {
            this.progressing = false;
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    this.progressing = true;
                });
            });
        }
    }));

    Alpine.data('timeseriesChart', () => ({
        chartData: null,

        init() {
            this.$watch('$store.chart.interval', () => this.loadAndDraw());
            this.$watch('$store.chart.metric', () => this.loadAndDraw());
            window.addEventListener('resize', () => this.draw());
        },

        handleData(event) {
            if (event.detail && event.detail.target && event.detail.target.id === 'activity-chart') {
                try {
                    this.chartData = JSON.parse(event.detail.xhr.responseText);
                    this.draw();
                } catch(e) {}
            }
        },

        async loadAndDraw() {
            try {
                const project = Alpine.store('chart').selectedProject || '';
                const interval = Alpine.store('chart').interval;
                const sep = project ? '&' : '';
                const projParam = project ? `project_id=${encodeURIComponent(project)}` : '';
                const r = await fetch(`/api/timeseries?interval=${interval}&days=30${sep}${projParam}`);
                this.chartData = await r.json();
                this.draw();
            } catch(e) {}
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
                ctx.fillText('No data for this period', W/2, H/2);
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
                const y = padT + cH - (i/4) * cH;
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

// Auto-remove toasts after 4 seconds using MutationObserver
(function() {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const MAX_TOASTS = 5;
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                setTimeout(() => {
                    node.classList.add('removing');
                    setTimeout(() => node.remove(), 200);
                }, 4000);
            }
        }
        while (container.children.length > MAX_TOASTS) {
            container.lastElementChild.remove();
        }
    });
    observer.observe(container, { childList: true });
})();

// Include project filter in all htmx requests
document.body.addEventListener('htmx:configRequest', (event) => {
    const projectSelect = document.querySelector('.project-select');
    if (projectSelect && projectSelect.value) {
        const separator = event.detail.path.includes('?') ? '&' : '?';
        event.detail.path += separator + 'project_id=' + encodeURIComponent(projectSelect.value);
    }
});
