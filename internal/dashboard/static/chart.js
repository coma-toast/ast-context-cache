// Shared helpers for pages that load chart.js without ws.js (e.g. Layout).
// Main dashboard uses ws.js for WebSocket + Alpine; avoid duplicating Alpine components here.

if (typeof fmtNum === 'undefined') {
    function fmtNum(n) {
        if (n == null) return '-';
        return n.toLocaleString();
    }
}

document.body.addEventListener('htmx:configRequest', (event) => {
    const projectSelect = document.querySelector('.project-select');
    if (projectSelect && projectSelect.value) {
        const separator = event.detail.path.includes('?') ? '&' : '?';
        event.detail.path += separator + 'project_id=' + encodeURIComponent(projectSelect.value);
    }
});
