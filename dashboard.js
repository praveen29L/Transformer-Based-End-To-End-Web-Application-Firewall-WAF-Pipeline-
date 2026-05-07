// WAF Dashboard Logic

const API_BASE = ""; // Relative path

// State
let isAutoRefresh = true;
let startTime = Date.now();

// Elements
const els = {
    statusBadge: document.getElementById('res-status-badge'),
    confVal: document.getElementById('res-confidence-val'),
    confBar: document.getElementById('res-confidence-bar'),
    resLabel: document.getElementById('res-label'),
    resThreat: document.getElementById('res-threat'),
    resReason: document.getElementById('res-reason'),
    resLatency: document.getElementById('res-latency'),
    resultPlaceholder: document.getElementById('result-placeholder'),
    resultContent: document.getElementById('result-content'),

    logsTable: document.getElementById('logs-table-body'),
    statTotal: document.getElementById('stat-total'),
    statBlocked: document.getElementById('stat-blocked'),
    uptime: document.getElementById('uptime'),

    inputs: {
        method: document.getElementById('sim-method'),
        path: document.getElementById('sim-path'),
        body: document.getElementById('sim-body'),
        preset: document.getElementById('preset-payloads')
    },

    btnSend: document.getElementById('btn-send-attack'),
    btnRefresh: document.getElementById('refresh-logs'),
    btnReset: document.getElementById('reset-stats'),
    bodyGroup: document.getElementById('body-input-group')
};

// Presets
const presets = {
    'safe': { method: 'GET', path: '/products?id=123&category=electronics', body: '' },
    'sqli': { method: 'GET', path: "/login?user=admin' OR '1'='1", body: '' },
    'xss': { method: 'GET', path: "/search?q=<script>alert('pwned')</script>", body: '' },
    'traversal': { method: 'GET', path: '/download?file=../../etc/passwd', body: '' },
    'cmd': { method: 'GET', path: '/ping?ip=8.8.8.8 | whoami', body: '' }
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    updateUptime();
    fetchStats();
    fetchLogs();

    // Interval for updates
    setInterval(() => {
        if (isAutoRefresh) {
            fetchStats();
            fetchLogs();
        }
    }, 2000);

    setInterval(updateUptime, 1000);

    // Events
    els.btnSend.addEventListener('click', sendAttack);
    els.btnRefresh.addEventListener('click', () => { fetchLogs(); fetchStats(); });
    els.btnReset.addEventListener('click', resetStats);

    els.inputs.preset.addEventListener('change', (e) => {
        const val = e.target.value;
        if (presets[val]) {
            els.inputs.method.value = presets[val].method;
            els.inputs.path.value = presets[val].path;
            els.inputs.body.value = presets[val].body;
            toggleBodyInput();
        }
    });

    els.inputs.method.addEventListener('change', toggleBodyInput);
});

function toggleBodyInput() {
    const method = els.inputs.method.value;
    if (method === 'POST' || method === 'PUT') {
        els.bodyGroup.style.display = 'flex';
    } else {
        els.bodyGroup.style.display = 'none';
        els.inputs.body.value = '';
    }
}

function updateUptime() {
    const diff = Math.floor((Date.now() - startTime) / 1000);
    const h = Math.floor(diff / 3600).toString().padStart(2, '0');
    const m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(diff % 60).toString().padStart(2, '0');
    els.uptime.textContent = `Uptime: ${h}:${m}:${s}`;
}

async function sendAttack() {
    const method = els.inputs.method.value;
    const path = els.inputs.path.value;
    const body = els.inputs.body.value;

    els.btnSend.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analying...';
    els.btnSend.disabled = true;

    try {
        const fetchOptions = {
            method: method,
            headers: {
                'Content-Type': 'application/json' // Defaulting to JSON for simplicity
            }
        };

        if (body && (method === 'POST' || method === 'PUT')) {
            // Try to parse JSON, otherwise send as plain text?
            // For WAF testing, plain text body is common.
            // But typical apps expect JSON. Let's try to send as is.
            fetchOptions.body = body;
        }

        const response = await fetch(API_BASE + path, fetchOptions);

        // WAF might return 403 (Blocked) or 200 (Allowed) or 500
        // Our WAF returns JSON in all cases roughly.
        // If blocked, status is 403, and body is JSON.

        let data;
        try {
            data = await response.json();
        } catch (e) {
            data = { status: `HTTP ${response.status}`, message: "Non-JSON response" };
        }

        displayResult(data, response.status);

        // Force refresh logs
        setTimeout(fetchLogs, 500);
        setTimeout(fetchStats, 500);

    } catch (err) {
        console.error("Attack failed", err);
        displayResult({ status: 'ERROR', message: err.message }, 0);
    } finally {
        els.btnSend.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Request';
        els.btnSend.disabled = false;
    }
}

function displayResult(data, httpStatus) {
    els.resultPlaceholder.style.display = 'none';
    els.resultContent.style.display = 'flex';

    // Status Logic
    let status = 'UNKNOWN';
    let isBlocked = false;

    if (httpStatus === 403 || data.status === "BLOCKED") {
        status = 'BLOCKED';
        isBlocked = true;
    } else if (data.status === "ALLOW" || data.status === "ALLOWED") {
        status = 'ALLOWED';
    } else {
        status = data.status || `HTTP ${httpStatus}`;
    }

    // Update UI
    els.statusBadge.textContent = status;
    els.statusBadge.className = `status-badge ${isBlocked ? 'BLOCKED' : 'ALLOWED'}`;

    // Confidence
    let confidence = 0;
    if (data.confidence) confidence = data.confidence;
    if (data.analysis && data.analysis.confidence) confidence = data.analysis.confidence;

    const confPct = Math.round(confidence * 100);
    els.confVal.textContent = `${confPct}%`;
    els.confBar.style.width = `${confPct}%`;

    // Details
    let label = 'N/A';
    if (data.attack_type) label = data.attack_type;
    if (data.analysis && data.analysis.prediction) label = data.analysis.prediction;

    els.resLabel.textContent = label;
    els.resReason.textContent = data.reason || data.message || '-';
    els.resThreat.textContent = data.threat_level || (data.analysis ? data.analysis.threat_level : 'None');

    const lat = data.latency_ms || 0;
    els.resLatency.textContent = `${lat.toFixed(1)}ms`;
}

async function fetchStats() {
    try {
        const res = await fetch('/stats');
        const data = await res.json();

        els.statTotal.textContent = data.total_requests || 0;
        els.statBlocked.textContent = data.blocked_requests || 0;
    } catch (e) {
        console.warn("Stats fetch failed");
    }
}

async function fetchLogs() {
    try {
        const res = await fetch('/logs/recent?limit=20');
        const data = await res.json();

        if (data.logs) {
            renderLogs(data.logs.reverse()); // Show newest first
        }
    } catch (e) {
        console.warn("Logs fetch failed");
    }
}

function renderLogs(logs) {
    const html = logs.map(log => {
        const time = new Date(log.timestamp).toLocaleTimeString();
        const methodClass = `method-${log.method}`;
        const actionClass = log.decision.action === 'BLOCK' ? 'action-BLOCK' : 'action-ALLOW';
        const label = log.prediction.label;
        const conf = Math.round(log.prediction.confidence * 100);

        return `
            <tr>
                <td>${time}</td>
                <td><span class="log-method ${methodClass}">${log.method}</span></td>
                <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${log.path}?${log.query}">${log.path}</td>
                <td>${label}</td>
                <td>${conf}%</td>
                <td><span class="log-action ${actionClass}">${log.decision.action}</span></td>
            </tr>
        `;
    }).join('');

    els.logsTable.innerHTML = html;
}

async function resetStats() {
    if (confirm("Reset all WAF statistics?")) {
        await fetch('/stats/reset', { method: 'POST' });
        fetchStats();
        fetchLogs();
    }
}