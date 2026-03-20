/* recaper web — shared utilities */

/* ===== Pipeline stage definitions ===== */
const PIPELINE_STAGES = [
    { key: 'unpack',    label: 'Распаковка', icon: '\uD83D\uDCE6' },
    { key: 'detect',    label: 'Детекция',   icon: '\uD83D\uDD0D' },
    { key: 'extract',   label: 'Извлечение', icon: '\u2702\uFE0F' },
    { key: 'analyze',   label: 'Анализ',     icon: '\uD83D\uDD2C' },
    { key: 'script',    label: 'Сценарий',   icon: '\uD83D\uDCDD' },
    { key: 'voiceover', label: 'Озвучка',    icon: '\uD83C\uDF99\uFE0F' },
    { key: 'render',    label: 'Рендер',     icon: '\uD83C\uDFAC' },
];

/* ===== Format file size ===== */
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
    return (bytes / 1073741824).toFixed(1) + ' GB';
}

/* ===== API call helper ===== */
async function apiCall(method, path, body) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
    }
    return res.json();
}

/* ===== Russian pluralization ===== */
function pluralize(n, one, few, many) {
    const abs = Math.abs(n);
    const mod10 = abs % 10;
    const mod100 = abs % 100;
    if (mod10 === 1 && mod100 !== 11) return one;
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
    return many;
}

/* ===== Relative time (Russian) ===== */
function timeAgo(timestamp) {
    const seconds = Math.floor(Date.now() / 1000 - timestamp);
    if (seconds < 30) return 'только что';
    if (seconds < 60) return seconds + ' сек назад';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + ' ' + pluralize(minutes, 'минуту', 'минуты', 'минут') + ' назад';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + ' ' + pluralize(hours, 'час', 'часа', 'часов') + ' назад';
    const days = Math.floor(hours / 24);
    return days + ' ' + pluralize(days, 'день', 'дня', 'дней') + ' назад';
}

/* ===== Elapsed time formatting ===== */
function formatElapsed(seconds) {
    seconds = Math.floor(seconds);
    if (seconds < 60) return seconds + ' сек';
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m < 60) return m + ' мин ' + s + ' сек';
    const h = Math.floor(m / 60);
    return h + ' ч ' + (m % 60) + ' мин';
}

/* ===== Format timestamp to HH:MM:SS ===== */
function formatTime(timestamp) {
    const d = new Date(timestamp * 1000);
    return d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/* ===== Toast notification system ===== */
function toast(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;
    const container = document.getElementById('toast-container');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'toast toast-' + type;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(function () {
        el.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(function () { el.remove(); }, 300);
    }, duration);
}

/* ===== Get pipeline stage index ===== */
function getStageIndex(stageKey) {
    for (let i = 0; i < PIPELINE_STAGES.length; i++) {
        if (PIPELINE_STAGES[i].key === stageKey) return i;
    }
    return -1;
}

/* ===== Build mini stepper HTML for job cards ===== */
function buildMiniStepper(job) {
    const currentIdx = getStageIndex(job.current_stage);
    let html = '<div class="mini-stepper">';
    for (let i = 0; i < PIPELINE_STAGES.length; i++) {
        let cls = 'mini-step';
        if (job.status === 'completed') {
            cls += ' done';
        } else if (job.status === 'running' || job.status === 'queued') {
            if (i < currentIdx) cls += ' done';
            else if (i === currentIdx) cls += ' active';
        } else if (job.status === 'failed') {
            if (i < currentIdx) cls += ' done';
            else if (i === currentIdx) cls += ' active';
        }
        html += '<div class="' + cls + '"></div>';
    }
    html += '</div>';
    return html;
}

/* ===== Render job list (for auto-refresh) ===== */
function renderJobList(jobs) {
    const container = document.getElementById('jobs-list');
    if (!container) return;

    if (!jobs.length) {
        container.innerHTML =
            '<div class="empty-state">' +
            '<div class="empty-icon">\uD83D\uDCDA</div>' +
            '<div class="empty-title">Нет заданий</div>' +
            '<div class="empty-hint">Создайте новое задание выше, указав путь к файлу манги</div>' +
            '</div>';
        return;
    }

    container.innerHTML = jobs.map(function (job) {
        let card = '<a href="/jobs/' + job.id + '/view" class="job-card">';
        card += '<div class="job-header">';
        card += '<span class="job-id">#' + job.id.substring(0, 8) + '</span>';
        card += '<span class="badge badge-' + job.status + '">' + job.status + '</span>';
        card += '</div>';
        card += '<div class="job-source">' + escapeHtml(job.source) + '</div>';
        if (job.title) card += '<div class="job-title">' + escapeHtml(job.title) + '</div>';
        if (job.status === 'running' || job.status === 'queued') {
            card += '<div class="progress-bar-wrap"><div class="progress-bar animated" style="width:' + job.progress.toFixed(1) + '%"></div></div>';
            card += '<div class="job-meta">';
            if (job.current_stage) card += '<span>' + job.current_stage + ' — ' + job.progress.toFixed(1) + '%</span>';
            card += buildMiniStepper(job);
            if (job.created_at) card += '<span>' + timeAgo(job.created_at) + '</span>';
            card += '</div>';
        } else {
            card += '<div class="job-meta">';
            card += buildMiniStepper(job);
            if (job.created_at) card += '<span>' + timeAgo(job.created_at) + '</span>';
            card += '</div>';
        }
        if (job.error) card += '<div class="job-error">' + escapeHtml(job.error) + '</div>';
        card += '</a>';
        return card;
    }).join('');
}

/* ===== Render stats ===== */
function renderStats(stats) {
    var el;
    el = document.getElementById('stat-total');
    if (el) el.textContent = stats.total;
    el = document.getElementById('stat-running');
    if (el) el.textContent = stats.running;
    el = document.getElementById('stat-completed');
    if (el) el.textContent = stats.completed;
    el = document.getElementById('stat-failed');
    if (el) el.textContent = stats.failed;
}

/* ===== Auto-refresh job list and stats ===== */
function startJobListRefresh(interval) {
    interval = interval || 5000;
    setInterval(async function () {
        try {
            const [jobsRes, statsRes] = await Promise.all([
                fetch('/api/jobs'),
                fetch('/api/jobs/stats'),
            ]);
            if (jobsRes.ok && statsRes.ok) {
                const jobs = await jobsRes.json();
                const stats = await statsRes.json();
                renderJobList(jobs);
                renderStats(stats);
            }
        } catch (e) {
            // silent
        }
    }, interval);
}

/* ===== Escape HTML ===== */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/* ===== Update pipeline stepper on job detail page ===== */
function updatePipelineStepper(currentStage, status) {
    const idx = getStageIndex(currentStage);
    const steps = document.querySelectorAll('.pipeline-step');
    const connectors = document.querySelectorAll('.pipeline-connector');

    steps.forEach(function (step, i) {
        step.classList.remove('completed', 'active', 'failed', 'pending');
        if (status === 'completed') {
            step.classList.add('completed');
        } else if (status === 'failed') {
            if (i < idx) step.classList.add('completed');
            else if (i === idx) step.classList.add('failed');
        } else {
            if (i < idx) step.classList.add('completed');
            else if (i === idx) step.classList.add('active');
        }
    });

    connectors.forEach(function (conn, i) {
        conn.classList.remove('completed', 'active');
        if (status === 'completed') {
            conn.classList.add('completed');
        } else {
            if (i < idx) conn.classList.add('completed');
            else if (i === idx) conn.classList.add('active');
        }
    });
}
