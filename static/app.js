const API_URL = '';

const btnRun = document.getElementById('btn-run');
const btnClearLogs = document.getElementById('btn-clear-logs');
const statusBadge = document.getElementById('status-badge');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const logTerminal = document.getElementById('log-terminal');
const attrMapSelector = document.getElementById('attr-map-selector');
const attrCanvas = document.getElementById('attr-canvas');

const statCacc = document.getElementById('metric-cacc');
const statAhl = document.getElementById('metric-ahl');
const statCsk = document.getElementById('metric-csk');
const statZk = document.getElementById('metric-zk');

const ctx = attrCanvas.getContext('2d');

let chartResilienceInstance = null;
let chartDistillationInstance = null;
let pollingInterval = null;

function initCanvasPlaceholder() {
    ctx.fillStyle = '#0a0e1b';
    ctx.fillRect(0, 0, attrCanvas.width, attrCanvas.height);
    ctx.fillStyle = '#9ca3af';
    ctx.font = '14px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText('No Map Loaded', attrCanvas.width / 2, attrCanvas.height / 2);
}

function drawAttributionMap(matrix) {
    if (!matrix || matrix.length === 0) {
        initCanvasPlaceholder();
        return;
    }
    const h = matrix.length;
    const w = matrix[0].length;
    const imgData = ctx.createImageData(attrCanvas.width, attrCanvas.height);

    const scaleX = attrCanvas.width / w;
    const scaleY = attrCanvas.height / h;

    let max = -1e9;
    let min = 1e9;
    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            if (matrix[r][c] > max) max = matrix[r][c];
            if (matrix[r][c] < min) min = matrix[r][c];
        }
    }

    const range = max - min + 1e-8;

    for (let y = 0; y < attrCanvas.height; y++) {
        for (let x = 0; x < attrCanvas.width; x++) {
            const matrixY = Math.floor(y / scaleY);
            const matrixX = Math.floor(x / scaleX);

            const val = matrix[matrixY][matrixX];
            const normVal = Math.floor(((val - min) / range) * 255);

            let r = 0, g = 0, b = 0;
            if (normVal < 128) {
                r = Math.floor(normVal * 0.9);
                g = 0;
                b = Math.floor(normVal * 1.5);
            } else {
                const ratio = (normVal - 128) / 127;
                r = Math.floor(115 * (1 - ratio) + 255 * ratio);
                g = Math.floor(242 * ratio);
                b = Math.floor(200 * (1 - ratio) + 254 * ratio);
            }

            const pixelIdx = (y * attrCanvas.width + x) * 4;
            imgData.data[pixelIdx] = r;
            imgData.data[pixelIdx+1] = g;
            imgData.data[pixelIdx+2] = b;
            imgData.data[pixelIdx+3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function appendLog(line) {
    const isScrollBottom = logTerminal.scrollHeight - logTerminal.clientHeight <= logTerminal.scrollTop + 5;

    const div = document.createElement('div');
    div.className = 'terminal-line';

    if (line.includes('[SYSTEM]')) {
        div.className += ' system-line';
    } else if (line.includes('Epoch')) {
        div.className += ' epoch-line';
    } else if (line.includes('Error') || line.includes('Traceback') || line.includes('RuntimeError')) {
        div.className += ' error-line';
    } else if (line.includes('complete') || line.includes('SUCCESS') || line.includes('ready')) {
        div.className += ' success-line';
    }

    div.innerText = line;
    logTerminal.appendChild(div);

    if (isScrollBottom) {
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }
}

btnClearLogs.addEventListener('click', () => {
    logTerminal.innerHTML = '';
});

function updateStatusBadge(status) {
    statusBadge.innerText = status;
    statusBadge.className = 'badge';

    if (status === 'idle') statusBadge.classList.add('badge-idle');
    else if (status === 'running') statusBadge.classList.add('badge-running');
    else if (status === 'done') statusBadge.classList.add('badge-done');
    else if (status === 'error') statusBadge.classList.add('badge-error');
}

async function startPipeline() {
    try {
        btnRun.disabled = true;
        progressContainer.style.display = 'block';
        updateStatusBadge('running');

        const response = await fetch(`${API_URL}/api/start-pipeline`, { method: 'POST' });
        const data = await response.json();

        logTerminal.innerHTML = '';
        appendLog(`[SYSTEM] ${data.msg}`);

        if (pollingInterval) clearInterval(pollingInterval);
        pollingInterval = setInterval(pollStatus, 1500);

    } catch (err) {
        appendLog(`[SYSTEM] Connection error: ${err}`);
        btnRun.disabled = false;
        updateStatusBadge('error');
    }
}

async function pollStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`);
        const data = await response.json();

        updateStatusBadge(data.status);
        progressBar.style.width = `${data.progress}%`;
        progressText.innerText = `${data.progress}%`;

        logTerminal.innerHTML = '';
        data.logs.forEach(line => appendLog(line));

        if (data.status === 'done' || data.status === 'error') {
            clearInterval(pollingInterval);
            btnRun.disabled = false;
            if (data.status === 'done') {
                progressContainer.style.display = 'none';
                fetchMetrics();
                fetchAttributionMap();
            }
        }
    } catch (err) {
        appendLog(`[SYSTEM] Status polling error: ${err}`);
    }
}

async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/api/metrics`);
        const metrics = await response.json();
        if (!metrics) return;

        const originalScenario = metrics.scenarios.find(s => s.scenario === "No Attack");
        if (originalScenario) {
            statCacc.innerText = `${(originalScenario.cacc * 100).toFixed(1)}%`;
            statAhl.innerText = `${(originalScenario.ahl * 100).toFixed(1)}%`;
            statCsk.innerText = `${(originalScenario.fama_d * 100).toFixed(1)}%`;
            statZk.innerText = originalScenario.zk_snark ? 'VALID' : 'INVALID';
            statZk.style.color = originalScenario.zk_snark ? 'var(--green)' : 'var(--red)';
        }

        renderResilienceChart(metrics.scenarios);
        renderDistillationChart(metrics.student);

    } catch (err) {
        console.error('Error fetching metrics:', err);
    }
}

async function fetchAttributionMap() {
    const selectedMap = attrMapSelector.value;
    try {
        const response = await fetch(`${API_URL}/api/attribution-map?name=${encodeURIComponent(selectedMap)}`);
        const data = await response.json();
        drawAttributionMap(data.map);
    } catch (err) {
        console.error('Error loading map:', err);
    }
}

attrMapSelector.addEventListener('change', fetchAttributionMap);
btnRun.addEventListener('click', startPipeline);

function renderResilienceChart(scenarios) {
    const labels = scenarios.map(s => s.scenario);
    const caccData = scenarios.map(s => s.cacc * 100);
    const ahlData = scenarios.map(s => s.ahl * 100);
    const famaData = scenarios.map(s => s.fama_d * 100);

    if (chartResilienceInstance) {
        chartResilienceInstance.destroy();
    }

    const ctxChart = document.getElementById('chart-resilience').getContext('2d');
    chartResilienceInstance = new Chart(ctxChart, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Clean Accuracy (CACC)',
                    data: caccData,
                    backgroundColor: 'rgba(0, 242, 254, 0.4)',
                    borderColor: '#00f2fe',
                    borderWidth: 1.5
                },
                {
                    label: 'Baseline AHL Match',
                    data: ahlData,
                    backgroundColor: 'rgba(157, 78, 221, 0.4)',
                    borderColor: '#9d4edd',
                    borderWidth: 1.5
                },
                {
                    label: 'FAMA-D CSK Match',
                    data: famaData,
                    backgroundColor: 'rgba(255, 0, 127, 0.4)',
                    borderColor: '#ff007f',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6', font: { family: 'Outfit', size: 10 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 10 } },
                    max: 100,
                    min: 0
                }
            }
        }
    });
}

function renderDistillationChart(student) {
    if (chartDistillationInstance) {
        chartDistillationInstance.destroy();
    }

    const ctxChart = document.getElementById('chart-distillation').getContext('2d');
    chartDistillationInstance = new Chart(ctxChart, {
        type: 'bar',
        data: {
            labels: ['Student CACC', 'Baseline AHL Match', 'FAMA-D CLADA Match'],
            datasets: [{
                label: 'Student Verification (%)',
                data: [student.cacc * 100, student.ahl * 100, student.clada * 100],
                backgroundColor: [
                    'rgba(0, 242, 254, 0.4)',
                    'rgba(157, 78, 221, 0.4)',
                    'rgba(0, 230, 118, 0.4)'
                ],
                borderColor: [
                    '#00f2fe',
                    '#9d4edd',
                    '#00e676'
                ],
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 10 } },
                    max: 100,
                    min: 0
                }
            }
        }
    });
}

const stressAttackType = document.getElementById('stress-attack-type');
const stressParam = document.getElementById('stress-param');
const stressParamVal = document.getElementById('stress-param-val');
const stressParamLabel = document.getElementById('stress-param-label');
const btnApplyStress = document.getElementById('btn-apply-stress');

stressAttackType.addEventListener('change', () => {
    const val = stressAttackType.value;
    if (val === 'prune') {
        stressParam.min = '0.0';
        stressParam.max = '0.95';
        stressParam.step = '0.05';
        stressParam.value = '0.50';
        stressParamLabel.innerText = 'Ratio:';
        stressParamVal.innerText = '0.50';
    } else if (val === 'quantize') {
        stressParam.min = '2';
        stressParam.max = '8';
        stressParam.step = '1';
        stressParam.value = '8';
        stressParamLabel.innerText = 'Bits:';
        stressParamVal.innerText = '8';
    }
});

stressParam.addEventListener('input', () => {
    stressParamVal.innerText = parseFloat(stressParam.value).toFixed(2);
});

btnApplyStress.addEventListener('click', async () => {
    try {
        btnApplyStress.disabled = true;
        const type = stressAttackType.value;
        const param = parseFloat(stressParam.value);

        appendLog(`[SYSTEM] Initiating interactive stress attack: Type=${type}, Param=${param}...`);

        const response = await fetch(`${API_URL}/api/interactive-attack`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ attack_type: type, param: param })
        });

        const data = await response.json();
        if (data.error) {
            appendLog(`[SYSTEM] Stress testing failed: ${data.error}`);
            btnApplyStress.disabled = false;
            return;
        }

        statCacc.innerText = `${(data.cacc * 100).toFixed(1)}%`;
        statAhl.innerText = `${(data.ahl * 100).toFixed(1)}%`;
        statZk.innerText = data.zk_snark ? 'VALID' : 'INVALID';
        statZk.style.color = data.zk_snark ? 'var(--green)' : 'var(--red)';

        drawAttributionMap(data.map);

        appendLog(`[SYSTEM] Stress attack complete! CACC: ${(data.cacc * 100).toFixed(1)}%, AHL Match: ${(data.ahl * 100).toFixed(1)}%, EaaW ASR: ${(data.eaaw * 100).toFixed(1)}%, zk-SNARK: ${data.zk_snark ? 'VALID' : 'INVALID'}`);

        btnApplyStress.disabled = false;

    } catch (err) {
        appendLog(`[SYSTEM] Stress test execution error: ${err}`);
        btnApplyStress.disabled = false;
    }
});

initCanvasPlaceholder();
fetchMetrics();
fetchAttributionMap();
