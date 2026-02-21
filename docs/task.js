// ============================================================
// Task Recreation: Three Compositional Categorization Tasks
// Based on Tafazoli et al., Nature 650, 164-172 (2026)
// ============================================================

// --- Constants ---
const MORPH_LEVELS = [0, 30, 50, 70, 100, 130, 150, 170]; // circular space
const TASKS = ['S1', 'C1', 'C2'];
const FIXATION_DUR = 600; // ms
const FEEDBACK_DUR = 800;
const ITI_DUR = 1200;
const SWITCH_THRESHOLD = 0.70;
const SWITCH_WINDOW = 15;

// Response positions (in normalized 0-1 coords of canvas)
const TARGETS = {
    UL: { x: 0.22, y: 0.22, label: 'UL' },
    UR: { x: 0.78, y: 0.22, label: 'UR' },
    LL: { x: 0.22, y: 0.78, label: 'LL' },
    LR: { x: 0.78, y: 0.78, label: 'LR' }
};

// Colour endpoints in HSL-ish space
function morphToColour(level) {
    // 0=red, 100=green, circular
    const t = level / 200; // normalize to 0-1 range (circular from 0-200)
    // Map: 0->red, 50->ambiguous, 100->green, 150->ambiguous, 200/0->red
    const angle = t * Math.PI * 2;
    const r = Math.round(180 + 75 * Math.cos(angle));
    const g = Math.round(100 + 75 * Math.cos(angle - Math.PI * 2 / 3));
    const b = Math.round(80 + 40 * Math.cos(angle + Math.PI * 2 / 3));
    return `rgb(${Math.max(0, Math.min(255, r))},${Math.max(0, Math.min(255, g))},${Math.max(0, Math.min(255, b))})`;
}

function morphToColourSimple(level) {
    // Simpler: 0=red(#ef4444), 100=green(#10b981), interpolate
    let t;
    if (level <= 100) {
        t = level / 100;
    } else {
        t = (200 - level) / 100;
    }
    const r = Math.round(239 * (1 - t) + 16 * t);
    const g = Math.round(68 * (1 - t) + 185 * t);
    const b = Math.round(68 * (1 - t) + 129 * t);
    return `rgb(${r},${g},${b})`;
}

function getColourCategory(morphLevel) {
    // Red category: morph 0-49 and 151-200 (closer to 0)
    // Green category: morph 51-149 (closer to 100)
    if (morphLevel < 50 || morphLevel > 150) return 'red';
    if (morphLevel > 50 && morphLevel < 150) return 'green';
    return 'ambiguous';
}

function getShapeCategory(morphLevel) {
    // Bunny: morph 0-49 and 151-200
    // Tee: morph 51-149
    if (morphLevel < 50 || morphLevel > 150) return 'bunny';
    if (morphLevel > 50 && morphLevel < 150) return 'tee';
    return 'ambiguous';
}

// --- State ---
let state = {
    running: false,
    phase: 'idle', // idle, fixation, stimulus, response, feedback, iti
    currentTask: 'C1',
    currentAxis: 1,
    blockNum: 1,
    trialNum: 0,
    trialInBlock: 0,
    stimOnTime: 0,
    colourMorph: 0,
    shapeMorph: 0,
    correctTarget: null,
    history: [], // {task, correct, rt, colourMorph, shapeMorph, response}
    recentCorrect: [],
    blockHistory: [],
    psychoData: { colour: {}, shape: {} },
    stimDuration: 1000,
    mode: 'auto',
    showRule: false,
    timeout: null
};

// --- Canvas Setup ---
let canvas, ctx, canvasW, canvasH;

function initCanvas() {
    canvas = document.getElementById('task-canvas');
    const container = document.getElementById('experiment-display');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    canvasW = canvas.width;
    canvasH = canvas.height;
    ctx = canvas.getContext('2d');
    canvas.addEventListener('click', handleClick);
    drawIdle();
}

function drawIdle() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = '#6b7280';
    ctx.font = '16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Click "Start" to begin', canvasW / 2, canvasH / 2 - 10);
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Respond by clicking a target location', canvasW / 2, canvasH / 2 + 15);
}

function drawFixation() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);
    // Fixation dot
    ctx.beginPath();
    ctx.arc(canvasW / 2, canvasH / 2, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    // Rule display
    if (state.showRule) {
        ctx.font = '13px JetBrains Mono, monospace';
        ctx.fillStyle = '#3b82f6';
        ctx.fillText(`Task: ${state.currentTask}`, canvasW / 2, 25);
    }
}

function drawTargets(highlight) {
    const targetSize = Math.min(canvasW, canvasH) * 0.06;
    Object.entries(TARGETS).forEach(([key, pos]) => {
        const x = pos.x * canvasW;
        const y = pos.y * canvasH;
        ctx.beginPath();
        ctx.arc(x, y, targetSize, 0, Math.PI * 2);
        if (highlight === key) {
            ctx.fillStyle = 'rgba(59,130,246,0.4)';
            ctx.strokeStyle = '#3b82f6';
        } else {
            ctx.fillStyle = 'rgba(255,255,255,0.08)';
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        }
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(key, x, y + 4);
    });
}

function drawStimulus(colourMorph, shapeMorph) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Draw targets
    drawTargets();

    // Draw fixation
    ctx.beginPath();
    ctx.arc(canvasW / 2, canvasH / 2, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fill();

    // Draw stimulus at center
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const colour = morphToColourSimple(colourMorph);
    const size = Math.min(canvasW, canvasH) * 0.12;

    // Shape: interpolate between bunny (round+ears) and tee (T-shape)
    const shapeCat = getShapeCategory(shapeMorph);
    let shapeT; // 0=bunny, 1=tee
    if (shapeMorph <= 100) {
        shapeT = shapeMorph / 100;
    } else {
        shapeT = (200 - shapeMorph) / 100;
    }

    ctx.save();
    ctx.translate(cx, cy);

    // Interpolate between circle (bunny) and T-shape
    const roundness = 1 - shapeT;
    const tness = shapeT;

    // Draw blended shape
    ctx.fillStyle = colour;
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;

    if (roundness > 0.5) {
        // More bunny-like: circle body with ears
        ctx.beginPath();
        ctx.arc(0, size * 0.1, size * (0.6 + roundness * 0.1), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Ears
        const earH = size * 0.5 * roundness;
        ctx.beginPath();
        ctx.ellipse(-size * 0.25, -size * 0.4, size * 0.12, earH, -0.15, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(size * 0.25, -size * 0.4, size * 0.12, earH, 0.15, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    } else {
        // More tee-like: T shape
        const barW = size * 1.2;
        const barH = size * 0.3;
        const stemW = size * 0.35;
        const stemH = size * 0.8;

        // Top bar
        ctx.beginPath();
        ctx.rect(-barW / 2, -size * 0.5, barW, barH);
        ctx.fill();
        ctx.stroke();
        // Stem
        ctx.beginPath();
        ctx.rect(-stemW / 2, -size * 0.5 + barH - 2, stemW, stemH);
        ctx.fill();
        ctx.stroke();
    }

    ctx.restore();

    // Rule display
    if (state.showRule) {
        ctx.font = '13px JetBrains Mono, monospace';
        ctx.fillStyle = '#3b82f6';
        ctx.textAlign = 'center';
        ctx.fillText(`Task: ${state.currentTask}`, canvasW / 2, 25);
    }
}

function drawFeedback(correct, correctTarget) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);
    drawTargets(correctTarget);

    ctx.textAlign = 'center';
    ctx.font = 'bold 28px Inter, sans-serif';
    if (correct) {
        ctx.fillStyle = '#10b981';
        ctx.fillText('Correct!', canvasW / 2, canvasH / 2);
    } else {
        ctx.fillStyle = '#ef4444';
        ctx.fillText('Incorrect', canvasW / 2, canvasH / 2);
        ctx.font = '14px Inter, sans-serif';
        ctx.fillStyle = '#9ca3af';
        ctx.fillText(`Correct: ${correctTarget}`, canvasW / 2, canvasH / 2 + 25);
    }
}

function drawBlockSwitch() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);
    // Yellow flash
    ctx.fillStyle = 'rgba(245,158,11,0.15)';
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.textAlign = 'center';
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('Block Switch!', canvasW / 2, canvasH / 2 - 15);
    ctx.font = '14px Inter, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText('Task may have changed. Keep responding!', canvasW / 2, canvasH / 2 + 15);
}

// --- Task Logic ---
function getCorrectTarget(task, colourMorph, shapeMorph) {
    if (task === 'S1') {
        const cat = getShapeCategory(shapeMorph);
        if (cat === 'bunny') return 'UL';
        if (cat === 'tee') return 'LR';
        return Math.random() < 0.5 ? 'UL' : 'LR'; // ambiguous
    } else if (task === 'C1') {
        const cat = getColourCategory(colourMorph);
        if (cat === 'green') return 'UL';
        if (cat === 'red') return 'LR';
        return Math.random() < 0.5 ? 'UL' : 'LR';
    } else { // C2
        const cat = getColourCategory(colourMorph);
        if (cat === 'red') return 'UR';
        if (cat === 'green') return 'LL';
        return Math.random() < 0.5 ? 'UR' : 'LL';
    }
}

function pickMorphLevel(controlVal) {
    if (controlVal < 0) {
        // Random
        return MORPH_LEVELS[Math.floor(Math.random() * MORPH_LEVELS.length)];
    }
    return MORPH_LEVELS[Math.min(controlVal, MORPH_LEVELS.length - 1)];
}

function pickTask() {
    if (state.mode !== 'auto') return state.mode;

    // Auto mode: alternate axis, random within axis 1
    if (state.currentAxis === 1) {
        // Switch to axis 2
        state.currentAxis = 2;
        return 'C2';
    } else {
        // Switch to axis 1: pick S1 or C1
        state.currentAxis = 1;
        return Math.random() < 0.5 ? 'S1' : 'C1';
    }
}

function checkBlockSwitch() {
    if (state.mode !== 'auto') return false;
    if (state.recentCorrect.length < SWITCH_WINDOW) return false;
    const recent = state.recentCorrect.slice(-SWITCH_WINDOW);
    const acc = recent.reduce((a, b) => a + b, 0) / recent.length;
    return acc >= SWITCH_THRESHOLD;
}

// --- Trial Flow ---
function startExperiment() {
    state.running = true;
    state.phase = 'idle';
    state.trialNum = 0;
    state.trialInBlock = 0;
    state.blockNum = 1;
    state.history = [];
    state.recentCorrect = [];
    state.blockHistory = [];
    state.psychoData = { colour: {}, shape: {} };

    state.mode = document.getElementById('mode-select').value;
    state.showRule = document.getElementById('show-rule').value === '1';
    state.stimDuration = parseInt(document.getElementById('stim-duration').value);

    if (state.mode === 'auto') {
        state.currentAxis = 2; // Will switch to axis 1 first
        state.currentTask = pickTask();
    } else {
        state.currentTask = state.mode;
    }

    updateStats();
    startTrial();
}

function startTrial() {
    if (!state.running) return;
    clearTimeout(state.timeout);

    state.trialNum++;
    state.trialInBlock++;

    const shapeMorphCtrl = parseInt(document.getElementById('shape-morph').value);
    const colourMorphCtrl = parseInt(document.getElementById('colour-morph').value);
    state.colourMorph = pickMorphLevel(colourMorphCtrl);
    state.shapeMorph = pickMorphLevel(shapeMorphCtrl);
    state.correctTarget = getCorrectTarget(state.currentTask, state.colourMorph, state.shapeMorph);

    document.getElementById('trial-indicator').textContent = `Trial ${state.trialInBlock} / Block ${state.blockNum}`;
    document.getElementById('feedback-text').innerHTML = '&nbsp;';

    // Fixation phase
    state.phase = 'fixation';
    drawFixation();
    state.timeout = setTimeout(() => {
        // Stimulus phase
        state.phase = 'stimulus';
        state.stimOnTime = performance.now();
        drawStimulus(state.colourMorph, state.shapeMorph);
    }, FIXATION_DUR);
}

function handleClick(e) {
    if (state.phase !== 'stimulus') return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Find closest target
    let closest = null;
    let minDist = Infinity;
    Object.entries(TARGETS).forEach(([key, pos]) => {
        const d = Math.hypot(x - pos.x, y - pos.y);
        if (d < minDist) {
            minDist = d;
            closest = key;
        }
    });

    if (minDist > 0.18) return; // Too far from any target

    const rt = performance.now() - state.stimOnTime;
    if (rt < 150) return; // Too fast

    const correct = closest === state.correctTarget;

    // Record
    state.history.push({
        task: state.currentTask,
        correct,
        rt,
        colourMorph: state.colourMorph,
        shapeMorph: state.shapeMorph,
        response: closest,
        correctTarget: state.correctTarget
    });
    state.recentCorrect.push(correct ? 1 : 0);

    // Update psychometric data
    recordPsychometric(state.currentTask, state.colourMorph, state.shapeMorph, closest);

    // Feedback
    state.phase = 'feedback';
    drawFeedback(correct, state.correctTarget);
    document.getElementById('feedback-text').innerHTML = correct
        ? '<span class="feedback-correct">Correct!</span>'
        : `<span class="feedback-incorrect">Incorrect</span> &mdash; correct was ${state.correctTarget}`;

    updateStats();
    updateCharts();

    // Check block switch
    state.timeout = setTimeout(() => {
        if (checkBlockSwitch()) {
            // Block switch
            state.blockNum++;
            state.trialInBlock = 0;
            state.recentCorrect = [];
            const prevTask = state.currentTask;
            state.currentTask = pickTask();
            state.blockHistory.push({ task: prevTask, trials: state.history.length });

            state.phase = 'block_switch';
            drawBlockSwitch();
            document.getElementById('feedback-text').innerHTML = '<span style="color:#f59e0b">Block switched! Task may have changed.</span>';

            state.timeout = setTimeout(() => {
                startTrial();
            }, 2000);
        } else {
            // ITI then next trial
            state.phase = 'iti';
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvasW, canvasH);
            state.timeout = setTimeout(startTrial, ITI_DUR);
        }
    }, FEEDBACK_DUR);
}

function recordPsychometric(task, colourMorph, shapeMorph, response) {
    // Record for colour psychometric
    if (task === 'C1' || task === 'C2') {
        if (!state.psychoData.colour[colourMorph]) {
            state.psychoData.colour[colourMorph] = { green: 0, total: 0 };
        }
        state.psychoData.colour[colourMorph].total++;
        // "Green" response: UL for C1, LL for C2
        if ((task === 'C1' && response === 'UL') || (task === 'C2' && response === 'LL')) {
            state.psychoData.colour[colourMorph].green++;
        }
    }
    // Record for shape psychometric
    if (task === 'S1') {
        if (!state.psychoData.shape[shapeMorph]) {
            state.psychoData.shape[shapeMorph] = { tee: 0, total: 0 };
        }
        state.psychoData.shape[shapeMorph].total++;
        if (response === 'LR') {
            state.psychoData.shape[shapeMorph].tee++;
        }
    }
}

// --- UI Updates ---
function updateStats() {
    const n = state.history.length;
    if (n === 0) {
        document.getElementById('stat-accuracy').textContent = '--';
        document.getElementById('stat-rt').textContent = '--';
    } else {
        const acc = state.history.filter(h => h.correct).length / n * 100;
        const avgRt = state.history.reduce((s, h) => s + h.rt, 0) / n;
        document.getElementById('stat-accuracy').textContent = acc.toFixed(0) + '%';
        document.getElementById('stat-rt').textContent = avgRt.toFixed(0);
    }
    document.getElementById('stat-block').textContent = state.blockNum;
    document.getElementById('stat-trials').textContent = n;

    const taskLabel = document.getElementById('task-label');
    if (state.showRule || state.phase === 'idle') {
        taskLabel.textContent = state.currentTask;
    } else {
        taskLabel.textContent = '???';
    }

    // Recent accuracy progress
    const recent = state.recentCorrect.slice(-SWITCH_WINDOW);
    const recentAcc = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length * 100 : 0;
    document.getElementById('recent-progress').style.width = recentAcc + '%';
    document.getElementById('recent-pct').textContent = recentAcc.toFixed(0) + '%';
    if (recentAcc >= 70) {
        document.getElementById('recent-progress').style.background = '#10b981';
    } else {
        document.getElementById('recent-progress').style.background = '#3b82f6';
    }
}

// --- Charts ---
let psychoColourChart, psychoShapeChart, accuracyChart;

function initCharts() {
    const defaultOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#9ca3af', font: { size: 11 } } }
        },
        scales: {
            x: {
                ticks: { color: '#6b7280', font: { size: 10 } },
                grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
                ticks: { color: '#6b7280', font: { size: 10 } },
                grid: { color: 'rgba(255,255,255,0.05)' },
                min: 0, max: 1
            }
        }
    };

    psychoColourChart = new Chart(document.getElementById('psycho-colour'), {
        type: 'line',
        data: {
            labels: ['0', '30', '50', '70', '100', '130', '150', '170'],
            datasets: [{
                label: 'P(Green Response)',
                data: [null, null, null, null, null, null, null, null],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16,185,129,0.1)',
                tension: 0.3,
                pointRadius: 5,
                fill: true
            }]
        },
        options: {
            ...defaultOpts,
            scales: {
                ...defaultOpts.scales,
                x: { ...defaultOpts.scales.x, title: { display: true, text: 'Colour Morph Level', color: '#9ca3af' } },
                y: { ...defaultOpts.scales.y, title: { display: true, text: 'P(Green Response)', color: '#9ca3af' } }
            }
        }
    });

    psychoShapeChart = new Chart(document.getElementById('psycho-shape'), {
        type: 'line',
        data: {
            labels: ['0', '30', '50', '70', '100', '130', '150', '170'],
            datasets: [{
                label: 'P(Tee Response)',
                data: [null, null, null, null, null, null, null, null],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.1)',
                tension: 0.3,
                pointRadius: 5,
                fill: true
            }]
        },
        options: {
            ...defaultOpts,
            scales: {
                ...defaultOpts.scales,
                x: { ...defaultOpts.scales.x, title: { display: true, text: 'Shape Morph Level', color: '#9ca3af' } },
                y: { ...defaultOpts.scales.y, title: { display: true, text: 'P(Tee Response)', color: '#9ca3af' } }
            }
        }
    });

    accuracyChart = new Chart(document.getElementById('accuracy-timeline'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy (15-trial window)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.1)',
                tension: 0.2,
                pointRadius: 0,
                fill: true,
                borderWidth: 2
            }]
        },
        options: {
            ...defaultOpts,
            scales: {
                ...defaultOpts.scales,
                x: { ...defaultOpts.scales.x, title: { display: true, text: 'Trial', color: '#9ca3af' } },
                y: { ...defaultOpts.scales.y, title: { display: true, text: 'Accuracy', color: '#9ca3af' }, min: 0, max: 1 }
            }
        }
    });
}

function updateCharts() {
    // Psychometric colour
    const colourData = MORPH_LEVELS.map(ml => {
        const d = state.psychoData.colour[ml];
        return d && d.total > 0 ? d.green / d.total : null;
    });
    psychoColourChart.data.datasets[0].data = colourData;
    psychoColourChart.update('none');

    // Psychometric shape
    const shapeData = MORPH_LEVELS.map(ml => {
        const d = state.psychoData.shape[ml];
        return d && d.total > 0 ? d.tee / d.total : null;
    });
    psychoShapeChart.data.datasets[0].data = shapeData;
    psychoShapeChart.update('none');

    // Accuracy timeline
    const accs = [];
    const labels = [];
    for (let i = 0; i < state.history.length; i++) {
        const start = Math.max(0, i - SWITCH_WINDOW + 1);
        const window = state.history.slice(start, i + 1);
        accs.push(window.filter(h => h.correct).length / window.length);
        labels.push(i + 1);
    }
    accuracyChart.data.labels = labels;
    accuracyChart.data.datasets[0].data = accs;
    accuracyChart.update('none');
}

// --- Control Listeners ---
document.getElementById('stim-duration').addEventListener('input', function () {
    document.getElementById('dur-val').textContent = this.value;
    state.stimDuration = parseInt(this.value);
});

document.getElementById('shape-morph').addEventListener('input', function () {
    const v = parseInt(this.value);
    document.getElementById('shape-val').textContent = v < 0 ? 'Random' : MORPH_LEVELS[Math.min(v, 7)];
});

document.getElementById('colour-morph').addEventListener('input', function () {
    const v = parseInt(this.value);
    document.getElementById('colour-val').textContent = v < 0 ? 'Random' : MORPH_LEVELS[Math.min(v, 7)];
});

document.getElementById('mode-select').addEventListener('change', function () {
    state.mode = this.value;
    if (this.value !== 'auto') {
        state.currentTask = this.value;
    }
});

document.getElementById('show-rule').addEventListener('change', function () {
    state.showRule = this.value === '1';
});

// --- Init ---
window.addEventListener('load', () => {
    initCanvas();
    initCharts();
});

window.addEventListener('resize', () => {
    initCanvas();
});
