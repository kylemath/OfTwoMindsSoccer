// ============================================================
// Theory & Computational Model Simulations
// Based on Tafazoli et al., Nature 650, 164-172 (2026)
// ============================================================

// --- Utility Functions ---
function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function smoothWindow(data, w) {
    const out = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - Math.floor(w / 2));
        const end = Math.min(data.length, i + Math.ceil(w / 2));
        let sum = 0;
        for (let j = start; j < end; j++) sum += data[j];
        out.push(sum / (end - start));
    }
    return out;
}

// Chart.js defaults
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.font.size = 11;

// ============================================================
// SIMULATION 1: Gain Modulation & Compression
// ============================================================

let sim1Chart = null;

function runSim1() {
    const task = document.getElementById('sim1-task').value;
    const beliefStrength = parseInt(document.getElementById('sim1-belief').value) / 100;
    let colourGain = parseInt(document.getElementById('sim1-colour-gain').value) / 100;
    let shapeGain = parseInt(document.getElementById('sim1-shape-gain').value) / 100;
    const noise = parseInt(document.getElementById('sim1-noise').value) / 100;
    const nNeurons = parseInt(document.getElementById('sim1-neurons').value);

    // Auto-adjust gains based on task belief
    if (task === 'C1' || task === 'C2') {
        colourGain *= (0.5 + 0.5 * beliefStrength);
        shapeGain *= (1.0 - 0.7 * beliefStrength);
    } else {
        shapeGain *= (0.5 + 0.5 * beliefStrength);
        colourGain *= (1.0 - 0.7 * beliefStrength);
    }

    // Generate neuron tuning (random projection weights)
    const colourWeights = Array.from({ length: nNeurons }, () => randn());
    const shapeWeights = Array.from({ length: nNeurons }, () => randn());

    // Stimulus conditions: [colour, shape]
    const conditions = [
        { colour: -1, shape: -1, label: 'Red-Bunny', color: '#ef4444' },
        { colour: -1, shape: 1, label: 'Red-Tee', color: '#f97316' },
        { colour: 1, shape: -1, label: 'Green-Bunny', color: '#10b981' },
        { colour: 1, shape: 1, label: 'Green-Tee', color: '#06b6d4' }
    ];

    const nTrials = 20;
    const datasets = [];

    conditions.forEach(cond => {
        const points = [];
        for (let t = 0; t < nTrials; t++) {
            // Generate population response
            const responses = [];
            for (let n = 0; n < nNeurons; n++) {
                const r = colourWeights[n] * cond.colour * colourGain
                    + shapeWeights[n] * cond.shape * shapeGain
                    + randn() * noise;
                responses.push(r);
            }
            // Project onto colour and shape axes (dot product with weights)
            let colourProj = 0, shapeProj = 0;
            for (let n = 0; n < nNeurons; n++) {
                colourProj += responses[n] * colourWeights[n] / nNeurons;
                shapeProj += responses[n] * shapeWeights[n] / nNeurons;
            }
            points.push({ x: colourProj, y: shapeProj });
        }
        datasets.push({
            label: cond.label,
            data: points,
            backgroundColor: cond.color,
            borderColor: cond.color,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    });

    // Calculate separability
    const colourSep = Math.abs(
        datasets[2].data.reduce((s, p) => s + p.x, 0) / nTrials -
        datasets[0].data.reduce((s, p) => s + p.x, 0) / nTrials
    );
    const shapeSep = Math.abs(
        datasets[1].data.reduce((s, p) => s + p.y, 0) / nTrials -
        datasets[0].data.reduce((s, p) => s + p.y, 0) / nTrials
    );
    const cpi = Math.log(colourSep / Math.max(shapeSep, 0.001));

    document.getElementById('sim1-cpi').textContent = cpi.toFixed(2);
    document.getElementById('sim1-colour-sep').textContent = colourSep.toFixed(2);
    document.getElementById('sim1-shape-sep').textContent = shapeSep.toFixed(2);
    document.getElementById('sim1-task-badge').textContent = task;

    // CPI color
    const cpiEl = document.getElementById('sim1-cpi');
    cpiEl.style.color = cpi > 0 ? '#3b82f6' : cpi < 0 ? '#f59e0b' : '#9ca3af';

    if (sim1Chart) sim1Chart.destroy();
    sim1Chart = new Chart(document.getElementById('sim1-scatter'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#9ca3af', font: { size: 10 } } }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Colour Axis Projection', color: '#3b82f6' },
                    ticks: { color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    title: { display: true, text: 'Shape Axis Projection', color: '#f59e0b' },
                    ticks: { color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// Sim1 event listeners
['sim1-task', 'sim1-belief', 'sim1-colour-gain', 'sim1-shape-gain', 'sim1-noise', 'sim1-neurons'].forEach(id => {
    document.getElementById(id).addEventListener('input', function () {
        // Update display values
        document.getElementById('sim1-belief-val').textContent = (parseInt(document.getElementById('sim1-belief').value) / 100).toFixed(2);
        document.getElementById('sim1-colour-gain-val').textContent = (parseInt(document.getElementById('sim1-colour-gain').value) / 100).toFixed(2);
        document.getElementById('sim1-shape-gain-val').textContent = (parseInt(document.getElementById('sim1-shape-gain').value) / 100).toFixed(2);
        document.getElementById('sim1-noise-val').textContent = (parseInt(document.getElementById('sim1-noise').value) / 100).toFixed(2);
        document.getElementById('sim1-neurons-val').textContent = document.getElementById('sim1-neurons').value;
        runSim1();
    });
});

// ============================================================
// SIMULATION 2: Sensory-Motor Transformation
// ============================================================

let sim2TimeChart = null;
let sim2TrajChart = null;
let sim2AnimFrame = null;

function runSim2() {
    cancelAnimationFrame(sim2AnimFrame);

    const task = document.getElementById('sim2-task').value;
    const speed = parseInt(document.getElementById('sim2-speed').value) / 100;
    const sensoryOnset = parseInt(document.getElementById('sim2-sensory-onset').value);
    const motorDelay = parseInt(document.getElementById('sim2-motor-delay').value);

    const nTimepoints = 400; // ms
    const dt = 1;
    const times = Array.from({ length: nTimepoints / dt }, (_, i) => i * dt);

    // Generate traces
    const colourTrace = [];
    const shapeTrace = [];
    const motorTrace = [];

    const isColourTask = task === 'C1' || task === 'C2';
    const sensoryGain = 1.0;
    const irrelevantGain = 0.2;
    const motorOnset = sensoryOnset + motorDelay;
    const tauRise = 20 / speed;
    const tauDecay = 80 / speed;

    times.forEach(t => {
        // Sensory: rises after onset, sustained
        const sensoryT = t - sensoryOnset;
        const colourAmp = isColourTask ? sensoryGain : irrelevantGain;
        const shapeAmp = !isColourTask ? sensoryGain : irrelevantGain;

        let cVal = 0, sVal = 0, mVal = 0;

        if (sensoryT > 0) {
            cVal = colourAmp * (1 - Math.exp(-sensoryT / tauRise)) * Math.exp(-Math.max(0, sensoryT - 150) / (tauDecay * 3));
            sVal = shapeAmp * (1 - Math.exp(-sensoryT / tauRise)) * Math.exp(-Math.max(0, sensoryT - 150) / (tauDecay * 3));
        }

        // Motor: rises after motor onset, sustained
        const motorT = t - motorOnset;
        if (motorT > 0) {
            mVal = (1 - Math.exp(-motorT / (tauRise * 1.2)));
        }

        // Add noise
        cVal += randn() * 0.03;
        sVal += randn() * 0.03;
        mVal += randn() * 0.03;

        colourTrace.push(Math.max(0, cVal));
        shapeTrace.push(Math.max(0, sVal));
        motorTrace.push(Math.max(0, mVal));
    });

    // Smooth
    const smoothC = smoothWindow(colourTrace, 10);
    const smoothS = smoothWindow(shapeTrace, 10);
    const smoothM = smoothWindow(motorTrace, 10);

    if (sim2TimeChart) sim2TimeChart.destroy();
    sim2TimeChart = new Chart(document.getElementById('sim2-timecourse'), {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                { label: 'Colour Subspace', data: smoothC, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Shape Subspace', data: smoothS, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.05)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Motor Subspace', data: smoothM, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.05)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' },
                    ticks: { color: '#6b7280', maxTicksLimit: 10 },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    title: { display: true, text: 'Encoding Strength', color: '#9ca3af' },
                    ticks: { color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    min: -0.05, max: 1.2
                }
            },
            animation: { duration: 800 }
        }
    });

    // 2D Trajectory (colour vs motor)
    const trajData = [];
    for (let i = 0; i < smoothC.length; i += 2) {
        trajData.push({ x: smoothC[i], y: smoothM[i] });
    }

    if (sim2TrajChart) sim2TrajChart.destroy();
    sim2TrajChart = new Chart(document.getElementById('sim2-trajectory'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Trajectory',
                data: trajData,
                borderColor: '#8b5cf6',
                backgroundColor: trajData.map((_, i) => {
                    const alpha = 0.2 + 0.8 * (i / trajData.length);
                    return `rgba(139,92,246,${alpha})`;
                }),
                pointRadius: trajData.map((_, i) => 2 + 3 * (i / trajData.length)),
                showLine: true,
                tension: 0.3,
                borderWidth: 1.5
            },
            {
                label: 'Start',
                data: [trajData[0]],
                backgroundColor: '#3b82f6',
                pointRadius: 8,
                pointStyle: 'triangle'
            },
            {
                label: 'End',
                data: [trajData[trajData.length - 1]],
                backgroundColor: '#10b981',
                pointRadius: 8,
                pointStyle: 'rect'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } } },
            scales: {
                x: {
                    title: { display: true, text: isColourTask ? 'Colour Subspace' : 'Shape Subspace', color: '#3b82f6' },
                    ticks: { color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    title: { display: true, text: 'Motor Subspace', color: '#10b981' },
                    ticks: { color: '#6b7280' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// Sim2 listeners
['sim2-task', 'sim2-speed', 'sim2-sensory-onset', 'sim2-motor-delay'].forEach(id => {
    document.getElementById(id).addEventListener('input', function () {
        document.getElementById('sim2-speed-val').textContent = (parseInt(document.getElementById('sim2-speed').value) / 100).toFixed(2);
        document.getElementById('sim2-onset-val').textContent = document.getElementById('sim2-sensory-onset').value;
        document.getElementById('sim2-delay-val').textContent = document.getElementById('sim2-motor-delay').value;
    });
});

document.getElementById('sim2-run').addEventListener('click', runSim2);
document.getElementById('sim2-reset').addEventListener('click', () => {
    if (sim2TimeChart) { sim2TimeChart.destroy(); sim2TimeChart = null; }
    if (sim2TrajChart) { sim2TrajChart.destroy(); sim2TrajChart = null; }
});

// ============================================================
// SIMULATION 3: Task Discovery Dynamics
// ============================================================

let sim3BeliefChart = null;
let sim3PerfChart = null;

function runSim3() {
    const lr = parseInt(document.getElementById('sim3-lr').value) / 100;
    const initBelief = parseInt(document.getElementById('sim3-init').value) / 100;
    const coupling = parseInt(document.getElementById('sim3-coupling').value) / 100;
    const nTrials = parseInt(document.getElementById('sim3-ntrials').value);

    // Simulate task discovery on C1 task after S1-C2-C1 sequence
    const belief = [initBelief]; // belief that current task is C1 (vs S1)
    const colourEngagement = [0.5 + 0.3 * initBelief];
    const shapeEngagement = [0.5 + 0.3 * (1 - initBelief)];
    const accuracy = [];
    const cpiTrace = [];

    for (let t = 0; t < nTrials; t++) {
        const b = belief[t];

        // Colour and shape engagement based on belief
        const cEng = 0.3 + 0.6 * b * coupling + randn() * 0.05;
        const sEng = 0.3 + 0.6 * (1 - b) * coupling + randn() * 0.05;

        // Accuracy depends on correct engagement (C1 needs colour)
        const pCorrect = sigmoid((cEng - 0.5) * 6 + (b - 0.5) * 4 + randn() * 0.5);
        const correct = Math.random() < pCorrect;
        accuracy.push(correct ? 1 : 0);

        // CPI
        const cpi = Math.log(Math.max(cEng, 0.01) / Math.max(sEng, 0.01));
        cpiTrace.push(cpi);

        // Update belief based on feedback
        const update = correct ? lr * (1 - b) : -lr * b * 0.3;
        const newBelief = Math.max(0, Math.min(1, b + update + randn() * 0.02));
        belief.push(newBelief);
        colourEngagement.push(Math.max(0, Math.min(1, cEng)));
        shapeEngagement.push(Math.max(0, Math.min(1, sEng)));
    }

    // Smooth accuracy
    const smoothAcc = smoothWindow(accuracy, 15);
    const smoothCPI = smoothWindow(cpiTrace, 10);
    const smoothBelief = smoothWindow(belief.slice(0, nTrials), 10);
    const smoothColour = smoothWindow(colourEngagement.slice(0, nTrials), 10);
    const smoothShape = smoothWindow(shapeEngagement.slice(0, nTrials), 10);

    const labels = Array.from({ length: nTrials }, (_, i) => i + 1);

    if (sim3BeliefChart) sim3BeliefChart.destroy();
    sim3BeliefChart = new Chart(document.getElementById('sim3-belief'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Task Belief (C1)', data: smoothBelief, borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Colour Engagement', data: smoothColour, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Shape Engagement', data: smoothShape, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.05)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Trial', color: '#9ca3af' }, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Strength', color: '#9ca3af' }, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' }, min: 0, max: 1 }
            },
            animation: { duration: 600 }
        }
    });

    if (sim3PerfChart) sim3PerfChart.destroy();
    sim3PerfChart = new Chart(document.getElementById('sim3-performance'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Accuracy', data: smoothAcc, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
                { label: 'CPI', data: smoothCPI, borderColor: '#06b6d4', tension: 0.3, pointRadius: 0, borderWidth: 2, yAxisID: 'y1' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Trial', color: '#9ca3af' }, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Accuracy', color: '#10b981' }, ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' }, min: 0, max: 1, position: 'left' },
                y1: { title: { display: true, text: 'CPI', color: '#06b6d4' }, ticks: { color: '#6b7280' }, grid: { display: false }, position: 'right' }
            },
            animation: { duration: 600 }
        }
    });
}

// Sim3 listeners
['sim3-lr', 'sim3-init', 'sim3-coupling', 'sim3-ntrials'].forEach(id => {
    document.getElementById(id).addEventListener('input', function () {
        document.getElementById('sim3-lr-val').textContent = (parseInt(document.getElementById('sim3-lr').value) / 100).toFixed(2);
        document.getElementById('sim3-init-val').textContent = (parseInt(document.getElementById('sim3-init').value) / 100).toFixed(2);
        document.getElementById('sim3-coupling-val').textContent = (parseInt(document.getElementById('sim3-coupling').value) / 100).toFixed(2);
        document.getElementById('sim3-ntrials-val').textContent = document.getElementById('sim3-ntrials').value;
    });
});

document.getElementById('sim3-run').addEventListener('click', runSim3);

// ============================================================
// Initialize
// ============================================================

window.addEventListener('load', () => {
    runSim1();
    runSim2();
    runSim3();
});
