// ============================================================
// Results & Data Analysis - Simulated Data Visualizations
// Based on Tafazoli et al., Nature 650, 164-172 (2026)
// ============================================================

function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function smoothData(arr, w) {
    const out = [];
    for (let i = 0; i < arr.length; i++) {
        const s = Math.max(0, i - Math.floor(w / 2));
        const e = Math.min(arr.length, i + Math.ceil(w / 2));
        let sum = 0;
        for (let j = s; j < e; j++) sum += arr[j];
        out.push(sum / (e - s));
    }
    return out;
}

function linspace(a, b, n) {
    const d = (b - a) / (n - 1);
    return Array.from({ length: n }, (_, i) => a + i * d);
}

Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.font.size = 11;

const defaultScales = {
    x: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    y: { ticks: { color: '#6b7280' }, grid: { color: 'rgba(255,255,255,0.05)' } }
};

// --- Tab System ---
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('tab-' + this.dataset.tab).classList.add('active');
    });
});

// ============================================================
// TAB 1: BEHAVIOUR
// ============================================================
function initBehaviour() {
    // Psychometric curves (simulated erf)
    const morphs = [0, 30, 50, 70, 100, 130, 150, 170];
    const morphNorm = morphs.map(m => (m <= 100 ? m / 100 : (200 - m) / 100)); // 0 to 1

    // S1: shape psychometric
    const s1Data = morphNorm.map(m => sigmoid((m - 0.5) * 8 + randn() * 0.1));
    // C1: colour psychometric
    const c1Data = morphNorm.map(m => sigmoid((m - 0.5) * 7 + randn() * 0.1));
    // C2: colour psychometric (steeper)
    const c2Data = morphNorm.map(m => sigmoid((m - 0.5) * 12 + randn() * 0.05));

    new Chart(document.getElementById('res-psychometric'), {
        type: 'line',
        data: {
            labels: morphs.map(String),
            datasets: [
                { label: 'S1', data: s1Data, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.05)', tension: 0.4, pointRadius: 5, fill: false, borderWidth: 2 },
                { label: 'C1', data: c1Data, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.05)', tension: 0.4, pointRadius: 5, fill: false, borderWidth: 2 },
                { label: 'C2', data: c2Data, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)', tension: 0.4, pointRadius: 5, fill: false, borderWidth: 2 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Morph Level', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'P(Category B)', color: '#9ca3af' }, min: 0, max: 1 }
            }
        }
    });

    // Learning curves
    const trials = linspace(1, 110, 50);
    const s1Learn = trials.map(t => 0.47 + 0.30 * sigmoid((t - 40) / 15) + randn() * 0.02);
    const c1Learn = trials.map(t => 0.62 + 0.18 * sigmoid((t - 30) / 15) + randn() * 0.02);
    const c2Learn = trials.map(t => 0.85 + 0.07 * sigmoid((t - 10) / 20) + randn() * 0.01);

    new Chart(document.getElementById('res-learning'), {
        type: 'line',
        data: {
            labels: trials.map(t => Math.round(t)),
            datasets: [
                { label: 'S1', data: smoothData(s1Learn, 3), borderColor: '#f59e0b', tension: 0.3, pointRadius: 0, borderWidth: 2, fill: false },
                { label: 'C1', data: smoothData(c1Learn, 3), borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2, fill: false },
                { label: 'C2', data: smoothData(c2Learn, 3), borderColor: '#3b82f6', tension: 0.3, pointRadius: 0, borderWidth: 2, fill: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Trial After Switch', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Accuracy', color: '#9ca3af' }, min: 0.3, max: 1 }
            }
        }
    });
}

// ============================================================
// TAB 2: CROSS-TASK DECODING
// ============================================================
function initDecoding() {
    const times = linspace(-100, 400, 100);
    const onset = 80;
    const makeTrace = (onset, peak, delay, noiseScale) =>
        times.map(t => {
            const trel = t - onset - delay;
            const v = trel > 0 ? peak * (1 - Math.exp(-trel / 25)) * Math.exp(-Math.max(0, trel - 200) / 400) : 0;
            return 0.5 + Math.max(0, v) + randn() * noiseScale;
        });

    const c1Within = smoothData(makeTrace(onset, 0.38, 0, 0.01), 5);
    const c2Within = smoothData(makeTrace(onset, 0.40, 5, 0.01), 5);
    const c2ToC1 = smoothData(makeTrace(onset, 0.28, 15, 0.015), 5);
    const c1ToC2 = smoothData(makeTrace(onset, 0.25, 20, 0.015), 5);

    new Chart(document.getElementById('res-colour-decode'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                { label: 'C1 within', data: c1Within, borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'C2 within', data: c2Within, borderColor: '#3b82f6', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'C2→C1', data: c2ToC1, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [5, 3] },
                { label: 'C1→C2', data: c1ToC2, borderColor: '#06b6d4', tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [5, 3] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, annotation: {} },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Classifier Accuracy', color: '#9ca3af' }, min: 0.45, max: 0.95 }
            }
        }
    });

    // Response decoding
    const s1Resp = smoothData(makeTrace(onset, 0.35, 35, 0.01), 5);
    const c1Resp = smoothData(makeTrace(onset, 0.33, 55, 0.01), 5);
    const s1ToC1Resp = smoothData(makeTrace(onset, 0.25, 50, 0.015), 5);
    const c1ToS1Resp = smoothData(makeTrace(onset, 0.23, 45, 0.015), 5);

    new Chart(document.getElementById('res-response-decode'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                { label: 'S1 within', data: s1Resp, borderColor: '#f59e0b', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'C1 within', data: c1Resp, borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'S1→C1', data: s1ToC1Resp, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [5, 3] },
                { label: 'C1→S1', data: c1ToS1Resp, borderColor: '#06b6d4', tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [5, 3] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Classifier Accuracy', color: '#9ca3af' }, min: 0.45, max: 0.90 }
            }
        }
    });
}

// ============================================================
// TAB 3: SENSORY-MOTOR TRANSFORMATION
// ============================================================
function initTransformation() {
    const times = linspace(-100, 400, 100);
    const colourTrace = smoothData(times.map(t => {
        const v = t > 65 ? 0.9 * (1 - Math.exp(-(t - 65) / 20)) * Math.exp(-Math.max(0, t - 250) / 300) : 0;
        return 0.5 + v + randn() * 0.015;
    }), 5);
    const motorTrace = smoothData(times.map(t => {
        const v = t > 128 ? 0.85 * (1 - Math.exp(-(t - 128) / 25)) : 0;
        return 0.5 + v + randn() * 0.015;
    }), 5);

    new Chart(document.getElementById('res-sequential'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                { label: 'Shared Colour', data: colourTrace, borderColor: '#3b82f6', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false },
                { label: 'Shared Motor', data: motorTrace, borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Normalized Accuracy', color: '#9ca3af' }, min: 0.4, max: 1.1 }
            }
        }
    });

    // Heatmap as scatter (color-coded grid)
    const heatData = [];
    const gridSize = 20;
    const heatTimes = linspace(-50, 350, gridSize);
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const ti = heatTimes[i], tj = heatTimes[j];
            // Correlation: positive when colour early predicts motor later
            let corr = 0;
            if (ti > 60 && tj > 120 && tj > ti) {
                corr = 0.3 * Math.exp(-Math.pow(ti - 150, 2) / 8000) * Math.exp(-Math.pow(tj - 250, 2) / 8000);
            }
            if (ti < 100 && tj < 150) corr -= 0.1 * Math.exp(-Math.pow(ti - 50, 2) / 3000);
            corr += randn() * 0.02;
            heatData.push({ x: Math.round(ti), y: Math.round(tj), v: corr });
        }
    }

    // Use scatter with varying colors
    new Chart(document.getElementById('res-heatmap'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Correlation',
                data: heatData.map(d => ({ x: d.x, y: d.y })),
                backgroundColor: heatData.map(d => {
                    const v = Math.max(-0.2, Math.min(0.3, d.v));
                    if (v > 0) return `rgba(59,130,246,${Math.min(1, v * 3)})`;
                    return `rgba(239,68,68,${Math.min(1, -v * 5)})`;
                }),
                pointRadius: 8,
                pointStyle: 'rect'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Colour Encoding Time (ms)', color: '#3b82f6' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Motor Encoding Time (ms)', color: '#10b981' } }
            }
        }
    });

    // TDR trajectories
    const nPts = 50;
    const tdrTimes = linspace(-100, 300, nPts);

    function makeTDR(task, colour) {
        const sign = colour === 'red' ? 1 : -1;
        return tdrTimes.map(t => {
            const colourDim = t > 60 ? sign * 0.8 * (1 - Math.exp(-(t - 60) / 20)) : randn() * 0.05;
            let motorDim = 0;
            if (task === 'C1') {
                motorDim = t > 120 ? -sign * 0.7 * (1 - Math.exp(-(t - 120) / 25)) : randn() * 0.05;
            }
            return { x: colourDim + randn() * 0.02, y: motorDim + randn() * 0.02 };
        });
    }

    new Chart(document.getElementById('res-tdr'), {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Red (C1)', data: makeTDR('C1', 'red'), showLine: true, borderColor: '#ef4444', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Green (C1)', data: makeTDR('C1', 'green'), showLine: true, borderColor: '#10b981', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Red (C2)', data: makeTDR('C2', 'red'), showLine: true, borderColor: 'rgba(239,68,68,0.4)', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1.5, borderDash: [4, 3] },
                { label: 'Green (C2)', data: makeTDR('C2', 'green'), showLine: true, borderColor: 'rgba(16,185,129,0.4)', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1.5, borderDash: [4, 3] }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Colour Dimension', color: '#3b82f6' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Axis 1 Response Dimension', color: '#f59e0b' } }
            }
        }
    });
}

// ============================================================
// TAB 4: TASK DISCOVERY
// ============================================================
function initDiscovery() {
    const trials = linspace(40, 110, 15);

    // Belief encoding
    const beliefSCCl = trials.map(t => 0.42 + 0.12 * sigmoid((t - 70) / 15) + randn() * 0.02);
    const beliefCCCl = trials.map(t => 0.67 + 0.08 * sigmoid((t - 60) / 20) + randn() * 0.015);

    new Chart(document.getElementById('res-belief-evolution'), {
        type: 'line',
        data: {
            labels: trials.map(t => Math.round(t)),
            datasets: [
                { label: 'S1→C2→C1', data: beliefSCCl, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 4, borderWidth: 2.5 },
                { label: 'C1→C2→C1', data: beliefCCCl, borderColor: '#06b6d4', tension: 0.3, pointRadius: 4, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Trial', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Task Classifier Accuracy', color: '#9ca3af' }, min: 0.35, max: 0.85 }
            }
        }
    });

    // Colour evolution
    const colourSCC = trials.map(t => 0.64 + 0.05 * sigmoid((t - 70) / 15) + randn() * 0.015);
    const colourCCC = trials.map(t => 0.63 - 0.01 * sigmoid((t - 70) / 20) + randn() * 0.015);

    new Chart(document.getElementById('res-colour-evolution'), {
        type: 'line',
        data: {
            labels: trials.map(t => Math.round(t)),
            datasets: [
                { label: 'S1→C2→C1', data: colourSCC, borderColor: '#3b82f6', tension: 0.3, pointRadius: 4, borderWidth: 2.5 },
                { label: 'C1→C2→C1', data: colourCCC, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 4, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Trial', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Colour Classifier Accuracy', color: '#9ca3af' }, min: 0.55, max: 0.75 }
            }
        }
    });

    // Shape evolution (decreases for S1-C2-C1)
    const shapeSCC = trials.map(t => 0.53 - 0.04 * sigmoid((t - 60) / 15) + randn() * 0.015);
    const shapeCCC = trials.map(t => 0.51 + 0.01 * sigmoid((t - 70) / 20) + randn() * 0.01);

    new Chart(document.getElementById('res-shape-evolution'), {
        type: 'line',
        data: {
            labels: trials.map(t => Math.round(t)),
            datasets: [
                { label: 'S1→C2→C1', data: shapeSCC, borderColor: '#f59e0b', tension: 0.3, pointRadius: 4, borderWidth: 2.5 },
                { label: 'C1→C2→C1', data: shapeCCC, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 4, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Trial', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Shape Classifier Accuracy', color: '#9ca3af' }, min: 0.44, max: 0.58 }
            }
        }
    });

    // Response (stable)
    const respSCC = trials.map(t => 0.72 + 0.03 * sigmoid((t - 70) / 30) + randn() * 0.015);
    const respCCC = trials.map(t => 0.73 + 0.03 * sigmoid((t - 70) / 30) + randn() * 0.015);

    new Chart(document.getElementById('res-response-evolution'), {
        type: 'line',
        data: {
            labels: trials.map(t => Math.round(t)),
            datasets: [
                { label: 'S1→C2→C1', data: respSCC, borderColor: '#10b981', tension: 0.3, pointRadius: 4, borderWidth: 2.5 },
                { label: 'C1→C2→C1', data: respCCC, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 4, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Trial', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Response Classifier Accuracy', color: '#9ca3af' }, min: 0.60, max: 0.85 }
            }
        }
    });
}

// ============================================================
// TAB 5: COMPRESSION
// ============================================================
function initCompression() {
    const times = linspace(-100, 400, 80);

    const cpiS1 = smoothData(times.map(t => {
        const v = t > 80 ? -0.6 * (1 - Math.exp(-(t - 80) / 30)) : 0;
        return v + randn() * 0.05;
    }), 5);
    const cpiC1 = smoothData(times.map(t => {
        const v = t > 80 ? 0.5 * (1 - Math.exp(-(t - 80) / 30)) : 0;
        return v + randn() * 0.05;
    }), 5);
    const cpiC2 = smoothData(times.map(t => {
        const v = t > 80 ? 0.7 * (1 - Math.exp(-(t - 80) / 25)) : 0;
        return v + randn() * 0.05;
    }), 5);

    new Chart(document.getElementById('res-cpi-time'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                { label: 'S1', data: cpiS1, borderColor: '#f59e0b', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false },
                { label: 'C1', data: cpiC1, borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false },
                { label: 'C2', data: cpiC2, borderColor: '#3b82f6', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'CPI (log colour/shape)', color: '#9ca3af' } }
            }
        }
    });

    // CPI vs Belief scatter
    const nPoints = 30;
    const beliefVals = Array.from({ length: nPoints }, () => 0.3 + Math.random() * 0.5);
    const cpiVals = beliefVals.map(b => 0.8 * (b - 0.5) + randn() * 0.15);

    new Chart(document.getElementById('res-cpi-belief'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'CPI vs Belief',
                data: beliefVals.map((b, i) => ({ x: b, y: cpiVals[i] })),
                backgroundColor: '#8b5cf6',
                pointRadius: 5
            }, {
                label: 'Trend',
                data: [{ x: 0.3, y: -0.16 }, { x: 0.8, y: 0.24 }],
                borderColor: '#8b5cf6',
                showLine: true,
                pointRadius: 0,
                borderWidth: 2,
                borderDash: [5, 3]
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Task Belief Strength', color: '#8b5cf6' } },
                y: { ...defaultScales.y, title: { display: true, text: 'CPI', color: '#06b6d4' } }
            }
        }
    });

    // Axis suppression
    const trialTimes = linspace(-200, 400, 60);
    const axis1Correct = smoothData(trialTimes.map(t => 8 + (t > 0 ? 4 * (1 - Math.exp(-t / 40)) : 0) + randn() * 0.5), 5);
    const axis1Incorrect = smoothData(trialTimes.map(t => 8 + (t > 0 ? 1.5 * (1 - Math.exp(-t / 50)) : 0) + randn() * 0.5), 5);

    new Chart(document.getElementById('res-axis-suppression'), {
        type: 'line',
        data: {
            labels: trialTimes.map(t => Math.round(t)),
            datasets: [
                { label: 'Axis 1 (correct)', data: axis1Correct, borderColor: '#10b981', tension: 0.3, pointRadius: 0, borderWidth: 2.5, backgroundColor: 'rgba(16,185,129,0.08)', fill: true },
                { label: 'Axis 2 (incorrect)', data: axis1Incorrect, borderColor: '#8b5cf6', tension: 0.3, pointRadius: 0, borderWidth: 2.5, backgroundColor: 'rgba(139,92,246,0.08)', fill: true }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x, title: { display: true, text: 'Time from Stimulus (ms)', color: '#9ca3af' } },
                y: { ...defaultScales.y, title: { display: true, text: 'Firing Rate (Hz)', color: '#9ca3af' } }
            }
        }
    });
}

// ============================================================
// TAB 6: BRAIN REGIONS
// ============================================================
function initRegions() {
    const regions = ['LPFC', 'FEF', 'PAR', 'aIT', 'STR'];

    // Cross-task colour generalization by region
    const colourGen = [0.72, 0.58, 0.56, 0.55, 0.51];
    const colourErr = [0.03, 0.04, 0.05, 0.04, 0.04];

    new Chart(document.getElementById('res-region-colour'), {
        type: 'bar',
        data: {
            labels: regions,
            datasets: [{
                label: 'Cross-Task Colour Accuracy',
                data: colourGen,
                backgroundColor: ['#3b82f6', 'rgba(59,130,246,0.6)', 'rgba(59,130,246,0.5)', 'rgba(59,130,246,0.4)', 'rgba(59,130,246,0.2)'],
                borderColor: '#3b82f6',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                annotation: {}
            },
            scales: {
                x: { ...defaultScales.x },
                y: { ...defaultScales.y, title: { display: true, text: 'Accuracy', color: '#9ca3af' }, min: 0.4, max: 0.85 }
            }
        }
    });

    // Motor generalization
    const motorGen = [0.75, 0.73, 0.71, 0.68, 0.65];

    new Chart(document.getElementById('res-region-motor'), {
        type: 'bar',
        data: {
            labels: regions,
            datasets: [{
                label: 'Cross-Task Motor Accuracy',
                data: motorGen,
                backgroundColor: ['#10b981', 'rgba(16,185,129,0.8)', 'rgba(16,185,129,0.7)', 'rgba(16,185,129,0.6)', 'rgba(16,185,129,0.5)'],
                borderColor: '#10b981',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ...defaultScales.x },
                y: { ...defaultScales.y, title: { display: true, text: 'Accuracy', color: '#9ca3af' }, min: 0.4, max: 0.85 }
            }
        }
    });
}

// ============================================================
// Initialize all tabs
// ============================================================
window.addEventListener('load', () => {
    initBehaviour();
    initDecoding();
    initTransformation();
    initDiscovery();
    initCompression();
    initRegions();
});
