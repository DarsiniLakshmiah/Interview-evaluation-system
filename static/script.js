// ══════════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════════
let ws                  = null;
let jobPosition         = '';
let mediaStream         = null;
let mediaRecorder       = null;
let audioChunks         = [];
let isRecording         = false;
let recordingStart      = null;
let recTimerInterval    = null;
let speakingTimer       = null;

let faceApiReady        = false;   // true once script tag fires onload
let faceApiLoaded       = false;   // true once models finish loading
let detectionInterval   = null;
let currentMetrics      = { eye_contact: 0, emotion: 'neutral', emotion_key: 'neutral', head_ok: true };
let frameMetrics        = [];      // accumulated during one recording session

let conversationHistory     = [];  // [{role, content}]
let answerMetricsSummaries  = [];  // one summary object per submitted answer

// face-api script fires this when the <script> tag finishes loading
function onFaceApiScriptLoaded() {
    faceApiReady = true;
}

// ══════════════════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════════════════
async function startInterview() {
    jobPosition = document.getElementById('jobRole').value.trim();
    if (!jobPosition) { alert('Please enter a job role.'); return; }

    document.getElementById('setup-screen').style.display  = 'none';
    document.getElementById('interview-screen').style.display = 'flex';
    document.getElementById('role-title').innerText = jobPosition + ' Interview';

    await initCamera();
    initFaceApi();      // fire-and-forget; degrades gracefully if CDN unavailable
    connectWebSocket();
}

// ── Camera ────────────────────────────────────────────────────────────────
async function initCamera() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true,
        });
        document.getElementById('webcam').srcObject = mediaStream;
        setFaceStatus('green', 'Camera active');
    } catch (err) {
        console.error('Camera error:', err);
        setFaceStatus('red', 'Camera unavailable');
    }
}

// ── Face-api.js ────────────────────────────────────────────────────────────
async function initFaceApi() {
    // Wait up to 8 s for the CDN script to finish parsing
    for (let i = 0; i < 80; i++) {
        if (faceApiReady && typeof faceapi !== 'undefined') break;
        await sleep(100);
    }

    if (typeof faceapi === 'undefined') {
        console.warn('face-api.js not available — face analysis disabled.');
        setMetrics({ eye_contact: null, emotion: 'N/A', head: 'N/A' });
        return;
    }

    try {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model';
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ]);
        faceApiLoaded = true;
        setFaceStatus('green', 'Face analysis ready');
        startDetectionLoop();
    } catch (err) {
        console.warn('Could not load face-api models:', err);
        setMetrics({ eye_contact: null, emotion: 'N/A', head: 'N/A' });
    }
}

// ── Detection loop (runs every 500 ms) ────────────────────────────────────
function startDetectionLoop() {
    const video = document.getElementById('webcam');

    detectionInterval = setInterval(async () => {
        if (!faceApiLoaded || video.readyState < 2) return;

        try {
            const det = await faceapi
                .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 }))
                .withFaceLandmarks(true)
                .withFaceExpressions();

            if (!det) {
                setFaceStatus('yellow', 'No face detected');
                setMetrics({ eye_contact: 0, emotion: '—', head: '—', face: false });
                return;
            }

            setFaceStatus('green', 'Face detected');

            const lm   = det.landmarks;
            const box  = det.detection.box;
            const expr = det.expressions;

            // ── Eye-contact: how centered is the nose tip over the face box ──
            const noseTipX   = lm.getNose()[3].x;
            const faceCenterX = box.x + box.width / 2;
            const deviation  = Math.abs(noseTipX - faceCenterX) / (box.width + 1e-6);
            const eyeContact = Math.max(0, Math.min(100, Math.round((1 - deviation * 3) * 100)));

            // ── Head pose: vertical ratio of nose-to-eyes vs face height ──
            const leftEye  = lm.getLeftEye();
            const rightEye = lm.getRightEye();
            const eyeMidY  = (leftEye[0].y + rightEye[3].y) / 2;
            const noseTipY = lm.getNose()[6].y;
            const vRatio   = (noseTipY - eyeMidY) / (box.height + 1e-6);
            const headOk   = vRatio > 0.15 && vRatio < 0.48;

            // ── Dominant expression ──
            const topExpr = Object.entries(expr).sort(([, a], [, b]) => b - a)[0];
            const emotionMap = {
                happy:     { label: '😊 Confident',    color: '#10b981' },
                neutral:   { label: '😐 Neutral',      color: '#94a3b8' },
                surprised: { label: '😮 Surprised',    color: '#f59e0b' },
                sad:       { label: '😔 Nervous',      color: '#f87171' },
                fearful:   { label: '😰 Anxious',      color: '#f87171' },
                disgusted: { label: '😒 Uncomfortable',color: '#f59e0b' },
                angry:     { label: '😠 Stressed',     color: '#f87171' },
            };
            const emoInfo = emotionMap[topExpr[0]] || { label: topExpr[0], color: '#94a3b8' };

            const metrics = {
                face:         true,
                eye_contact:  eyeContact,
                head_ok:      headOk,
                head:         headOk ? '✓ Straight' : '⚠ Turned',
                emotion:      emoInfo.label,
                emotion_key:  topExpr[0],
                emotion_color: emoInfo.color,
            };

            currentMetrics = metrics;
            setMetrics(metrics);

            if (isRecording) frameMetrics.push({ ...metrics });

        } catch (_) { /* silent — individual frame errors are OK */ }
    }, 500);
}

// ══════════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ══════════════════════════════════════════════════════════════════════════
function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${window.location.host}/ws/interview?pos=${encodeURIComponent(jobPosition)}`);

    ws.onopen  = () => setStatus('green', 'Live');

    ws.onmessage = ({ data }) => {
        if (data.startsWith('SYSTEM_TURN:USER')) {
            enableRecording(true);
            return;
        }
        if (data.startsWith('SYSTEM_INFO:')) {
            addSystemMsg(data.slice('SYSTEM_INFO:'.length));
            return;
        }
        if (data.startsWith('SYSTEM_END:')) {
            addSystemMsg('Interview complete — generating your report...');
            enableRecording(false);
            setStatus('gray', 'Finished');
            document.getElementById('new-interview-btn').style.display = 'block';
            ws.close();
            generateReport();
            return;
        }
        if (data.startsWith('SYSTEM_ERROR:')) {
            addSystemMsg('⚠ Error: ' + data.slice('SYSTEM_ERROR:'.length));
            return;
        }

        const colon   = data.indexOf(':');
        if (colon > -1) {
            const source  = data.slice(0, colon);
            const content = data.slice(colon + 1);
            if (source !== 'Candidate') {
                addMessage(source, content);
                conversationHistory.push({ role: source, content });
            }
        }
    };

    ws.onclose = () => setStatus('gray', 'Disconnected');
    ws.onerror = () => setStatus('red',  'Connection error');
}

// ══════════════════════════════════════════════════════════════════════════
// RECORDING
// ══════════════════════════════════════════════════════════════════════════
function enableRecording(on) {
    document.getElementById('record-btn').disabled = !on;
    document.getElementById('hint-text').innerText  = on
        ? 'Click the microphone to start recording your answer.'
        : 'Waiting for question...';
    if (!on && isRecording) stopRecording();
}

function toggleRecording() {
    isRecording ? stopRecording() : startRecording();
}

function startRecording() {
    if (!mediaStream) { alert('Microphone not available.'); return; }

    audioChunks  = [];
    frameMetrics = [];

    const audioStream = new MediaStream(mediaStream.getAudioTracks());
    const mimeType    = supportedMime();
    mediaRecorder     = new MediaRecorder(audioStream, { mimeType });

    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.start(200);
    isRecording    = true;
    recordingStart = Date.now();

    // UI
    document.getElementById('record-btn').innerHTML = '⏹ Stop Recording';
    document.getElementById('record-btn').classList.add('recording');
    document.getElementById('recording-indicator').style.display = 'flex';
    document.getElementById('submit-btn').disabled  = true;
    document.getElementById('hint-text').innerText  = 'Recording... click Stop when done speaking.';
    document.getElementById('transcription-preview').innerText = '';

    recTimerInterval = setInterval(tickTimer, 1000);
    animateBars(true);
}

function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

    isRecording = false;
    mediaRecorder.stop();

    document.getElementById('record-btn').innerHTML = '🎤 Re-record';
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('recording-indicator').style.display = 'none';
    document.getElementById('hint-text').innerText = 'Transcribing your answer...';
    clearInterval(recTimerInterval);
    animateBars(false);

    mediaRecorder.onstop = processAudio;
}

async function processAudio() {
    if (!audioChunks.length) {
        document.getElementById('hint-text').innerText = 'No audio recorded — please try again.';
        return;
    }

    try {
        const mime     = supportedMime();
        const ext      = mime.includes('mp4') ? 'mp4' : mime.includes('ogg') ? 'ogg' : 'webm';
        const blob     = new Blob(audioChunks, { type: mime });
        const form     = new FormData();
        form.append('audio', blob, `recording.${ext}`);

        document.getElementById('transcription-preview').innerText = '⏳ Transcribing...';

        const res    = await fetch('/transcribe', { method: 'POST', body: form });
        const result = await res.json();

        if (result.text && result.text.trim()) {
            document.getElementById('transcription-preview').innerText = `"${result.text}"`;
            document.getElementById('submit-btn').disabled      = false;
            document.getElementById('submit-btn').dataset.text  = result.text;
            document.getElementById('hint-text').innerText      = 'Review your answer and click Send.';
        } else {
            document.getElementById('transcription-preview').innerText = '⚠ Could not transcribe — please re-record.';
            document.getElementById('hint-text').innerText = 'Please re-record your answer.';
        }
    } catch (err) {
        console.error('Transcription error:', err);
        document.getElementById('transcription-preview').innerText = '⚠ Transcription failed.';
        document.getElementById('hint-text').innerText = 'Please try again.';
    }
}

// ══════════════════════════════════════════════════════════════════════════
// SUBMIT ANSWER
// ══════════════════════════════════════════════════════════════════════════
function submitAnswer() {
    const text = document.getElementById('submit-btn').dataset.text;
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    // Build per-answer metrics summary
    let metricsPrefix = '';
    if (faceApiLoaded && frameMetrics.length > 0) {
        const avgEye = Math.round(
            frameMetrics.reduce((s, m) => s + (m.eye_contact || 0), 0) / frameMetrics.length
        );
        const emotionCounts = {};
        frameMetrics.forEach(m => {
            if (m.emotion_key) emotionCounts[m.emotion_key] = (emotionCounts[m.emotion_key] || 0) + 1;
        });
        const domEmotion = Object.entries(emotionCounts).sort(([, a], [, b]) => b - a)[0]?.[0] || 'neutral';
        const headPct    = Math.round(
            frameMetrics.filter(m => m.head_ok).length / frameMetrics.length * 100
        );
        const headLabel  = headPct >= 60 ? 'straight' : 'frequently turned';

        answerMetricsSummaries.push({ eye_contact: avgEye, emotion: domEmotion, head: headLabel });
        metricsPrefix = `[METRICS: Eye Contact: ${avgEye}% | Emotion: ${domEmotion} | Head: ${headLabel}]\n`;
    }

    // Show user bubble (text only, without metrics prefix)
    addMessage('You', text);
    conversationHistory.push({ role: 'Candidate', content: text });

    ws.send(metricsPrefix + text);

    // Reset controls
    document.getElementById('transcription-preview').innerText = '';
    document.getElementById('submit-btn').disabled     = true;
    document.getElementById('submit-btn').dataset.text = '';
    const rb = document.getElementById('record-btn');
    rb.disabled   = true;
    rb.innerHTML  = '🎤 Start Recording';
    document.getElementById('hint-text').innerText = 'Waiting for response...';
    audioChunks = [];
}

// ══════════════════════════════════════════════════════════════════════════
// REPORT
// ══════════════════════════════════════════════════════════════════════════
async function generateReport() {
    const modal = document.getElementById('report-modal');
    modal.style.display = 'flex';

    try {
        const res    = await fetch('/generate-report', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({
                conversation:  conversationHistory,
                metrics:       answerMetricsSummaries,
                job_position:  jobPosition,
            }),
        });
        const report = await res.json();
        renderReport(report);
    } catch (err) {
        document.getElementById('report-content').innerHTML =
            '<p style="color:var(--text-muted)">Could not generate report. Please refresh and try again.</p>';
    }
}

function renderReport(r) {
    const recColor = {
        'Strong Hire': '#10b981',
        'Hire':        '#3b82f6',
        'Maybe':       '#f59e0b',
        'No Hire':     '#ef4444',
    }[r.recommendation] || '#94a3b8';

    const bar = (score) => {
        const pct   = Math.round((score / 10) * 100);
        const color = score >= 7 ? '#10b981' : score >= 5 ? '#f59e0b' : '#ef4444';
        return `<div class="score-bar-wrap">
                  <div class="score-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>`;
    };

    const li = (arr) =>
        (arr || []).map(s => `<li>${s}</li>`).join('');

    document.getElementById('report-content').innerHTML = `
        <div class="report-recommendation" style="color:${recColor}">${r.recommendation || '—'}</div>
        <p class="report-summary">${r.summary || ''}</p>

        <div class="scores-grid">
            <div class="score-item">
                <div class="score-label">Overall</div>
                <div class="score-num">${r.overall_score ?? '—'}/10</div>
                ${bar(r.overall_score)}
            </div>
            <div class="score-item">
                <div class="score-label">Technical</div>
                <div class="score-num">${r.technical_score ?? '—'}/10</div>
                ${bar(r.technical_score)}
            </div>
            <div class="score-item">
                <div class="score-label">Communication</div>
                <div class="score-num">${r.communication_score ?? '—'}/10</div>
                ${bar(r.communication_score)}
            </div>
            <div class="score-item">
                <div class="score-label">Confidence</div>
                <div class="score-num">${r.confidence_score ?? '—'}/10</div>
                ${bar(r.confidence_score)}
            </div>
        </div>

        <div class="report-columns">
            <div>
                <h4>💪 Strengths</h4>
                <ul class="report-list strengths">${li(r.strengths)}</ul>
            </div>
            <div>
                <h4>🎯 Areas to Improve</h4>
                <ul class="report-list improvements">${li(r.improvements)}</ul>
            </div>
        </div>
    `;
}

function closeReport() {
    document.getElementById('report-modal').style.display = 'none';
}

// ══════════════════════════════════════════════════════════════════════════
// UI HELPERS
// ══════════════════════════════════════════════════════════════════════════
function addMessage(source, content) {
    const div   = document.getElementById('messages');
    const bubble = document.createElement('div');

    let type = 'interviewer';
    if (source === 'You' || source === 'Candidate') type = 'user';
    else if (source === 'Evaluator') type = 'evaluator';

    bubble.className = `message ${type}`;

    if (type !== 'user') {
        const name = document.createElement('div');
        name.className  = 'sender-name';
        name.innerText  = source;
        bubble.appendChild(name);
    }

    const text = document.createElement('div');
    text.innerText = content;
    bubble.appendChild(text);

    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;
}

function addSystemMsg(text) {
    const div  = document.getElementById('messages');
    const span = document.createElement('div');
    span.className = 'system-message';
    span.innerText  = text;
    div.appendChild(span);
    div.scrollTop = div.scrollHeight;
}

function setStatus(color, text) {
    const map = { green: '#10b981', gray: '#9ca3af', red: '#ef4444' };
    document.getElementById('status-dot').style.color  = map[color] || '#9ca3af';
    document.getElementById('status-text').innerText   = text;
}

function setFaceStatus(color, label) {
    const map = { green: '#10b981', yellow: '#f59e0b', red: '#ef4444' };
    document.getElementById('face-indicator').style.color = map[color] || '#9ca3af';
    document.getElementById('face-label').innerText        = label;
}

function setMetrics({ eye_contact, emotion, emotion_color, head, head_ok }) {
    // Eye contact bar
    if (eye_contact !== null && eye_contact !== undefined) {
        const pct   = eye_contact;
        const color = pct > 60 ? '#10b981' : pct > 30 ? '#f59e0b' : '#ef4444';
        document.getElementById('eye-bar').style.width       = pct + '%';
        document.getElementById('eye-bar').style.background  = color;
        document.getElementById('eye-value').innerText       = pct + '%';
    }
    if (emotion) {
        const el = document.getElementById('emotion-display');
        el.innerText    = emotion;
        el.style.color  = emotion_color || 'var(--text-muted)';
    }
    if (head) {
        const el = document.getElementById('pose-display');
        el.innerText   = head;
        el.style.color = head_ok !== false ? '#10b981' : '#f59e0b';
    }
}

function tickTimer() {
    const s    = Math.floor((Date.now() - recordingStart) / 1000);
    const m    = Math.floor(s / 60);
    const sec  = s % 60;
    document.getElementById('rec-timer').innerText = `${m}:${sec.toString().padStart(2, '0')}`;
}

function animateBars(on) {
    clearTimeout(speakingTimer);
    const ids = ['bar1','bar2','bar3','bar4','bar5'];
    if (!on) { ids.forEach(id => { document.getElementById(id).style.height = '4px'; }); return; }
    (function tick() {
        if (!isRecording) { ids.forEach(id => { document.getElementById(id).style.height = '4px'; }); return; }
        ids.forEach(id => {
            document.getElementById(id).style.height = (Math.random() * 18 + 4) + 'px';
        });
        speakingTimer = setTimeout(tick, 140);
    })();
}

function supportedMime() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || 'audio/webm';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Role quick-select ──────────────────────────────────────────────────────
function setRole(role) {
    document.getElementById('jobRole').value = role;
    document.querySelectorAll('.chip').forEach(c => {
        c.classList.toggle('active', c.textContent.trim() === role ||
            role.startsWith(c.textContent.trim().replace(' Dev', '')));
    });
}

// ── Restart / New Interview ────────────────────────────────────────────────
function restartInterview() {
    // Stop any active streams
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (detectionInterval) { clearInterval(detectionInterval); detectionInterval = null; }
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();

    // Reset state
    faceApiLoaded = false; faceApiReady = false;
    conversationHistory = []; answerMetricsSummaries = []; frameMetrics = []; audioChunks = [];
    isRecording = false;

    // Reset UI
    document.getElementById('messages').innerHTML = '';
    document.getElementById('transcription-preview').innerText = '';
    document.getElementById('new-interview-btn').style.display = 'none';
    document.getElementById('interview-screen').style.display = 'none';
    document.getElementById('setup-screen').style.display = 'flex';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    document.getElementById('jobRole').value = '';
}
