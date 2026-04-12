// ══════════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════════
let ws                  = null;
let jobPosition         = '';
let companyName         = '';
let interviewType       = 'technical';   // 'technical' | 'behavioral' | 'case_study'
let mediaStream         = null;
let mediaRecorder       = null;
let audioChunks         = [];
let isRecording         = false;
let recordingStart      = null;
let recTimerInterval    = null;
let speakingTimer       = null;

let faceApiReady        = false;
let faceApiLoaded       = false;
let detectionInterval   = null;
let currentMetrics      = { eye_contact: 0, emotion: 'neutral', emotion_key: 'neutral', head_ok: true };
let frameMetrics        = [];
let conversationHistory     = [];
let answerMetricsSummaries  = [];

function onFaceApiScriptLoaded() {
    faceApiReady = true;
}

// ══════════════════════════════════════════════════════════════════════════
// SETUP
// ══════════════════════════════════════════════════════════════════════════
function setCompany(name) {
    companyName = name;
    document.getElementById('companyName').value = name;
    document.querySelectorAll('.company-chip').forEach(c => {
        const match = c.textContent.trim().toLowerCase() === name.toLowerCase();
        c.classList.toggle('active', match);
    });
}

// Called on every keystroke / paste in the company input field
function syncCompanyChip(value) {
    companyName = value.trim();
    const lower = companyName.toLowerCase();
    document.querySelectorAll('.company-chip').forEach(c => {
        const match = c.textContent.trim().toLowerCase() === lower;
        c.classList.toggle('active', match);
    });
}

function setRole(role) {
    jobPosition = role;
    document.getElementById('jobRole').value = role;
    document.querySelectorAll('.role-chip').forEach(c => {
        c.classList.toggle('active', c.textContent.trim() === role ||
            role.startsWith(c.textContent.trim().replace(' Dev', '')));
    });
}

function setInterviewType(type) {
    interviewType = type;
    document.querySelectorAll('.type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });
}

async function startInterview() {
    jobPosition = document.getElementById('jobRole').value.trim();
    companyName = document.getElementById('companyName').value.trim();

    if (!jobPosition) { alert('Please enter a job role.'); return; }

    document.getElementById('setup-screen').style.display  = 'none';
    document.getElementById('interview-screen').style.display = 'flex';

    const typeLabel = { technical: 'Technical', behavioral: 'Behavioral', case_study: 'Case Study' }[interviewType] || 'Technical';
    const titleParts = [typeLabel, 'Interview —', jobPosition];
    if (companyName) titleParts.push(`@ ${companyName}`);
    document.getElementById('role-title').innerText = titleParts.join(' ');

    await initCamera();
    initFaceApi();
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

// ── Detection loop ────────────────────────────────────────────────────────
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

            const noseTipX    = lm.getNose()[3].x;
            const faceCenterX = box.x + box.width / 2;
            const deviation   = Math.abs(noseTipX - faceCenterX) / (box.width + 1e-6);
            const eyeContact  = Math.max(0, Math.min(100, Math.round((1 - deviation * 3) * 100)));

            const leftEye  = lm.getLeftEye();
            const rightEye = lm.getRightEye();
            const eyeMidY  = (leftEye[0].y + rightEye[3].y) / 2;
            const noseTipY = lm.getNose()[6].y;
            const vRatio   = (noseTipY - eyeMidY) / (box.height + 1e-6);
            const headOk   = vRatio > 0.15 && vRatio < 0.48;

            const topExpr = Object.entries(expr).sort(([, a], [, b]) => b - a)[0];
            const emotionMap = {
                happy:     { label: '😊 Confident',     color: '#10b981' },
                neutral:   { label: '😐 Neutral',       color: '#94a3b8' },
                surprised: { label: '😮 Surprised',     color: '#f59e0b' },
                sad:       { label: '😔 Nervous',       color: '#f87171' },
                fearful:   { label: '😰 Anxious',       color: '#f87171' },
                disgusted: { label: '😒 Uncomfortable', color: '#f59e0b' },
                angry:     { label: '😠 Stressed',      color: '#f87171' },
            };
            const emoInfo = emotionMap[topExpr[0]] || { label: topExpr[0], color: '#94a3b8' };

            const metrics = {
                face:          true,
                eye_contact:   eyeContact,
                head_ok:       headOk,
                head:          headOk ? '✓ Straight' : '⚠ Turned',
                emotion:       emoInfo.label,
                emotion_key:   topExpr[0],
                emotion_color: emoInfo.color,
            };

            currentMetrics = metrics;
            setMetrics(metrics);

            if (isRecording) frameMetrics.push({ ...metrics });

        } catch (_) { /* silent */ }
    }, 500);
}

// ══════════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ══════════════════════════════════════════════════════════════════════════
function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const params = new URLSearchParams({
        pos:            jobPosition,
        company:        companyName,
        interview_type: interviewType,
    });
    ws = new WebSocket(`${proto}://${window.location.host}/ws/interview?${params}`);

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
        const mime = supportedMime();
        const ext  = mime.includes('mp4') ? 'mp4' : mime.includes('ogg') ? 'ogg' : 'webm';
        const blob = new Blob(audioChunks, { type: mime });
        const form = new FormData();
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

    addMessage('You', text);
    conversationHistory.push({ role: 'Candidate', content: text });

    ws.send(metricsPrefix + text);

    document.getElementById('transcription-preview').innerText = '';
    document.getElementById('submit-btn').disabled     = true;
    document.getElementById('submit-btn').dataset.text = '';
    const rb = document.getElementById('record-btn');
    rb.disabled  = true;
    rb.innerHTML = '🎤 Start Recording';
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
                conversation:   conversationHistory,
                metrics:        answerMetricsSummaries,
                job_position:   jobPosition,
                company:        companyName,
                interview_type: interviewType,
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

    const scoreCard = (label, score, reason, extra) => `
        <div class="score-item">
            <div class="score-label">${label}</div>
            <div class="score-num">${score ?? '—'}/10</div>
            ${bar(score)}
            ${reason ? `<div class="score-reason">${reason}</div>` : ''}
            ${extra || ''}
        </div>`;

    const li = (arr, cls) =>
        (arr || []).map(s => `<li class="${cls}">${s}</li>`).join('');

    const companyLine = companyName ? ` <span class="report-company">@ ${companyName}</span>` : '';
    const typeLabel   = { technical: '💻 Technical', behavioral: '🤝 Behavioral', case_study: '📊 Case Study' }[interviewType] || interviewType;

    // Filler word pills
    const fillerPills = (r.filler_words || []).length
        ? (r.filler_words).map(w => `<span class="filler-pill">${w}</span>`).join('')
        : '<span class="filler-pill none">None detected</span>';

    // Grammar observations
    const grammarRows = (r.grammar_observations || []).length
        ? (r.grammar_observations).map(g => `<li class="grammar-item">${g}</li>`).join('')
        : '<li class="grammar-item none">No major issues observed</li>';

    // Fluency tips
    const fluencyTips = (r.fluency_tips || []).map(t => `<li class="improvement-item">${t}</li>`).join('');

    // Extra filler info inside the fluency score card
    const fillerExtra = `<div class="filler-row"><span class="filler-label">Filler words:</span>${fillerPills}</div>`;

    document.getElementById('report-content').innerHTML = `
        <div class="report-meta">${typeLabel} Interview — ${jobPosition}${companyLine}</div>

        <div class="report-recommendation" style="color:${recColor}">${r.recommendation || '—'}</div>
        ${r.recommendation_reason ? `<p class="report-rec-reason" style="color:${recColor}">${r.recommendation_reason}</p>` : ''}
        <p class="report-summary">${r.summary || ''}</p>

        <div class="scores-grid">
            ${scoreCard('Overall',        r.overall_score,        r.overall_score_reason)}
            ${scoreCard('Technical',      r.technical_score,      r.technical_score_reason)}
            ${scoreCard('Communication',  r.communication_score,  r.communication_score_reason)}
            ${scoreCard('Confidence',     r.confidence_score,     r.confidence_score_reason)}
        </div>

        <div class="fluency-section">
            <div class="fluency-header">
                <div>
                    <h4>🗣 English Fluency</h4>
                    <p class="fluency-subheading">Based on your raw, verbatim speech patterns</p>
                </div>
                <div class="fluency-score-badge ${r.english_fluency_score >= 7 ? 'good' : r.english_fluency_score >= 5 ? 'fair' : 'weak'}">
                    ${r.english_fluency_score ?? '—'}/10
                </div>
            </div>
            <p class="fluency-reason">${r.english_fluency_reason || ''}</p>

            ${fillerExtra}

            <div class="fluency-columns">
                <div>
                    <h5>Grammar Observations</h5>
                    <ul class="report-list">${grammarRows}</ul>
                </div>
                <div>
                    <h5>Tips to Improve</h5>
                    <ul class="report-list">${fluencyTips}</ul>
                </div>
            </div>
        </div>

        <div class="report-columns">
            <div>
                <h4>💪 Strengths</h4>
                <ul class="report-list">${li(r.strengths, 'strength-item')}</ul>
            </div>
            <div>
                <h4>🎯 Areas to Improve</h4>
                <ul class="report-list">${li(r.improvements, 'improvement-item')}</ul>
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

/**
 * Parse evaluator content for POSITIVE: / NEEDS WORK: sentiment markers.
 * Returns { sentiment: 'positive'|'negative'|'neutral', text: string }
 */
function parseEvaluatorContent(content) {
    const trimmed = content.trim();

    if (trimmed.startsWith('POSITIVE:')) {
        return { sentiment: 'positive', text: trimmed.slice('POSITIVE:'.length).trim() };
    }
    if (trimmed.startsWith('NEEDS WORK:')) {
        return { sentiment: 'negative', text: trimmed.slice('NEEDS WORK:'.length).trim() };
    }
    // Fallback: scan first line
    const firstLine = trimmed.split('\n')[0].toUpperCase();
    if (firstLine.includes('POSITIVE')) {
        return { sentiment: 'positive', text: trimmed };
    }
    if (firstLine.includes('NEEDS WORK') || firstLine.includes('NEGATIVE') || firstLine.includes('IMPROVE')) {
        return { sentiment: 'negative', text: trimmed };
    }
    return { sentiment: 'neutral', text: trimmed };
}

function addMessage(source, content) {
    const div    = document.getElementById('messages');
    const bubble = document.createElement('div');

    let type = 'interviewer';
    if (source === 'You' || source === 'Candidate') type = 'user';
    else if (source === 'Evaluator') type = 'evaluator';

    if (type === 'evaluator') {
        const { sentiment, text } = parseEvaluatorContent(content);
        bubble.className = `message evaluator ${sentiment}`;

        const badge = document.createElement('div');
        badge.className = 'eval-badge';
        if (sentiment === 'positive') {
            badge.innerHTML = '<span class="eval-sentiment positive-badge">✓ Positive Feedback</span>';
        } else if (sentiment === 'negative') {
            badge.innerHTML = '<span class="eval-sentiment negative-badge">✗ Needs Improvement</span>';
        } else {
            badge.innerHTML = '<span class="eval-sentiment neutral-badge">Evaluator</span>';
        }
        bubble.appendChild(badge);

        const textEl = document.createElement('div');
        textEl.innerText = text;
        bubble.appendChild(textEl);
    } else {
        bubble.className = `message ${type}`;

        if (type !== 'user') {
            const name = document.createElement('div');
            name.className = 'sender-name';
            name.innerText = source;
            bubble.appendChild(name);
        }

        const text = document.createElement('div');
        text.innerText = content;
        bubble.appendChild(text);
    }

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
    if (eye_contact !== null && eye_contact !== undefined) {
        const pct   = eye_contact;
        const color = pct > 60 ? '#10b981' : pct > 30 ? '#f59e0b' : '#ef4444';
        document.getElementById('eye-bar').style.width      = pct + '%';
        document.getElementById('eye-bar').style.background = color;
        document.getElementById('eye-value').innerText      = pct + '%';
    }
    if (emotion) {
        const el = document.getElementById('emotion-display');
        el.innerText   = emotion;
        el.style.color = emotion_color || 'var(--text-muted)';
    }
    if (head) {
        const el = document.getElementById('pose-display');
        el.innerText   = head;
        el.style.color = head_ok !== false ? '#10b981' : '#f59e0b';
    }
}

function tickTimer() {
    const s   = Math.floor((Date.now() - recordingStart) / 1000);
    const m   = Math.floor(s / 60);
    const sec = s % 60;
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

// ── Restart ────────────────────────────────────────────────────────────────
function restartInterview() {
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (detectionInterval) { clearInterval(detectionInterval); detectionInterval = null; }
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();

    faceApiLoaded = false; faceApiReady = false;
    conversationHistory = []; answerMetricsSummaries = []; frameMetrics = []; audioChunks = [];
    isRecording = false; jobPosition = ''; companyName = ''; interviewType = 'technical';

    document.getElementById('messages').innerHTML = '';
    document.getElementById('transcription-preview').innerText = '';
    document.getElementById('new-interview-btn').style.display = 'none';
    document.getElementById('interview-screen').style.display = 'none';
    document.getElementById('setup-screen').style.display = 'flex';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.type-btn').forEach(b => b.classList.toggle('active', b.dataset.type === 'technical'));
    document.getElementById('jobRole').value = '';
    document.getElementById('companyName').value = '';
}
