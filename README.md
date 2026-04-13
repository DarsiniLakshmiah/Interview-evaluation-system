# AI Interview Coach

A real-time AI-powered interview practice system that simulates company-specific interviews, analyzes body language through your webcam, transcribes your spoken answers verbatim, and generates a detailed performance report with English fluency feedback.
## Live webpage - https://interview-evaluation-system-g9j6.onrender.com/
---

## What It Does

You pick a company (Google, Amazon, Meta, etc.), a job role, and an interview type — Technical, Behavioral, or Case Study. The system starts a live interview session where:

- An AI **Interviewer** asks 5 company-specific questions one at a time
- An AI **Evaluator** gives real-time feedback after every answer, color-coded green for positive and red for needs improvement
- Your **webcam** tracks eye contact, head position, and facial emotion throughout
- Your **voice** is recorded and transcribed verbatim (filler words and all)
- At the end, a detailed **performance report** breaks down your scores with specific reasons, English fluency analysis, grammar observations, and hiring recommendation

---

## How to Run

**1. Clone the repository and set up the virtual environment**

```bash
git clone <repo-url>
cd Interview-evaluation-system
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Create a `.env` file in the project root**

```
OPENAI_API_KEY=your_openai_api_key_here
```

**4. Start the server**

```bash
uvicorn app:app --reload
```

**5. Open the app**

Navigate to `http://127.0.0.1:8000` in your browser. Allow camera and microphone access when prompted.

---

## Project Structure

```
Interview-evaluation-system/
├── app.py                  # FastAPI backend — agents, WebSocket, transcription, report
├── requirements.txt
├── .env                    # API key (not committed)
├── templates/
│   └── index.html          # Single-page app UI
└── static/
    ├── script.js           # Frontend logic — WebSocket, recording, face analysis, report
    └── style.css           # Dark theme UI styles
```

---

## Architecture

This project is built around three separate AI agents that communicate in real time over a WebSocket connection. This is what makes it meaningfully different from a typical chatbot or Q&A app.

```
Browser (WebSocket client)
    │
    ├── MediaRecorder API  →  /transcribe  →  Whisper API
    ├── face-api.js        →  Live webcam metrics
    │
    └── /ws/interview  ←──────────────────────────────────┐
                                                           │
                              FastAPI WebSocket Handler    │
                                        │                  │
                              AutoGen RoundRobinGroupChat  │
                              ┌─────────────────────┐      │
                              │  Interviewer Agent  │      │
                              │  (AssistantAgent)   │      │
                              ├─────────────────────┤      │
                              │  Candidate Agent    │──────┘
                              │  (UserProxyAgent)   │  input_func bridges
                              ├─────────────────────┤  WebSocket ↔ AutoGen
                              │  Evaluator Agent    │
                              │  (AssistantAgent)   │
                              └─────────────────────┘
                                        │
                              /generate-report  →  GPT-4o JSON report
```

The `UserProxyAgent` acts as the bridge between the human (browser) and the AutoGen team. Every time it is the Candidate's turn, the backend sends a `SYSTEM_TURN:USER` signal over WebSocket, the browser enables the recording button, and the typed/spoken answer is sent back through the same socket into the agent pipeline.

---

## Technologies Used and What I Learned

### FastAPI
FastAPI was used for the backend. I learned how to build both REST endpoints (`/transcribe`, `/generate-report`) and a persistent **WebSocket endpoint** (`/ws/interview`) in the same application. The key learning was understanding how `async/await` works with streaming — the `async for message in team.run_stream(...)` loop keeps the connection open and pushes each agent message to the browser the moment it is generated, rather than waiting for the full interview to complete.

### AutoGen AgentChat (Microsoft)
This was the most technically challenging and rewarding part of the project. AutoGen is a multi-agent orchestration framework. I learned:

- The difference between `AssistantAgent` (LLM-backed, responds autonomously) and `UserProxyAgent` (represents a human, delegates input to a function)
- How `RoundRobinGroupChat` coordinates multiple agents in a fixed rotation — Interviewer → Candidate → Evaluator → repeat
- How `TextMentionTermination` works as a **stop condition**: the interviewer says `TERMINATE` after the final evaluation and the loop ends cleanly
- How `max_turns=35` acts as a hard ceiling to prevent runaway agent loops
- How to inject a custom `input_func` into `UserProxyAgent` to bridge an async WebSocket into the synchronous agent input interface

### OpenAI GPT-4o
Used for two purposes — driving the Interviewer and Evaluator agents, and generating the final report. I learned how `response_format={"type": "json_object"}` forces GPT-4o to return valid JSON every time without needing to parse markdown code fences. I also learned how careful **system message engineering** shapes agent behavior — the evaluator's `POSITIVE:` / `NEEDS WORK:` sentiment markers are entirely a result of prompt design, not post-processing.

### OpenAI Whisper (whisper-1)
Used for speech-to-text transcription. I learned something non-obvious: Whisper silently "cleans" speech by default — it removes filler words, corrects grammar, and normalizes hesitations. To get **verbatim transcription** (which is needed for genuine fluency analysis), I learned to use the `prompt` parameter as a style seed. Seeding the prompt with real filler words (`"Um, uh, like, you know, I mean..."`) signals to Whisper that this style of speech should be preserved rather than corrected.

### face-api.js (vladmandic fork)
A browser-side face detection library based on TensorFlow.js models. I learned how to:
- Load three separate neural network models (face detector, 68-point landmark detector, expression classifier) asynchronously from a CDN
- Run inference on live webcam frames at 500ms intervals using `setInterval`
- Calculate **eye contact** from nose tip deviation relative to face bounding box center
- Estimate **head pose** from the vertical ratio of nose-to-eye distance vs face height
- Map raw expression probabilities to human-readable labels

The key learning was that the face analysis runs entirely **in the browser** — no video frames are ever sent to the server, which is both a privacy benefit and a latency benefit.

### WebSockets (native browser + FastAPI)
I learned the full WebSocket lifecycle: connection, message passing, and graceful disconnection. A key challenge was designing a **message protocol** over a single text channel. Since WebSocket sends plain strings, I designed a prefix system:

| Prefix | Meaning |
|---|---|
| `SYSTEM_TURN:USER` | Backend is waiting for the candidate's answer |
| `SYSTEM_INFO:...` | Informational system message |
| `SYSTEM_END:...` | Interview finished, generate report |
| `SYSTEM_ERROR:...` | Something went wrong |
| `Interviewer:...` | Interviewer's question |
| `Evaluator:...` | Evaluator's feedback |

This pattern of multiplexing structured events over a single WebSocket channel is a real-world pattern used in live collaboration tools.

### Prompt Engineering as a Guardrail
I learned that **how you write a system message is a form of application logic**. Examples from this project:

- Telling the evaluator to start every response with exactly `POSITIVE:` or `NEEDS WORK:` on its own line makes sentiment parsing on the frontend completely reliable — no NLP needed
- Telling the report generator "do NOT mentally correct errors when scoring language fluency" prevents the model from silently fixing the candidate's speech before analyzing it
- Telling the interviewer to "Ignore any [METRICS] tags you see in candidate answers" prevents the face-analysis data injected into answers from confusing the question flow
- Telling the interviewer to "say TERMINATE after the Evaluator has given feedback on the 5th answer" gives AutoGen a deterministic exit condition

### Jinja2 Templating
Used to serve the single HTML page from FastAPI. I learned how `TemplateResponse` works and how static files are mounted separately under `/static`.

---

## What Makes This Different from Typical Projects

Most AI interview tools are simple chatbots — one prompt in, one answer out. This project is different in several ways:

**Multi-agent pipeline, not a single LLM call.** Three specialized agents with separate roles and system messages run concurrently. The Interviewer doesn't know what the Evaluator says (and vice versa) because they have separate context windows and instructions. This mirrors how a real panel interview works.

**Real-time behavioral analysis alongside text.** Most projects either do text analysis OR video analysis. Here, facial emotion, eye contact percentage, and head position are tracked frame-by-frame during the answer and aggregated into a per-question behavioral summary that is appended to the answer text before the Evaluator sees it.

**Verbatim transcription for honest feedback.** Typical voice apps transcribe "clean" text. This project deliberately preserves filler words and grammar errors in the transcript so the English fluency analysis in the report reflects what was actually said, not a polished version of it.

**Company-specific question generation.** Instead of generic interview questions, the system instructs GPT-4o to draw from known interview patterns at specific companies. This uses the model's training knowledge of public interview experiences shared on forums like Glassdoor and LeetCode Discuss — making it a lightweight knowledge-retrieval approach without a vector database.

**Interview type specialization.** The same system handles three structurally different interview formats — Technical (algorithms, system design), Behavioral (STAR format), and Case Study (structured problem analysis) — each with its own tailored agent instructions.

**Detailed score reasoning.** The report doesn't just give a number. Every score comes with 2–3 sentences that cite specific moments from the transcript. The hiring recommendation also includes a reason. This makes the feedback actionable rather than decorative.

---

## Guardrails Implemented

| Guardrail | Why It Matters |
|---|---|
| `max_turns=35` hard limit on the AutoGen team | Prevents infinite agent loops if the termination condition is never triggered |
| `TextMentionTermination("TERMINATE")` | Clean, deterministic stop condition that doesn't rely on the LLM deciding when to stop |
| `response_format={"type": "json_object"}` on the report endpoint | Prevents GPT-4o from wrapping JSON in markdown or prose, which would break `json.loads()` |
| Sentiment markers (`POSITIVE:` / `NEEDS WORK:`) enforced in the evaluator system message | Makes feedback coloring on the frontend a simple string prefix check — no unreliable sentiment classification model needed |
| Whisper verbatim prompt seeding | Prevents the transcription layer from silently "fixing" the candidate's speech, which would corrupt the fluency analysis |
| `[METRICS]` tag isolation in interviewer system message | Prevents raw telemetry data appended to answers from leaking into the question flow |
| `os.unlink(tmp_path)` in a `finally` block | Ensures temporary audio files are always deleted from disk even if transcription fails |
| WebSocket disconnect → `return "TERMINATE"` | Gracefully exits the AutoGen loop if the browser disconnects mid-interview instead of hanging |
| `try/except` around the final WebSocket error send | Prevents a secondary exception when trying to report an error on an already-closed socket |

---

## Requirements

```
autogen-agentchat==0.5.7
autogen-core
autogen-ext
python-dotenv
openai
tiktoken
pydantic
fastapi
uvicorn
websockets
jinja2
aiofiles
python-multipart
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key — used for GPT-4o (agents + report) and Whisper (transcription) |

---

## Known Limitations

- Face analysis requires a modern browser with WebGL support. It degrades gracefully if unavailable.
- Company-specific questions are generated from GPT-4o's training knowledge, not a live database. Questions for very niche or new companies fall back to role-appropriate general questions.
- Whisper verbatim mode is approximate — very fast or heavily accented speech may still be partially normalized.
- The interview runs entirely in one WebSocket session; if the connection drops mid-interview, the session cannot be resumed.
