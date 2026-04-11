from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv
from typing import Optional
import openai
import json
import os
import tempfile

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY)
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio using OpenAI Whisper API."""
    try:
        audio_bytes = await audio.read()

        # Pick file extension so Whisper detects the format correctly
        content_type = audio.content_type or ""
        if "mp4" in content_type:
            suffix = ".mp4"
        elif "wav" in content_type:
            suffix = ".wav"
        elif "ogg" in content_type:
            suffix = ".ogg"
        else:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                transcript = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            return JSONResponse({"text": transcript.text})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"Transcription error: {e}")
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


@app.post("/generate-report")
async def generate_report(data: dict):
    """Generate a final performance report using GPT-4o."""
    try:
        conversation = data.get("conversation", [])
        metrics = data.get("metrics", [])
        job_position = data.get("job_position", "the position")

        convo_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation
        )

        metrics_text = ""
        for i, m in enumerate(metrics, 1):
            metrics_text += (
                f"Q{i}: Eye Contact {m.get('eye_contact', 'N/A')}% | "
                f"Emotion: {m.get('emotion', 'N/A')} | "
                f"Head: {m.get('head', 'N/A')}\n"
            )

        prompt = f"""You evaluated a candidate for a {job_position} position.
Based on the interview transcript and behavioral metrics below, produce a JSON performance report.

TRANSCRIPT:
{convo_text}

BEHAVIORAL METRICS PER ANSWER:
{metrics_text or "No behavioral data collected."}

Return ONLY valid JSON with these exact keys:
- overall_score: integer 1-10
- technical_score: integer 1-10
- communication_score: integer 1-10
- confidence_score: integer 1-10
- strengths: list of exactly 3 short bullet strings
- improvements: list of exactly 3 short bullet strings
- summary: 2-3 sentence overall summary string
- recommendation: one of "Strong Hire" | "Hire" | "Maybe" | "No Hire"
"""

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        report = json.loads(response.choices[0].message.content)
        return JSONResponse(report)

    except Exception as e:
        print(f"Report generation error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── AutoGen team ─────────────────────────────────────────────────────────────

class WebSocketInputHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def get_input(self, prompt: str, cancellation_token: Optional[object] = None) -> str:
        try:
            await self.websocket.send_text("SYSTEM_TURN:USER")
            data = await self.websocket.receive_text()
            return data
        except WebSocketDisconnect:
            return "TERMINATE"


async def create_interview_team(websocket: WebSocket, job_position: str):
    handler = WebSocketInputHandler(websocket)

    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"Interviewer for {job_position}",
        system_message=f"""You are a professional interviewer for a {job_position} position.
Ask exactly 3 questions, one at a time, covering:
1. Technical skills and experience
2. Problem-solving ability
3. Cultural fit and motivation

After the Evaluator has given feedback on the 3rd answer, say TERMINATE.
Keep each question under 50 words.
Ignore any [METRICS] tags you see in candidate answers.""",
    )

    candidate = UserProxyAgent(
        name="Candidate",
        description="The interview candidate",
        input_func=handler.get_input,
    )

    evaluator = AssistantAgent(
        name="Evaluator",
        model_client=model_client,
        description="Interview coach",
        system_message=f"""You are an interview coach evaluating a {job_position} candidate.
After each candidate answer provide brief feedback (max 60 words):
- Comment on answer quality and relevance
- Note communication clarity
- If a [METRICS] tag is present, briefly mention body language observations

Be specific, constructive, and encouraging.""",
    )

    return RoundRobinGroupChat(
        participants=[interviewer, candidate, evaluator],
        termination_condition=TextMentionTermination(text="TERMINATE"),
        max_turns=20,
    )


@app.websocket("/ws/interview")
async def websocket_endpoint(websocket: WebSocket, pos: str = "AI Engineer"):
    await websocket.accept()
    try:
        team = await create_interview_team(websocket, pos)
        await websocket.send_text(f"SYSTEM_INFO:Starting interview for {pos}...")

        async for message in team.run_stream(task="Start the interview with the first question."):
            if isinstance(message, TaskResult):
                await websocket.send_text(f"SYSTEM_END:{message.stop_reason}")
            else:
                await websocket.send_text(f"{message.source}:{message.content}")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_text(f"SYSTEM_ERROR:{str(e)}")
        except Exception:
            pass
