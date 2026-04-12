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
                    # Seed prompt keeps filler words and preserves natural speech
                    # as-spoken — Whisper uses this as a style/context hint.
                    prompt=(
                        "Um, uh, like, you know, I mean, so, actually, basically, "
                        "right, hmm, let me think, kind of, sort of, well, okay so, "
                        "I guess, I think, you know what I mean"
                    ),
                )
            return JSONResponse({"text": transcript.text})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"Transcription error: {e}")
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


@app.post("/generate-report")
async def generate_report(data: dict):
    """Generate a detailed final performance report using GPT-4o."""
    try:
        conversation  = data.get("conversation", [])
        metrics       = data.get("metrics", [])
        job_position  = data.get("job_position", "the position")
        company       = data.get("company", "")
        interview_type = data.get("interview_type", "technical")

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

        company_context = f" at {company}" if company else ""
        type_label = interview_type.replace("_", " ").title()

        prompt = f"""You evaluated a candidate for a {job_position}{company_context} position in a {type_label} interview.
Based on the interview transcript and behavioral metrics below, produce a detailed JSON performance report.

TRANSCRIPT:
{convo_text}

BEHAVIORAL METRICS PER ANSWER:
{metrics_text or "No behavioral data collected."}

IMPORTANT: The transcript was captured verbatim from speech — preserve and analyze all filler words, grammar errors, and natural speech patterns exactly as written. Do NOT mentally correct errors when scoring language fluency.

Return ONLY valid JSON with these exact keys:
- overall_score: integer 1-10
- overall_score_reason: string explaining why this score was given (2-3 sentences citing specific moments from the transcript)
- technical_score: integer 1-10
- technical_score_reason: string explaining why this score was given (2-3 sentences citing specific answers)
- communication_score: integer 1-10
- communication_score_reason: string explaining why this score was given (2-3 sentences with examples)
- confidence_score: integer 1-10
- confidence_score_reason: string explaining why this score was given (mention behavioral metrics if available)
- english_fluency_score: integer 1-10 (assess spoken grammar, vocabulary range, sentence structure, and clarity based on the raw transcript)
- english_fluency_reason: string (2-3 sentences explaining the score — mention specific transcript examples of strong or weak language use)
- filler_words: list of filler words/phrases actually observed in the transcript (e.g. "um", "uh", "like", "you know", "I mean") — list only what actually appeared
- grammar_observations: list of up to 4 specific grammar or vocabulary issues observed, each with a brief example from the transcript and a correction
- fluency_tips: list of exactly 3 actionable, personalized tips to improve spoken English fluency based on patterns seen in this interview
- strengths: list of exactly 3 detailed bullet strings (each 1-2 sentences, cite specific answers)
- improvements: list of exactly 3 detailed bullet strings (each 1-2 sentences with actionable advice)
- summary: 3-4 sentence overall summary string referencing specific interview moments
- recommendation: one of "Strong Hire" | "Hire" | "Maybe" | "No Hire"
- recommendation_reason: 1-2 sentence explanation of the hiring recommendation
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


def _interview_type_instructions(interview_type: str, job_position: str, company: str) -> str:
    """Return interviewer instructions tailored to interview type and company."""
    company_ctx = f" at {company}" if company else ""
    company_q   = f"commonly asked by {company}" if company else "commonly asked by top companies"

    if interview_type == "behavioral":
        return f"""You are conducting a BEHAVIORAL interview for a {job_position}{company_ctx} position.
Ask exactly 5 behavioral questions, one at a time, using the STAR format style {company_q}.
Focus on:
1. Teamwork and collaboration
2. Handling conflict or failure
3. Leadership and initiative
4. Adaptability under pressure
5. Motivation and cultural fit

Draw from real behavioral questions {company_q} for this role.
After the Evaluator has given feedback on the 5th answer, say TERMINATE.
Keep each question under 60 words.
Ignore any [METRICS] tags you see in candidate answers."""

    elif interview_type == "case_study":
        return f"""You are conducting a CASE STUDY interview for a {job_position}{company_ctx} position.
Present exactly 5 case study prompts, one at a time, {company_q}.
Focus on:
1. Problem framing and clarification
2. Structured analysis and data interpretation
3. Business or technical trade-offs
4. Solution design and feasibility
5. Communication of recommendations

Draw from real case study questions {company_q} for this role.
After the Evaluator has given feedback on the 5th answer, say TERMINATE.
Keep each case prompt under 80 words.
Ignore any [METRICS] tags you see in candidate answers."""

    else:  # technical (default)
        return f"""You are conducting a TECHNICAL interview for a {job_position}{company_ctx} position.
Ask exactly 5 technical questions, one at a time, {company_q}.
Focus on:
1. Core technical fundamentals relevant to {job_position}
2. Data structures or system design
3. Problem-solving / algorithm thinking
4. Tools, technologies, and hands-on experience
5. Architecture or scalability considerations

Draw from real technical questions {company_q} for this role to make them authentic and specific.
After the Evaluator has given feedback on the 5th answer, say TERMINATE.
Keep each question under 70 words.
Ignore any [METRICS] tags you see in candidate answers."""


async def create_interview_team(
    websocket: WebSocket,
    job_position: str,
    company: str = "",
    interview_type: str = "technical",
):
    handler = WebSocketInputHandler(websocket)

    company_ctx  = f" at {company}" if company else ""
    type_label   = interview_type.replace("_", " ").title()

    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"{type_label} Interviewer for {job_position}{company_ctx}",
        system_message=_interview_type_instructions(interview_type, job_position, company),
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
        system_message=f"""You are an expert interview coach evaluating a {job_position}{company_ctx} candidate in a {type_label} interview.

After EACH candidate answer provide concise feedback (max 80 words) with this EXACT format:

Start your response with one of these two sentiment markers on its own line:
POSITIVE: (use when the answer is good, strong, or meets expectations)
NEEDS WORK: (use when the answer is weak, incomplete, or needs significant improvement)

Then give your feedback:
- Comment on the quality and relevance of the answer
- Note communication clarity and structure
- If a [METRICS] tag is present, briefly mention body language observations
- Be specific and cite what was said

Be honest and direct — do not soften negative feedback.""",
    )

    return RoundRobinGroupChat(
        participants=[interviewer, candidate, evaluator],
        termination_condition=TextMentionTermination(text="TERMINATE"),
        max_turns=35,
    )


@app.websocket("/ws/interview")
async def websocket_endpoint(
    websocket: WebSocket,
    pos: str = "AI Engineer",
    company: str = "",
    interview_type: str = "technical",
):
    await websocket.accept()
    try:
        team = await create_interview_team(websocket, pos, company, interview_type)

        label = f"{pos}" + (f" @ {company}" if company else "")
        type_label = interview_type.replace("_", " ").title()
        await websocket.send_text(f"SYSTEM_INFO:Starting {type_label} interview for {label}...")

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
