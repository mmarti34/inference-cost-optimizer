"""parse_import.py

POST /api/parse-import
Parses an LLM SDK code snippet and extracts the call config using GPT-4o-mini.
Requires SYSTEM_OPENAI_API_KEY env var.
"""
import json
import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth_dependency import require_auth, AuthenticatedUser

logger = logging.getLogger(__name__)
router = APIRouter()

SYSTEM_OPENAI_API_KEY = os.environ.get("SYSTEM_OPENAI_API_KEY", "")

PARSE_PROMPT = """\
Extract the AI API call config from the code below. Return ONLY valid JSON — no markdown, no explanation.

STEP 1 — Identify the api_type:
- "chat"             → text chat/completion (openai.chat.completions.create, anthropic.messages.create, responses.create, groq/mistral/cohere/together chat, etc.)
- "vision"           → chat call where messages include an image_url or base64 image in the content array
- "image_generation" → image generation (openai.images.generate, stability, replicate image models, etc.)
- "tts"              → text-to-speech (openai.audio.speech.create, elevenlabs, etc.)
- "stt"              → speech-to-text / transcription (openai.audio.transcriptions.create, whisper, deepgram, etc.)
- "embeddings"       → vector embeddings (openai.embeddings.create, cohere embed, etc.)

STEP 2 — Apply rules by api_type:

For "chat" and "vision":
- "system_prompt": HARDCODED text from system/developer/instructions role. Use "" if none. Do NOT include user-role content.
- "user_variable": snake_case variable NAME for dynamic user input (e.g. userMessage → user_message). If hardcoded string literal, use "user_message".
- Treat role "developer", "system", "instructions" identically — all are system prompts.
- For responses.create / Responses API: "input" array = same as "messages".
- For Anthropic: "system" param = system prompt; user role in messages[] = user input.
- For "vision": also set "image_variable" to the snake_case name of the image variable (e.g. imageUrl → image_url). If hardcoded, use "image_url".
- "stream": true if streaming, else false or null.

For "image_generation":
- "user_variable": snake_case name of the text prompt variable (e.g. promptText → prompt_text). If hardcoded, use "prompt".

For "tts":
- "user_variable": snake_case name of the input text variable (e.g. inputText → input_text). If hardcoded, use "text".

For "stt":
- "user_variable": snake_case name of the audio input variable (e.g. audioFile → audio_file). If hardcoded, use "audio".

For "embeddings":
- "user_variable": snake_case name of the text-to-embed variable (e.g. inputText → input_text). If hardcoded, use "text".

Return this JSON shape:
{
  "api_type": "chat" | "vision" | "image_generation" | "tts" | "stt" | "embeddings",
  "provider": "openai" | "anthropic" | "gemini" | "mistral" | "cohere" | "groq" | "together" | "deepseek" | "fireworks" | "elevenlabs" | "stability" | "replicate",
  "model": "exact model string from the code",
  "system_prompt": "hardcoded system/developer role text, or empty string",
  "user_variable": "snake_case variable name for primary dynamic input",
  "image_variable": "snake_case variable name for image input (vision only), or null",
  "temperature": number or null,
  "max_tokens": number or null,
  "stream": true | false | null,
  "suggestedName": "short-kebab-case endpoint name based on the apparent purpose"
}

Use null for fields not applicable to the api_type. If this is not an AI API call, return:
{"error": "not an LLM call"}

Code:
"""


class ParseImportRequest(BaseModel):
    code: str


@router.post("/parse-import")
async def parse_import(
    body: ParseImportRequest,
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    code = body.code.strip()
    if len(code) < 20:
        return JSONResponse(status_code=400, content={"error": "code is too short"})
    if len(code) > 5000:
        return JSONResponse(status_code=400, content={"error": "code is too long (max 5000 chars)"})

    if not SYSTEM_OPENAI_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "import parsing is not available (SYSTEM_OPENAI_API_KEY not configured)"},
        )

    prompt = PARSE_PROMPT + code
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SYSTEM_OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 600,
                },
            )
            resp.raise_for_status()

        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        logger.warning("parse_import: could not parse model JSON response: %s", e)
        return JSONResponse(status_code=422, content={"error": "could not parse model response as JSON"})
    except httpx.HTTPStatusError as e:
        logger.error("parse_import: OpenAI API error %s: %s", e.response.status_code, e.response.text)
        return JSONResponse(status_code=502, content={"error": "upstream API error"})
    except Exception as e:
        logger.error("parse_import: unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")
