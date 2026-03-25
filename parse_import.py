"""parse_import.py

POST /api/parse-import
Parses an LLM SDK code snippet and extracts the call config using GPT-4o-mini.
Requires SYSTEM_OPENAI_API_KEY env var.
"""
import json
import logging
import os
import re

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth_dependency import require_auth, AuthenticatedUser

logger = logging.getLogger(__name__)
router = APIRouter()

SYSTEM_OPENAI_API_KEY = os.environ.get("SYSTEM_OPENAI_API_KEY", "")

PARSE_PROMPT = """\
Extract the AI API call config from the code below. Return ONLY valid JSON — no markdown, no explanation.

STEP 1 — Identify the api_type. Read the ENTIRE call carefully before deciding.
- "chat"             → text completion with NO tools parameter. Includes: chat.completions.create, anthropic.messages.create, responses.create (with or without a messages/input array), any provider chat endpoint — as long as there is NO explicit tools/functions parameter.
- "vision"           → same as "chat" (no tools) but the messages/input contain an image_url or base64 image in the content.
- "tool_call"        → ONLY when THIS snippet contains tools= / tools: / functions= / tool_choice= passing tool definitions. If those keys are absent, use "chat". The method name "responses.create" alone is NOT tool_call.
- CRITICAL — OpenAI Responses API: client.responses.create({ model, input, reasoning, ... }) where reasoning is e.g. { effort: "low" } is ALWAYS "chat". "reasoning" is a model/reasoning-effort field, NOT tool definitions.
- "agent"            → autonomous multi-step loop: Assistants API (beta.assistants / beta.threads.runs), LangChain AgentExecutor, while-loop that calls LLM + executes tool results iteratively.
- "image_generation" → image generation (openai.images.generate, stability, replicate image models, etc.)
- "tts"              → text-to-speech (openai.audio.speech.create, elevenlabs, etc.)
- "stt"              → speech-to-text / transcription (openai.audio.transcriptions.create, whisper, deepgram, etc.)
- "embeddings"       → vector embeddings (openai.embeddings.create, cohere embed, etc.)

STEP 2 — Apply rules by api_type:

For "chat", "vision", "tool_call", "agent":
- "system_prompt": HARDCODED text from system/developer/instructions role. Use "" if none. Do NOT include user-role content.
- "user_variable": if user-role content is a runtime VARIABLE (identifier, not a string literal), its snake_case name (e.g. userMessage → user_message). Set to null if the content is a hardcoded string literal.
- "user_content": if user-role content is a HARDCODED string literal, that exact text. Set to null if it is a variable.
- Exactly one of user_variable / user_content will be non-null.
- Treat role "developer", "system", "instructions" identically — all are system prompts.
- For responses.create / Responses API: "input" can be a plain string (treat as user content) OR an array of role objects (developer/system = system_prompt, user = user content). Handle both forms.
- For Anthropic: "system" param = system prompt; user role in messages[] = user input.
- For Assistants API / agent loops: "user_variable" is the user message fed into the thread/run.
- For "vision": also set "image_variable" to the snake_case name of the image variable (e.g. imageUrl → image_url). If hardcoded, use "image_url".
- "stream": true if streaming, else false or null.

For "image_generation":
- "user_variable": snake_case name of the prompt variable if dynamic; null if hardcoded.
- "user_content": the hardcoded prompt text if hardcoded; null if dynamic. Default variable name: "prompt".

For "tts":
- "user_variable": snake_case name of the input text variable if dynamic; null if hardcoded.
- "user_content": the hardcoded text-to-speak if hardcoded; null if dynamic. Default variable name: "text".
- "instructions": hardcoded text from the "instructions" parameter (speaking style / tone). Use "" if not present.
- "voice": the voice name/id string from the code (e.g. "coral", "alloy", "nova"). Use null if not specified.

For "stt":
- "user_variable": snake_case name of the audio variable if dynamic; null if hardcoded. Default variable name: "audio".
- "user_content": null (audio is always a runtime input).
- "instructions": hardcoded text from the "prompt" parameter (context hint for transcription). Use "" if not present.

For "embeddings":
- "user_variable": snake_case name of the text variable if dynamic; null if hardcoded.
- "user_content": the hardcoded text if hardcoded; null if dynamic. Default variable name: "text".

STEP 3 — Workflow-ready fields (for OptiML studio + deploy):
- "input_summary": One line describing what the app passes in: dynamic vs hardcoded user text, developer/system instructions, multimodal parts. Example: "developer system string + user question (literal); output read as .output_text".
- "output_kind": Infer how the APPLICATION consumes the model result (read the code AFTER the API call):
  - "text" → plain string usage: .output_text, .choices[0].message.content, string concatenation, logging text, etc.
  - "json" or "json_object" → .json(), JSON.parse, structured output / response_format json, tool arguments.
  - "image" → image bytes, b64, or image URL from an image generation call.
  - "audio" → audio buffer or URL from TTS.
  - "embeddings" → embedding vector/array from embeddings API.
  - "unknown" → cannot tell from snippet.

Return this JSON shape:
{
  "api_type": "chat" | "vision" | "tool_call" | "agent" | "image_generation" | "tts" | "stt" | "embeddings",
  "provider": "openai" | "anthropic" | "gemini" | "mistral" | "cohere" | "groq" | "together" | "deepseek" | "fireworks" | "elevenlabs" | "stability" | "replicate",
  "model": "exact model string from the code (or assistant model for agent)",
  "system_prompt": "hardcoded system/developer role text, or empty string",
  "user_variable": "snake_case variable name if input is dynamic, or null if hardcoded",
  "user_content": "the hardcoded input text if user_variable is null, otherwise null",
  "image_variable": "snake_case variable name for image input (vision only), or null",
  "instructions": "speaking style for tts, transcription context for stt, or null for other types",
  "voice": "voice name/id for tts (e.g. 'coral'), or null",
  "temperature": number or null,
  "max_tokens": number or null,
  "stream": true | false | null,
  "input_summary": "one line — inputs and how they flow",
  "output_kind": "text" | "json" | "json_object" | "image" | "audio" | "embeddings" | "unknown",
  "suggestedName": "short-kebab-case endpoint name based on the apparent purpose"
}

Use null for fields not applicable to the api_type. If this is not an AI API call, return:
{"error": "not an LLM call"}

Code:
"""


def _coerce_misclassified_tool_call(code: str, result: dict) -> dict:
    """
    The classifier sometimes returns api_type tool_call for plain Responses API or chat calls
    (e.g. client.responses.create with reasoning: { effort: "low" }). If the snippet has no
    tools/functions/tool_choice parameters, treat as chat.
    """
    if not isinstance(result, dict) or result.get("error"):
        return result
    if result.get("api_type") != "tool_call":
        return result
    if re.search(r"\btools\s*=", code) or re.search(r"\btools\s*:", code):
        return result
    if re.search(r"\bfunctions\s*=", code) or re.search(r"\btool_choice\s*=", code):
        return result
    out = dict(result)
    out["api_type"] = "chat"
    logger.info("parse_import: coerced api_type tool_call -> chat (no tools/functions in snippet)")
    return out


class ParseImportRequest(BaseModel):
    code: str


async def _run_parse_import(code: str) -> JSONResponse:
    code = code.strip()
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
                    "max_tokens": 1100,
                },
            )
            resp.raise_for_status()

        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        if isinstance(result, dict):
            result = _coerce_misclassified_tool_call(code, result)
            if not result.get("error"):
                result.setdefault("output_kind", "unknown")
        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        logger.warning("parse_import: could not parse model JSON response: %s", e)
        return JSONResponse(status_code=422, content={"error": "could not parse model response as JSON"})
    except httpx.HTTPStatusError as e:
        logger.error("parse_import: OpenAI API error %s: %s", e.response.status_code, e.response.text)
        return JSONResponse(status_code=502, content={"error": "upstream API error"})
    except Exception as e:
        logger.error("parse_import: unexpected error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Parse error: {str(e)}"},
        )


@router.post("/parse-import")
async def parse_import(
    body: ParseImportRequest,
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    return await _run_parse_import(body.code)


@router.post("/public/parse-import-preview")
async def parse_import_preview(body: ParseImportRequest):
    """
    Unauthenticated preview for marketing / landing page.
    Same limits and parsing as /api/parse-import; does not create workflows.
    """
    return await _run_parse_import(body.code)
