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
Extract the LLM call config from the code below. Return ONLY valid JSON — no markdown, no explanation.

{
  "provider": "openai" | "anthropic" | "gemini" | "mistral" | "cohere" | "groq" | "together" | "deepseek" | "fireworks",
  "model": "exact model string from the code",
  "system_prompt": "system prompt text, or empty string if none",
  "user_variable": "variable name used for user input, converted to snake_case",
  "temperature": number or null,
  "max_tokens": number or null,
  "stream": true | false | null,
  "suggestedName": "short-kebab-case endpoint name based on the apparent purpose"
}

Use null for any field you cannot determine. If this is not an LLM API call, return:
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
                    "max_tokens": 500,
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
