"""github_migration.py

GitHub repo scanning and PR-based migration endpoints.

- GET  /github/auth-url        — Returns GitHub OAuth authorization URL
- POST /github/exchange-token  — Exchanges OAuth code for access token
- POST /github/scan-repo       — Scans a repo for AI SDK calls
- POST /github/migrate         — Creates OptiML workflows and opens a PR
- POST /github/migrate-presignup — Opens a migration PR without auth (pre-signup)
"""

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth_dependency import require_org_member, AuthenticatedUser
from parse_import import _run_parse_import
from supabase_client import supabase

logger = logging.getLogger(__name__)
router = APIRouter()

GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_API_BASE = "https://api.github.com"
GITHUB_REDIRECT_URI = "https://optiml.one/api/github/callback"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExchangeTokenRequest(BaseModel):
    code: str


class ScanRepoRequest(BaseModel):
    github_token: str
    repo_full_name: str


class CallToMigrate(BaseModel):
    file_path: str
    code_snippet: str
    line_number: int


class MigrateRequest(BaseModel):
    github_token: str
    repo_full_name: str
    org_id: str
    calls_to_migrate: List[CallToMigrate]


class CallToMigratePresignup(BaseModel):
    file_path: str
    code_snippet: str
    line_number: int
    endpoint_slug: str


class MigratePresignupRequest(BaseModel):
    github_token: str
    repo_full_name: str
    org_name: str
    calls_to_migrate: List[CallToMigratePresignup]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _github_headers(token: str) -> dict:
    # GitHub App user-to-server tokens work with both "Bearer" and "token" prefix.
    # Use "token" which is more universally supported across GitHub API versions.
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# Patterns to detect AI SDK call sites
_CALL_PATTERNS = [
    # OpenAI (Python + JS/TS)
    (re.compile(r"client\.chat\.completions\.create\s*\("), "openai"),
    (re.compile(r"openai\.ChatCompletion\.create\s*\("), "openai"),
    (re.compile(r"openai\.chat\.completions\.create\s*\("), "openai"),
    (re.compile(r"\.chat\.completions\.create\s*\("), "openai"),
    # Anthropic (Python + JS/TS)
    (re.compile(r"anthropic\.messages\.create\s*\("), "anthropic"),
    (re.compile(r"client\.messages\.create\s*\("), "anthropic"),
    (re.compile(r"\.messages\.create\s*\("), "anthropic"),
    # Google Gemini
    (re.compile(r"genai\.GenerativeModel\s*\("), "gemini"),
    (re.compile(r"model\.generate_content\s*\("), "gemini"),
    (re.compile(r"\.generate_content\s*\("), "gemini"),
    (re.compile(r"generativeai\.GenerativeModel\s*\("), "gemini"),
    # Mistral
    (re.compile(r"mistral\.chat\.complete\s*\("), "mistral"),
    (re.compile(r"client\.chat\.complete\s*\("), "mistral"),
    (re.compile(r"MistralClient\s*\("), "mistral"),
    (re.compile(r"\.chat\.complete\s*\("), "mistral"),
    # Cohere
    (re.compile(r"cohere\.chat\s*\("), "cohere"),
    (re.compile(r"co\.chat\s*\("), "cohere"),
    (re.compile(r"client\.chat\s*\("), "cohere"),
    (re.compile(r"CohereClient\s*\("), "cohere"),
    # Azure OpenAI
    (re.compile(r"AzureOpenAI\s*\("), "azure_openai"),
    (re.compile(r"azure_openai\.chat\.completions\.create\s*\("), "azure_openai"),
    # Together AI / Groq / OpenRouter (OpenAI-compatible)
    (re.compile(r"Together\s*\("), "together"),
    (re.compile(r"Groq\s*\("), "groq"),
    # Swift / mobile — OpenAI via URLSession or Swift SDK
    (re.compile(r"api\.openai\.com/v1/chat/completions"), "openai"),
    (re.compile(r"api\.openai\.com/v1/completions"), "openai"),
    (re.compile(r"api\.anthropic\.com/v1/messages"), "anthropic"),
    (re.compile(r"generativelanguage\.googleapis\.com"), "gemini"),
    (re.compile(r"OpenAI\s*\(\s*apiToken"), "openai"),
    (re.compile(r"\.chats\s*\(\s*query"), "openai"),
    # Generic HTTP calls to AI provider endpoints
    (re.compile(r"api\.openai\.com"), "openai"),
    (re.compile(r"api\.anthropic\.com"), "anthropic"),
    (re.compile(r"api\.mistral\.ai"), "mistral"),
    (re.compile(r"api\.cohere\.ai"), "cohere"),
    (re.compile(r"api\.together\.xyz"), "together"),
    (re.compile(r"api\.groq\.com"), "groq"),
]

# Search queries for the GitHub code search API
_SEARCH_QUERIES = [
    "openai",
    "anthropic",
    "from openai import",
    "import Anthropic",
    "@anthropic-ai/sdk",
    "google.generativeai",
    "google-genai",
    "mistralai",
    "MistralClient",
    "cohere",
    "CohereClient",
    "AzureOpenAI",
    "together",
    "groq",
]


def _detect_language(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "jsx": "javascript",
        "mjs": "javascript",
        "swift": "swift",
        "kt": "kotlin",
        "java": "java",
        "go": "go",
        "rb": "ruby",
        "rs": "rust",
        "cs": "csharp",
        "php": "php",
        "dart": "dart",
        "r": "r",
        "scala": "scala",
    }
    return mapping.get(ext, "unknown")


def _extract_call_block(content: str, start: int) -> str:
    """Extract the full call expression starting at *start* (the position of the
    opening paren). Finds the matching closing paren/bracket accounting for
    nesting and string literals."""
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = set(openers.values())
    stack: list[str] = []
    i = start
    in_string: Optional[str] = None

    while i < len(content):
        ch = content[i]

        # Handle string literals (skip contents)
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'", "`"):
            in_string = ch
            i += 1
            continue

        if ch in openers:
            stack.append(openers[ch])
        elif ch in closers:
            if stack and stack[-1] == ch:
                stack.pop()
            if not stack:
                return content[start : i + 1]
        i += 1

    # Fallback: return up to 500 chars
    return content[start : start + 500]


def _extract_model_from_snippet(snippet: str) -> Optional[str]:
    """Try to pull out the model string from a code snippet."""
    m = re.search(r"""model\s*[:=]\s*["']([^"']+)["']""", snippet)
    return m.group(1) if m else None


def _apply_replacement(content: str, old_snippet: str, replacement: str) -> str:
    """Replace *old_snippet* with *replacement* in *content*, preserving the
    indentation of the original code."""
    pos = content.find(old_snippet)
    if pos == -1:
        return content

    # Detect indentation: walk back from pos to find the leading whitespace
    line_start = content.rfind("\n", 0, pos)
    line_start = line_start + 1 if line_start != -1 else 0
    indent = ""
    for ch in content[line_start:pos]:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break

    # Indent each line of the replacement to match
    replacement_lines = replacement.split("\n")
    indented = replacement_lines[0]  # first line inherits position
    for line in replacement_lines[1:]:
        indented += "\n" + (indent + line if line.strip() else line)

    return content[:pos] + indented + content[pos + len(old_snippet):]


def _find_enclosing_statement(content: str, match_start: int) -> tuple[int, int]:
    """Find the full statement/expression that encloses a match position.

    Walks backward to find the statement start, counting any open brackets
    we pass through so the forward scan knows the correct nesting depth.
    Then walks forward to the end of the statement.  Returns (start, end)
    indices into *content*.
    """
    # Walk backward to find statement start, tracking brackets we cross
    # so we know how deep we are when we start scanning forward.
    start = match_start
    enclosing_depth = 0
    while start > 0:
        prev = content[start - 1]
        if prev in ("\n", ";"):
            break
        if prev in (")", "]", "}"):
            enclosing_depth += 1
        elif prev in ("(", "[", "{"):
            if enclosing_depth > 0:
                enclosing_depth -= 1
            else:
                # This open bracket is part of the enclosing statement
                start -= 1
                enclosing_depth += 1  # count it for forward scan
                # Keep going back to find the function call name (e.g. axios.post)
                while start > 0 and content[start - 1] not in ("\n", ";", "{"):
                    start -= 1
                break
        start -= 1

    # Skip leading newlines
    while start < match_start and content[start] == "\n":
        start += 1

    # Count actual bracket depth from start to match_start
    depth = 0
    for ch in content[start:match_start]:
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1

    # Walk forward from match_start to find statement end
    i = match_start
    while i < len(content):
        ch = content[i]
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
            if depth < 0:
                break
        elif ch == ";" and depth <= 0:
            i += 1  # include the semicolon
            break
        elif ch == "\n" and depth <= 0:
            rest = content[i + 1:i + 40].lstrip()
            if not rest or rest[0] not in (".", ",", ")", "]", "}"):
                break
        i += 1

    return start, i


def _fix_downstream_references(content: str) -> str:
    """Fix downstream code that still references OpenAI-style response shapes
    after the call site has been migrated to OptiML.

    Common patterns replaced:
    - ``response.data.choices[0].message.content`` → ``response.data.final_output``
    - ``response.choices[0].message.content`` → ``response.final_output``
    - ``json["choices"][0]["message"]["content"]`` → ``json["final_output"]``
    """
    replacements = [
        # JS axios: response.data.choices[0].message.content
        (re.compile(r'(\w+)\.data\.choices\[0\]\.message\.content'), r'\1.data.final_output'),
        # JS fetch / generic: var.choices[0].message.content
        (re.compile(r'(\w+)\.choices\[0\]\.message\.content'), r'\1.final_output'),
        # Python/Swift dict: ["choices"][0]["message"]["content"]
        (re.compile(r'\["choices"\]\[0\]\["message"\]\["content"\]'), '["final_output"]'),
        # Swift optional chain: .choices.first?.message.content
        (re.compile(r'\.choices\.first\?\.message\.content'), '.final_output'),
    ]
    for pattern, repl in replacements:
        content = pattern.sub(repl, content)
    return content


def _extract_url_var_name(snippet: str) -> Optional[str]:
    """Extract the variable name from a URL-only declaration.
    e.g. ``let baseURL = "https://..."`` → ``baseURL``
    """
    m = re.search(
        r"""(?:let|var|const|final)\s+(\w+)\s*[:=]\s*['"]https?://""",
        snippet,
    )
    return m.group(1) if m else None


def _find_function_using_var(content: str, var_name: str) -> Optional[tuple[int, int, str]]:
    """Find the function body that uses *var_name* to make an HTTP call.

    Returns (start, end, function_text) or None.
    Looks for function/method definitions that reference the variable AND
    contain HTTP call indicators (URLSession, httpBody, httpMethod, fetch,
    requests, etc.).
    """
    # Match function definitions (Swift, JS/TS, Python)
    func_pattern = re.compile(
        r'(?:func\s+\w+|(?:async\s+)?function\s+\w+|def\s+\w+|'
        r'(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()'
        r'[^{]*\{'
    )
    http_indicators = [
        "httpMethod", "httpBody", "URLSession", ".dataTask",
        "fetch(", ".post(", ".get(", "requests.", "httpx.",
        "axios.", "HttpClient",
    ]

    for m in func_pattern.finditer(content):
        # Find the matching closing brace for this function
        brace_start = content.index("{", m.start())
        depth = 0
        i = brace_start
        while i < len(content):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        func_body = content[m.start():i + 1]

        # Check if this function uses our URL variable AND makes HTTP calls
        if var_name in func_body and any(ind in func_body for ind in http_indicators):
            line_number = content[:m.start()].count("\n") + 1
            return (m.start(), i + 1, func_body)

    return None


def _find_calls_in_content(file_path: str, content: str) -> list[dict]:
    """Scan file content for AI SDK call sites and return structured results."""
    results: list[dict] = []
    seen_lines: set[int] = set()  # deduplicate by line number
    language = _detect_language(file_path)

    # URL-based patterns need special handling — flag them
    # Use escaped dots to match the regex pattern strings
    _URL_PATTERNS = {
        r"api\.openai\.com", r"api\.anthropic\.com", r"api\.mistral\.ai",
        r"api\.cohere\.ai", r"api\.together\.xyz", r"api\.groq\.com",
        r"generativelanguage\.googleapis\.com",
    }

    url_only_vars: list[str] = []  # track URL variable names for second pass

    for pattern, provider in _CALL_PATTERNS:
        is_url_pattern = any(url in pattern.pattern for url in _URL_PATTERNS)

        for match in pattern.finditer(content):
            line_number = content[: match.start()].count("\n") + 1
            if line_number in seen_lines:
                continue
            seen_lines.add(line_number)

            if is_url_pattern:
                stmt_start, stmt_end = _find_enclosing_statement(content, match.start())
                full_snippet = content[stmt_start:stmt_end]

                # Track URL-only declarations for second pass
                if _is_url_only_declaration(full_snippet):
                    var_name = _extract_url_var_name(full_snippet)
                    if var_name:
                        url_only_vars.append(var_name)
            else:
                paren_pos = content.find("(", match.start())
                if paren_pos == -1:
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 200)
                    full_snippet = content[start:end]
                else:
                    snippet = _extract_call_block(content, paren_pos)
                    full_snippet = content[match.start() : match.start() + len(match.group()) + len(snippet)]

            model = _extract_model_from_snippet(full_snippet)

            results.append({
                "file_path": file_path,
                "line_number": line_number,
                "language": language,
                "code_snippet": full_snippet,
                "detected_provider": provider,
                "detected_model": model,
            })

    # ── Second pass: find HTTP functions that use detected URL variables ──
    # When we detect a URL-only declaration like `let baseURL = "..."`, the
    # actual HTTP call (sendRequest, etc.) uses baseURL by name and won't
    # match URL patterns.  Find those functions and add them as call sites
    # so the migration replaces the full HTTP function, not just the URL.
    for var_name in url_only_vars:
        func_result = _find_function_using_var(content, var_name)
        if func_result:
            func_start, func_end, func_text = func_result
            func_line = content[:func_start].count("\n") + 1
            if func_line not in seen_lines:
                seen_lines.add(func_line)
                model = _extract_model_from_snippet(func_text)
                results.append({
                    "file_path": file_path,
                    "line_number": func_line,
                    "language": language,
                    "code_snippet": func_text,
                    "detected_provider": results[0]["detected_provider"] if results else "openai",
                    "detected_model": model or (results[0]["detected_model"] if results else None),
                    "_is_http_function": True,  # flag for replacement generation
                })

    return results


def _is_url_only_declaration(snippet: str) -> bool:
    """Return True if the snippet is a simple variable assignment containing a
    URL string (e.g. ``const url = "https://api.openai.com/..."``), as opposed
    to a full HTTP call (``axios.post(...)`` or ``fetch(...)``).

    For URL-only declarations we just swap the URL value; for full HTTP calls
    we replace the entire call block.
    """
    http_call_indicators = [
        "fetch(", "axios.", "requests.", "httpx.", "http.",
        "URLSession", "HttpClient", "urllib",
        ".post(", ".get(", ".put(", ".patch(", ".delete(",
    ]
    for indicator in http_call_indicators:
        if indicator in snippet:
            return False
    return True


def _replace_url_in_snippet(snippet: str, new_url: str, workflow_id: str) -> str:
    """For URL-only declarations, swap the provider URL with the OptiML
    OpenAI-compatible endpoint and swap the API key reference."""
    url_pattern = re.compile(
        r"""(['"])https?://(?:api\.openai\.com|api\.anthropic\.com|api\.mistral\.ai|"""
        r"""api\.cohere\.ai|api\.together\.xyz|api\.groq\.com|"""
        r"""generativelanguage\.googleapis\.com)[^'"]*\1"""
    )
    m = url_pattern.search(snippet)
    if m:
        quote = m.group(1)
        replaced = snippet[:m.start()] + f'{quote}{new_url}{quote}' + snippet[m.end():]
        comment_char = "//" if any(kw in snippet for kw in ("let ", "var ", "const ")) else "#"
        return f'{comment_char} Migrated to OptiML — workflow {workflow_id}\n{replaced}'
    return f'// Migrated to OptiML — workflow {workflow_id}\n{snippet}'


def _generate_http_function_replacement(
    original_snippet: str, workflow_id: str, url: str, language: str,
) -> str:
    """Generate a replacement for an entire HTTP function (like sendRequest).

    Extracts the function signature (name, params, return type) from the
    original and generates a clean OptiML version that:
    - Sends ``input_text`` to the OptiML endpoint
    - Reads ``final_output`` from the response
    - Preserves the function's public interface (name, params, completion handler)
    """
    if language == "swift":
        # Extract function name and signature
        sig_match = re.search(
            r'func\s+(\w+)\s*\(([^)]*)\)\s*(.*?)\{',
            original_snippet,
            re.DOTALL,
        )
        if sig_match:
            func_name = sig_match.group(1)
            params = sig_match.group(2).strip()
            rest = sig_match.group(3).strip()
            # Detect if it uses completion handler or async
            is_async = "async" in rest
            has_completion = "completion" in params or "@escaping" in params

            if has_completion:
                return (
                    f'// Migrated to OptiML — workflow {workflow_id}\n'
                    f'// Model, prompt template & parameters are managed in OptiML\n'
                    f'// Edit at: https://optiml.one/studio\n'
                    f'func {func_name}({params}) {rest}{{\n'
                    f'    guard let url = URL(string: "{url}") else {{\n'
                    f'        completion(.failure(OpenAIError.invalidURL))\n'
                    f'        return\n'
                    f'    }}\n'
                    f'\n'
                    f'    var request = URLRequest(url: url)\n'
                    f'    request.httpMethod = "POST"\n'
                    f'    request.setValue("application/json", forHTTPHeaderField: "Content-Type")\n'
                    f'    request.setValue("Bearer \\(ProcessInfo.processInfo.environment["OPTIML_API_KEY"] ?? "")", forHTTPHeaderField: "Authorization")\n'
                    f'\n'
                    f'    do {{\n'
                    f'        request.httpBody = try JSONSerialization.data(withJSONObject: ["input_text": prompt])\n'
                    f'    }} catch {{\n'
                    f'        completion(.failure(error))\n'
                    f'        return\n'
                    f'    }}\n'
                    f'\n'
                    f'    URLSession.shared.dataTask(with: request) {{ data, response, error in\n'
                    f'        if let error = error {{\n'
                    f'            completion(.failure(OpenAIError.networkError(error)))\n'
                    f'            return\n'
                    f'        }}\n'
                    f'        guard let data = data else {{\n'
                    f'            completion(.failure(OpenAIError.invalidResponse))\n'
                    f'            return\n'
                    f'        }}\n'
                    f'        do {{\n'
                    f'            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],\n'
                    f'               let output = json["final_output"] as? String {{\n'
                    f'                completion(.success(output))\n'
                    f'            }} else {{\n'
                    f'                completion(.failure(OpenAIError.invalidResponse))\n'
                    f'            }}\n'
                    f'        }} catch {{\n'
                    f'            completion(.failure(OpenAIError.decodingError(error)))\n'
                    f'        }}\n'
                    f'    }}.resume()\n'
                    f'}}'
                )

        # Fallback: generic Swift function replacement
        return (
            f'// Migrated to OptiML — workflow {workflow_id}\n'
            f'// Model, prompt template & parameters are managed in OptiML\n'
            f'// Edit at: https://optiml.one/studio\n'
            f'// TODO: Update this function to call OptiML at:\n'
            f'//   {url}\n'
            f'// Send: {{"input_text": "<user input>"}}\n'
            f'// Read: response["final_output"]\n'
            + original_snippet
        )

    elif language == "python":
        # Extract function name
        func_match = re.search(r'(?:async\s+)?def\s+(\w+)', original_snippet)
        func_name = func_match.group(1) if func_match else "send_request"
        is_async = "async " in original_snippet

        if is_async:
            return (
                f'# Migrated to OptiML — workflow {workflow_id}\n'
                f'# Model, prompt template & parameters are managed in OptiML\n'
                f'# Edit at: https://optiml.one/studio\n'
                f'async def {func_name}(prompt: str) -> str:\n'
                f'    async with httpx.AsyncClient(timeout=30.0) as client:\n'
                f'        response = await client.post(\n'
                f'            "{url}",\n'
                f'            json={{"input_text": prompt}},\n'
                f'            headers={{"Authorization": f"Bearer {{os.environ[\'OPTIML_API_KEY\']}}"}},\n'
                f'        )\n'
                f'        return response.json()["final_output"]'
            )
        else:
            return (
                f'# Migrated to OptiML — workflow {workflow_id}\n'
                f'# Model, prompt template & parameters are managed in OptiML\n'
                f'# Edit at: https://optiml.one/studio\n'
                f'def {func_name}(prompt: str) -> str:\n'
                f'    response = requests.post(\n'
                f'        "{url}",\n'
                f'        json={{"input_text": prompt}},\n'
                f'        headers={{"Authorization": f"Bearer {{os.environ[\'OPTIML_API_KEY\']}}"}},\n'
                f'        timeout=30,\n'
                f'    )\n'
                f'    return response.json()["final_output"]'
            )

    else:
        # JS/TS function replacement
        func_match = re.search(r'(?:async\s+)?function\s+(\w+)', original_snippet)
        func_name = func_match.group(1) if func_match else "sendRequest"

        return (
            f'// Migrated to OptiML — workflow {workflow_id}\n'
            f'// Model, prompt template & parameters are managed in OptiML\n'
            f'// Edit at: https://optiml.one/studio\n'
            f'async function {func_name}(prompt) {{\n'
            f'  const response = await fetch("{url}", {{\n'
            f'    method: "POST",\n'
            f'    headers: {{\n'
            f'      "Content-Type": "application/json",\n'
            f'      Authorization: `Bearer ${{process.env.OPTIML_API_KEY}}`,\n'
            f'    }},\n'
            f'    body: JSON.stringify({{ input_text: prompt }}),\n'
            f'  }});\n'
            f'  const {{ final_output }} = await response.json();\n'
            f'  return final_output;\n'
            f'}}'
        )


def _generate_replacement_code(
    original_snippet: str, workflow_id: str, deployment_id: str, language: str,
    endpoint_url: Optional[str] = None,
    is_http_function: bool = False,
) -> str:
    """Generate code that replaces the original AI provider call with a
    proper OptiML integration.

    The generated code sends just the dynamic user input to the OptiML
    managed endpoint.  The model, system prompt, temperature, and all
    other AI parameters are configured in the OptiML workflow — NOT in
    the code.  This is the core value prop: prompts and model config
    become manageable, versionable, and A/B-testable without code
    deploys.
    """
    url = endpoint_url or f"https://api.optiml.one/api/public/execute/{deployment_id}"

    # ── HTTP function replacement (e.g. sendRequest in Swift) ──────────
    # When we detect a full function that uses a URL variable to make
    # HTTP calls, replace the entire function with a clean OptiML version.
    # Detect by checking if snippet is a function definition or via flag.
    snippet_stripped = original_snippet.strip()
    _is_func = (
        is_http_function
        or snippet_stripped.startswith("func ")
        or snippet_stripped.startswith("function ")
        or snippet_stripped.startswith("async function ")
        or snippet_stripped.startswith("def ")
        or snippet_stripped.startswith("async def ")
    )
    if _is_func:
        return _generate_http_function_replacement(
            original_snippet, workflow_id, url, language,
        )

    # ── URL-only declarations ──────────────────────────────────────────
    if _is_url_only_declaration(original_snippet):
        return _replace_url_in_snippet(original_snippet, url, workflow_id)

    # ── Full HTTP call replacements ────────────────────────────────────
    # Replace with a clean OptiML call.  The AI config (model, prompt,
    # temperature) lives in the workflow — code just sends user input.

    if language == "python":
        if "requests." in original_snippet:
            return (
                f'# Migrated to OptiML — workflow {workflow_id}\n'
                f'# Model, prompt template & parameters are managed in OptiML\n'
                f'# Edit at: https://optiml.one/studio\n'
                f'response = requests.post(\n'
                f'    "{url}",\n'
                f'    json={{"input_text": prompt}},\n'
                f'    headers={{\n'
                f'        "Authorization": f"Bearer {{os.environ[\'OPTIML_API_KEY\']}}",\n'
                f'        "Content-Type": "application/json",\n'
                f'    }},\n'
                f'    timeout=30,\n'
                f')\n'
                f'result = response.json()["final_output"]'
            )
        else:
            return (
                f'# Migrated to OptiML — workflow {workflow_id}\n'
                f'# Model, prompt template & parameters are managed in OptiML\n'
                f'# Edit at: https://optiml.one/studio\n'
                f'response = await client.post(\n'
                f'    "{url}",\n'
                f'    json={{"input_text": prompt}},\n'
                f'    headers={{\n'
                f'        "Authorization": f"Bearer {{os.environ[\'OPTIML_API_KEY\']}}",\n'
                f'        "Content-Type": "application/json",\n'
                f'    }},\n'
                f')\n'
                f'result = response.json()["final_output"]'
            )

    elif language == "swift":
        return (
            f'// Migrated to OptiML — workflow {workflow_id}\n'
            f'// Model, prompt template & parameters are managed in OptiML\n'
            f'// Edit at: https://optiml.one/studio\n'
            f'let optimlURL = URL(string: "{url}")!\n'
            f'var optimlRequest = URLRequest(url: optimlURL)\n'
            f'optimlRequest.httpMethod = "POST"\n'
            f'optimlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")\n'
            f'optimlRequest.setValue("Bearer \\(ProcessInfo.processInfo.environment["OPTIML_API_KEY"] ?? "")", forHTTPHeaderField: "Authorization")\n'
            f'optimlRequest.httpBody = try JSONSerialization.data(withJSONObject: ["input_text": prompt])\n'
            f'let (optimlData, _) = try await URLSession.shared.data(for: optimlRequest)\n'
            f'let optimlJSON = try JSONSerialization.jsonObject(with: optimlData) as? [String: Any]\n'
            f'let finalOutput = optimlJSON?["final_output"] as? String ?? ""'
        )

    else:
        if "axios" in original_snippet:
            return (
                f'// Migrated to OptiML — workflow {workflow_id}\n'
                f'// Model, prompt template & parameters are managed in OptiML\n'
                f'// Edit at: https://optiml.one/studio\n'
                f'const response = await axios.post(\n'
                f'  "{url}",\n'
                f'  {{ input_text: prompt }},\n'
                f'  {{\n'
                f'    headers: {{\n'
                f'      Authorization: `Bearer ${{process.env.OPTIML_API_KEY}}`,\n'
                f'      "Content-Type": "application/json",\n'
                f'    }},\n'
                f'  }}\n'
                f');\n'
                f'const result = response.data.final_output;'
            )
        else:
            return (
                f'// Migrated to OptiML — workflow {workflow_id}\n'
                f'// Model, prompt template & parameters are managed in OptiML\n'
                f'// Edit at: https://optiml.one/studio\n'
                f'const response = await fetch("{url}", {{\n'
                f'  method: "POST",\n'
                f'  headers: {{\n'
                f'    "Content-Type": "application/json",\n'
                f'    Authorization: `Bearer ${{process.env.OPTIML_API_KEY}}`,\n'
                f'  }},\n'
                f'  body: JSON.stringify({{ input_text: prompt }}),\n'
                f'}});\n'
                f'const {{ final_output }} = await response.json();'
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/github/auth-url")
async def github_auth_url():
    """Return a GitHub OAuth authorization URL."""
    if not GITHUB_CLIENT_ID:
        return JSONResponse(
            status_code=503,
            content={"error": "GitHub OAuth is not configured (GITHUB_CLIENT_ID missing)."},
        )

    state = str(uuid.uuid4())
    url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={GITHUB_REDIRECT_URI}"
        f"&scope=repo"
        f"&state={state}"
    )
    return {"auth_url": url, "state": state}


@router.post("/github/exchange-token")
async def github_exchange_token(body: ExchangeTokenRequest):
    """Exchange a GitHub OAuth code for an access token."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        return JSONResponse(
            status_code=503,
            content={"error": "GitHub OAuth is not configured."},
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                json={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": body.code,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Log full response for debugging (redact token)
        log_data = {k: (v[:8] + "..." if k == "access_token" and v else v) for k, v in data.items()}
        logger.info("GitHub token exchange response keys: %s, data: %s", list(data.keys()), log_data)

        if "error" in data:
            logger.warning("GitHub token exchange error: %s", data.get("error_description", data["error"]))
            return JSONResponse(
                status_code=400,
                content={"error": data.get("error_description", data["error"])},
            )

        access_token = data.get("access_token", "")
        if not access_token:
            logger.error("No access_token in GitHub response: %s", list(data.keys()))
            return JSONResponse(
                status_code=400,
                content={"error": "GitHub did not return an access token. Keys received: " + ", ".join(data.keys())},
            )

        # Fetch the authenticated user's username
        async with httpx.AsyncClient(timeout=10.0) as client:
            user_resp = await client.get(
                f"{GITHUB_API_BASE}/user",
                headers=_github_headers(access_token),
            )
            user_resp.raise_for_status()
            github_username = user_resp.json().get("login", "")

        return {"access_token": access_token, "github_username": github_username}

    except httpx.HTTPStatusError as e:
        logger.error("GitHub API error during token exchange: %s %s", e.response.status_code, e.response.text)
        return JSONResponse(status_code=502, content={"error": "GitHub API error during token exchange."})
    except Exception as e:
        logger.error("Unexpected error during GitHub token exchange: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Token exchange failed: {str(e)}"})


# File extensions we care about when scanning repos
_SCANNABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".mts",
    ".swift", ".kt", ".java", ".go", ".rb", ".rs", ".cs",
    ".php", ".dart", ".r", ".R", ".scala",
}

# Import-like strings that hint a file might contain AI SDK calls
_IMPORT_HINTS = [
    "openai", "anthropic", "genai", "generativeai", "mistral",
    "cohere", "azure", "together", "groq",
    "api.openai.com", "api.anthropic.com", "api.mistral.ai",
    "api.cohere.ai", "api.together.xyz", "api.groq.com",
    "generativelanguage.googleapis.com",
]


@router.post("/github/scan-repo")
async def github_scan_repo(body: ScanRepoRequest):
    """Scan a GitHub repo for AI SDK call sites.

    Strategy:
    1. Try the GitHub Code Search API first (fast but unreliable — repos may
       not be indexed).
    2. If code search returns nothing, fall back to the Git Trees API to list
       all files, filter to scannable extensions, fetch each file, and scan
       for AI call patterns directly.
    """
    repo = body.repo_full_name
    token = body.github_token
    headers = _github_headers(token)

    all_file_paths: set[str] = set()

    try:
        # ── Strategy 1: Code Search API (skip if auth fails) ─────────
        code_search_failed = False
        async with httpx.AsyncClient(timeout=10.0) as client:
            for query in _SEARCH_QUERIES:
                search_url = f"{GITHUB_API_BASE}/search/code"
                params = {"q": f"{query} repo:{repo}"}
                try:
                    resp = await client.get(search_url, headers=headers, params=params)
                except httpx.TimeoutException:
                    logger.warning("Code search timed out for query '%s' in %s", query, repo)
                    continue

                if resp.status_code == 401:
                    # Bad token — skip remaining queries, go to tree fallback
                    logger.warning("GitHub token invalid for code search, skipping to tree scan")
                    code_search_failed = True
                    break
                if resp.status_code == 403:
                    logger.warning("GitHub code search rate limited for repo %s", repo)
                    code_search_failed = True
                    break
                if resp.status_code == 422:
                    # Repo not indexed — skip to tree fallback
                    logger.info("Repo %s not indexed for code search (422)", repo)
                    code_search_failed = True
                    break
                if resp.status_code != 200:
                    logger.warning(
                        "GitHub code search returned %s for query '%s' in %s",
                        resp.status_code, query, repo,
                    )
                    continue

                items = resp.json().get("items", [])
                if items:
                    logger.info("Code search found %d files for query '%s' in %s", len(items), query, repo)
                for item in items:
                    all_file_paths.add(item["path"])

        # ── Strategy 2: Git Trees API fallback ───────────────────────
        if not all_file_paths:
            logger.info("Code search returned 0 results for %s, falling back to tree scan", repo)
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    # Verify token works first
                    user_resp = await client.get(f"{GITHUB_API_BASE}/user", headers=headers)
                    logger.info("Token check for %s: user endpoint returned %s", repo, user_resp.status_code)
                    if user_resp.status_code == 401:
                        return JSONResponse(
                            status_code=401,
                            content={"error": "GitHub token is invalid or expired. Please re-authenticate with GitHub."},
                        )

                    # Get default branch
                    repo_resp = await client.get(f"{GITHUB_API_BASE}/repos/{repo}", headers=headers)
                    logger.info("Repo info for %s: status=%s", repo, repo_resp.status_code)
                    if repo_resp.status_code != 200:
                        resp_text = repo_resp.text[:200]
                        logger.warning("Could not fetch repo info for %s: %s — %s", repo, repo_resp.status_code, resp_text)
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Could not access repo: HTTP {repo_resp.status_code}. Check the repo URL and your GitHub permissions."},
                        )
                    default_branch = repo_resp.json().get("default_branch", "main")

                    # Get recursive tree
                    tree_resp = await client.get(
                        f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{default_branch}",
                        headers=headers,
                        params={"recursive": "1"},
                    )
                    if tree_resp.status_code != 200:
                        logger.warning("Could not fetch tree for %s: %s", repo, tree_resp.status_code)
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Could not read repo file tree: HTTP {tree_resp.status_code}"},
                        )
                    tree_data = tree_resp.json()
            except httpx.TimeoutException:
                logger.warning("Tree scan timed out for %s", repo)
                return JSONResponse(
                    status_code=504,
                    content={"error": "Scan timed out. The repository may be too large."},
                )

            tree_items = tree_data.get("tree", [])
            blob_count = sum(1 for it in tree_items if it.get("type") == "blob")
            logger.info("Tree for %s: %d total items, %d blobs", repo, len(tree_items), blob_count)
            for item in tree_items:
                if item.get("type") != "blob":
                    continue
                path = item.get("path", "")
                # Skip vendor / dependency / build directories
                if any(seg in path.split("/") for seg in (
                    "node_modules", ".next", "dist", "build", "__pycache__",
                    ".git", "vendor", "venv", ".venv", "env",
                )):
                    continue
                ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
                if ext in _SCANNABLE_EXTENSIONS:
                    all_file_paths.add(path)
            logger.info("Found %d scannable files in %s", len(all_file_paths), repo)

        if not all_file_paths:
            return {
                "repo": repo,
                "found_calls": [],
                "total_files_scanned": 0,
                "total_calls_found": 0,
            }

        # Cap at 150 files to avoid timeout on large repos
        file_list = list(all_file_paths)[:150]
        logger.info("Scanning %d files in %s (of %d found)", len(file_list), repo, len(all_file_paths))

        # ── Fetch files concurrently (batches of 10) ─────────────────

        found_calls: list[dict] = []
        files_scanned = 0

        async def _fetch_and_scan(client: httpx.AsyncClient, file_path: str) -> list[dict]:
            content_url = f"{GITHUB_API_BASE}/repos/{repo}/contents/{file_path}"
            try:
                resp = await client.get(content_url, headers={
                    **headers,
                    "Accept": "application/vnd.github.raw+json",
                })
                if resp.status_code != 200:
                    return []
                content = resp.text
                # Quick pre-filter: skip files that don't mention any AI SDK
                content_lower = content.lower()
                if not any(hint in content_lower for hint in _IMPORT_HINTS):
                    return []
                return _find_calls_in_content(file_path, content)
            except Exception:
                return []

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Process in batches of 10 concurrent requests
            batch_size = 10
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i : i + batch_size]
                results = await asyncio.gather(
                    *[_fetch_and_scan(client, fp) for fp in batch]
                )
                for result in results:
                    files_scanned += 1
                    found_calls.extend(result)

        return {
            "repo": repo,
            "found_calls": found_calls,
            "total_files_scanned": files_scanned,
            "total_calls_found": len(found_calls),
        }

    except httpx.HTTPStatusError as e:
        logger.error("GitHub API error scanning repo %s: %s", repo, e.response.text)
        return JSONResponse(status_code=502, content={"error": f"GitHub API error: {e.response.status_code}"})
    except Exception as e:
        logger.error("Unexpected error scanning repo %s: %s", repo, e)
        return JSONResponse(status_code=500, content={"error": f"Scan failed: {str(e)}"})


@router.post("/github/migrate")
async def github_migrate(
    body: MigrateRequest,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create OptiML workflows for each AI call and open a PR with migrated code."""
    repo = body.repo_full_name
    token = body.github_token
    org_id = body.org_id
    headers = _github_headers(token)
    timestamp = int(datetime.now(timezone.utc).timestamp())
    branch_name = f"optiml-migration-{timestamp}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # ------------------------------------------------------------------
            # 1. Get the default branch SHA to base our new branch on
            # ------------------------------------------------------------------
            repo_resp = await client.get(f"{GITHUB_API_BASE}/repos/{repo}", headers=headers)
            repo_resp.raise_for_status()
            repo_data = repo_resp.json()
            default_branch = repo_data.get("default_branch", "main")

            ref_resp = await client.get(
                f"{GITHUB_API_BASE}/repos/{repo}/git/ref/heads/{default_branch}",
                headers=headers,
            )
            ref_resp.raise_for_status()
            base_sha = ref_resp.json()["object"]["sha"]

            # ------------------------------------------------------------------
            # 2. Create the new branch
            # ------------------------------------------------------------------
            create_ref_resp = await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/git/refs",
                headers=headers,
                json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )
            create_ref_resp.raise_for_status()

            # ------------------------------------------------------------------
            # 3. Process each call: parse → create workflow → create deployment
            #    → generate replacement code
            # ------------------------------------------------------------------
            workflows_created = 0
            # Group changes by file path so we update each file once
            file_changes: dict[str, list[tuple[CallToMigrate, str]]] = {}

            for call in body.calls_to_migrate:
                # Detect if this is an HTTP function snippet (sendRequest etc.)
                # These don't need their own workflow — they share the URL
                # declaration's workflow.  Just generate replacement code.
                snippet_stripped = call.code_snippet.strip()
                is_function_snippet = (
                    snippet_stripped.startswith("func ")
                    or snippet_stripped.startswith("function ")
                    or snippet_stripped.startswith("async function ")
                    or snippet_stripped.startswith("def ")
                    or snippet_stripped.startswith("async def ")
                )

                if is_function_snippet:
                    # Generate replacement using the last created workflow
                    language = _detect_language(call.file_path)
                    last_wf_id = workflow_id if workflows_created > 0 else "migrated"
                    last_dep_id = deployment_id if workflows_created > 0 else "migrated"
                    replacement = _generate_replacement_code(
                        call.code_snippet, last_wf_id, last_dep_id, language,
                        is_http_function=True,
                    )
                    file_changes.setdefault(call.file_path, []).append((call, replacement))
                    continue

                # 3a. Parse the snippet using the existing parse logic
                try:
                    parse_resp = await _run_parse_import(call.code_snippet)
                    parse_body = json.loads(parse_resp.body.decode())
                except Exception as e:
                    logger.warning("Parse failed for %s:%s — %s", call.file_path, call.line_number, e)
                    parse_body = {"error": str(e)}

                if "error" in parse_body:
                    logger.warning(
                        "Skipping call at %s:%s — parse error: %s",
                        call.file_path, call.line_number, parse_body["error"],
                    )
                    continue

                suggested_name = parse_body.get("suggestedName", f"migrated-call-{workflows_created + 1}")
                api_type = parse_body.get("api_type", "chat")

                # 3b. Create a workflow in Supabase
                workflow_data = {
                    "org_id": org_id,
                    "name": suggested_name,
                    "description": f"Auto-migrated from {repo} — {call.file_path}:{call.line_number}",
                    "graph": {
                        "nodes": [
                            {"id": "input", "type": "input"},
                            {
                                "id": "ai-step",
                                "type": "ai",
                                "config": {
                                    "provider": parse_body.get("provider", "openai"),
                                    "model": parse_body.get("model", "gpt-4o"),
                                    "system_prompt": parse_body.get("system_prompt", ""),
                                    "temperature": parse_body.get("temperature"),
                                    "max_tokens": parse_body.get("max_tokens"),
                                    "api_type": api_type,
                                },
                            },
                        ],
                        "edges": [{"from": "input", "to": "ai-step"}],
                    },
                    "status": "active",
                    "created_by": auth_user.user_id,
                }

                wf_result = supabase.table("workflows").insert(workflow_data).execute()
                if not wf_result.data:
                    logger.error("Failed to create workflow for %s:%s", call.file_path, call.line_number)
                    continue

                workflow_id = wf_result.data[0]["id"]
                workflows_created += 1

                # 3c. Create a deployment for the workflow
                deployment_data = {
                    "workflow_id": workflow_id,
                    "org_id": org_id,
                    "status": "active",
                    "created_by": auth_user.user_id,
                }

                dep_result = supabase.table("deployments").insert(deployment_data).execute()
                deployment_id = dep_result.data[0]["id"] if dep_result.data else workflow_id

                # 3d. Generate replacement code
                language = _detect_language(call.file_path)
                replacement = _generate_replacement_code(
                    call.code_snippet, workflow_id, deployment_id, language,
                )

                file_changes.setdefault(call.file_path, []).append((call, replacement))

            # ------------------------------------------------------------------
            # 4. For each modified file, fetch current content and apply changes,
            #    then update on the new branch via the GitHub Contents API.
            # ------------------------------------------------------------------
            for file_path, changes in file_changes.items():
                # Fetch current file content + SHA
                file_resp = await client.get(
                    f"{GITHUB_API_BASE}/repos/{repo}/contents/{file_path}",
                    headers=headers,
                    params={"ref": branch_name},
                )
                file_resp.raise_for_status()
                file_data = file_resp.json()
                file_sha = file_data["sha"]

                # Decode content

                raw_content = base64.b64decode(file_data["content"]).decode("utf-8")

                # Apply replacements (process in reverse line order to preserve positions)
                modified_content = raw_content
                for call, replacement in sorted(changes, key=lambda c: c[0].line_number, reverse=True):
                    modified_content = _apply_replacement(modified_content, call.code_snippet, replacement)

                # Fix downstream references (e.g. choices[0].message.content → final_output)
                modified_content = _fix_downstream_references(modified_content)

                # Push updated file
                encoded = base64.b64encode(modified_content.encode("utf-8")).decode("utf-8")
                update_resp = await client.put(
                    f"{GITHUB_API_BASE}/repos/{repo}/contents/{file_path}",
                    headers=headers,
                    json={
                        "message": f"chore(optiml): migrate AI call in {file_path}",
                        "content": encoded,
                        "sha": file_sha,
                        "branch": branch_name,
                    },
                )
                update_resp.raise_for_status()

            # ------------------------------------------------------------------
            # 5. Open a pull request
            # ------------------------------------------------------------------
            owner = repo.split("/")[0]
            pr_body = (
                "## Migrate AI calls to OptiML managed endpoints\n\n"
                f"This PR was auto-generated by [OptiML](https://optiml.one) to migrate "
                f"**{workflows_created}** AI call(s) to managed OptiML workflows.\n\n"
                "### What changed\n"
                "Each direct AI provider call (OpenAI, Anthropic, etc.) has been replaced with a call "
                "to your OptiML deployment endpoint. This gives you:\n\n"
                "- **Version control** — roll back to any previous prompt or model config\n"
                "- **A/B testing** — experiment with models and prompts without code changes\n"
                "- **Knowledge base** — attach context assets to enrich responses\n"
                "- **Cost tracking** — per-call cost and latency analytics\n"
                "- **Human-in-the-loop** — optional approval gates for sensitive calls\n\n"
                "### Next steps\n"
                f"1. Set the `OPTIML_API_KEY` environment variable in your deployment\n"
                f"2. Visit [OptiML Dashboard](https://optiml.one/dashboard) to configure your workflows\n"
                "3. Review the changes in this PR and merge when ready\n"
            )

            pr_resp = await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/pulls",
                headers=headers,
                json={
                    "title": "Migrate AI calls to OptiML managed endpoints",
                    "body": pr_body,
                    "head": branch_name,
                    "base": default_branch,
                },
            )
            pr_resp.raise_for_status()
            pr_url = pr_resp.json().get("html_url", "")

        return {"pr_url": pr_url, "workflows_created": workflows_created}

    except httpx.HTTPStatusError as e:
        logger.error("GitHub API error during migration for %s: %s %s", repo, e.response.status_code, e.response.text)
        return JSONResponse(
            status_code=502,
            content={"error": f"GitHub API error: {e.response.status_code} — {e.response.text[:200]}"},
        )
    except Exception as e:
        logger.error("Unexpected error during migration for %s: %s", repo, e)
        return JSONResponse(status_code=500, content={"error": f"Migration failed: {str(e)}"})


@router.post("/github/migrate-presignup")
async def github_migrate_presignup(body: MigratePresignupRequest):
    """Create a migration PR without requiring authentication or creating
    Supabase workflows/deployments.  Uses placeholder endpoint URLs based on
    org_name and endpoint_slug so the user can sign up later to activate them."""
    repo = body.repo_full_name
    token = body.github_token
    org_name = body.org_name
    headers = _github_headers(token)
    timestamp = int(datetime.now(timezone.utc).timestamp())
    branch_name = f"optiml-migration-{timestamp}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # ------------------------------------------------------------------
            # 1. Get the default branch SHA to base our new branch on
            # ------------------------------------------------------------------
            repo_resp = await client.get(f"{GITHUB_API_BASE}/repos/{repo}", headers=headers)
            repo_resp.raise_for_status()
            repo_data = repo_resp.json()
            default_branch = repo_data.get("default_branch", "main")

            ref_resp = await client.get(
                f"{GITHUB_API_BASE}/repos/{repo}/git/ref/heads/{default_branch}",
                headers=headers,
            )
            ref_resp.raise_for_status()
            base_sha = ref_resp.json()["object"]["sha"]

            # ------------------------------------------------------------------
            # 2. Create the new branch
            # ------------------------------------------------------------------
            create_ref_resp = await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/git/refs",
                headers=headers,
                json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )
            create_ref_resp.raise_for_status()

            # ------------------------------------------------------------------
            # 3. Process each call: generate replacement code using slug-based
            #    endpoint URLs (no Supabase interaction)
            # ------------------------------------------------------------------
            calls_processed = 0
            file_changes: dict[str, list[tuple[CallToMigratePresignup, str]]] = {}

            for call in body.calls_to_migrate:
                endpoint_url = f"https://api.optiml.one/api/public/{org_name}/{call.endpoint_slug}"
                language = _detect_language(call.file_path)
                replacement = _generate_replacement_code(
                    call.code_snippet,
                    workflow_id=call.endpoint_slug,
                    deployment_id=call.endpoint_slug,
                    language=language,
                    endpoint_url=endpoint_url,
                )
                file_changes.setdefault(call.file_path, []).append((call, replacement))
                calls_processed += 1

            # ------------------------------------------------------------------
            # 4. For each modified file, fetch current content, apply changes,
            #    and push to the new branch.
            # ------------------------------------------------------------------
            import base64

            for file_path, changes in file_changes.items():
                file_resp = await client.get(
                    f"{GITHUB_API_BASE}/repos/{repo}/contents/{file_path}",
                    headers=headers,
                    params={"ref": branch_name},
                )
                file_resp.raise_for_status()
                file_data = file_resp.json()
                file_sha = file_data["sha"]

                raw_content = base64.b64decode(file_data["content"]).decode("utf-8")

                modified_content = raw_content
                for call, replacement in sorted(changes, key=lambda c: c[0].line_number, reverse=True):
                    modified_content = _apply_replacement(modified_content, call.code_snippet, replacement)

                encoded = base64.b64encode(modified_content.encode("utf-8")).decode("utf-8")
                update_resp = await client.put(
                    f"{GITHUB_API_BASE}/repos/{repo}/contents/{file_path}",
                    headers=headers,
                    json={
                        "message": f"chore(optiml): migrate AI call in {file_path}",
                        "content": encoded,
                        "sha": file_sha,
                        "branch": branch_name,
                    },
                )
                update_resp.raise_for_status()

            # ------------------------------------------------------------------
            # 5. Open a pull request
            # ------------------------------------------------------------------
            pr_body = (
                "## Migrate AI calls to OptiML managed endpoints\n\n"
                f"This PR was auto-generated by [OptiML](https://optiml.one) to migrate "
                f"**{calls_processed}** AI call(s) to managed OptiML endpoints.\n\n"
                "### What changed\n"
                "Each direct AI provider call (OpenAI, Anthropic, etc.) has been replaced with a call "
                "to your OptiML endpoint. This gives you:\n\n"
                "- **Version control** — roll back to any previous prompt or model config\n"
                "- **A/B testing** — experiment with models and prompts without code changes\n"
                "- **Knowledge base** — attach context assets to enrich responses\n"
                "- **Cost tracking** — per-call cost and latency analytics\n"
                "- **Human-in-the-loop** — optional approval gates for sensitive calls\n\n"
                "### ⚠️ Action required\n"
                "These endpoints are **not yet active**. To activate them:\n\n"
                "1. **Sign up** at [optiml.one](https://optiml.one) and create your organization\n"
                "2. Set the `OPTIML_API_KEY` environment variable in your deployment "
                "(use `process.env.OPTIML_API_KEY`)\n"
                "3. Visit the [OptiML Dashboard](https://optiml.one/dashboard) to configure and "
                "activate your endpoints\n"
                "4. Review the changes in this PR and merge when ready\n"
            )

            pr_resp = await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/pulls",
                headers=headers,
                json={
                    "title": "Migrate AI calls to OptiML managed endpoints",
                    "body": pr_body,
                    "head": branch_name,
                    "base": default_branch,
                },
            )
            pr_resp.raise_for_status()
            pr_url = pr_resp.json().get("html_url", "")

        return {"pr_url": pr_url, "calls_migrated": calls_processed}

    except httpx.HTTPStatusError as e:
        logger.error("GitHub API error during presignup migration for %s: %s %s", repo, e.response.status_code, e.response.text)
        return JSONResponse(
            status_code=502,
            content={"error": f"GitHub API error: {e.response.status_code} — {e.response.text[:200]}"},
        )
    except Exception as e:
        logger.error("Unexpected error during presignup migration for %s: %s", repo, e)
        return JSONResponse(status_code=500, content={"error": f"Migration failed: {str(e)}"})
