"""
Write-time redaction for persisted customer request content (evidence capture).

THE BOUNDARY MAP  —  source value → redaction boundary → persisted field
------------------------------------------------------------------------
Every route by which customer-controlled request content reaches durable
storage is listed here. There is deliberately NO fourth column: a route that
does not pass through this module is a bug, not a variant.

  A. NAMED VARIABLES (the v12 evidence-capture columns)

     caller `variables` mapping
       → capture_variables()                      [this module]
       → workflow_runs.variables            (redacted value)
       + workflow_runs.variables_capture    (provenance: what was removed)

     Write sites, all of which call capture_variables() and nothing else:
       workflow_runtime.execute_workflow    success path      (~2589)
       workflow_runtime.execute_workflow    HTTPException path(~2663)
       workflow_runtime.execute_workflow    generic-error path(~2737)
       workflow_streaming._insert_workflow_run_linear         (~61)

  B. FREE-TEXT / JSON-SERIALISED INPUT  (the older, wider ingress)

     caller `input_text`  — OR `json.dumps(variables)`, which
     workflow_runtime.execute_workflow substitutes for an empty input_text
     when the workflow has an Input node (the path that produced 683 of the
     685 stored inputs, with no redaction at all before this change)
       → persist_input_text()                     [this module]
           → capture_input_text()
               → _walk()      when the text parses as a JSON object/array,
                              so KEY HINTS survive and the result is
                              IDENTICAL to what capture_variables() would
                              have produced from the same request
               → redact_text() otherwise
       → workflow_runs.input_text                     (redacted value)
       + workflow_runs.variables_capture["input_text_capture"]  (provenance)

     Write sites, all of which call persist_input_text() and nothing else:
       workflow_runtime.execute_workflow    success path      (~2589)
       workflow_runtime.execute_workflow    HTTPException path(~2663)
       workflow_runtime.execute_workflow    generic-error path(~2737)
       workflow_streaming._insert_workflow_run_linear         (~61)
       routers/public_execution             pre-exec HTTP fail(~649)
       routers/public_execution             pre-exec generic  (~682)

     THE VALUE THAT EXECUTES IS NEVER TOUCHED. `input_text` drives the Input
     node and is what the model actually receives. Redaction happens ONLY on
     the local passed to the insert, at the persist site. Nothing upstream of
     the database sees a redacted value.

  C. EXECUTION TRACE (the THIRD copy of the same request, on the SAME ROW)

     node_results — the Input node records `input_text[:200]`, the Prompt node
     records the variable-interpolated template `[:200]`, and error entries
     record `error_detail`, which providers routinely echo credentials into
       → persist_node_results()                   [this module]
           → capture_node_results()
               → _walk(..., preserve_keys=_TRACE_STRUCTURAL_KEYS)
       → workflow_runs.node_results                    (redacted trace)
       + workflow_runs.variables_capture["node_results_capture"] (provenance)

     Write sites, all of which call persist_node_results() and nothing else:
       workflow_runtime.execute_workflow    success path      (~2589)
       workflow_runtime.execute_workflow    HTTPException path(~2663)
       workflow_runtime.execute_workflow    generic-error path(~2737)
       workflow_streaming._insert_workflow_run_linear         (~61)

     Redacting `input_text` and leaving this alone would have stored the same
     secret 200 characters away on the same row. STRUCTURE IS LOAD-BEARING
     HERE: the entry count is the trace's step count, `status`/`error` produce
     every error rate in the product, and `node_id` is the join key other
     entries in the same trace refer to by value. All three survive; only what
     a step SAYS is redacted. The list handed to the boundary is not modified,
     so the trace returned to the caller and streamed over SSE is verbatim.

  D. REPLAY CASES (golden_inputs — an equivalent persisted evidence field)

     API payload / promoted production row
       → persist_golden_input()                   [this module]
           → capture_input_text() for input_text
           → capture_variables()   for variables
       → golden_inputs.input_text, golden_inputs.variables

     Write sites:
       workflow_management.create_golden_input                (~1350)
       workflow_management.update_golden_input                (~1378)
       workflow_management.import_golden_input_from_production(~1427)

     The promotion path additionally consults replay_gate() across ALL of the
     above — variables, input_text AND the trace: a row whose persisted content
     was modified by redaction anywhere is NOT automatically promoted, because
     the removed value may be what drove the behaviour the case claims to
     reproduce. See REPLAY SEMANTICS below.

  READS, not writes — these must NOT be redacted, and are listed so nobody
  "closes" them by mistake: workflow_management._golden_row_to_response
  (~1291) and the eval replay loop (~1785) read back what was already
  sanitised at write time.

REPLAY SEMANTICS
----------------
Redaction must never turn a sanitised request into supposedly faithful replay
evidence. `replay_gate()` reads the capture provenance and returns a
structured verdict. When redaction modified the persisted content the verdict
is ineligible with reason ``redacted_input_requires_review``; the sanitised row
is left exactly where it is, so a human can inspect it, approve the case
explicitly, or substitute a safe fixture. An unredacted secret is NEVER
retained in order to preserve replayability. Privacy beats automatic replay.

WHY THIS EXISTS
---------------
`workflow_runs` records `input_text` but never recorded the named `variables`
that actually drove a variable-driven workflow. A workflow whose input is five
named variables therefore stored nothing about its inputs, and none of its
traffic could ever be promoted into an evaluation case — `golden_inputs`
already has `variables`, `source` and `source_run_id`, so the consumer side of
that bridge was built and only the capture side was missing.

Capturing real customer input means capturing whatever the customer put in it.
So the values are redacted HERE, at write time, before they reach the database
— not at read time, not on export. A value that is never written cannot leak.

THREE PROPERTIES THAT ARE NOT NEGOTIABLE
----------------------------------------
1. REDACTION IS VISIBLE, NEVER SILENT. A redacted value is replaced by a marker
   that names what was removed (``[redacted:email]``) and the run records which
   paths were touched and why. A case with redacted fields may not reproduce
   the original behaviour on replay; the curation step downstream has to be able
   to SEE that, rather than discover it later as a mystery.
2. VALUES ARE REDACTED, STRUCTURE IS PRESERVED. Keys, nesting and types survive,
   so the SHAPE of the input stays analysable even when the content is gone. The
   one type change that can occur — a numeric value that is a valid card number
   becomes a marker string — is recorded explicitly as `type_changed`.
3. CONSERVATIVE ON AMBIGUITY. A false positive costs one replay case. A false
   negative writes a customer's PII into the database. When the two are
   balanced, redact.

WHAT IS DELIBERATELY *NOT* DONE
-------------------------------
No naive "any 16 digits is a card" and no "any run of digits is a phone
number". Both of those destroy far more evidence than they protect:
  * Card candidates must pass the LUHN checksum AND carry a recognised issuer
    prefix. An order number of the same length fails Luhn ~90% of the time and
    fails the prefix check besides.
  * Phone numbers are matched only in E.164 form, in an explicitly grouped
    form (3-3-4 with separators, or parenthesised area code), or when the KEY
    says the value is a phone number. A bare 13-digit millisecond timestamp, a
    version string like `2.10.4`, an IPv4 address and a date are all left alone.
  * Generic high-entropy detection applies only to a value that is ENTIRELY one
    whitespace-free token with mixed case and digits. A lowercase slug, a UUID
    and ordinary prose can never trigger it.

EXACTLY WHAT THE NATIONAL-ID RULE DOES, SINCE IT IS EASY TO GET WRONG
---------------------------------------------------------------------
`_SSN_RE` matches the punctuated 3-2-4 shape and excludes only what no scheme
issues. Concretely, in UNKEYED free text:

    123-45-6789   redacted   ordinary SSN
    912-70-1234   redacted   ITIN — 9xx area with a real ITIN group
    900-45-6789   kept       9xx area, group 45 is not an ITIN group
    666-12-3456   kept       area never issued
    000-12-3456   kept       area never issued

The 9xx block is NOT excluded wholesale: the IRS issues ITINs there, and an
ITIN is real PII. Only a 9xx value whose group digits fall outside the issued
ITIN ranges (50-65, 70-88, 90-92, 94-99) is treated as a formatted identifier
and left alone.

The two remaining pass-throughs are covered by the KEY when one is present:
rule 7 in `redact_text` removes ANY short value under a key matching
`_NATIONAL_ID_KEY_RE` (`ssn`, `national_id`, `tax_id`, `passport`, ...)
whatever its shape, so `{"ssn": "666-12-3456"}` IS redacted — including through
`capture_input_text`'s json mode, which preserves key hints.

Nothing in this module performs I/O, imports the database, reads configuration
or logs a value. It is pure and it is import-safe.
"""
from __future__ import annotations

import math
import re
from typing import Any

# Redaction ruleset version. Bump when patterns change so a downstream reader
# can tell which ruleset produced a given row.
#   1 — initial ruleset (migration v12).
#   2 — `input_text` and `node_results` brought inside the boundary; the SSN
#       rule narrowed so ITIN-shaped values (9xx area with a real ITIN group)
#       are redacted instead of being passed through as free text.
REDACTION_VERSION = 2

# Matches the existing `input_text[:5000]` convention in workflow_runtime.py.
MAX_VALUE_CHARS = 5000
# Structural caps. Each one, when it bites, is recorded — never silent.
MAX_LIST_ITEMS = 200
MAX_MAPPING_KEYS = 500
MAX_DEPTH = 8
# Whole-blob guard. Past this the row is recorded as unavailable-with-reason
# rather than writing an unbounded document into every run.
MAX_SERIALIZED_CHARS = 200_000

KIND_EMAIL = "email"
KIND_PHONE = "phone"
KIND_CARD = "payment_card"
KIND_NATIONAL_ID = "national_id"
KIND_CREDENTIAL = "credential"

TRUNCATION_MARKER = "[truncated:{original} chars]"
DEPTH_MARKER = "[truncated:depth]"
LIST_MARKER = "[truncated:{omitted} more items]"
MAPPING_MARKER = "[truncated:{omitted} more keys]"


def marker(kind: str) -> str:
    """The visible replacement written in place of a redacted value."""
    return f"[redacted:{kind}]"


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

def luhn_ok(digits: str) -> bool:
    """Luhn (mod-10) checksum. This is what separates a card number from an
    order number of the same length."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Issuer identification prefixes. A number that passes Luhn but belongs to no
# scheme is far more likely to be a checksum-carrying internal identifier.
def _card_scheme(digits: str) -> str | None:
    n = len(digits)
    if not (13 <= n <= 19):
        return None
    try:
        p2, p3, p4, p6 = int(digits[:2]), int(digits[:3]), int(digits[:4]), int(digits[:6])
    except ValueError:
        return None
    if digits[0] == "4" and n in (13, 16, 19):
        return "visa"
    if (51 <= p2 <= 55 or 2221 <= p4 <= 2720) and n == 16:
        return "mastercard"
    if p2 in (34, 37) and n == 15:
        return "amex"
    if (p4 == 6011 or p2 == 65 or 644 <= p3 <= 649 or 622126 <= p6 <= 622925) and 16 <= n <= 19:
        return "discover"
    if 3528 <= p4 <= 3589 and 16 <= n <= 19:
        return "jcb"
    if (300 <= p3 <= 305 or p4 == 3095 or p2 in (36, 38, 39)) and 14 <= n <= 19:
        return "diners"
    if p2 in (62, 81) and 16 <= n <= 19:
        return "unionpay"
    if p4 in (5018, 5020, 5038, 5893, 6304, 6759, 6761, 6762, 6763) and 12 <= n <= 19:
        return "maestro"
    return None


def looks_like_payment_card(raw: str) -> bool:
    digits = re.sub(r"[ \-]", "", raw)
    if not digits.isdigit():
        return False
    return _card_scheme(digits) is not None and luhn_ok(digits)


# ---------------------------------------------------------------------------
# Key-name hints
#
# The KEY is evidence. `{"phone": "5551234567"}` is a phone number; the same
# ten digits under `{"order_id": ...}` are not. Using the key lets the value
# patterns stay strict instead of being widened until they eat everything.
# ---------------------------------------------------------------------------

_SECRET_KEY_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:"
    r"pass(?:word|wd|phrase)|secret|api[_\-]?key|apikey|access[_\-]?key|"
    r"private[_\-]?key|secret[_\-]?key|auth[_\-]?token|access[_\-]?token|"
    r"refresh[_\-]?token|bearer|credential|authorization|client[_\-]?secret|"
    r"session[_\-]?(?:id|token)|cookie|otp|passcode|security[_\-]?code|cvv|cvc"
    r")(?:$|[^a-z])"
)
# `token` on its own is excluded above on purpose: `max_tokens`, `token_count`,
# `output_tokens` are counters, and redacting them would corrupt replay while
# protecting nothing.

_PHONE_KEY_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:phone|telephone|tel|mobile|cell|fax|msisdn|whatsapp)(?:$|[^a-z])"
)
_NATIONAL_ID_KEY_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:ssn|social[_\-]?security|national[_\-]?id|nin|nino|"
    r"sin|aadhaar|aadhar|tax[_\-]?id|tin|passport|id[_\-]?number)(?:$|[^a-z])"
)
_CARD_KEY_RE = re.compile(
    r"(?i)(?:^|[^a-z])(?:card[_\-]?number|cardnumber|credit[_\-]?card|cc[_\-]?num(?:ber)?|pan)(?:$|[^a-z])"
)


# ---------------------------------------------------------------------------
# Value patterns, applied in order. Order matters: a credential blob can
# contain something email-shaped, and a card written with dashes is phone-ish.
# ---------------------------------------------------------------------------

_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN[ A-Z0-9]*PRIVATE KEY[ A-Z]*-----[\s\S]*?-----END[ A-Z0-9]*PRIVATE KEY[ A-Z]*-----"
)
# An unterminated block still has to go: redact from BEGIN to end of value.
_PRIVATE_KEY_OPEN_RE = re.compile(r"-----BEGIN[ A-Z0-9]*PRIVATE KEY[ A-Z]*-----[\s\S]*\Z")

_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?")

_BEARER_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}")

_PROVIDER_KEY_RE = re.compile(
    r"(?:"
    r"sk-ant-[A-Za-z0-9_\-]{16,}"
    r"|sk-proj-[A-Za-z0-9_\-]{16,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|A(?:KIA|SIA)[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprse]-[A-Za-z0-9\-]{10,}"
    r"|AIza[0-9A-Za-z_\-]{35}"
    r"|SG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"
    r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|glpat-[A-Za-z0-9_\-]{16,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|dop_v1_[a-f0-9]{40,}"
    r"|shpat_[a-fA-F0-9]{28,}"
    r"|ya29\.[A-Za-z0-9._\-]{20,}"
    r")"
)

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.\-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*"
    r"\.[A-Za-z]{2,24}\b"
)

# Card CANDIDATES only. Every hit is then checked with Luhn + issuer prefix,
# and only survivors are redacted.
_CARD_CANDIDATE_RE = re.compile(r"(?<![\d.\-])\d(?:[ \-]?\d){11,18}(?![\d.\-])")

# US SSN / ITIN in punctuated form. Matches on shape alone, with only the
# ranges that are STRUCTURALLY IMPOSSIBLE excluded.
#
# `000` and `666` are never issued as an SSN area and are not valid as an ITIN
# area either, so excluding them is precision: a 3-2-4 grouping starting with
# one of them is far likelier to be a formatted internal identifier.
#
# `9xx` is a DIFFERENT CASE and is NOT excluded wholesale. The IRS issues
# INDIVIDUAL TAXPAYER IDENTIFICATION NUMBERS in exactly that range —
# 9XX-XX-XXXX with the group digits in 50-65, 70-88, 90-92 or 94-99. An ITIN is
# a real taxpayer identifier and real PII; treating the whole 9xx block as
# "administratively invalid" was a false negative, not a conservative choice.
# The lookahead below rejects a 9xx area ONLY when its group is not an ITIN
# group, keeping the false-positive protection for genuinely unissued 9xx
# shapes while redacting every real ITIN.
#
# `00` group and `0000` serial are never issued in either scheme.
_ITIN_GROUP = r"(?:5\d|6[0-5]|7\d|8[0-8]|9[0-2]|9[4-9])"
_SSN_RE = re.compile(
    r"(?<!\d)"
    r"(?!000|666)"                                # areas no scheme issues
    r"(?!9\d\d[ \-](?!" + _ITIN_GROUP + r"))"     # 9xx only with an ITIN group
    r"\d{3}([ \-])(?!00)\d{2}\1(?!0000)\d{4}(?!\d)"
)
# UK National Insurance number.
_NINO_RE = re.compile(
    r"(?<![A-Za-z0-9])[ABCEGHJ-PRSTW-Z][ABCEGHJ-NPRSTW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D](?![A-Za-z0-9])"
)

# Phone: E.164, i.e. an explicit leading '+'. Digit count is verified after.
_PHONE_E164_RE = re.compile(r"(?<![\w.+])\+\d[\d\s().\-]{6,20}\d(?![\w])")
# Phone: explicitly GROUPED national form. Requires real separators, so a bare
# run of digits (timestamp, order number, account number) cannot match, and the
# leading `(?<![\w.])` stops it firing inside an IPv4 address or a version
# string such as `10.155.123.4567`.
_PHONE_GROUPED_RE = re.compile(
    r"(?<![\w.])(?:\(\d{3}\)[ .\-]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\d.\-])"
)

_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_like_opaque_secret(value: str) -> bool:
    """A whole-value heuristic for credential-shaped strings that carry no
    recognisable vendor prefix.

    Deliberately narrow. The value must be ONE whitespace-free token of
    24..256 characters carrying upper case, lower case AND digits, with high
    per-character entropy, and must not be a UUID or a URL. Prose cannot reach
    this test (whitespace), a lowercase slug cannot pass it (no upper case),
    a UUID cannot pass it (no upper case in canonical form, and excluded
    explicitly), and a base64 image payload cannot pass it (length cap).
    """
    v = value.strip()
    if not (24 <= len(v) <= 256):
        return False
    if any(ch.isspace() for ch in v):
        return False
    if _UUID_RE.match(v):
        return False
    if "://" in v or v.startswith("/") or "@" in v:
        return False
    if not (any(c.isupper() for c in v) and any(c.islower() for c in v) and any(c.isdigit() for c in v)):
        return False
    if not all(c.isalnum() or c in "-_=+/.~" for c in v):
        return False
    return _shannon_entropy(v) >= 3.5


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


# ---------------------------------------------------------------------------
# String redaction
# ---------------------------------------------------------------------------

def redact_text(value: str, key_hint: str = "") -> tuple[str, list[str]]:
    """Redact sensitive spans inside one string.

    Returns (redacted_value, kinds_found). Surrounding text is preserved so the
    shape of the input stays readable; only the sensitive span is replaced.
    """
    kinds: list[str] = []
    out = value

    def _sub(pattern: re.Pattern[str], kind: str, text: str) -> str:
        nonlocal kinds
        if not pattern.search(text):
            return text
        if kind not in kinds:
            kinds.append(kind)
        return pattern.sub(marker(kind), text)

    # 1. Whole-value redaction when the KEY declares the value a secret.
    if key_hint and _SECRET_KEY_RE.search(key_hint):
        stripped = out.strip()
        if len(stripped) >= 8 and not any(ch.isspace() for ch in stripped):
            return marker(KIND_CREDENTIAL), [KIND_CREDENTIAL]

    # 2. Credential material, longest/most structured first.
    out = _sub(_PRIVATE_KEY_BLOCK_RE, KIND_CREDENTIAL, out)
    out = _sub(_PRIVATE_KEY_OPEN_RE, KIND_CREDENTIAL, out)
    out = _sub(_JWT_RE, KIND_CREDENTIAL, out)
    out = _sub(_BEARER_RE, KIND_CREDENTIAL, out)
    out = _sub(_PROVIDER_KEY_RE, KIND_CREDENTIAL, out)

    # 3. Email before any digit pattern: an address can contain digits.
    out = _sub(_EMAIL_RE, KIND_EMAIL, out)

    # 4. Payment cards: candidates verified with Luhn + issuer prefix.
    def _card_repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        if looks_like_payment_card(raw):
            if KIND_CARD not in kinds:
                kinds.append(KIND_CARD)
            return marker(KIND_CARD)
        return raw

    out = _CARD_CANDIDATE_RE.sub(_card_repl, out)

    # 5. National identifiers.
    out = _sub(_SSN_RE, KIND_NATIONAL_ID, out)
    out = _sub(_NINO_RE, KIND_NATIONAL_ID, out)

    # 6. Phone numbers: E.164 first, then explicitly grouped national form.
    def _e164_repl(m: re.Match[str]) -> str:
        n = len(_digits_only(m.group(0)))
        if 8 <= n <= 15:
            if KIND_PHONE not in kinds:
                kinds.append(KIND_PHONE)
            return marker(KIND_PHONE)
        return m.group(0)

    out = _PHONE_E164_RE.sub(_e164_repl, out)
    out = _sub(_PHONE_GROUPED_RE, KIND_PHONE, out)

    # 7. Key-directed checks for values that carry no formatting of their own.
    #    These fire ONLY because the key says what the value is.
    if key_hint:
        digits = _digits_only(out)
        if _PHONE_KEY_RE.search(key_hint) and 7 <= len(digits) <= 15 and re.fullmatch(r"[\d\s().+\-]+", out.strip() or "x"):
            return marker(KIND_PHONE), (kinds + [KIND_PHONE] if KIND_PHONE not in kinds else kinds)
        if _NATIONAL_ID_KEY_RE.search(key_hint) and out.strip() and len(out.strip()) <= 64:
            return marker(KIND_NATIONAL_ID), (kinds + [KIND_NATIONAL_ID] if KIND_NATIONAL_ID not in kinds else kinds)
        if _CARD_KEY_RE.search(key_hint) and 12 <= len(digits) <= 19 and re.fullmatch(r"[\d\s\-]+", out.strip() or "x"):
            return marker(KIND_CARD), (kinds + [KIND_CARD] if KIND_CARD not in kinds else kinds)

    # 8. Opaque credential-shaped token as the entire value.
    if not kinds and looks_like_opaque_secret(out):
        return marker(KIND_CREDENTIAL), [KIND_CREDENTIAL]

    return out, kinds


def redact_number(value: int | float, key_hint: str = "") -> tuple[Any, list[str]]:
    """A card number or an identifier can arrive as a JSON number. Checking it
    means rendering it as digits; a hit therefore changes the type, which the
    caller records as `type_changed`."""
    if isinstance(value, bool):
        return value, []
    if isinstance(value, float):
        if not value.is_integer():
            return value, []
        text = str(int(value))
    else:
        text = str(value)
    digits = text.lstrip("-")
    if not digits.isdigit():
        return value, []
    if looks_like_payment_card(digits):
        return marker(KIND_CARD), [KIND_CARD]
    if key_hint and _CARD_KEY_RE.search(key_hint) and 12 <= len(digits) <= 19:
        return marker(KIND_CARD), [KIND_CARD]
    if key_hint and _NATIONAL_ID_KEY_RE.search(key_hint):
        return marker(KIND_NATIONAL_ID), [KIND_NATIONAL_ID]
    if key_hint and _PHONE_KEY_RE.search(key_hint) and 7 <= len(digits) <= 15:
        return marker(KIND_PHONE), [KIND_PHONE]
    return value, []


# ---------------------------------------------------------------------------
# Structure walk
# ---------------------------------------------------------------------------

class _Report:
    def __init__(self) -> None:
        self.redactions: list[dict[str, Any]] = []
        self.truncations: list[dict[str, Any]] = []
        self.kinds: set[str] = set()
        self.type_changed = False

    def record_redaction(self, path: str, kinds: list[str], type_changed: bool) -> None:
        self.redactions.append({"path": path, "kinds": sorted(set(kinds))})
        self.kinds.update(kinds)
        if type_changed:
            self.type_changed = True

    def record_truncation(self, path: str, reason: str, original: int, kept: int) -> None:
        self.truncations.append(
            {"path": path, "reason": reason, "original": original, "kept": kept}
        )


def _walk(
    value: Any,
    path: str,
    key_hint: str,
    depth: int,
    report: _Report,
    preserve_keys: frozenset[str] = frozenset(),
) -> Any:
    """Redact a structure. `preserve_keys` names mapping keys whose values are
    copied through UNCHANGED.

    That parameter exists for exactly one caller — the execution trace, whose
    entries are keyed by backend-generated identifiers that other entries in the
    same trace refer to by value. Redacting one of those would break the trace's
    internal consistency (a router entry pointing at a node id that no longer
    appears) while protecting nothing, because an identifier the backend
    generated is not customer request content. The set is closed, explicit, and
    contains ONLY identifiers and enumerations. Everything else — including any
    key a future node type introduces — is redacted by default.
    """
    if depth > MAX_DEPTH:
        report.record_truncation(path, "depth", depth, MAX_DEPTH)
        return DEPTH_MARKER

    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, str):
        redacted, kinds = redact_text(value, key_hint)
        if kinds:
            report.record_redaction(path, kinds, type_changed=False)
        if len(redacted) > MAX_VALUE_CHARS:
            original_len = len(redacted)
            redacted = redacted[:MAX_VALUE_CHARS] + TRUNCATION_MARKER.format(original=original_len)
            report.record_truncation(path, "value_length", original_len, MAX_VALUE_CHARS)
        return redacted

    if isinstance(value, (int, float)):
        redacted, kinds = redact_number(value, key_hint)
        if kinds:
            report.record_redaction(path, kinds, type_changed=True)
        return redacted

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        items = list(value.items())
        kept = items[:MAX_MAPPING_KEYS]
        if len(items) > MAX_MAPPING_KEYS:
            report.record_truncation(path or "$", "mapping_keys", len(items), MAX_MAPPING_KEYS)
        for k, v in kept:
            key_str = k if isinstance(k, str) else str(k)
            child_path = f"{path}.{key_str}" if path else key_str
            if key_str in preserve_keys:
                out[key_str] = v
                continue
            out[key_str] = _walk(v, child_path, key_str, depth + 1, report, preserve_keys)
        if len(items) > MAX_MAPPING_KEYS:
            out[MAPPING_MARKER.format(omitted=len(items) - MAX_MAPPING_KEYS)] = None
        return out

    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        seq = list(value)
        kept_seq = seq[:MAX_LIST_ITEMS]
        if len(seq) > MAX_LIST_ITEMS:
            report.record_truncation(path or "$", "list_items", len(seq), MAX_LIST_ITEMS)
        for i, v in enumerate(kept_seq):
            out_list.append(_walk(v, f"{path}[{i}]", key_hint, depth + 1, report, preserve_keys))
        if len(seq) > MAX_LIST_ITEMS:
            out_list.append(LIST_MARKER.format(omitted=len(seq) - MAX_LIST_ITEMS))
        return out_list

    # Anything else (set, bytes, custom object) is rendered as text and put
    # through the same redaction, so no unknown type escapes unchecked.
    coerced = str(value)
    redacted, kinds = redact_text(coerced, key_hint)
    if kinds:
        report.record_redaction(path, kinds, type_changed=True)
    if len(redacted) > MAX_VALUE_CHARS:
        original_len = len(redacted)
        redacted = redacted[:MAX_VALUE_CHARS] + TRUNCATION_MARKER.format(original=original_len)
        report.record_truncation(path, "value_length", original_len, MAX_VALUE_CHARS)
    return redacted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def capture_variables(variables: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Turn the execution's `variables` mapping into (stored_value, capture_meta).

    `stored_value` is what belongs in `workflow_runs.variables`; `capture_meta`
    is what belongs in `workflow_runs.variables_capture`.

    NEVER RAISES. Every failure mode resolves to `(None, {...status: ...})` so a
    capture problem can never turn into a failed customer request. There is also
    no path that returns `{}` meaning "we don't know": absent, empty and
    unavailable are three distinct, recorded statuses.
    """
    try:
        if variables is None:
            return None, {
                "status": "absent",
                "reason": "no_variables_supplied",
                "redacted": False,
                "truncated": False,
                "redaction_version": REDACTION_VERSION,
            }

        if not isinstance(variables, dict):
            return None, {
                "status": "unavailable",
                "reason": "variables_not_a_mapping",
                "observed_type": type(variables).__name__,
                "redacted": False,
                "truncated": False,
                "redaction_version": REDACTION_VERSION,
            }

        if len(variables) == 0:
            return None, {
                "status": "empty",
                "reason": "caller_supplied_empty_mapping",
                "redacted": False,
                "truncated": False,
                "key_count": 0,
                "redaction_version": REDACTION_VERSION,
            }

        report = _Report()
        safe = _walk(variables, "", "", 1, report)

        import json as _json

        serialized = _json.dumps(safe, default=str)
        if len(serialized) > MAX_SERIALIZED_CHARS:
            return None, {
                "status": "unavailable",
                "reason": "oversize",
                "serialized_chars": len(serialized),
                "limit_chars": MAX_SERIALIZED_CHARS,
                "key_count": len(variables),
                "keys": sorted(str(k) for k in variables)[:MAX_MAPPING_KEYS],
                "redacted": bool(report.redactions),
                "truncated": True,
                "redaction_version": REDACTION_VERSION,
            }

        meta: dict[str, Any] = {
            "status": "captured",
            "redacted": bool(report.redactions),
            "truncated": bool(report.truncations),
            "key_count": len(variables),
            "redaction_version": REDACTION_VERSION,
        }
        if report.redactions:
            meta["redactions"] = report.redactions
            meta["redacted_paths"] = sorted({r["path"] for r in report.redactions})
            meta["redacted_kinds"] = sorted(report.kinds)
            meta["type_changed"] = report.type_changed
        if report.truncations:
            meta["truncations"] = report.truncations
            meta["truncated_paths"] = sorted({t["path"] for t in report.truncations})
        return safe, meta

    except BaseException as exc:  # noqa: BLE001 — capture must never break a run
        try:
            reason_type = type(exc).__name__
        except BaseException:
            reason_type = "unknown"
        return None, {
            "status": "unavailable",
            "reason": "capture_failed",
            "error_type": reason_type,
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }


# ---------------------------------------------------------------------------
# Free-text / JSON-serialised input  (boundary map, route B)
#
# `workflow_runs.input_text` is the OLDER and WIDER ingress: it predates the
# v12 capture columns and it is where 683 of the 685 stored inputs arrived,
# because workflow_runtime substitutes `json.dumps(variables)` for an empty
# input_text. Nothing redacted it. These entry points close that route using
# the SAME primitives as capture_variables — `_walk` and `redact_text` — so
# there is exactly one ruleset in the process and the two columns can never
# disagree about the same request.
# ---------------------------------------------------------------------------

# Namespaced key under which the input_text provenance is nested inside the
# existing `workflow_runs.variables_capture` JSONB. Chosen over a new column so
# this fix needs no structural migration; v13 is a COMMENT-only migration that
# records the widened meaning.
INPUT_TEXT_CAPTURE_KEY = "input_text_capture"

# Structured reason surfaced when redaction modified content that a replay case
# would otherwise claim to reproduce faithfully. Backend returns the code; the
# frontend owns the wording.
REVIEW_REDACTED_INPUT = "redacted_input_requires_review"
#: The row's redaction provenance could not be read, or capture ran and failed.
#: Distinct from REVIEW_REDACTED_INPUT on purpose: "we removed something" and
#: "we cannot tell whether anything needed removing" are different facts, and
#: collapsing them would hide the second behind the first.
REVIEW_UNKNOWN_PROVENANCE = "capture_provenance_unavailable"

# Every namespaced capture record nested inside `variables_capture`. The replay
# gate walks all of them, so adding a persisted field to the boundary means
# adding its key here and nowhere else.
_NESTED_CAPTURE_KEYS = ("input_text_capture", "node_results_capture")


def _meta_from_report(report: "_Report") -> dict[str, Any]:
    """Provenance fields shared by every capture entry point, so `variables`
    and `input_text` describe their redactions in exactly the same shape."""
    meta: dict[str, Any] = {}
    if report.redactions:
        meta["redactions"] = report.redactions
        meta["redacted_paths"] = sorted({r["path"] for r in report.redactions})
        meta["redacted_kinds"] = sorted(report.kinds)
        meta["type_changed"] = report.type_changed
    if report.truncations:
        meta["truncations"] = report.truncations
        meta["truncated_paths"] = sorted({t["path"] for t in report.truncations})
    return meta


def capture_input_text(value: Any) -> tuple[str | None, dict[str, Any]]:
    """Redact one free-text (or JSON-serialised) input destined for a persisted
    evidence field. Returns (stored_value, capture_meta).

    NEVER RAISES, and never returns a placeholder: absent, empty and
    unavailable are three distinct recorded statuses, exactly as in
    capture_variables.

    Two modes, one ruleset:

    * ``json``  — the text parses as a JSON object or array, which is what
      ``json.dumps(variables)`` produces. The parsed structure goes through the
      SAME `_walk` used for the `variables` column, so key-directed rules
      (``{"api_key": "..."}``, ``{"ssn": "..."}``) fire identically and the two
      persisted fields cannot disagree about the same request.
    * ``text``  — anything else goes through `redact_text`, the same span-level
      primitive `_walk` calls for every string it visits.

    Redaction runs BEFORE truncation: a secret straddling the 5000-character
    storage cap must not survive as a usable prefix.
    """
    try:
        if value is None:
            return None, {
                "status": "absent",
                "reason": "no_input_text_supplied",
                "redacted": False,
                "truncated": False,
                "redaction_version": REDACTION_VERSION,
            }

        coerced_from: str | None = None
        if isinstance(value, str):
            text = value
        else:
            coerced_from = type(value).__name__
            text = str(value)

        if text == "":
            meta = {
                "status": "empty",
                "reason": "caller_supplied_empty_text",
                "redacted": False,
                "truncated": False,
                "source_chars": 0,
                "redaction_version": REDACTION_VERSION,
            }
            if coerced_from:
                meta["coerced_from"] = coerced_from
            return None, meta

        source_chars = len(text)
        # Bound the work. Only the first MAX_VALUE_CHARS are ever stored, so a
        # window far larger than that cannot lose a secret from the stored
        # prefix, and it keeps regex cost bounded on a multi-megabyte body.
        oversize = source_chars > MAX_SERIALIZED_CHARS
        work = text[:MAX_SERIALIZED_CHARS] if oversize else text

        report = _Report()
        mode = "text"
        stripped = work.lstrip()
        if stripped[:1] in ("{", "["):
            try:
                import json as _json

                parsed = _json.loads(work)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                safe = _walk(parsed, "", "", 1, report)
                try:
                    import json as _json

                    redacted = _json.dumps(safe, default=str)
                    mode = "json"
                except Exception:
                    # Re-serialisation failed: fall back to the text ruleset
                    # rather than storing the unredacted original.
                    report = _Report()
                    redacted = None
                if redacted is None:
                    mode = "text"
            else:
                redacted = None
        else:
            redacted = None

        if mode == "text":
            redacted_text, kinds = redact_text(work, "")
            if kinds:
                report.record_redaction("$", kinds, type_changed=False)
            redacted = redacted_text

        stored = redacted[:MAX_VALUE_CHARS]
        truncated = oversize or len(redacted) > MAX_VALUE_CHARS
        if truncated:
            report.record_truncation(
                "$",
                "oversize_source" if oversize else "value_length",
                source_chars,
                len(stored),
            )

        meta: dict[str, Any] = {
            "status": "captured",
            "mode": mode,
            "redacted": bool(report.redactions),
            "truncated": truncated,
            "source_chars": source_chars,
            "stored_chars": len(stored),
            "redaction_version": REDACTION_VERSION,
        }
        if coerced_from:
            meta["coerced_from"] = coerced_from
        meta.update(_meta_from_report(report))
        # The existing column convention is a plain [:5000] slice with no
        # inline marker, so truncation is recorded here instead of being
        # written into the stored text.
        return (stored or None), meta

    except BaseException as exc:  # noqa: BLE001 — capture must never break a run
        try:
            reason_type = type(exc).__name__
        except BaseException:
            reason_type = "unknown"
        return None, {
            "status": "unavailable",
            "reason": "capture_failed",
            "error_type": reason_type,
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }


def persist_input_text(
    value: Any,
    variables_capture: Any = None,
) -> tuple[str | None, dict[str, Any]]:
    """THE persist-time boundary for `workflow_runs.input_text`.

    Call this at the INSERT, never in the execution path: the returned value is
    for the database only. The caller's own `input_text` must keep driving the
    workflow verbatim.

    Returns (value_to_store, variables_capture_with_input_provenance). The
    provenance is nested under `INPUT_TEXT_CAPTURE_KEY` inside the existing
    `variables_capture` JSONB, so no new column is needed. NEVER RAISES.
    """
    try:
        stored, meta = capture_input_text(value)
    except BaseException:  # noqa: BLE001 — belt and braces; capture_input_text
        stored, meta = None, {                # already guards itself
            "status": "unavailable",
            "reason": "capture_failed",
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }
    try:
        out = dict(variables_capture) if isinstance(variables_capture, dict) else {}
        if not isinstance(variables_capture, dict) and variables_capture is not None:
            out = {
                "status": "unavailable",
                "reason": "variables_capture_not_a_mapping",
                "redacted": False,
                "truncated": False,
                "redaction_version": REDACTION_VERSION,
            }
        out[INPUT_TEXT_CAPTURE_KEY] = meta
        return stored, out
    except BaseException:  # noqa: BLE001
        return stored, {INPUT_TEXT_CAPTURE_KEY: meta}


# ---------------------------------------------------------------------------
# Replay eligibility  (boundary map, route C)
# ---------------------------------------------------------------------------

def _capture_says_redacted(capture: Any) -> bool:
    return bool(isinstance(capture, dict) and capture.get("redacted"))


def _capture_kinds(capture: Any) -> list[str]:
    if not isinstance(capture, dict):
        return []
    kinds = capture.get("redacted_kinds")
    return [str(k) for k in kinds] if isinstance(kinds, list) else []


def _capture_paths(capture: Any) -> list[str]:
    if not isinstance(capture, dict):
        return []
    paths = capture.get("redacted_paths")
    return [str(p) for p in paths] if isinstance(paths, list) else []


def replay_gate(*captures: Any) -> dict[str, Any]:
    """Decide whether redacted evidence may become a replay case automatically.

    Accepts any number of capture-provenance mappings (a
    `workflow_runs.variables_capture`, which may itself nest an
    `input_text_capture` and a `node_results_capture`, and/or a freshly
    computed capture from the promotion boundary). Every persisted field on the
    row is therefore considered by one gate: a redaction anywhere — variables,
    input_text or the execution trace — makes the row ineligible, because the
    removed value may be exactly what drove the behaviour the case claims to
    reproduce. Returns a structured verdict — codes and measured facts only,
    no customer-facing prose:

        {"eligible": bool, "reasons": [...], "redacted_kinds": [...],
         "redacted_paths": [...]}

    Ineligible means: do not promote automatically. It does NOT mean discard.
    The sanitised row stays exactly where it is for inspection and curation, and
    a human may approve the case or substitute a safe fixture. Truncation alone
    is not a gate — the [:5000] cap predates this boundary and applies to every
    input equally — only redaction is, because only redaction removes a value
    that may be what drove the behaviour the case claims to reproduce.

    NEVER RAISES.
    """
    reasons: list[str] = []
    kinds: set[str] = set()
    paths: set[str] = set()
    try:
        readable = 0
        for capture in captures:
            if not isinstance(capture, dict):
                continue
            readable += 1
            parts = [capture]
            for nested_key in _NESTED_CAPTURE_KEYS:
                parts.append(capture.get(nested_key))
            for part in parts:
                if not isinstance(part, dict):
                    continue
                # UNKNOWN IS NOT CLEAN. `unavailable` means capture was
                # attempted and produced nothing usable — capture_failed,
                # oversize, variables_not_a_mapping. Those are precisely the
                # cases where a secret is most likely to have gone unexamined,
                # so `redacted: False` on them is an absence of evidence, not
                # evidence of absence. Treating them as promotable would let
                # the one row nobody could inspect become replay evidence.
                # `absent` and `empty` are REAL observations and stay eligible.
                if part.get("status") == "unavailable":
                    if REVIEW_UNKNOWN_PROVENANCE not in reasons:
                        reasons.append(REVIEW_UNKNOWN_PROVENANCE)
                if _capture_says_redacted(part):
                    if REVIEW_REDACTED_INPUT not in reasons:
                        reasons.append(REVIEW_REDACTED_INPUT)
                    kinds.update(_capture_kinds(part))
                    paths.update(_capture_paths(part))
        # No readable provenance at all — a legacy row whose fresh capture also
        # failed, or a caller that passed nothing. Same rule: fail closed.
        if readable == 0 and REVIEW_UNKNOWN_PROVENANCE not in reasons:
            reasons.append(REVIEW_UNKNOWN_PROVENANCE)
    except BaseException:  # noqa: BLE001 — a gate that crashes must fail CLOSED
        return {
            "eligible": False,
            "reasons": [REVIEW_REDACTED_INPUT],
            "redacted_kinds": [],
            "redacted_paths": [],
        }
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "redacted_kinds": sorted(kinds),
        "redacted_paths": sorted(paths),
    }


def persist_golden_input(
    input_text: Any = None,
    variables: Any = None,
) -> tuple[str | None, Any, dict[str, Any]]:
    """THE persist-time boundary for `golden_inputs`.

    `golden_inputs` holds customer request content just as `workflow_runs`
    does, and it is written from two customer-controlled sources: an API
    payload, and promotion of a production run. A run persisted BEFORE this
    boundary shipped carries unredacted `input_text` (history is immutable and
    is never rewritten), so promotion re-runs the same ruleset at this
    boundary rather than trusting the age of the row.

    Returns (stored_input_text, stored_variables, capture_meta), where
    capture_meta carries the same provenance shape used on `workflow_runs` and
    is what `replay_gate()` consumes. NEVER RAISES.
    """
    try:
        stored_text, text_meta = capture_input_text(input_text)

        stored_vars: Any = variables
        vars_meta: dict[str, Any]
        if isinstance(variables, dict) and variables:
            stored_vars, vars_meta = capture_variables(variables)
            if stored_vars is None:
                # capture declined (oversize / failure). Storing the raw
                # mapping would route around this boundary, so store nothing.
                stored_vars = None
        else:
            _unused, vars_meta = capture_variables(variables)
            stored_vars = variables if isinstance(variables, dict) else None

        meta = dict(vars_meta)
        meta[INPUT_TEXT_CAPTURE_KEY] = text_meta
        return stored_text, stored_vars, meta
    except BaseException:  # noqa: BLE001 — fail CLOSED: store nothing, say why
        return None, None, {
            "status": "unavailable",
            "reason": "capture_failed",
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }


# ---------------------------------------------------------------------------
# Execution trace  (boundary map, route D)
#
# `workflow_runs.node_results` is the third persisted copy of the same request.
# The Input node records `input_text[:200]` and the Prompt node records the
# variable-interpolated template `[:200]`, so redacting `input_text` while
# leaving the trace alone stored the same secret 200 characters away, on the
# SAME ROW. That is the inconsistent-guarantee problem this module exists to
# prevent, so the trace goes through the same `_walk` as everything else.
# ---------------------------------------------------------------------------

NODE_RESULTS_CAPTURE_KEY = "node_results_capture"

# Keys whose values are BACKEND- OR GRAPH-GENERATED identifiers and
# enumerations. They cannot carry customer request content, and redacting one
# would corrupt the trace rather than protect anything:
#   * `node_id` is the join key back to the workflow graph, and a router entry
#     refers to other entries by that same value via `selected`, `target`,
#     `candidates` and `router_selected_node_id`. Redact one and the trace
#     stops referring to itself.
#   * `type`/`status`/`mode`/`strategy`/`source`/`content_type` are closed
#     enumerations the runtime writes.
#   * `model`/`provider` (and their router-selected forms) come from the
#     workflow's own configuration and are what cost and routing analysis
#     correlate on.
# This list is a CLOSED EXCEPTION, not a policy: every other key — `output`,
# `error_detail`, `prompt`, `input`, `draft`, `final_output`, `tool_call`,
# `output_warning`, and anything a future node type adds — is redacted by
# default. Adding a key here means asserting it can never hold request content.
_TRACE_STRUCTURAL_KEYS = frozenset({
    "node_id", "type", "status", "content_type",
    "selected", "candidates", "branch_taken",
    "router_selected_node_id", "router_selected_model", "router_selected_provider",
    "model", "provider", "strategy", "mode",
})
# `source` and `target` were on this list and have been REMOVED. They are emitted
# only by `executed_edges`, which is a sibling of `node_results` in the response
# and is never persisted — so inside a trace entry they were dead entries. They
# are also the two most generic names here: a future node type putting an
# interpolated recipient, URL or filename under `target` would have been exempted
# silently. An unused exemption on a generic name is a latent hole, not caution.
#
# Every key above is an assertion that it can NEVER hold request content. Adding
# one is a privacy decision, not a formatting decision: prefer teaching the
# redactor about a real identifier over exempting the key that carries it.


def capture_node_results(node_results: Any) -> tuple[Any, dict[str, Any]]:
    """Redact the execution trace destined for `workflow_runs.node_results`.

    Returns (stored_value, capture_meta). NEVER RAISES.

    THE ENTRY COUNT IS PRESERVED EXACTLY. The top-level list is walked entry by
    entry rather than as one value, so `MAX_LIST_ITEMS` cannot silently drop
    steps: `len(node_results)` is the trace's step count in the UI, and
    `node_result_has_error` is evaluated per entry to produce every error rate
    in the product. Redaction changes what a step SAYS, never how many there
    were, and never whether one was an error — `status`, `error` and
    `error_status` are structural.

    Fails CLOSED. A capture failure yields (None, unavailable-with-reason)
    rather than the unredacted original; the column is nullable and the insert
    still proceeds, so the run is recorded either way.
    """
    try:
        if node_results is None:
            return None, {
                "status": "absent",
                "reason": "no_node_results_supplied",
                "redacted": False,
                "truncated": False,
                "redaction_version": REDACTION_VERSION,
            }

        if not isinstance(node_results, list):
            report = _Report()
            safe = _walk(node_results, "", "", 1, report, _TRACE_STRUCTURAL_KEYS)
            meta = {
                "status": "captured",
                "reason": "node_results_not_a_list",
                "observed_type": type(node_results).__name__,
                "redacted": bool(report.redactions),
                "truncated": bool(report.truncations),
                "redaction_version": REDACTION_VERSION,
            }
            meta.update(_meta_from_report(report))
            return safe, meta

        if len(node_results) == 0:
            return [], {
                "status": "empty",
                "reason": "no_steps_executed",
                "redacted": False,
                "truncated": False,
                "entry_count": 0,
                "redaction_version": REDACTION_VERSION,
            }

        report = _Report()
        safe = [
            _walk(entry, f"[{i}]", "", 1, report, _TRACE_STRUCTURAL_KEYS)
            for i, entry in enumerate(node_results)
        ]

        meta: dict[str, Any] = {
            "status": "captured",
            "redacted": bool(report.redactions),
            "truncated": bool(report.truncations),
            "entry_count": len(safe),
            "redaction_version": REDACTION_VERSION,
        }
        meta.update(_meta_from_report(report))
        return safe, meta

    except BaseException as exc:  # noqa: BLE001 — capture must never break a run
        try:
            reason_type = type(exc).__name__
        except BaseException:
            reason_type = "unknown"
        return None, {
            "status": "unavailable",
            "reason": "capture_failed",
            "error_type": reason_type,
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }


def persist_node_results(
    node_results: Any,
    variables_capture: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """THE persist-time boundary for `workflow_runs.node_results`.

    Call this at the INSERT. The list passed in is NOT modified: `_walk` builds
    new containers and never writes to its input, so the in-memory trace that
    drove execution — and the trace already streamed to the caller — stays
    verbatim. Only the database copy is redacted. NEVER RAISES.
    """
    try:
        stored, meta = capture_node_results(node_results)
    except BaseException:  # noqa: BLE001 — belt and braces
        stored, meta = None, {
            "status": "unavailable",
            "reason": "capture_failed",
            "redacted": False,
            "truncated": False,
            "redaction_version": REDACTION_VERSION,
        }
    try:
        out = dict(variables_capture) if isinstance(variables_capture, dict) else {}
        out[NODE_RESULTS_CAPTURE_KEY] = meta
        return stored, out
    except BaseException:  # noqa: BLE001
        return stored, {NODE_RESULTS_CAPTURE_KEY: meta}
