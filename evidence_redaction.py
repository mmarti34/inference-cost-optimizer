"""
Write-time redaction for production input variables (evidence capture).

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

Nothing in this module performs I/O, imports the database, reads configuration
or logs a value. It is pure and it is import-safe.
"""
from __future__ import annotations

import math
import re
from typing import Any

# Redaction ruleset version. Bump when patterns change so a downstream reader
# can tell which ruleset produced a given row.
REDACTION_VERSION = 1

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

# US SSN in its punctuated form, with the administratively-invalid ranges
# excluded (000/666/9xx area, 00 group, 0000 serial). Strong enough to match on
# shape alone.
_SSN_RE = re.compile(
    r"(?<!\d)(?!000|666|9\d\d)\d{3}([ \-])(?!00)\d{2}\1(?!0000)\d{4}(?!\d)"
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


def _walk(value: Any, path: str, key_hint: str, depth: int, report: _Report) -> Any:
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
            out[key_str] = _walk(v, child_path, key_str, depth + 1, report)
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
            out_list.append(_walk(v, f"{path}[{i}]", key_hint, depth + 1, report))
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
