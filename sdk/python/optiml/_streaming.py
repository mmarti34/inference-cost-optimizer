"""Server-Sent Events (SSE) stream parser."""

from __future__ import annotations

import json
from typing import Iterator

from .types import StreamEvent


def parse_sse_stream(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Parse SSE lines into StreamEvent objects.

    Expects the standard SSE format:
        event: token
        data: {"delta": "Hello"}

        event: done
        data: [DONE]
    """
    current_event = ""
    for line in lines:
        line = line.rstrip("\n").rstrip("\r")
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                return
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            yield StreamEvent(event=current_event or "message", data=data)
        elif line == "":
            # Empty line is the SSE event boundary; reset event type.
            current_event = ""
