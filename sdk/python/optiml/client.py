"""Synchronous client for the OptiML API."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import httpx

from ._streaming import parse_sse_stream
from ._version import __version__
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    OptiMLError,
    RateLimitError,
)
from .types import FeedbackResponse, StreamEvent, WorkflowResponse

_DEFAULT_BASE_URL = "https://optiml-production.up.railway.app"


class OptiMLClient:
    """Synchronous client for the OptiML API.

    Usage::

        from optiml import OptiMLClient

        client = OptiMLClient(api_key="sk-...", org="my-org")
        result = client.run("my-endpoint", input_text="Hello")
        print(result.final_output)
    """

    def __init__(
        self,
        api_key: str,
        org: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.org = org
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"optiml-python/{__version__}",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_error(response: httpx.Response) -> None:
        """Map HTTP errors to typed exceptions."""
        body = response.text[:500]
        code = response.status_code
        if code == 401:
            raise AuthenticationError(body)
        if code == 403:
            raise AuthorizationError(body)
        if code == 404:
            raise NotFoundError(body)
        if code == 429:
            retry = response.headers.get("Retry-After")
            raise RateLimitError(body, int(retry) if retry and retry.isdigit() else None)
        if code >= 400:
            raise OptiMLError(body, code)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run(
        self,
        endpoint: str,
        *,
        input_text: str = "",
        variables: Optional[dict[str, Any]] = None,
        version: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> WorkflowResponse:
        """Execute a deployed workflow.

        Args:
            endpoint: The endpoint slug (e.g. ``"summarize"``, ``"chat"``).
            input_text: Simple text input. Use *variables* for multi-field workflows.
            variables: Dict of workflow variables (takes precedence over *input_text*).
            version: Pin to a specific deployment version (e.g. ``3``).
            conversation_id: UUID for multi-turn conversations.

        Returns:
            A :class:`WorkflowResponse` with ``final_output``, ``total_cost``, etc.
        """
        body: dict[str, Any] = {"stream": False}
        if variables:
            body["variables"] = variables
        else:
            body["input_text"] = input_text
        if conversation_id:
            body["conversation_id"] = conversation_id

        params: dict[str, str] = {}
        if version is not None:
            params["version"] = f"v{version}"

        url = f"/api/public/{self.org}/{endpoint}"
        resp = self._client.post(url, json=body, params=params)
        if resp.status_code >= 400:
            self._handle_error(resp)
        return WorkflowResponse(**resp.json())

    def stream(
        self,
        endpoint: str,
        *,
        input_text: str = "",
        variables: Optional[dict[str, Any]] = None,
        version: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        """Execute a deployed workflow with streaming.

        Yields :class:`StreamEvent` objects as they arrive.

        Usage::

            for event in client.stream("my-endpoint", input_text="Hello"):
                if event.event == "token":
                    print(event.data.get("delta", ""), end="")
        """
        body: dict[str, Any] = {"stream": True}
        if variables:
            body["variables"] = variables
        else:
            body["input_text"] = input_text
        if conversation_id:
            body["conversation_id"] = conversation_id

        params: dict[str, str] = {}
        if version is not None:
            params["version"] = f"v{version}"

        url = f"/api/public/{self.org}/{endpoint}"
        with self._client.stream("POST", url, json=body, params=params) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._handle_error(resp)
            yield from parse_sse_stream(resp.iter_lines())

    def feedback(
        self,
        endpoint: str,
        request_id: str,
        metrics: dict[str, Any],
    ) -> FeedbackResponse:
        """Submit feedback/custom metrics for a previous request.

        Args:
            endpoint: The endpoint slug.
            request_id: The ``request_id`` from a previous :meth:`run` response.
            metrics: Dict of custom metrics (e.g. ``{"helpfulness_score": 4}``).
        """
        url = f"/api/public/{self.org}/{endpoint}/feedback"
        resp = self._client.post(url, json={
            "request_id": request_id,
            "metrics": metrics,
        })
        if resp.status_code >= 400:
            self._handle_error(resp)
        return FeedbackResponse(**resp.json())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OptiMLClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
