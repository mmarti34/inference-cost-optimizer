"""Synchronous client for the OptiML API."""

from __future__ import annotations

import json
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

#: OptiML's public API origin. This is the stable, documented hostname — NOT
#: the underlying Railway origin, which is an implementation detail that can be
#: re-pointed without notice.
_DEFAULT_BASE_URL = "https://api.optiml.one"


class OptiMLClient:
    """Synchronous client for the OptiML API.

    Usage::

        from optiml import OptiMLClient

        client = OptiMLClient(api_key="sk-...", org="my-org")
        result = client.run("my-endpoint", input_text="Hello")
        print(result.final_output)

    For **direct inference** — sending your existing chat traffic through
    OptiML without building a workflow — you do not need an ``org``::

        client = OptiMLClient(api_key="sk-...")
        reply = client.chat("gpt-4o", [{"role": "user", "content": "Hello"}])

    Most applications should skip this SDK for direct inference entirely and
    point the official OpenAI client at :meth:`openai_base_url`; that path needs
    no OptiML-specific code at all.
    """

    def __init__(
        self,
        api_key: str,
        org: Optional[str] = None,
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
    def _error_message(response: httpx.Response) -> str:
        """
        Best-effort human-readable message.

        ``/v1/chat/completions`` replies in the OpenAI error envelope
        (``{"error": {"message": ...}}``) and the native API replies with
        ``{"detail": ...}``; surface whichever is present rather than a wall of
        raw JSON.
        """
        try:
            payload = response.json()
        except Exception:
            return response.text[:500]
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
            if isinstance(err, str):
                return err[:500]
            if payload.get("detail"):
                return str(payload["detail"])[:500]
        return response.text[:500]

    @staticmethod
    def _handle_error(response: httpx.Response) -> None:
        """Map HTTP errors to typed exceptions."""
        body = OptiMLClient._error_message(response)
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

    def _require_org(self) -> str:
        """Workflow calls are addressed by org slug; direct inference is not."""
        if not self.org:
            raise OptiMLError(
                "This call targets a deployed workflow and needs an org slug: "
                "OptiMLClient(api_key=..., org='my-org'). Direct inference "
                "(client.chat / the OpenAI SDK) does not need one.",
                400,
            )
        return self.org

    @property
    def openai_base_url(self) -> str:
        """
        The base URL to hand to an OpenAI-compatible client.

        This is the whole direct-inference integration::

            from openai import OpenAI
            oai = OpenAI(base_url=client.openai_base_url, api_key="<optiml_service_key>")
        """
        return f"{self.base_url}/v1"

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

        url = f"/api/public/{self._require_org()}/{endpoint}"
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

        url = f"/api/public/{self._require_org()}/{endpoint}"
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
        url = f"/api/public/{self._require_org()}/{endpoint}/feedback"
        resp = self._client.post(url, json={
            "request_id": request_id,
            "metrics": metrics,
        })
        if resp.status_code >= 400:
            self._handle_error(resp)
        return FeedbackResponse(**resp.json())


    # ------------------------------------------------------------------
    # Direct inference — OpenAI-compatible passthrough
    # ------------------------------------------------------------------

    def _chat_body(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        workload: Optional[str],
        user_id: Optional[str],
        conversation_id: Optional[str],
        experiment_tags: Optional[list[str]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        body.update({k: v for k, v in extra.items() if v is not None})
        optiml = {
            k: v
            for k, v in {
                "workload": workload,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "experiment_tags": experiment_tags,
            }.items()
            if v
        }
        if optiml:
            metadata = dict(body.get("metadata") or {})
            metadata["optiml"] = optiml
            body["metadata"] = metadata
        return body

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        workload: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        experiment_tags: Optional[list[str]] = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Chat completion through OptiML, in OpenAI request/response shape.

        Two modes, decided entirely by *model*:

        * ``"gpt-4o"``, ``"claude-sonnet-4-5"``, … — **direct inference**. The
          call runs against that provider using your own provider key, and
          OptiML observes cost, latency, workload and strategy. No workflow and
          no deployment are involved.
        * ``"optiml/<endpoint_slug>"`` — an OptiML **deployed workflow**.

        A bare model id is ALWAYS direct inference; it never resolves to a
        workflow. The ``optiml/`` namespace is reserved.

        Every OptiML field is optional. Without them, OptiML still identifies
        the workload structurally from the model, system prompt, tool signature
        and response format.

        Args:
            model: Provider model id, or ``optiml/<endpoint_slug>``.
            messages: OpenAI ``messages`` array.
            workload: Your own name for this workload, e.g. ``"support-refund"``.
            user_id: Your end user's id (not an OptiML user).
            conversation_id: Groups multi-turn traffic.
            experiment_tags: Free-form tags for your own experiments.
            **params: Any other OpenAI parameter — ``temperature``,
                ``max_tokens``, ``tools``, ``response_format``, … — forwarded
                to the provider unchanged. On ``optiml/<slug>`` these are
                REFUSED rather than accepted-and-discarded: a workflow's
                parameters come from its graph.

        Returns:
            The OpenAI ``chat.completion`` object, plus an ``optiml`` block with
            the request id, resolved workload and cost.
        """
        body = self._chat_body(
            model, messages, stream=False, workload=workload, user_id=user_id,
            conversation_id=conversation_id, experiment_tags=experiment_tags, extra=params,
        )
        resp = self._client.post("/v1/chat/completions", json=body)
        if resp.status_code >= 400:
            self._handle_error(resp)
        return resp.json()

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        workload: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        experiment_tags: Optional[list[str]] = None,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        """Streaming chat completion. Yields OpenAI ``chat.completion.chunk`` dicts.

        Usage::

            for chunk in client.chat_stream("gpt-4o", [{"role": "user", "content": "hi"}]):
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)

        The terminal ``[DONE]`` sentinel is consumed, not yielded. If the
        provider fails mid-stream the HTTP status is already 200, so the failure
        arrives as a chunk containing an ``error`` key — check for it rather
        than assuming every chunk has ``choices``.
        """
        body = self._chat_body(
            model, messages, stream=True, workload=workload, user_id=user_id,
            conversation_id=conversation_id, experiment_tags=experiment_tags, extra=params,
        )
        with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._handle_error(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except ValueError:
                    continue

    def models(self) -> list[dict[str, Any]]:
        """List callable models: provider models plus ``optiml/<slug>`` deployments.

        Each entry carries ``optiml_mode`` (``"direct"`` or ``"workflow"``) so
        the two surfaces are distinguishable without string matching.
        """
        resp = self._client.get("/v1/models")
        if resp.status_code >= 400:
            self._handle_error(resp)
        return resp.json().get("data", [])

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
