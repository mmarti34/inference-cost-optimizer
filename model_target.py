"""
ModelTarget: typed resolution object for workflow execution.

When a workflow node needs to call an LLM, the runtime resolves the node's
configuration into a ModelTarget.  This object carries everything the router
layer needs — provider, model name, optional base_url override, auth, and
extra headers — without leaking secrets to the frontend or logs.

Two target types:
  1. provider_model   — a model on a known provider (OpenAI, Anthropic, …).
                        May be a stock model or a fine-tune ID.
  2. openai_compatible_endpoint — a custom endpoint (vLLM, Anyscale, Together
                        self-hosted, etc.) that speaks the OpenAI chat-completions
                        API.  Reuses the openai_router formatting logic with a
                        custom base_url and credentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class ModelTarget:
    """Immutable resolution object passed from runtime → router."""

    target_type: Literal["provider_model", "openai_compatible_endpoint"]

    # Which router module to use (always lowercase).
    # For provider_model: "openai", "anthropic", etc.
    # For openai_compatible_endpoint: always "openai" (reuse formatting).
    provider: str

    # The model identifier sent to the API  (e.g. "gpt-4o", "ft:gpt-3.5-turbo:org:my-ft:abc123").
    model_name: str

    # --- fields only used for openai_compatible_endpoint ---
    base_url: Optional[str] = None
    auth_header_name: Optional[str] = None      # e.g. "Authorization" or "X-Api-Key"
    auth_header_value: Optional[str] = None      # DECRYPTED at resolution time; never log/serialize
    default_headers: dict[str, str] = field(default_factory=dict)

    # --- provenance (for logging / observability) ---
    model_registry_id: Optional[str] = None
    endpoint_id: Optional[str] = None

    def log_safe_dict(self) -> dict:
        """Return a dict safe for logging (no secrets)."""
        d: dict = {
            "target_type": self.target_type,
            "provider": self.provider,
            "model_name": self.model_name,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        if self.model_registry_id:
            d["model_registry_id"] = self.model_registry_id
        if self.endpoint_id:
            d["endpoint_id"] = self.endpoint_id
        # Never include auth_header_value or default_headers (may contain tokens)
        return d
