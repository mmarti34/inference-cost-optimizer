"""OptiML Python SDK.

Two ways in:

* **Direct inference** — point an OpenAI-compatible client at
  ``https://api.optiml.one/v1`` (or use :meth:`OptiMLClient.chat`) and keep your
  existing application exactly as it is. No workflow, no migration.
* **Deployed workflows** — call a Studio deployment by endpoint slug with
  :meth:`OptiMLClient.run`.
"""

from .client import OptiMLClient
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    OptiMLError,
    RateLimitError,
)
from .types import FeedbackResponse, StreamEvent, WorkflowResponse
from ._version import __version__

__all__ = [
    "OptiMLClient",
    "WorkflowResponse",
    "StreamEvent",
    "FeedbackResponse",
    "OptiMLError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "__version__",
]
