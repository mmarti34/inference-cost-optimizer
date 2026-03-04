"""Typed exceptions for the OptiML SDK."""

from __future__ import annotations


class OptiMLError(Exception):
    """Base exception for all OptiML SDK errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(OptiMLError):
    """401 — Invalid or missing API key."""

    def __init__(self, message: str = "Invalid or missing API key."):
        super().__init__(message, 401)


class AuthorizationError(OptiMLError):
    """403 — API key does not belong to this organization."""

    def __init__(self, message: str = "API key does not belong to this organization."):
        super().__init__(message, 403)


class NotFoundError(OptiMLError):
    """404 — Endpoint or organization not found."""

    def __init__(self, message: str = "Not found."):
        super().__init__(message, 404)


class RateLimitError(OptiMLError):
    """429 — Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded.", retry_after: int | None = None):
        super().__init__(message, 429)
        self.retry_after = retry_after
