"""
Observability middleware for request ID and trace ID generation.
"""
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from utils.observability import generate_request_id, generate_trace_id


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request_id and trace_id to every request.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate IDs
        request_id = generate_request_id()
        trace_id = generate_trace_id()
        
        # Add to request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        
        return response

