"""
Base router interface for pluggable routing engine architecture.
Enables integration of emerging routing research (quality predictors, bandits, offline replay)
without rewriting the API gateway.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class Outcome(str, Enum):
    """Request outcome status"""
    SUCCESS = "success"
    FALLBACK_SUCCESS = "fallback_success"
    FAILURE = "failure"
    REFUSED = "refused"


class FailureReason(str, Enum):
    """Failure reason codes"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    PROVIDER_5XX = "provider_5xx"
    INVALID_JSON = "invalid_json"
    TOOL_ERROR = "tool_error"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class RouteInput(BaseModel):
    """Input for routing decision"""
    prompt: str
    model: Optional[str] = None  # If None, router should select
    provider: Optional[str] = None  # If None, router should select
    org_id: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    prompt_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Additional context for routing
    capabilities_required: Optional[List[str]] = None  # e.g., ["json", "function_calling"]
    cost_budget: Optional[float] = None
    latency_budget_ms: Optional[int] = None


class RouteDecision(BaseModel):
    """Routing decision output"""
    provider: str
    model: str
    fallback_chain: List[Dict[str, str]] = []  # Ordered list of {provider, model} for fallbacks
    parameters: Dict[str, Any] = {}  # Additional parameters (temperature, max_tokens, etc.)
    reason_code: str  # Human-readable reason for the decision
    route_policy_name: str  # Name of the routing policy used
    route_policy_version: str  # Version of the routing policy


class Router(ABC):
    """Base interface for all routing engines"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the router (e.g., 'baseline', 'experimental')"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the router"""
        pass
    
    @abstractmethod
    def select_model(self, route_input: RouteInput) -> RouteDecision:
        """
        Select the best model/provider combination for the given input.
        
        Args:
            route_input: Input parameters for routing decision
            
        Returns:
            RouteDecision with chosen provider/model and fallback chain
        """
        pass


# ---------------------------------------------------------------------------
# REMOVED: BaselineRouter and ExperimentalRouter.
#
# BaselineRouter was a heuristic that ran on the live /v1/prompt path and whose
# decision was then discarded — main.py built the provider call from the prompt
# template, not from the decision. Its only lasting effect was stamping a
# reason_code such as "heuristic_cost_optimized" onto request_logs rows for
# calls where no routing decision was ever made, which made that column
# unanalysable. Its model table was also stale (gpt-3.5-turbo, claude-3-haiku,
# gemini-1.5-flash) and one reason_code was an f-string with no placeholders.
#
# ExperimentalRouter was a stub that delegated straight back to BaselineRouter
# with reason_code="experimental_fallback_to_baseline", reachable only via
# OPTIML_ROUTER=experimental.
#
# routers/router_manager.py (get_router()) went with them; it existed only to
# choose between the two.
#
# The Router / RouteInput / RouteDecision interfaces above are kept: they are
# the extension point a real, evidence-based optimizer should implement. When
# one exists, wire its decision into the actual provider call — do not
# reintroduce a decision that is computed and thrown away.
# ---------------------------------------------------------------------------
