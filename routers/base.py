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


class BaselineRouter(Router):
    """
    Baseline heuristic router using:
    - Capabilities filters
    - Cost estimates
    - Latency priors
    """
    
    def __init__(self):
        self._name = "baseline"
        self._version = "1.0.0"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def select_model(self, route_input: RouteInput) -> RouteDecision:
        """
        Heuristic routing based on prompt characteristics and requirements.
        """
        from utils.pricing import get_pricing, suggest_model
        
        # If provider/model are explicitly specified, use them
        if route_input.provider and route_input.model:
            return RouteDecision(
                provider=route_input.provider.lower(),
                model=route_input.model,
                fallback_chain=[],
                parameters={},
                reason_code="explicit_provider_model_specified",
                route_policy_name=self.name,
                route_policy_version=self.version
            )
        
        # If only provider is specified, select cheapest model from that provider
        if route_input.provider:
            provider = route_input.provider.lower()
            # Select cheapest model for the provider
            if provider == "openai":
                model = "gpt-3.5-turbo"
            elif provider == "anthropic":
                model = "claude-3-haiku"
            elif provider == "mistral":
                model = "mistral-small"
            elif provider == "cohere":
                model = "command-r"
            elif provider == "gemini":
                model = "gemini-1.5-flash"
            else:
                model = "gpt-3.5-turbo"  # Default fallback
            
            return RouteDecision(
                provider=provider,
                model=model,
                fallback_chain=[],
                parameters={},
                reason_code=f"provider_specified_cheapest_model_selected",
                route_policy_name=self.name,
                route_policy_version=self.version
            )
        
        # Use existing suggest_model heuristic
        suggestion = suggest_model(route_input.prompt)
        provider = suggestion.get("provider", "openai")
        model = suggestion.get("model", "gpt-3.5-turbo")
        
        # Build fallback chain (cheaper alternatives)
        fallback_chain = []
        if provider == "openai" and model != "gpt-3.5-turbo":
            fallback_chain.append({"provider": "openai", "model": "gpt-3.5-turbo"})
        elif provider == "anthropic" and model != "claude-3-haiku":
            fallback_chain.append({"provider": "anthropic", "model": "claude-3-haiku"})
        
        # Add cross-provider fallback (cheapest option)
        if provider != "openai":
            fallback_chain.append({"provider": "openai", "model": "gpt-3.5-turbo"})
        
        # Check capabilities requirements
        reason_code = "heuristic_cost_optimized"
        if route_input.capabilities_required:
            if "json" in route_input.capabilities_required:
                # Prefer models with better JSON mode
                if provider == "openai" and model == "gpt-3.5-turbo":
                    model = "gpt-4-turbo"  # Better JSON support
                    reason_code = "json_capability_required"
        
        # Apply cost budget constraint if specified
        if route_input.cost_budget:
            pricing = get_pricing(provider, model)
            estimated_input_tokens = len(route_input.prompt.split()) * 1.3  # Rough estimate
            estimated_output_tokens = route_input.max_tokens or 500
            estimated_cost = (estimated_input_tokens * pricing["input"] + estimated_output_tokens * pricing["output"]) / 1000
            
            if estimated_cost > route_input.cost_budget:
                # Switch to cheaper model
                if provider == "openai" and model != "gpt-3.5-turbo":
                    model = "gpt-3.5-turbo"
                    reason_code = "cost_budget_constraint"
                elif provider == "anthropic" and model != "claude-3-haiku":
                    model = "claude-3-haiku"
                    reason_code = "cost_budget_constraint"
        
        return RouteDecision(
            provider=provider,
            model=model,
            fallback_chain=fallback_chain,
            parameters={
                "temperature": route_input.temperature or 0.7,
                "max_tokens": route_input.max_tokens or 1024
            },
            reason_code=reason_code,
            route_policy_name=self.name,
            route_policy_version=self.version
        )


class ExperimentalRouter(Router):
    """
    Experimental router stub for future research integrations.
    Can load JSON/DB config and call learned predictors or bandit policies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._name = "experimental"
        self._version = "1.0.0"
        self.config_path = config_path
        # TODO: Load config from JSON/DB
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def select_model(self, route_input: RouteInput) -> RouteDecision:
        """
        Experimental routing - currently falls back to baseline.
        TODO: Implement learned predictor or bandit policy.
        """
        # For now, delegate to baseline
        baseline = BaselineRouter()
        decision = baseline.select_model(route_input)
        decision.route_policy_name = self.name
        decision.route_policy_version = self.version
        decision.reason_code = "experimental_fallback_to_baseline"
        return decision

