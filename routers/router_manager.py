"""
Router manager - selects and manages routing engines based on configuration.
"""
import os
from routers.base import Router, BaselineRouter, ExperimentalRouter


def get_router() -> Router:
    """
    Get the configured router based on OPTIML_ROUTER environment variable.
    Defaults to baseline router.
    """
    router_type = os.getenv("OPTIML_ROUTER", "baseline").lower()
    
    if router_type == "experimental":
        config_path = os.getenv("OPTIML_ROUTER_CONFIG_PATH")
        return ExperimentalRouter(config_path=config_path)
    else:
        return BaselineRouter()

