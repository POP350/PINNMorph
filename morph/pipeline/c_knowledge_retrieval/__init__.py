"""Static PINNacle PDE capability registry used by problem adapters."""

from .pinnacle_registry import (
    get_pinnacle_profile,
    list_pinnacle_problems,
    pinnacle_registry_summary,
)
from .execution_evidence import build_online_execution_evidence
from .prior_knowledge import (
    resolve_legacy_knowledge_base,
    retrieve_online_prior_context,
)

__all__ = [
    "get_pinnacle_profile",
    "list_pinnacle_problems",
    "pinnacle_registry_summary",
    "build_online_execution_evidence",
    "resolve_legacy_knowledge_base",
    "retrieve_online_prior_context",
]
