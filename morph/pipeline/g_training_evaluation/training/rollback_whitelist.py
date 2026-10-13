"""Primitive-based rollback exemptions for committed online adaptations."""

from collections.abc import Mapping
from typing import Any

from .adaptation_program import (
    ADD_ADAPTIVE_FOURIER_BRANCH,
    ADD_CORRECTION_BRANCH,
    ADD_FOURIER_BRANCH,
    ADD_LOCALIZED_EXPERT,
    ADD_RESIDUAL_BRANCH,
    ADAPT_SAMPLING_DISTRIBUTION,
    BALANCE_COMPONENT_GRADIENTS,
    EMPHASIZE_CONSTRAINTS,
    MITIGATE_GRADIENT_CONFLICT,
    PROJECT_CONFLICTING_GRADIENTS,
    REBALANCE_LOSS_WEIGHTS,
    REWEIGHT_RESIDUAL_REGIONS,
)


ROLLBACK_WHITELIST_GROUPS = {
    "Loss Reweighting": frozenset({
        REBALANCE_LOSS_WEIGHTS, REWEIGHT_RESIDUAL_REGIONS, EMPHASIZE_CONSTRAINTS,
    }),
    "Adaptive Sampling": frozenset({ADAPT_SAMPLING_DISTRIBUTION}),
    "Gradient Adjustment": frozenset({
        BALANCE_COMPONENT_GRADIENTS, PROJECT_CONFLICTING_GRADIENTS,
        MITIGATE_GRADIENT_CONFLICT,
    }),
    "Residual/Fourier Branch": frozenset({
        ADD_RESIDUAL_BRANCH, ADD_CORRECTION_BRANCH, ADD_FOURIER_BRANCH,
        ADD_ADAPTIVE_FOURIER_BRANCH,
    }),
    "Localized Expert": frozenset({ADD_LOCALIZED_EXPERT}),
}
ROLLBACK_WHITELIST_PRIMITIVES = frozenset().union(
    *ROLLBACK_WHITELIST_GROUPS.values()
)
ROLLBACK_ACTION_WHITELIST = frozenset({
    "replace_sampling_subset", "update_sampling_allocation", "update_loss_weights",
})

# These older action families are unambiguous even without their primitive list.
_LEGACY_ACTION_PRIMITIVES = {
    "OBJECTIVE_REBALANCING": REBALANCE_LOSS_WEIGHTS,
    "REGION_AWARE_REBALANCING": REWEIGHT_RESIDUAL_REGIONS,
    "CONSTRAINT_EMPHASIS": EMPHASIZE_CONSTRAINTS,
    "GRADIENT_SURGERY": PROJECT_CONFLICTING_GRADIENTS,
    "GRADIENT_CONFLICT_MITIGATION": MITIGATE_GRADIENT_CONFLICT,
    "SAMPLING_ADAPTATION": ADAPT_SAMPLING_DISTRIBUTION,
    "SPECTRAL_ADAPTATION": ADD_ADAPTIVE_FOURIER_BRANCH,
    "LOCAL_SPECIALIZATION": ADD_LOCALIZED_EXPERT,
}


def executed_primitive_names(value: Any) -> frozenset[str]:
    """Prefer committed executor operations over an AUTO policy's requested family."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return frozenset()
    if value.get("executed_primitives"):
        return frozenset(str(name) for name in value["executed_primitives"])
    for key in ("architecture_instruction_audit", "objective_instruction_audit", "sampling_instruction_audit"):
        audit = value.get(key) or {}
        if isinstance(audit, Mapping) and audit.get("executed_operations"):
            return frozenset(
                str(operation.get("primitive") or operation.get("action") or "")
                for operation in audit["executed_operations"]
                if isinstance(operation, Mapping)
            )
    payload = value.get("adaptation_program") or value.get("program") or value
    if not isinstance(payload, Mapping):
        return frozenset()
    operations = payload.get("operations") or ()
    names = {
        str(operation.get("primitive") or "")
        for operation in operations if isinstance(operation, Mapping)
    }
    if not names:
        mechanism = str(payload.get("mechanism") or "")
        action = str(payload.get("action") or value.get("operator") or "")
        primitive = (
            mechanism if mechanism in ROLLBACK_WHITELIST_PRIMITIVES
            else _LEGACY_ACTION_PRIMITIVES.get(action)
        )
        names = {primitive} if primitive else set()
    return frozenset(names)


def rollback_whitelist_groups(value: Any) -> tuple[str, ...]:
    """Exempt complete programs only; mixed protected/unprotected actions stay guarded."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping) and value.get("action_type") in ROLLBACK_ACTION_WHITELIST:
        group = (
            "Loss Reweighting" if value["action_type"] == "update_loss_weights"
            else "Adaptive Sampling"
        )
        return (group,)
    names = executed_primitive_names(value)
    if not names or not names.issubset(ROLLBACK_WHITELIST_PRIMITIVES):
        return ()
    return tuple(
        group for group, primitives in ROLLBACK_WHITELIST_GROUPS.items()
        if names.intersection(primitives)
    )


def is_rollback_whitelisted(value: Any) -> bool:
    return bool(rollback_whitelist_groups(value))
