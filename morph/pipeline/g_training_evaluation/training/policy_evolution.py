"""Unified EOH population for one continuously trained PINN.

The population contains five homogeneous policy candidates.  At a decision
point every candidate proposes one declarative :class:`AdaptationProgram`; the
population selects exactly one program from execution-grounded evidence.
"""

from __future__ import annotations

from .adaptive_strategy import bottleneck_contract, program_rejection

from copy import deepcopy
import hashlib
import json
import math
import random
from statistics import median
from typing import Any, Mapping, Sequence

from morph.pipeline.d_algorithm_generation.llm.json_extractor import (
    extract_json_object,
)

from .adaptation_program import (
    ADD_FOURIER_BRANCH,
    ADD_LAYER,
    ADD_LOCALIZED_EXPERT,
    ACTION_TARGETS,
    AUTO_MECHANISM,
    SAMPLING_ACTIONS,
    OPTIMIZATION_ACTIONS,
    ONLINE_ADAPTATION_ACTIONS,
    OBJECTIVE_ACTIONS,
    OBJECTIVE_REBALANCING,
    GRADIENT_CONFLICT_MITIGATION,
    REBALANCE_LOSS_WEIGHTS,
    MITIGATE_GRADIENT_CONFLICT,
    CAPABILITY_REGISTRY,
    CONSERVATIVE_GROWTH,
    CONDITION_STATE_PATHS,
    EXPLORATORY_MORPH,
    KEEP,
    LOCAL_REINIT,
    LOCAL_SPECIALIZATION,
    MAXIMUM_CONDITION_COUNT,
    MAXIMUM_DIAGNOSIS_CHARS,
    MAXIMUM_REASON_CHARS,
    MECHANISMS_BY_ACTION,
    PROGRAM_TRIGGER_FIELDS,
    PRUNE,
    REDISTRIBUTE,
    REALLOCATION,
    REPRESENTATION_AUGMENTATION,
    REGROW,
    TARGET_REGIONS,
    TARGET_SELECTORS,
    WIDEN,
    AdaptationProgram,
    PolicyCandidate,
    ProgramProposal,
    candidate_applicability_reason,
    conditions_rejection_reason,
)

from dataclasses import asdict, replace
import time

import torch

from .intervention_tracker import FEEDBACK_HORIZONS, InterventionTracker
from .online_diagnostics import (
    OnlineMorphDiagnostics,
    build_budget_diagnostics,
    build_frequency_diagnostics,
    build_gradient_diagnostics,
    build_optimization_diagnostics,
    build_physics_health_diagnostics,
    build_spatial_residual_diagnostics,
)
from .online_morph import (
    PRIMITIVE_LIBRARY,
    DeterministicArchitectureExecutor,
    InterventionClass,
    OnlineMorphAction,
    OnlineMorphDecisionProvider,
    SafeExecutionEngine,
)
from .validation_probe import fit_log_linear_continuation


POLICY_POPULATION_SIZE = 5
KEEP_CONTINUATION_BASELINE_POLICY_ID = "__keep_continuation_baseline__"
DEFAULT_KEEP_BASELINE_MARGIN = 0.05
PROMPT_HISTORY_MAX_CHARS = 12_000
EVOLUTION_MODES = (
    "refine",
    "physics_guided",
    "architecture_guided",
    "synthesis",
    "novelty",
)

# These are mutation operators over policies, not labels for architecture
# actions.  Keeping their contracts here makes every generation prompt state
# what must be inherited and what evidence is allowed to drive a mutation.
EVOLUTION_MODE_GUIDANCE = {
    "refine": (
        "Refine one execution-grounded parent. Preserve its useful decision "
        "logic and make a small, evidence-backed mutation (normally trigger, "
        "target, strength, recovery, or cooldown). Do not rewrite the policy "
        "from scratch merely to appear different. This role owns local "
        "exploitation; it should not imitate a more radical peer."
    ),
    "physics_guided": (
        "Mutate one parent from PDE/BC/IC residual trajectories, gradient "
        "interaction, frequency evidence, and the dominant physical failure. "
        "Explain which physical signal requires each changed policy field. "
        "This role owns a physics-failure hypothesis, not generic capacity "
        "growth when a more specific spatial or spectral response is supported."
    ),
    "architecture_guided": (
        "Mutate one parent using current topology, parameter allocation, layer "
        "utility, and causal outcomes of earlier morphs. Change mechanism or "
        "target only when architecture evidence supports that change. This "
        "role owns the resource-allocation/topology hypothesis and should not "
        "copy the physics-guided response without independent architecture evidence."
    ),
    "synthesis": (
        "Recombine two assigned complementary parents. State in the policy "
        "reason which useful decision logic comes from each parent; resolve "
        "conflicts using real physics feedback rather than averaging fields. "
        "Prefer a coherent ordered multi-operation program when both mechanisms "
        "are compatible; otherwise make the tradeoff explicit."
    ),
    "novelty": (
        "Use the assigned anchor parent and identify a state, response, or "
        "registered capability that the current population does not cover. "
        "Explore that evidence-backed gap; novelty is not random difference "
        "and may retain the parent's action only when evidence strongly requires "
        "convergence. Search the registered capability menu for an overlooked "
        "but physically defensible hypothesis before returning KEEP."
    ),
}

_CAPABILITY_SCIENTIFIC_GUIDE = {
    "WIDEN": "increase capacity along the current representation when gradients remain usable",
    "REGROW": "restore capacity after pruning or under-allocation",
    "ADD_LAYER": "increase compositional depth when a topology bottleneck is plausible",
    "LOCAL_REINIT": "replace a low-utility local topology under persistent gradient conflict",
    "PRUNE": "remove low-utility capacity",
    "REDISTRIBUTE": "move capacity between hidden regions at roughly fixed total size",
    "ADD_RESIDUAL_BRANCH": "add a zero-output residual correction while preserving the current function",
    "ADD_CORRECTION_BRANCH": "add a zero-output global input-to-output correction branch",
    "ADD_FOURIER_BRANCH": "add deterministic multi-band sine/cosine features for spectral bias or oscillatory physics",
    "ADD_GATED_BRANCH": "add a zero-output gated representation branch for difficult feature interactions",
    "ADD_LOCALIZED_EXPERT": "add zero-output overlapping experts centered from localized PDE residual evidence",
    "ADD_ADAPTIVE_FOURIER_BRANCH": "add trainable multi-band Fourier features, optionally localized around high-residual evidence",
    "ADD_COMPONENT_EXPERT": "add function-preserving capacity for one coupled output component",
    "REALLOCATE_COMPONENT_CAPACITY": "move a fixed parameter budget from low-utility shared units into one under-served output component",
    "SPLIT_CONFLICTING_TOPOLOGY": "add a gated component branch when shared gradients remain persistently conflicting",
    "REBALANCE_LOSS_WEIGHTS": "rebalance PDE/BC/IC weights from persistent registered component-gradient imbalance",
    "BALANCE_COMPONENT_GRADIENTS": "balance registered component weights from their measured gradient norms",
    "REWEIGHT_RESIDUAL_REGIONS": "emphasize registered residual components associated with a difficult interface, boundary, or high-residual region",
    "PROJECT_CONFLICTING_GRADIENTS": "activate deterministic PCGrad-style projection for conflicting registered physics components",
    "EMPHASIZE_CONSTRAINTS": "protect boundary and initial constraints when PDE gradients starve them",
    "MITIGATE_GRADIENT_CONFLICT": "apply bounded constraint-protective weighting when PDE and constraint gradients conflict",
    "ADAPT_SAMPLING_DISTRIBUTION": "replace a fixed-budget subset with high-residual and exploratory collocation points",
    "ENTER_LBFGS_PHASE": "enter the registered second-order polishing phase when Adam shows a genuine physics plateau with stable gradients; do not use before there is plateau evidence",
}

# The generation prompt must state the full schema vocabulary that
# AdaptationProgram/PolicyCandidate validation enforces; it is derived from
# the same constants so prompt and validator cannot drift apart.
_GENERATION_SCHEMA_CONTRACT = (
    "The returned object must contain policy_spec. policy_spec must contain "
    "exactly conditions and program. The returned object may additionally "
    "contain capability_gap_proposals for audit only; this field is never "
    "executed or scored.\n"
    f"conditions: object with at most {MAXIMUM_CONDITION_COUNT} entries; each "
    "key must be one of "
    f"[{', '.join(CONDITION_STATE_PATHS)}], mapped to an exact scalar value "
    'or to a numeric range {"min": x} / {"max": x} / {"min": x, "max": x}. '
    "Range keys other than min/max (for example lt, gt, eq) are rejected, as "
    "are keys outside the list. budget.cooldown_active is a system hard "
    "constraint and may never appear in conditions. Candidates compete at "
    "the current diagnostic moment: when action is not KEEP, every "
    "condition must already be TRUE in the Current state JSON below; only an "
    "explicit KEEP candidate may carry conditions that do "
    "not hold now.\n"
    "program keys and constraints:\n"
    f"- diagnosis: non-empty string, at most {MAXIMUM_DIAGNOSIS_CHARS} characters.\n"
    "- trigger: object whose keys may only be: "
    f"{', '.join(PROGRAM_TRIGGER_FIELDS)}.\n"
    "- action: exactly one of "
    f"[{', '.join(ONLINE_ADAPTATION_ACTIONS)}].\n"
    "- mechanism: exactly one registered deterministic capability compatible "
    "with action. Compatibility map: "
    f"{json.dumps({key: sorted(value) for key, value in MECHANISMS_BY_ACTION.items()}, sort_keys=True)}. "
    "AUTO delegates the choice to the executor; NONE is valid only for KEEP.\n"
    "- Choose action, program target, and each operation together from this "
    "joint compatibility contract. program target.region must belong to "
    "program_targets; each operation target.region must belong to that "
    "primitive's targets (explicitly override it when the program target is "
    "not compatible). AUTO does not override target or primitive checks: "
    f"{json.dumps({action: {'program_targets': sorted(ACTION_TARGETS[action]), 'operations': {primitive: {'targets': CAPABILITY_REGISTRY[primitive]['targets']} for primitive in sorted(mechanisms) if primitive in CAPABILITY_REGISTRY}} for action, mechanisms in MECHANISMS_BY_ACTION.items()}, sort_keys=True)}.\n"
    "Registered per-operation parameter contracts (all fields optional; "
    "strength supplies the default magnitude): "
    f"{json.dumps({key: value['parameters'] for key, value in CAPABILITY_REGISTRY.items()}, sort_keys=True)}.\n"
    "- scientific mechanism guide: "
    f"{json.dumps(_CAPABILITY_SCIENTIFIC_GUIDE, sort_keys=True)}. Select a "
    "mechanism only when the current reference-free diagnostic evidence "
    "supports its stated inductive bias.\n"
    "- target: object with region one of "
    f"[{', '.join(TARGET_REGIONS)}] and selector one of "
    f"[{', '.join(TARGET_SELECTORS)}]; required whenever action is not KEEP.\n"
    "- strength: number in (0, 1] for an online adaptation and 0 for KEEP.\n"
    "- operations: the authoritative ordered deterministic request. KEEP must "
    "use []; every non-KEEP proposal should provide one or more operation objects. "
    'Each operation is {"primitive": <registered capability>, "strength": '
    '<number in (0,1]>, "target": {"region": ..., "selector": ...} (optional; '
    'inherits program target), "parameters": {...} (optional)}. Operations '
    "execute in list order, may repeat a primitive, and are one atomic "
    "transaction. Every primitive must be compatible with action; use "
    "EXPLORATORY_MORPH only for the structural primitives explicitly listed "
    "under that action. It cannot mix objective, sampling or optimizer "
    "primitives. Parameter contracts "
    "come only from the registered capability table above. Do not invent "
    "parameter keys or executable code.\n"
    "- mechanism must equal the primitive of a single-operation request "
    "(or AUTO); use AUTO for a "
    "multi-operation sequence. The operation list, when present, is more "
    "specific and is what the executor compiles.\n"
    "- recovery_steps: integer in [0, 2000]; must be 0 when action is KEEP.\n"
    "- cooldown_steps: integer in [0, 5000].\n"
    '- fallback: {} or exactly {"condition": <dotted state path>, '
    '"action": <one high-level action>}.\n'
    f"- reason: short string, at most {MAXIMUM_REASON_CHARS} characters.\n"
    "KEEP is a first-class decision. Return KEEP when the current causal "
    "evidence does not justify an intervention; never manufacture "
    "an intervention merely to make the generation non-KEEP.\n"
    "If the registered menu cannot express a scientifically important "
    "response, still return a valid registered fallback in policy_spec and "
    "optionally add capability_gap_proposals (at most 3), each shaped as "
    '{"name": <short proposed capability>, "category": <architecture|objective|sampling|constraint|optimizer|representation>, '
    '"target": <where it should act>, "rationale": <why registered capabilities are insufficient>, '
    '"trigger_evidence": [<current reference-free signals>], "required_executor_behavior": <deterministic behavior needed>}. '
    "Do not put executable code in a gap proposal. Return exactly one JSON "
    "object. No code, no markdown fences."
)

_GENERATION_EXAMPLE = {
    "policy_spec": {
        "conditions": {"optimization.plateau": True},
        "program": {
            "diagnosis": "capacity_bottleneck",
            "trigger": {"plateau_steps": 600},
            "action": "CONSERVATIVE_GROWTH",
            "mechanism": "WIDEN",
            "target": {"region": "middle_layers", "selector": "highest_utility"},
            "strength": 0.2,
            "operations": [
                {
                    "primitive": "WIDEN",
                    "strength": 0.2,
                    "target": {
                        "region": "middle_layers",
                        "selector": "highest_utility",
                    },
                    "parameters": {"width_fraction": 0.2},
                }
            ],
            "recovery_steps": 500,
            "cooldown_steps": 800,
            "fallback": {},
            "reason": "Plateau with stable gradients suggests a capacity bottleneck.",
        },
    },
    "capability_gap_proposals": [],
}


def _schema_repair_directive(
    errors: Sequence[str], rejected_payload: str | None
) -> str:
    """Convert a blind retry into a targeted schema repair request."""

    parts = [
        "\nYour previous response was rejected by schema validation. Return "
        "the corrected complete JSON object, changing only what the errors "
        "require.",
        "Rejection errors: " + "; ".join(errors),
    ]
    if rejected_payload:
        parts.append("Rejected response (truncated): " + rejected_payload)
    return "\n".join(parts)


_PROGRAM_SCHEMA_FIELDS = frozenset(
    {
        "diagnosis",
        "trigger",
        "target",
        "operations",
        "recovery_steps",
        "cooldown_steps",
        "fallback",
        "reason",
        "action",
        "strength",
        "mechanism",
    }
)


class _TransientPolicyProviderFailure(RuntimeError):
    """A retryable transport/service failure, distinct from schema rejection."""


def _is_transient_policy_provider_error(error: Any) -> bool:
    text = str(error or "").casefold()
    return any(
        token in text
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "temporarily unavailable",
            "rate limit",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
        )
    )


def _normalize_generated_policy_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Repair only the common harmless flattening of program fields.

    Unknown fields remain visible to strict validation. This is intentionally
    not a permissive schema fixer and cannot invent an action or condition.
    """

    outer = dict(raw or {})
    nested = outer.get("policy_spec")
    spec = dict(nested) if isinstance(nested, Mapping) else dict(outer)
    program_payload = spec.pop("program", {})
    program = dict(program_payload) if isinstance(program_payload, Mapping) else {}
    for key in tuple(spec):
        if key in _PROGRAM_SCHEMA_FIELDS:
            program.setdefault(key, spec.pop(key))
    if program:
        spec["program"] = program
    return spec


def extract_capability_gap_observations(
    raw: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    mode: str,
    generation_kind: str,
    attempt: int,
) -> list[dict[str, Any]]:
    """Audit executor gaps without weakening executable-policy validation."""

    outer = dict(raw or {})
    nested = outer.get("policy_spec")
    spec = dict(nested) if isinstance(nested, Mapping) else dict(outer)
    program_raw = spec.get("program")
    program = dict(program_raw) if isinstance(program_raw, Mapping) else {
        key: spec.get(key) for key in _PROGRAM_SCHEMA_FIELDS if key in spec
    }
    compact_state = compact_policy_state(state)
    common = {
        "step": int(compact_state.get("step", 0) or 0),
        "state_id": state_id(compact_state),
        "mode": str(mode),
        "generation_kind": str(generation_kind),
        "attempt": int(attempt),
        "context": {
            key: deepcopy(dict(compact_state.get(key) or {}))
            for key in ("pde", "physics", "gradient", "spatial", "frequency")
        },
    }
    observations: list[dict[str, Any]] = []

    explicit = outer.get("capability_gap_proposals", ())
    if isinstance(explicit, Mapping):
        explicit = [explicit]
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        for proposal in list(explicit)[:3]:
            if not isinstance(proposal, Mapping):
                continue
            bounded = {
                key: deepcopy(proposal.get(key))
                for key in (
                    "name", "category", "target", "rationale",
                    "trigger_evidence", "required_executor_behavior",
                )
                if proposal.get(key) not in (None, "", [], {})
            }
            observations.append(
                {
                    **common,
                    "kind": "explicit_capability_gap",
                    "proposed": str(
                        bounded.get("name") or "unspecified_capability"
                    )[:160],
                    "proposal": bounded,
                    "source": "llm_audit_field",
                }
            )

    action = str(program.get("action") or "").strip()
    mechanism = str(program.get("mechanism") or "").strip()
    target = (
        dict(program.get("target") or {})
        if isinstance(program.get("target"), Mapping)
        else {}
    )

    def add(kind: str, proposed: Any, **details: Any) -> None:
        if proposed in (None, ""):
            return
        observations.append(
            {
                **common,
                "kind": kind,
                "proposed": str(proposed)[:160],
                "source": "invalid_executable_vocabulary",
                "action": action or None,
                "mechanism": mechanism or None,
                **details,
            }
        )

    if action and action not in ONLINE_ADAPTATION_ACTIONS:
        add("unknown_action", action)
    if mechanism and mechanism not in {"AUTO", "NONE"}:
        if mechanism not in CAPABILITY_REGISTRY:
            add("unknown_mechanism", mechanism)
        elif (
            action in ONLINE_ADAPTATION_ACTIONS
            and mechanism not in MECHANISMS_BY_ACTION.get(action, ())
        ):
            add("action_mechanism_mismatch", mechanism)
    region = target.get("region")
    selector = target.get("selector")
    if region and region not in TARGET_REGIONS:
        add("unknown_target_region", region)
    elif (
        mechanism in CAPABILITY_REGISTRY
        and region not in CAPABILITY_REGISTRY[mechanism]["targets"]
    ):
        add("mechanism_target_mismatch", region)
    if selector and selector not in TARGET_SELECTORS:
        add("unknown_selector", selector)

    operations = program.get("operations") or ()
    if isinstance(operations, Mapping):
        operations = [operations]
    if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes)):
        for index, operation_raw in enumerate(operations):
            if not isinstance(operation_raw, Mapping):
                continue
            operation = dict(operation_raw)
            primitive = str(operation.get("primitive") or "").strip()
            if primitive and primitive not in CAPABILITY_REGISTRY:
                add("unknown_primitive", primitive, operation_index=index)
                continue
            if (
                primitive
                and action in ONLINE_ADAPTATION_ACTIONS
                and action not in CAPABILITY_REGISTRY[primitive]["action_families"]
            ):
                add("action_primitive_mismatch", primitive, operation_index=index)
            parameters = operation.get("parameters") or {}
            if primitive in CAPABILITY_REGISTRY and isinstance(parameters, Mapping):
                allowed_parameters = CAPABILITY_REGISTRY[primitive]["parameters"]
                for parameter in parameters:
                    if parameter not in allowed_parameters:
                        add(
                            "unknown_parameter",
                            parameter,
                            operation_index=index,
                            primitive=primitive,
                        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in observations:
        key = (
            str(item.get("kind")),
            str(item.get("proposed")),
            str(item.get("action")),
            str(item.get("mechanism")),
        )
        if key in seen:
            continue
        seen.add(key)
        identity = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        item["gap_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        unique.append(item)
    return unique


def state_id(state: Mapping[str, Any]) -> str:
    """Stable identity for causal state/action/outcome attribution."""

    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def compact_morph_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded, non-recursive history exposed to policy candidates."""

    source = dict(event)
    summary: dict[str, Any] = {}
    for key in (
        "event",
        "intervention_id",
        "iteration",
        "step",
        "operator",
        "feedback_status",
        "selected_policy_id",
        "state_id",
        "utility",
        "parameter_count_before",
        "parameter_count_after",
        "parameter_growth_ratio",
    ):
        value = source.get(key)
        if value is not None and isinstance(value, (str, int, float, bool)):
            summary[key] = value
    reason = source.get("reason")
    if reason is not None:
        summary["reason"] = str(reason)[:240]
    program = source.get("adaptation_program") or source.get("program") or {}
    if isinstance(program, Mapping):
        if program.get("action") is not None:
            summary["architecture_action"] = str(program.get("action"))
        operations = program.get("operations") or ()
        summary["primitives"] = [
            str(item.get("primitive"))
            for item in operations
            if isinstance(item, Mapping) and item.get("primitive")
        ]
    physics = source.get("physics_utility") or {}
    if isinstance(physics, Mapping):
        if physics.get("accepted") is not None:
            summary["accepted"] = bool(physics.get("accepted"))
        aggregate = physics.get("aggregate_continuation_utility")
        if isinstance(aggregate, (int, float)):
            summary["aggregate_continuation_utility"] = float(aggregate)
    instruction_audit = source.get("architecture_instruction_audit") or {}
    if isinstance(instruction_audit, Mapping):
        summary["instruction_execution"] = {
            "transaction_status": instruction_audit.get("transaction_status"),
            "semantic_fidelity": instruction_audit.get("semantic_fidelity"),
            "requested": [
                str(item.get("primitive"))
                for item in list(instruction_audit.get("requested_operations") or ())[:8]
                if isinstance(item, Mapping) and item.get("primitive")
            ],
            "compiled": [
                str(item.get("action"))
                for item in list(instruction_audit.get("compiled_operations") or ())[:8]
                if isinstance(item, Mapping) and item.get("action")
            ],
            "executed": [
                str(item.get("action"))
                for item in list(instruction_audit.get("executed_operations") or ())[:8]
                if isinstance(item, Mapping) and item.get("action")
            ],
            "substitutions": deepcopy(
                list(instruction_audit.get("substitutions") or ())[:8]
            ),
        }
    return summary


def compact_policy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return decision-relevant state without cumulative topology history."""

    source = dict(state or {})
    result: dict[str, Any] = {
        "step": source.get("step", source.get("global_step", 0))
    }
    for name in (
        "optimization", "gradient", "spatial", "frequency", "budget", "physics"
    ):
        value = source.get(name)
        if isinstance(value, Mapping):
            result[name] = deepcopy(dict(value))
    pde = dict(source.get("pde") or {})
    if pde:
        result["pde"] = {
            key: deepcopy(pde[key])
            for key in (
                "benchmark_id", "problem_id", "pde_type", "pde_family",
                "equation_order", "spatial_dimension", "state_dimension",
                "time_dependent", "has_initial_condition", "has_shock",
                "is_stiff", "is_multiscale", "convection_dominated",
                "diffusion_dominated", "has_interface", "has_coupled_fields",
                "has_piecewise_coefficient", "has_oscillatory_solution",
                "has_periodicity", "geometry", "challenge_tags",
                "constraint_enforcement_options", "hard_constraint_transforms",
            )
            if key in pde
        }
    architecture = dict(source.get("current_architecture") or {})
    initial = dict(architecture.get("initial_network_spec") or {})
    history = list(architecture.get("online_architecture_state") or [])
    latest = dict(history[-1]) if history else {}
    result["current_architecture"] = {
        "architecture": initial.get("architecture"),
        "activation": deepcopy(initial.get("activation")),
        "initial_hidden_layers": deepcopy(initial.get("hidden_layers")),
        "current_hidden_layers": deepcopy(
            latest.get("hidden_layers_after", initial.get("hidden_layers"))
        ),
        "morph_count": len(history),
        "last_operator": latest.get("operator"),
    }
    result["previous_morph_events"] = [
        compact_morph_event(item)
        for item in list(source.get("previous_morph_events") or ())[-6:]
        if isinstance(item, Mapping)
    ]
    for forecast_key in ("continuation_forecast", "audit_continuation_forecast"):
        forecast = source.get(forecast_key)
        if isinstance(forecast, Mapping):
            result[forecast_key] = _compact_continuation_forecast(forecast)
    return result


def _compact_continuation_forecast(forecast: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only forecast fields needed to score KEEP vs intervention."""

    source = dict(forecast or {})
    result: dict[str, Any] = {}
    for key in (
        "valid", "reason", "credit_eligible", "forecast_confidence",
        "confidence_aggregation", "minimum_confidence",
        "maximum_extrapolation_factor", "forecast_clipped",
        "history_points_used",
    ):
        value = source.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    history_steps = source.get("history_steps")
    if isinstance(history_steps, Sequence) and not isinstance(history_steps, (str, bytes)):
        result["history_steps"] = [int(item) for item in list(history_steps)[-8:]]
    predictions = source.get("predictions")
    if isinstance(predictions, Mapping):
        compact_predictions: dict[str, Any] = {}
        for step, prediction_raw in list(predictions.items())[:4]:
            if not isinstance(prediction_raw, Mapping):
                continue
            prediction = dict(prediction_raw)
            compact: dict[str, Any] = {}
            for key in (
                "step", "total_score", "forecast_confidence",
                "credit_eligible", "forecast_clipped",
            ):
                value = prediction.get(key)
                if isinstance(value, (int, float, bool)):
                    compact[key] = value
            for metric_key in (
                "component_metrics", "group_forecast_confidences",
                "region_metrics",
            ):
                value = prediction.get(metric_key)
                if isinstance(value, Mapping):
                    compact[metric_key] = {
                        str(name): float(item)
                        for name, item in list(value.items())[:16]
                        if isinstance(item, (int, float))
                        and math.isfinite(float(item))
                    }
            compact_predictions[str(step)] = compact
        result["predictions"] = compact_predictions
    return result


def _compact_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(feedback or {})
    result: dict[str, Any] = {}
    for key in (
        "credit", "accepted", "reason", "aggregate_continuation_utility",
        "max_component_degradation", "raw_before_after_improvement", "confounded",
        "credit_eligible", "feedback_status", "withheld_kind",
        "directional_evidence", "decision_probe_direction",
        "audit_probe_direction", "probe_direction_agreement",
        "forecast_confidence", "decision_worst_region_id",
        "decision_worst_region_utility", "audit_worst_region_id",
        "audit_worst_region_utility", "worst_region_id",
        "worst_region_label",
        "worst_region_continuation_utility", "worst_region_degradation",
        "worst_region_direction_agreement", "qualitative_credit_hint",
        "qualitative_credit_bound", "llm_feedback_summary",
    ):
        value = source.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    for key in (
        "group_continuation_utility", "component_continuation_utility",
        "region_continuation_utility", "region_degradations",
    ):
        value = source.get(key)
        if isinstance(value, Mapping):
            result[key] = {
                str(name): float(item)
                for name, item in list(value.items())[:16]
                if isinstance(item, (int, float)) and math.isfinite(float(item))
            }
    labels = source.get("region_labels")
    if isinstance(labels, Mapping):
        result["region_labels"] = {
            str(name): str(label)
            for name, label in list(labels.items())[:16]
        }
    directions = source.get("worst_region_probe_directions")
    if isinstance(directions, Mapping):
        result["worst_region_probe_directions"] = {
            str(region_id): {
                "decision": str(dict(value).get("decision")),
                "audit": str(dict(value).get("audit")),
            }
            for region_id, value in list(directions.items())[:16]
            if isinstance(value, Mapping)
        }
    conflicts = source.get("conflicting_worst_region_ids")
    if isinstance(conflicts, (list, tuple)):
        result["conflicting_worst_region_ids"] = [
            str(value) for value in conflicts[:16]
        ]
    return result


def causal_feedback_status(
    physics_utility: Mapping[str, Any] | None,
    *,
    rolled_back: bool = False,
) -> str:
    """Classify outcomes without treating uncertain causal credit as failure."""

    payload = dict(physics_utility or {})
    if not bool(payload.get("credit_eligible", False)):
        return "credit_withheld"
    if rolled_back:
        return "failure"
    return "success" if bool(payload.get("accepted", False)) else "failure"


def adaptation_program_fingerprint(
    value: AdaptationProgram | Mapping[str, Any],
) -> str:
    """Return a stable identity for the executable part of a program.

    Diagnostic prose, triggers, recovery windows and fallback text do not
    change the structural intervention itself and are intentionally excluded.
    This keeps causal credit attached to the actual ordered primitive request.
    """

    program = (
        value if isinstance(value, AdaptationProgram)
        else AdaptationProgram.from_dict(value)
    )
    payload = program.to_dict()
    canonical = {
        "action": payload.get("action"),
        "operations": [
            {
                key: deepcopy(item.get(key))
                for key in ("primitive", "strength", "target", "parameters")
                if item.get(key) is not None
            }
            for item in list(payload.get("operations") or ())
            if isinstance(item, Mapping)
        ],
    }
    if not canonical["operations"]:
        canonical.update(
            {
                "mechanism": payload.get("mechanism"),
                "target": deepcopy(payload.get("target")),
                "strength": payload.get("strength"),
            }
        )
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def adaptation_hypothesis_signature(
    value: AdaptationProgram | Mapping[str, Any],
) -> str:
    """Describe a coarse scientific intervention hypothesis.

    Unlike the executable fingerprint, this intentionally ignores numeric
    strength, recovery and cooldown. Two residual branches that differ only by
    0.10 versus 0.15 strength are therefore reported as the same hypothesis.
    The signature is used for prompt context and audit only, never rejection.
    """

    program = (
        value if isinstance(value, AdaptationProgram)
        else AdaptationProgram.from_dict(value)
    )
    if not program.has_intervention():
        return KEEP
    parts: list[str] = []
    for operation in program.operations:
        payload = operation.to_dict()
        target = dict(payload.get("target") or program.target or {})
        parts.append(
            ":".join(
                (
                    str(payload.get("primitive") or "UNKNOWN"),
                    str(target.get("region") or "unspecified"),
                    str(target.get("selector") or "all"),
                )
            )
        )
    return f"{program.action}|" + "->".join(parts)


def compact_shared_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded, non-recursive state/action/outcome record."""

    source = dict(evidence or {})
    program = dict(source.get("program") or {})
    feedback = dict(source.get("feedback") or {})
    result = {
        "intervention_id": source.get("intervention_id"),
        "policy_id": source.get("policy_id"),
        "policy_version": source.get("policy_version"),
        "state_id": source.get("state_id"),
        "start_step": source.get("start_step"),
        "state_before": compact_policy_state(
            dict(source.get("state_before") or {})
        ),
        "program": {
            **{
                key: deepcopy(program.get(key))
                for key in (
                    "diagnosis",
                    "action",
                    "mechanism",
                    "target",
                    "strength",
                    "reason",
                )
                if program.get(key) is not None
            },
            "operations": [
                {
                    key: deepcopy(item.get(key))
                    for key in ("primitive", "strength", "target", "parameters")
                    if item.get(key) is not None
                }
                for item in list(program.get("operations") or ())[:8]
                if isinstance(item, Mapping)
            ],
        },
        "feedback": {
            str(horizon): _compact_feedback(item)
            for horizon, item in feedback.items()
            if isinstance(item, Mapping)
        },
    }
    if program:
        try:
            result["program_fingerprint"] = adaptation_program_fingerprint(program)
        except (TypeError, ValueError):
            # Preserve historical evidence even when a legacy program cannot
            # be normalized under the current capability registry.
            pass
    return result


def compact_shared_history(
    history: Sequence[Mapping[str, Any]],
    *,
    maximum_items: int = 6,
    maximum_chars: int = PROMPT_HISTORY_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Keep the newest complete compact records under a hard prompt budget."""

    selected: list[dict[str, Any]] = []
    used = 2  # JSON list brackets
    for item in reversed(list(history)[-max(0, int(maximum_items)) :]):
        compact = compact_shared_evidence(item)
        size = len(json.dumps(compact, ensure_ascii=False, default=str)) + 1
        if selected and used + size > max(256, int(maximum_chars)):
            break
        if not selected and size > max(256, int(maximum_chars)):
            # This should be rare after compaction; retain causal identity and
            # outcome instead of slicing JSON into an invalid document.
            compact = {
                "intervention_id": compact.get("intervention_id"),
                "policy_id": compact.get("policy_id"),
                "state_id": compact.get("state_id"),
                "start_step": compact.get("start_step"),
                "program": compact.get("program"),
                "feedback": {
                    horizon: {
                        key: payload.get(key)
                        for key in (
                            "credit", "accepted", "reason",
                            "aggregate_continuation_utility",
                            "max_component_degradation", "confounded",
                        )
                        if payload.get(key) is not None
                    }
                    for horizon, payload in dict(compact.get("feedback") or {}).items()
                },
            }
            size = len(json.dumps(compact, ensure_ascii=False, default=str)) + 1
        selected.append(compact)
        used += size
    selected.reverse()
    return selected


def compact_policy_candidate(candidate: PolicyCandidate) -> dict[str, Any]:
    """Expose policy credit without duplicating full feedback telemetry."""

    return {
        "policy_id": candidate.policy_id,
        "version": candidate.version,
        "parent_ids": list(candidate.parent_ids),
        "creation_step": candidate.creation_step,
        "creation_mode": candidate.creation_mode,
        "policy_spec": deepcopy(dict(candidate.policy_spec)),
        "execution_count": candidate.execution_count,
        "success_count": candidate.success_count,
        "failure_count": candidate.failure_count,
        "global_credit": candidate.global_credit,
        "recent_results": [
            {
                "intervention_id": item.get("intervention_id"),
                "horizon": item.get("horizon"),
                "credit": item.get("credit"),
                "feedback": _compact_feedback(dict(item.get("feedback") or {})),
            }
            for item in candidate.recent_results[-4:]
        ],
    }


def compact_policy_credit(candidate: PolicyCandidate) -> dict[str, Any]:
    """Expose causal credit once; the proposal already carries the full program."""

    return {
        "policy_id": candidate.policy_id,
        "version": candidate.version,
        "parent_ids": list(candidate.parent_ids),
        "creation_mode": candidate.creation_mode,
        "execution_count": candidate.execution_count,
        "success_count": candidate.success_count,
        "failure_count": candidate.failure_count,
        "global_credit": candidate.global_credit,
        "recent_results": [
            {
                "intervention_id": item.get("intervention_id"),
                "horizon": item.get("horizon"),
                "credit": item.get("credit"),
                "feedback": _compact_feedback(dict(item.get("feedback") or {})),
            }
            for item in candidate.recent_results[-4:]
        ],
    }


def _program(
    diagnosis: str,
    primitive: str | None,
    *,
    region: str = "middle_layers",
    selector: str = "highest_utility",
    strength: float = 0.2,
    recovery: int = 500,
    cooldown: int = 800,
) -> dict[str, Any]:
    action = (
        KEEP
        if primitive is None
        else REPRESENTATION_AUGMENTATION
        if primitive == ADD_FOURIER_BRANCH
        else LOCAL_SPECIALIZATION
        if primitive == ADD_LOCALIZED_EXPERT
        else OBJECTIVE_REBALANCING
        if primitive == REBALANCE_LOSS_WEIGHTS
        else GRADIENT_CONFLICT_MITIGATION
        if primitive == MITIGATE_GRADIENT_CONFLICT
        else CONSERVATIVE_GROWTH
        if primitive in {WIDEN, REGROW}
        else REALLOCATION
        if primitive in {PRUNE, REDISTRIBUTE}
        else EXPLORATORY_MORPH
    )
    return {
        "diagnosis": diagnosis,
        "trigger": {},
        "target": ({"region": region, "selector": selector} if primitive else {}),
        "action": action,
        "mechanism": "NONE" if primitive is None else str(primitive),
        "strength": float(strength if primitive else 0.0),
        "operations": (
            []
            if primitive is None
            else [
                {
                    "primitive": str(primitive),
                    "strength": float(strength),
                    "target": {"region": region, "selector": selector},
                    "parameters": {},
                }
            ]
        ),
        "recovery_steps": int(recovery if primitive else 0),
        "cooldown_steps": int(cooldown if primitive else 0),
        "fallback": {},
        "reason": "built-in cold-start EOH policy",
    }


def default_policy_population(*, creation_step: int = 0) -> list[PolicyCandidate]:
    """Five diverse cold-start policies; they are peers, not fixed roles."""

    specs = (
        (
            "policy-stability",
            {},
            _program("evidence_insufficient_or_learning", None),
        ),
        (
            "policy-capacity",
            {"optimization.plateau": True, "gradient.gradient_pathology": False},
            _program("capacity_bottleneck", WIDEN, strength=0.20),
        ),
        (
            "policy-spectral-representation",
            {
                "optimization.plateau": True,
                "frequency.has_multiscale_or_oscillatory_evidence": True,
            },
            _program(
                "spectral_representation_bottleneck",
                ADD_FOURIER_BRANCH,
                region="input_representation",
                strength=0.15,
                recovery=700,
            ),
        ),
        (
            "policy-objective-balance",
            {"gradient.gradient_imbalance_detected": True},
            _program(
                "physics_objective_gradient_imbalance",
                REBALANCE_LOSS_WEIGHTS,
                region="all_loss_components",
                selector="highest_utility",
                strength=0.20,
            ),
        ),
        (
            "policy-local-specialization",
            {
                "optimization.plateau": True,
                "spatial.localized_residual_concentration": True,
            },
            _program(
                "localized_physics_error",
                ADD_LOCALIZED_EXPERT,
                region="localized_regions",
                selector="highest_utility",
                strength=0.10,
                recovery=1000,
            ),
        ),
    )
    return [
        PolicyCandidate(
            policy_id=policy_id,
            version=1,
            parent_ids=(),
            creation_step=int(creation_step),
            creation_mode="initialization",
            policy_spec={"conditions": conditions, "program": program},
        )
        for policy_id, conditions, program in specs
    ]


class PolicyPopulation:
    """One fixed-size, execution-grounded population of adaptation policies."""

    def __init__(
        self,
        policies: Sequence[PolicyCandidate] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        selected = list(policies or default_policy_population())
        if len(selected) != POLICY_POPULATION_SIZE:
            raise ValueError("The unified policy population must contain exactly 5 candidates")
        ids = [item.policy_id for item in selected]
        if len(set(ids)) != len(ids):
            raise ValueError("Policy ids must be unique")
        self.policies: list[PolicyCandidate] = selected
        self.policy_archive: dict[str, PolicyCandidate] = {}
        self.shared_history: list[dict[str, Any]] = []
        self.prior_context: list[dict[str, Any]] = []
        self.generation = 0
        self.llm_initialized = False
        self.last_evolution_evidence_sequence = 0
        self.evolution_events: list[dict[str, Any]] = []
        self.capability_gap_events: list[dict[str, Any]] = []
        self.last_scoring_audit: dict[str, Any] | None = None
        self.last_selection_provider_calls: int = 0
        self.last_evolution_provider_calls: int = 0
        self.last_evolution_transient_failures: int = 0
        self.last_repeat_gate_audit: list[dict[str, Any]] = []
        self.applicability_repair_pending: bool = False
        self.applicability_repair_count: int = 0

    def set_prior_context(
        self, principles: Sequence[Mapping[str, Any]]
    ) -> None:
        """Install bounded, non-binding static guidance for both LLM roles."""

        self.prior_context = [
            deepcopy(dict(item)) for item in list(principles)[:6]
        ]

    def request_applicability_repair(self) -> None:
        """Flag that the population cannot act at the states it is seeing.

        Called when all five candidates expand to KEEP, which without a
        repair leads to a permanent idle loop: no intervention, no feedback,
        and therefore no evolution trigger ever again.
        """

        self.applicability_repair_pending = True

    def applicability_repair_due(self, maximum_repairs: int) -> bool:
        return bool(self.applicability_repair_pending) and self.applicability_repair_count < max(
            0, int(maximum_repairs)
        )

    def consume_applicability_repair(self) -> None:
        self.applicability_repair_pending = False
        self.applicability_repair_count += 1

    def propose_all(self, state: Mapping[str, Any]) -> list[ProgramProposal]:
        sid = state_id(state)
        proposals: list[ProgramProposal] = []
        for policy in self.policies:
            program = policy.propose(state=state, history=self.shared_history)
            match = 1.0 if program.diagnosis != "policy_trigger_not_satisfied" else 0.0
            proposals.append(
                ProgramProposal(
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    state_id=sid,
                    program=program,
                    state_match=match,
                    selection_score=0.0,
                    reason="awaiting independent LLM scorer",
                )
            )
        return proposals

    def gate_uncredited_repeats(
        self, proposals: Sequence[ProgramProposal]
    ) -> list[ProgramProposal]:
        """Block an identical structural program until delayed positive credit.

        A short-horizon success is insufficient. The latest application must
        have positive delayed credit, and its persistent outcome supersedes
        its medium outcome. An older success cannot erase a later failure. Other candidates
        remain available, so this is a causal safety gate rather than population
        de-duplication.
        """

        history_by_fingerprint: dict[str, list[Mapping[str, Any]]] = {}
        for evidence in self.shared_history:
            raw_program = evidence.get("program")
            if not isinstance(raw_program, Mapping) or not raw_program:
                continue
            try:
                fingerprint = adaptation_program_fingerprint(raw_program)
            except (TypeError, ValueError):
                continue
            history_by_fingerprint.setdefault(fingerprint, []).append(evidence)

        gated: list[ProgramProposal] = []
        audits: list[dict[str, Any]] = []
        for proposal in proposals:
            if not proposal.program.has_intervention():
                gated.append(proposal)
                continue
            fingerprint = adaptation_program_fingerprint(proposal.program)
            matching = history_by_fingerprint.get(fingerprint, [])
            if not matching:
                gated.append(proposal)
                continue
            positive_delayed = False
            observed: list[dict[str, Any]] = []
            for evidence in matching:
                feedback = dict(evidence.get("feedback") or {})
                for horizon in ("medium", "persistent"):
                    raw = feedback.get(horizon)
                    if not isinstance(raw, Mapping):
                        continue
                    payload = dict(raw)
                    status = str(
                        payload.get("feedback_status")
                        or causal_feedback_status(payload)
                    )
                    eligible = bool(payload.get("credit_eligible", False))
                    observed.append(
                        {
                            "intervention_id": evidence.get("intervention_id"),
                            "horizon": horizon,
                            "feedback_status": status,
                            "credit_eligible": eligible,
                        }
                    )
            latest = max(enumerate(matching), key=lambda item: (
                int(item[1].get("start_step") or 0), item[0]))[1]
            latest_feedback = dict(latest.get("feedback") or {})
            decisive = latest_feedback.get("persistent", latest_feedback.get("medium"))
            if isinstance(decisive, Mapping):
                positive_delayed = bool(
                    decisive.get("credit_eligible", False)
                    and not decisive.get("confounded", False)
                    and str(decisive.get("feedback_status") or causal_feedback_status(decisive)) == "success"
                )
            if positive_delayed:
                gated.append(proposal)
                continue
            keep = AdaptationProgram.keep(
                diagnosis="repeat_without_positive_delayed_credit",
                reason=(
                    "identical executable program was already applied without "
                    "positive credit at the latest available delayed horizon"
                ),
            )
            gated.append(
                replace(
                    proposal,
                    program=keep,
                    state_match=0.0,
                    reason="causal repeat gate",
                )
            )
            audits.append(
                {
                    "policy_id": proposal.policy_id,
                    "program_fingerprint": fingerprint,
                    "matching_interventions": len(matching),
                    "delayed_feedback": observed,
                    "status": "blocked",
                }
            )
        self.last_repeat_gate_audit = audits
        return gated

    def select(
        self,
        state: Mapping[str, Any],
        scoring_provider: Any,
        *,
        temperature: float = 0.0,
        keep_margin: float = DEFAULT_KEEP_BASELINE_MARGIN,
        proposals: Sequence[ProgramProposal] | None = None,
    ) -> tuple[ProgramProposal, list[ProgramProposal]]:
        """Have an independent LLM score proposals against a KEEP baseline.

        No project-side weighted formula participates in candidate ranking.  A
        malformed or incomplete score response raises so the controller can
        fail closed to KEEP instead of silently restoring heuristic ranking.
        Pre-computed ``proposals`` (e.g. reused by the controller after an
        all-KEEP short-circuit check) are scored as-is instead of re-expanding
        the population.  The synthetic KEEP baseline is not an evolvable policy;
        it represents the counterfactual of continuing the current optimizer
        trajectory without online adaptation.
        """

        if scoring_provider is None or not callable(
            getattr(scoring_provider, "complete_text", None)
        ):
            raise TypeError("Candidate scoring requires a provider with complete_text(...)")
        if proposals is None:
            proposals = self.propose_all(state)
        augmented_proposals = self._with_keep_continuation_baseline(state, proposals)
        self.last_selection_provider_calls = 1
        response = scoring_provider.complete_text(
            [
                {
                    "role": "user",
                    "content": self._scoring_prompt(state, augmented_proposals),
                }
            ],
            temperature=float(temperature),
            max_tokens=1800,
            task="online_candidate_scoring",
        )
        if not bool(getattr(response, "success", True)):
            raise ValueError(str(getattr(response, "error", "candidate scorer failed")))
        expected_ids = [item.policy_id for item in augmented_proposals]
        score_by_id, _ = self._parse_scoring_response(
            response, expected_ids=expected_ids, context="candidate scorer"
        )
        scored = [
            replace(
                proposal,
                selection_score=score_by_id[proposal.policy_id][0],
                reason=score_by_id[proposal.policy_id][1] or "independent LLM score",
            )
            for proposal in augmented_proposals
        ]
        maximum_score = max(item.selection_score for item in scored)
        tied = [
            item
            for item in scored
            if math.isclose(
                item.selection_score, maximum_score, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        tie_break_audit: dict[str, Any] | None = None
        if len(tied) > 1:
            self.last_selection_provider_calls += 1
            tie_response = scoring_provider.complete_text(
                [
                    {
                        "role": "user",
                        "content": self._tie_break_scoring_prompt(
                            state, tied, score_by_id
                        ),
                    }
                ],
                temperature=float(temperature),
                max_tokens=1400,
                task="online_candidate_scoring_tiebreak",
            )
            if not bool(getattr(tie_response, "success", True)):
                raise ValueError(
                    str(getattr(tie_response, "error", "candidate tie-break scorer failed"))
                )
            tie_ids = [item.policy_id for item in tied]
            tie_scores, tie_raw = self._parse_scoring_response(
                tie_response,
                expected_ids=tie_ids,
                context="candidate tie-break scorer",
            )
            second_maximum = max(value[0] for value in tie_scores.values())
            second_tied_ids = [
                policy_id
                for policy_id, value in tie_scores.items()
                if math.isclose(value[0], second_maximum, rel_tol=0.0, abs_tol=1e-12)
            ]
            winner_id = str(tie_raw.get("winner_policy_id") or "")
            if len(second_tied_ids) == 1:
                winner_id = second_tied_ids[0]
            elif winner_id not in second_tied_ids:
                raise ValueError(
                    "candidate tie-break scorer must name one of the tied "
                    "highest-scoring policies as winner_policy_id"
                )
            best = next(item for item in tied if item.policy_id == winner_id)
            tie_break_audit = {
                "triggered": True,
                "initial_tied_policy_ids": tie_ids,
                "scores": [
                    {
                        "policy_id": policy_id,
                        "score": value[0],
                        "reason": value[1],
                    }
                    for policy_id, value in tie_scores.items()
                ],
                "selected_policy_id": winner_id,
                "winner_reason": str(tie_raw.get("winner_reason") or ""),
            }
        else:
            best = tied[0]
        keep_baseline = next(
            item
            for item in scored
            if item.policy_id == KEEP_CONTINUATION_BASELINE_POLICY_ID
        )
        intervention_candidates = [
            item for item in scored if item.program.has_intervention()
        ]
        best_intervention = (
            max(intervention_candidates, key=lambda item: item.selection_score)
            if intervention_candidates
            else None
        )
        effective_keep_margin = max(0.0, float(keep_margin))
        blocked_by_keep_baseline = False
        if (
            best.program.has_intervention()
            and best.selection_score
            <= keep_baseline.selection_score + effective_keep_margin
        ):
            blocked_by_keep_baseline = True
            best = replace(
                keep_baseline,
                reason=(
                    "KEEP baseline selected because the scorer-preferred "
                    f"intervention ({best.policy_id}) did not exceed KEEP "
                    f"by margin {effective_keep_margin:.3f}"
                ),
            )
        if best.policy_id != KEEP_CONTINUATION_BASELINE_POLICY_ID:
            policy = self.get(best.policy_id)
            policy.last_selected_step = int(
                state.get("step", state.get("global_step", 0)) or 0
            )
        self.last_scoring_audit = {
            "scoring_source": "DUAL_LLM_2_MODEL",
            "selected_policy_id": best.policy_id,
            "keep_baseline_policy_id": KEEP_CONTINUATION_BASELINE_POLICY_ID,
            "keep_baseline_score": keep_baseline.selection_score,
            "keep_baseline_margin": effective_keep_margin,
            "best_intervention_policy_id": (
                best_intervention.policy_id if best_intervention is not None else None
            ),
            "best_intervention_score": (
                best_intervention.selection_score
                if best_intervention is not None
                else None
            ),
            "intervention_blocked_by_keep_baseline": blocked_by_keep_baseline,
            "scores": [
                {
                    "policy_id": item.policy_id,
                    "score": item.selection_score,
                    "reason": item.reason,
                }
                for item in scored
            ],
            "tie_break": tie_break_audit,
            "provider_call_count": self.last_selection_provider_calls,
        }
        return best, scored

    @staticmethod
    def _with_keep_continuation_baseline(
        state: Mapping[str, Any],
        proposals: Sequence[ProgramProposal],
    ) -> list[ProgramProposal]:
        """Append the non-evolvable KEEP counterfactual baseline once."""

        proposal_list = list(proposals)
        if any(
            item.policy_id == KEEP_CONTINUATION_BASELINE_POLICY_ID
            for item in proposal_list
        ):
            return proposal_list
        sid = state_id(state)
        keep = AdaptationProgram.keep(
            diagnosis="explicit_keep_continuation_baseline",
            reason=(
                "Continue optimizing the current PINN without online adaptation; "
                "compare intervention proposals against the continuation forecast."
            ),
        )
        proposal_list.append(
            ProgramProposal(
                policy_id=KEEP_CONTINUATION_BASELINE_POLICY_ID,
                policy_version=1,
                state_id=sid,
                program=keep,
                state_match=1.0,
                selection_score=0.0,
                reason="explicit KEEP continuation baseline",
            )
        )
        return proposal_list

    @staticmethod
    def _parse_scoring_response(
        response: Any,
        *,
        expected_ids: Sequence[str],
        context: str,
    ) -> tuple[dict[str, tuple[float, str]], dict[str, Any]]:
        raw, parse_errors = extract_json_object(
            str(getattr(response, "text", response)), preferred_keys=("scores",)
        )
        if raw is None:
            raise ValueError(f"{context} returned invalid JSON: " + "; ".join(parse_errors))
        raw_scores = raw.get("scores")
        if not isinstance(raw_scores, list):
            raise ValueError(f"{context} response must contain a scores list")
        score_by_id: dict[str, tuple[float, str]] = {}
        for item in raw_scores:
            if not isinstance(item, Mapping):
                raise ValueError(f"every {context} score must be an object")
            policy_id = str(item.get("policy_id") or "")
            if policy_id in score_by_id:
                raise ValueError(f"duplicate {context} score for {policy_id!r}")
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric {context} score for {policy_id!r}") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{context} score for {policy_id!r} must be finite and in [0, 1]"
                )
            score_by_id[policy_id] = (score, str(item.get("reason") or ""))
        expected = set(expected_ids)
        if set(score_by_id) != expected or len(score_by_id) != len(expected_ids):
            missing = sorted(expected - set(score_by_id))
            unknown = sorted(set(score_by_id) - expected)
            raise ValueError(
                f"{context} must score each requested candidate exactly once; "
                f"missing={missing}, unknown={unknown}"
            )
        return score_by_id, raw

    def _scoring_prompt(
        self,
        state: Mapping[str, Any],
        proposals: Sequence[ProgramProposal],
    ) -> str:
        compact_state = compact_policy_state(state)
        if state.get("online_adaptation"):
            compact_state["online_adaptation"] = deepcopy(state["online_adaptation"])
        history = compact_shared_history(self.shared_history)
        policy_evidence = {
            item.policy_id: compact_policy_credit(item) for item in self.policies
        }
        return (
            "You are the independent scorer for online PINN adaptation "
            "proposals produced by another model plus one explicit KEEP "
            "continuation baseline. The KEEP baseline is not an evolvable "
            "policy: it represents continuing the current optimizer trajectory "
            "without online adaptation under continuation_forecast and "
            "audit_continuation_forecast in Current state. Evaluate predicted physical "
            "benefit and execution risk using only PDE/BC/IC residual evidence, "
            "optimization state, architecture, budget, and causal action-outcome "
            "history. Feedback with credit_eligible=false or feedback_status="
            "credit_withheld is neutral uncertainty, never evidence that the "
            "action failed. Read withheld_kind precisely: "
            "same_direction_low_confidence means both probes support the same "
            "qualitative trend but numeric credit is unsafe; "
            "opposite_probe_directions means the action did not generalize "
            "consistently across probes; opposite_worst_region_directions means "
            "global averages agree but a worst physical region does not. Use the "
            "bounded qualitative_credit_hint only as weak directional evidence, "
            "never as formal causal credit. Burgers region_continuation_utility "
            "and worst-region fields are held-out shock-window PDE-residual "
            "evidence, not solution error. Never use reference-solution error "
            "or MSE. Score every "
            "proposal exactly once, including the explicit KEEP baseline. Prefer "
            "KEEP when the current residual trajectory is still improving, when "
            "forecast evidence is weak, or when an intervention's recovery risk "
            "is not clearly outweighed by state-specific physical benefit. Do "
            "not score an intervention above KEEP merely because it is more "
            "active, novel, or larger. Treat each operations list as an ordered "
            "atomic request: assess the physical rationale and interaction of "
            "the complete sequence, its targets, and its explicit parameters, "
            "not merely the high-level action label. Return only JSON with this schema: "
            '{"scores":[{"policy_id":"exact id","score":0.0,'
            '"reason":"brief evidence-based reason"}]}. Scores must be finite '
            "numbers in [0,1]. Do not invent or modify candidate programs; your "
            "only job is scoring.\n"
            "Executor capability contract: "
            f"{json.dumps(CAPABILITY_REGISTRY, ensure_ascii=False, default=str)}\n"
            "Static prior knowledge below is non-binding and contains no "
            "reference-solution score. Use it only to assess physical fit and "
            "risk; actual state/action/outcome evidence overrides it.\n"
            f"Current state: {json.dumps(compact_state, ensure_ascii=False, default=str)}\n"
            f"Proposals: {json.dumps([item.to_dict() for item in proposals], ensure_ascii=False, default=str)}\n"
            f"Per-policy causal evidence: {json.dumps(policy_evidence, ensure_ascii=False, default=str)}\n"
            f"Static prior knowledge: {json.dumps(self.prior_context, ensure_ascii=False, default=str)}\n"
            f"Shared execution history: {json.dumps(history, ensure_ascii=False, default=str)}"
        )

    def _tie_break_scoring_prompt(
        self,
        state: Mapping[str, Any],
        tied: Sequence[ProgramProposal],
        initial_scores: Mapping[str, tuple[float, str]],
    ) -> str:
        """Ask the scorer to discriminate only among first-round top ties."""

        compact_state = compact_policy_state(state)
        if state.get("online_adaptation"):
            compact_state["online_adaptation"] = deepcopy(state["online_adaptation"])
        history = compact_shared_history(self.shared_history)
        evidence = {
            item.policy_id: compact_policy_credit(item)
            for item in self.policies
            if item.policy_id in {proposal.policy_id for proposal in tied}
        }
        if any(
            proposal.policy_id == KEEP_CONTINUATION_BASELINE_POLICY_ID
            for proposal in tied
        ):
            evidence[KEEP_CONTINUATION_BASELINE_POLICY_ID] = {
                "role": "explicit_keep_continuation_baseline",
                "credit": "not_evolvable",
                "interpretation": (
                    "continuing the current optimizer trajectory without online "
                    "adaptation"
                ),
            }
        initial = [
            {
                "policy_id": item.policy_id,
                "score": initial_scores[item.policy_id][0],
                "reason": initial_scores[item.policy_id][1],
            }
            for item in tied
        ]
        return (
            "The first independent scoring round produced an exact tie for the "
            "highest score. Re-evaluate only the tied online PINN adaptation "
            "proposals more discriminatively. The explicit KEEP baseline, when "
            "present, means continuing current training without online adaptation. "
            "Compare causal physics fit, "
            "state-conditioned execution evidence, intervention risk, and "
            "remaining recovery budget. Never use MSE or any reference solution. "
            "Do not reward a candidate merely for its list position or evolution "
            "mode name. Prefer KEEP if intervention benefit is uncertain or does "
            "not clearly exceed continuation. Return only JSON: "
            '{"scores":[{"policy_id":"exact id","score":0.0,'
            '"reason":"specific discriminating reason"}],'
            '"winner_policy_id":"one highest-scoring tied id",'
            '"winner_reason":"why it wins the tie"}. '
            "Score every tied candidate exactly once. Prefer distinct numeric "
            "scores; if they remain numerically equal, winner_policy_id is "
            "mandatory and must identify one of the highest-scoring candidates.\n"
            "Executor capability contract: "
            f"{json.dumps(CAPABILITY_REGISTRY, ensure_ascii=False, default=str)}\n"
            f"Current state: {json.dumps(compact_state, ensure_ascii=False, default=str)}\n"
            f"First-round tied scores: {json.dumps(initial, ensure_ascii=False, default=str)}\n"
            f"Tied proposals: {json.dumps([item.to_dict() for item in tied], ensure_ascii=False, default=str)}\n"
            f"Tied-policy causal evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}\n"
            f"Shared execution history: {json.dumps(history, ensure_ascii=False, default=str)}"
        )

    def get(self, policy_id: str) -> PolicyCandidate:
        active = next(
            (item for item in self.policies if item.policy_id == policy_id),
            None,
        )
        if active is not None:
            return active
        archived = self.policy_archive.get(str(policy_id))
        if archived is not None:
            return archived
        raise KeyError(f"Unknown policy_id: {policy_id}")

    def append_shared_evidence(self, evidence: Mapping[str, Any]) -> None:
        intervention_id = str(evidence.get("intervention_id") or "")
        if not intervention_id:
            raise ValueError("Shared evidence requires intervention_id")
        existing = next(
            (item for item in self.shared_history if item["intervention_id"] == intervention_id),
            None,
        )
        if existing is None:
            self.shared_history.append(deepcopy(dict(evidence)))
        else:
            existing.update(deepcopy(dict(evidence)))

    def apply_direct_credit(
        self,
        *,
        intervention_id: str,
        policy_id: str,
        state_id_value: str,
        horizon: str,
        credit: float,
        feedback: Mapping[str, Any],
    ) -> None:
        """Update only the policy that produced the executed program."""

        if not bool(feedback.get("credit_eligible", False)) or feedback.get("confounded"):
            return
        policy = self.get(policy_id)
        raw_value = float(credit)
        if not math.isfinite(raw_value):
            raise ValueError("Policy credit must be finite")
        feedback_payload = deepcopy(dict(feedback))
        status = str(
            feedback_payload.get("feedback_status")
            or causal_feedback_status(feedback_payload)
        )
        if status in {"credit_withheld", "confounded"}:
            return
        if status not in {"success", "failure"}:
            raise ValueError(f"Unsupported policy feedback_status: {status}")
        # Outcome classification, not the sign of a noisy aggregate utility,
        # owns causal credit.  Cross-probe/component guards can reject a trial
        # even when its aggregate utility remains positive; such a trial must
        # never be learned as a success.
        magnitude = abs(raw_value)
        value = magnitude if status == "success" else -magnitude
        if horizon == "short":
            policy.execution_count += 1
        if status == "success":
            policy.success_count += 1
        else:
            policy.failure_count += 1
        rate = 1.0 / max(1, policy.success_count + policy.failure_count)
        policy.global_credit += rate * (value - policy.global_credit)
        old = float(policy.state_conditioned_credit.get(state_id_value, 0.0))
        policy.state_conditioned_credit[state_id_value] = old + rate * (value - old)
        policy.recent_results.append(
            {
                "intervention_id": str(intervention_id),
                "horizon": str(horizon),
                "credit": value,
                "raw_credit": raw_value,
                "feedback_status": status,
                "feedback": feedback_payload,
            }
        )
        del policy.recent_results[:-12]

    def can_evolve(self) -> bool:
        for index, item in enumerate(self.shared_history, start=1):
            if index <= self.last_evolution_evidence_sequence:
                continue
            feedback = dict(item.get("feedback") or {})
            delayed_feedback_available = (
                "medium" in feedback or "persistent" in feedback
            )
            causal_feedback_available = any(
                bool(dict(value).get("credit_eligible", False))
                for value in feedback.values()
                if isinstance(value, Mapping)
            )
            qualitative_feedback_available = any(
                abs(float(dict(value).get("qualitative_credit_hint", 0.0))) > 0.0
                for value in feedback.values()
                if isinstance(value, Mapping)
                and isinstance(
                    dict(value).get("qualitative_credit_hint", 0.0),
                    (int, float),
                )
            )
            if delayed_feedback_available and (
                causal_feedback_available or qualitative_feedback_available
            ):
                return True
        return False

    @staticmethod
    def _parent_evidence_key(candidate: PolicyCandidate) -> tuple[float, ...]:
        completed = candidate.success_count + candidate.failure_count
        success_balance = (
            (candidate.success_count - candidate.failure_count) / completed
            if completed
            else 0.0
        )
        return (
            1.0 if candidate.execution_count > 0 else 0.0,
            float(candidate.global_credit),
            float(success_balance),
            float(candidate.execution_count),
            float(candidate.last_selected_step or -1),
        )

    def _select_evolution_parents(
        self,
        *,
        mode: str,
        state: Mapping[str, Any],
        generation_kind: str,
    ) -> tuple[PolicyCandidate, ...]:
        """Deterministically assign parents before asking the LLM to mutate.

        Parent choice is system-owned so lineage cannot disappear when a model
        omits or invents ``parent_ids``.  Modes may legitimately converge on
        the same parent; diversity comes from their mutation objectives rather
        than a duplicate-rejection rule.
        """

        if generation_kind == "initialization":
            return ()
        candidates = list(self.policies)
        if not candidates:
            return ()
        base_rank = {
            item.policy_id: self._parent_evidence_key(item) for item in candidates
        }
        ranked = sorted(
            candidates,
            key=lambda item: (base_rank[item.policy_id], item.policy_id),
            reverse=True,
        )
        if mode == "physics_guided":
            lineage = [
                item for item in candidates if item.creation_mode == "physics_guided"
            ] or candidates
            physics = dict(state.get("physics") or {})
            dominant = str(physics.get("dominant_category") or "")
            condition_key = {
                "pde": "physics.pde_residual_dominant",
                "boundary": "physics.boundary_condition_dominant",
                "initial": "physics.initial_condition_dominant",
            }.get(dominant)

            def physics_key(item: PolicyCandidate) -> tuple[Any, ...]:
                conditions = dict(item.policy_spec.get("conditions") or {})
                aligned = bool(condition_key and conditions.get(condition_key) is True)
                program = item.propose(state=state, history=self.shared_history)
                applicable = program.has_intervention()
                return (
                    1.0 if aligned else 0.0,
                    1.0 if applicable else 0.0,
                    base_rank[item.policy_id],
                    item.policy_id,
                )

            return (max(lineage, key=physics_key),)
        if mode == "architecture_guided":
            # Prefer a policy with real structural evidence, including negative
            # evidence that can teach a target/mechanism correction.
            lineage = [
                item
                for item in candidates
                if item.creation_mode == "architecture_guided"
            ] or candidates
            return (
                max(
                    lineage,
                    key=lambda item: (
                        1.0 if item.recent_results else 0.0,
                        float(item.execution_count),
                        abs(float(item.global_credit)),
                        base_rank[item.policy_id],
                        item.policy_id,
                    ),
                ),
            )
        if mode == "synthesis":
            first = ranked[0]
            first_program = dict(first.policy_spec.get("program") or {})
            complement = next(
                (
                    item
                    for item in ranked[1:]
                    if (
                        dict(item.policy_spec.get("program") or {}).get("action"),
                        dict(item.policy_spec.get("program") or {}).get("mechanism"),
                    )
                    != (first_program.get("action"), first_program.get("mechanism"))
                ),
                ranked[1] if len(ranked) > 1 else first,
            )
            return (first, complement) if complement.policy_id != first.policy_id else (first,)
        if mode == "novelty":
            lineage = [
                item for item in candidates if item.creation_mode == "novelty"
            ] or candidates
            return (
                max(
                    lineage,
                    key=lambda item: (base_rank[item.policy_id], item.policy_id),
                ),
            )
        # REFINE uses the strongest execution-grounded parent.
        return (ranked[0],)

    @staticmethod
    def _mutation_audit(
        candidate: PolicyCandidate,
        parents: Sequence[PolicyCandidate],
    ) -> dict[str, Any]:
        child_spec = dict(candidate.policy_spec)
        parent_changes: dict[str, list[str]] = {}
        for parent in parents:
            parent_spec = dict(parent.policy_spec)
            changed: list[str] = []
            if parent_spec.get("conditions") != child_spec.get("conditions"):
                changed.append("conditions")
            parent_program = dict(parent_spec.get("program") or {})
            child_program = dict(child_spec.get("program") or {})
            changed.extend(
                f"program.{field}"
                for field in sorted(set(parent_program) | set(child_program))
                if parent_program.get(field) != child_program.get(field)
            )
            parent_changes[parent.policy_id] = changed
        return {
            "parent_ids": [item.policy_id for item in parents],
            "changed_fields_by_parent": parent_changes,
        }

    def evolve(
        self,
        provider: Any,
        *,
        state: Mapping[str, Any],
        maximum_attempts_per_mode: int = 5,
        maximum_transient_attempts_per_mode: int = 2,
        transient_retry_initial_seconds: float = 1.0,
        transient_retry_max_seconds: float = 4.0,
        temperature: float = 0.35,
        generation_kind: str = "evolution",
        force: bool = False,
        random_direction: bool = False,
    ) -> list[PolicyCandidate]:
        """Generate one candidate per EOH mode, then replace the population.

        ``force`` regenerates without waiting for new execution feedback; the
        controller uses it for applicability repairs after the whole
        population expanded to KEEP.
        """

        if generation_kind not in {"initialization", "evolution"}:
            raise ValueError(f"Unsupported generation kind: {generation_kind}")
        if generation_kind == "evolution" and not self.can_evolve() and not force:
            return []
        if provider is None or not callable(getattr(provider, "complete_text", None)):
            raise TypeError("Policy evolution requires a provider with complete_text(...)")
        next_generation: list[PolicyCandidate] = []
        audits: list[dict[str, Any]] = []
        self.last_evolution_provider_calls = 0
        self.last_evolution_transient_failures = 0
        transient_circuit_open = False
        parent_snapshot = [compact_policy_candidate(item) for item in self.policies]
        generation_gap_start = len(self.capability_gap_events)
        repair_directive = (
            "\nThis is an applicability repair regeneration: the previous "
            "population could not act at the current state. Every candidate "
            "must target the current diagnostic moment with conditions that "
            "already hold in the Current state JSON below."
            if force
            else ""
        )
        parent_plan = {
            mode: (
                (random.choice(self.policies),) if generation_kind != "initialization" else ()
            ) if random_direction else self._select_evolution_parents(
                mode=mode, state=state, generation_kind=generation_kind
            )
            for mode in EVOLUTION_MODES
        }
        for mode_index, mode in enumerate(EVOLUTION_MODES):
            accepted: PolicyCandidate | None = None
            errors: list[str] = []
            schema_errors: list[str] = []
            rejected_payload: str | None = None
            assigned_parents = parent_plan[mode]
            randomized_action = (
                random.choice(tuple(action for action in ONLINE_ADAPTATION_ACTIONS if action != KEEP))
                if random_direction else None
            )
            transient_attempts = 0
            attempt_limit = max(1, int(maximum_attempts_per_mode))
            for attempt in range(1, attempt_limit + 1):
                gap_attempt_start = len(self.capability_gap_events)
                if transient_circuit_open:
                    errors.append(
                        "provider circuit open after repeated transient failure; "
                        "role-preserving fallback used without another LLM call"
                    )
                    break
                prompt = self._evolution_prompt(
                    mode=mode,
                    state=state,
                    generation_kind=generation_kind,
                    parents=assigned_parents,
                    generation_peers=next_generation,
                    random_direction=random_direction,
                ) + repair_directive
                if randomized_action is not None:
                    prompt += (
                        f"\nRandomized evolution direction: set policy_spec.program.action "
                        f"to {randomized_action}. Select a compatible mechanism and target. "
                        "This direction was sampled independently of execution outcomes."
                    )
                if schema_errors:
                    # Repair, not reinvent: feed the rejection back so the
                    # generator fixes its own schema violations.
                    prompt += _schema_repair_directive(
                        schema_errors[-2:], rejected_payload
                    )
                response: Any | None = None
                try:
                    if isinstance(provider, BudgetedProvider) and budget_reason(provider.state, provider.config):
                        raise LLMBudgetExceeded("generation stopped at shared LLM budget")
                    self.last_evolution_provider_calls += 1
                    response = provider.complete_text(
                        [{"role": "user", "content": prompt}],
                        temperature=float(temperature),
                        max_tokens=2200,
                        task=f"online_policy_{generation_kind}_{mode}",
                    )
                    if not bool(getattr(response, "success", True)):
                        provider_error = str(
                            getattr(response, "error", "provider failure")
                        )
                        if _is_transient_policy_provider_error(provider_error):
                            raise _TransientPolicyProviderFailure(provider_error)
                        raise ValueError(provider_error)
                    raw, parse_errors = extract_json_object(
                        str(getattr(response, "text", response)),
                        preferred_keys=("policy_spec", "program"),
                    )
                    if raw is None:
                        raise ValueError(
                            "candidate generator returned invalid JSON: "
                            + "; ".join(parse_errors)
                        )
                    self.capability_gap_events.extend(
                        extract_capability_gap_observations(
                            raw,
                            state=state,
                            mode=mode,
                            generation_kind=generation_kind,
                            attempt=attempt,
                        )
                    )
                    spec = _normalize_generated_policy_spec(raw)
                    contract_error = program_rejection(spec["program"], state.get("online_adaptation") or {}, next_generation)
                    if contract_error:
                        raise ValueError(contract_error)
                    condition_problem = conditions_rejection_reason(
                        dict(spec.get("conditions") or {})
                    )
                    if condition_problem is not None:
                        raise ValueError(condition_problem)
                    candidate = PolicyCandidate(
                        policy_id=f"g{self.generation + 1}-{mode}",
                        version=(
                            max(item.version for item in assigned_parents) + 1
                            if assigned_parents
                            else 1
                        ),
                        # Lineage is selected by the system, not delegated to
                        # an unreliable free-form model field.
                        parent_ids=tuple(item.policy_id for item in assigned_parents),
                        creation_step=int(state.get("step", state.get("global_step", 0)) or 0),
                        creation_mode=mode,
                        policy_spec=spec,
                    )
                    # Candidates compete at the diagnostic moment they are
                    # generated for, so an intervention must be executable
                    # right now; only explicit KEEP is exempt.
                    applicability_problem = candidate_applicability_reason(
                        state,
                        dict(spec.get("conditions") or {}),
                        AdaptationProgram.from_dict(spec["program"]),
                    )
                    if applicability_problem is not None:
                        raise ValueError(applicability_problem)
                    for gap in self.capability_gap_events[gap_attempt_start:]:
                        gap["executable_policy_status"] = "accepted"
                    accepted = candidate
                    break
                except LLMBudgetExceeded:
                    raise
                except _TransientPolicyProviderFailure as exc:
                    transient_attempts += 1
                    self.last_evolution_transient_failures += 1
                    errors.append(
                        f"attempt {attempt}: {type(exc).__name__}: {exc}"
                    )
                    rejected_payload = str(
                        getattr(response, "text", "") or ""
                    )[:1200]
                    if transient_attempts >= max(
                        1, int(maximum_transient_attempts_per_mode)
                    ):
                        transient_circuit_open = True
                        errors.append(
                            "transient provider retry budget exhausted; "
                            "circuit opened for the remaining generation"
                        )
                        break
                    delay = min(
                        max(0.0, float(transient_retry_max_seconds)),
                        max(0.0, float(transient_retry_initial_seconds))
                        * (2.0 ** max(0, transient_attempts - 1)),
                    )
                    if delay > 0.0:
                        time.sleep(delay)
                except Exception as exc:
                    error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                    for gap in self.capability_gap_events[gap_attempt_start:]:
                        gap["executable_policy_status"] = "rejected"
                        gap["validation_error"] = error[:600]
                    errors.append(error)
                    schema_errors.append(error)
                    rejected_payload = str(
                        getattr(response, "text", "") or ""
                    )[:1200]
            if accepted is None:
                # Keep the five semantic roles intact during provider outages.
                # This is not candidate de-duplication: a role reuses its own
                # previous lineage before falling back to its indexed seed.
                role_parent = next(
                    (
                        item
                        for item in self.policies
                        if item.creation_mode == mode
                    ),
                    None,
                )
                parent = role_parent or (
                    assigned_parents[0]
                    if assigned_parents
                    else self.policies[mode_index]
                )
                accepted = PolicyCandidate(
                    policy_id=f"g{self.generation + 1}-{mode}-fallback",
                    version=parent.version + 1,
                    parent_ids=(parent.policy_id,),
                    creation_step=int(state.get("step", state.get("global_step", 0)) or 0),
                    creation_mode=mode,
                    policy_spec=deepcopy(dict(parent.policy_spec)),
                    global_credit=parent.global_credit,
                    state_conditioned_credit=deepcopy(parent.state_conditioned_credit),
                )
            if program_rejection(accepted.policy_spec["program"], state.get("online_adaptation") or {}, next_generation):
                accepted.policy_spec = {**accepted.policy_spec, "conditions": {},
                    "program": AdaptationProgram.keep(diagnosis="unsupported_bottleneck_fallback", reason="no legal evidence-backed candidate").to_dict()}
            next_generation.append(accepted)
            lineage_parents = tuple(
                item
                for item in self.policies
                if item.policy_id in accepted.parent_ids
            )
            audits.append(
                {
                    "mode": mode,
                    "policy_id": accepted.policy_id,
                    "errors": errors,
                    "fallback_used": accepted.policy_id.endswith("-fallback"),
                    "provider_calls": (
                        sum(1 for error in errors if error.startswith("attempt "))
                    ),
                    "mutation": self._mutation_audit(accepted, lineage_parents),
                }
            )
        for parent in self.policies:
            self.policy_archive[parent.policy_id] = parent
        self.policies = next_generation
        self.generation += 1
        self.llm_initialized = True
        self.last_evolution_evidence_sequence = len(self.shared_history)
        hypothesis_groups: dict[str, list[str]] = {}
        for candidate in next_generation:
            signature = adaptation_hypothesis_signature(
                candidate.policy_spec["program"]
            )
            hypothesis_groups.setdefault(signature, []).append(
                candidate.policy_id
            )
        self.evolution_events.append(
            {
                "generation": self.generation,
                "generation_kind": generation_kind,
                "step": int(state.get("step", state.get("global_step", 0)) or 0),
                "modes": list(EVOLUTION_MODES),
                "parents": parent_snapshot,
                "candidates": [item.to_dict() for item in next_generation],
                "audit": audits,
                "diversity_audit": {
                    "candidate_count": len(next_generation),
                    "distinct_hypothesis_count": len(hypothesis_groups),
                    "hypothesis_groups": hypothesis_groups,
                    "enforcement": "prompt_guidance_only",
                },
                "capability_gaps": deepcopy(
                    self.capability_gap_events[generation_gap_start:]
                ),
            }
        )
        return next_generation

    def _evolution_prompt(
        self,
        *,
        mode: str,
        state: Mapping[str, Any],
        generation_kind: str = "evolution",
        parents: Sequence[PolicyCandidate] | None = None,
        generation_peers: Sequence[PolicyCandidate] = (),
        random_direction: bool = False,
    ) -> str:
        compact_state = compact_policy_state(state)
        if state.get("online_adaptation"):
            compact_state["online_adaptation"] = deepcopy(state["online_adaptation"])
        if random_direction:
            compact_state.pop("previous_morph_events", None)
        history = [] if random_direction else compact_shared_history(self.shared_history)
        assigned_parents = tuple(
            parents
            if parents is not None
            else self._select_evolution_parents(
                mode=mode, state=state, generation_kind=generation_kind
            )
        )
        parent_payload = [
            {"policy_id": item.policy_id, "policy_spec": deepcopy(item.policy_spec)}
            if random_direction else compact_policy_candidate(item)
            for item in assigned_parents
        ]
        assigned_ids = {item.policy_id for item in assigned_parents}
        population = [
            ({"policy_id": item.policy_id, "policy_spec": deepcopy(item.policy_spec)}
             if random_direction else compact_policy_candidate(item))
            for item in self.policies
            if item.policy_id not in assigned_ids
        ]
        mode_guidance = (
            "Choose a valid mutation direction at random from the registered intervention space; "
            "do not infer direction from previous action outcomes or policy credit."
            if random_direction else EVOLUTION_MODE_GUIDANCE[mode]
        )
        accepted_peer_payload = [
            {
                "policy_id": item.policy_id,
                "creation_mode": item.creation_mode,
                "hypothesis_signature": adaptation_hypothesis_signature(
                    item.policy_spec["program"]
                ),
                "conditions": deepcopy(
                    dict(item.policy_spec.get("conditions") or {})
                ),
                "program": deepcopy(
                    dict(item.policy_spec.get("program") or {})
                ),
            }
            for item in generation_peers
        ]
        diversity_instruction = (
            "No peer has been accepted yet; establish a scientifically clear "
            "hypothesis for your assigned role."
            if not accepted_peer_payload
            else (
                "Inspect the candidates already accepted in this generation. "
                "After satisfying evidence and your role contract, search for a "
                "scientifically defensible hypothesis they do not cover. A "
                "change only to strength, recovery_steps, cooldown_steps, or "
                "wording is cosmetic and does not create diversity. Convergence "
                "on an existing action/mechanism/target is allowed when the "
                "evidence independently compels it, but the program reason must "
                "state that evidence. This is generation guidance, not a hard "
                "uniqueness constraint."
            )
        )
        lineage_instruction = (
            "This is generation-0 initialization, so there are no parents. "
            "Create a seed policy with this mode's scientific emphasis."
            if generation_kind == "initialization"
            else (
                "The system has already selected the authoritative parent IDs "
                f"{[item.policy_id for item in assigned_parents]}. Mutate these "
                "assigned parents; do not choose different parents. The system "
                "records lineage itself, so return policy_spec only (a top-level "
                "parent_ids field, if emitted, is ignored)."
            )
        )
        feedback_instruction = (
            "Random evolution ablation: execution feedback is withheld. Choose a mutation "
            "without using previous outcomes or policy credit. Never use MSE or L2.\n"
            if random_direction else
            "Current causal physics feedback overrides prior guidance. Never use MSE, L2, or "
            "any reference-solution metric. Treat credit_eligible=false or "
            "feedback_status=credit_withheld as neutral uncertainty, not as a "
            "failed action. A withheld_kind of same_direction_low_confidence "
            "still supplies a qualitative direction and a bounded weak hint; "
            "opposite_probe_directions or opposite_worst_region_directions "
            "signals poor cross-probe generalization. For Burgers, explicitly "
            "use region_continuation_utility and worst-region fields as "
            "reference-free shock-window PDE evidence.\n"
        )
        return (
            "Return exactly one JSON object for an online PINN adaptation "
            f"policy. Evolution mode: {mode}.\n"
            f"Mode-specific mutation contract: {mode_guidance}\n"
            f"Within-generation diversity contract: {diversity_instruction}\n"
            f"Lineage contract: {lineage_instruction}\n"
            "Static prior knowledge is non-binding guidance. Translate it only "
            f"into a registered action in [{', '.join(ONLINE_ADAPTATION_ACTIONS)}] "
            "and a compatible registered mechanism for policy_spec. You may "
            "describe a missing mechanism only in capability_gap_proposals; "
            "never place it in the executable policy or emit executable code. "
            f"{feedback_instruction}"
            f"Schema contract:\n{_GENERATION_SCHEMA_CONTRACT}\n"
            "Valid example (structure only; adapt to the evidence):\n"
            f"{json.dumps(_GENERATION_EXAMPLE, ensure_ascii=False)}\n"
            f"Current state: {json.dumps(compact_state, ensure_ascii=False, default=str)}\n"
            f"Static prior knowledge: {json.dumps(self.prior_context, ensure_ascii=False, default=str)}\n"
            f"Shared execution history: {json.dumps(history, ensure_ascii=False, default=str)}\n"
            f"Assigned parent policies: {json.dumps(parent_payload, ensure_ascii=False, default=str)}\n"
            f"Other current policies: {json.dumps(population, ensure_ascii=False, default=str)}\n"
            "Candidates already accepted in this generation: "
            f"{json.dumps(accepted_peer_payload, ensure_ascii=False, default=str)}"
        )

    def _repair_all_keep_generation(
        self,
        provider: Any,
        *,
        state: Mapping[str, Any],
        temperature: float,
        generation_kind: str,
        next_generation: list[PolicyCandidate],
        audits: list[dict[str, Any]],
        repair_directive: str,
    ) -> None:
        """One bounded round to inject an executable intervention.

        Called when every candidate of a fresh generation came back KEEP.
        Each mode gets a single extra attempt whose response is accepted only
        if it carries a non-KEEP action executable at the current state;
        the round stops at the first success.  Best-effort: on failure the
        all-KEEP generation stands and the controller-side applicability
        repair handles it.
        """

        directive = (
            "\nYour previous response was a KEEP program, "
            "but this generation must contain at least one executable "
            "intervention. Return the corrected complete JSON object with "
            "a non-KEEP high-level action whose conditions already hold in the "
            "Current state JSON."
        )
        for index, mode in enumerate(EVOLUTION_MODES):
            repair_parent = next_generation[index]
            prompt = (
                self._evolution_prompt(
                    mode=mode,
                    state=state,
                    generation_kind="evolution",
                    parents=(repair_parent,),
                )
                + repair_directive
                + directive
            )
            try:
                response = provider.complete_text(
                    [{"role": "user", "content": prompt}],
                    temperature=float(temperature),
                    max_tokens=2200,
                    task=f"online_policy_{generation_kind}_{mode}_intervention_repair",
                )
                if not bool(getattr(response, "success", True)):
                    raise ValueError(str(getattr(response, "error", "provider failure")))
                raw, parse_errors = extract_json_object(
                    str(getattr(response, "text", response)),
                    preferred_keys=("policy_spec", "program"),
                )
                if raw is None:
                    raise ValueError(
                        "candidate generator returned invalid JSON: "
                        + "; ".join(parse_errors)
                    )
                spec = _normalize_generated_policy_spec(raw)
                condition_problem = conditions_rejection_reason(
                    dict(spec.get("conditions") or {})
                )
                if condition_problem is not None:
                    raise ValueError(condition_problem)
                candidate = PolicyCandidate(
                    policy_id=f"g{self.generation + 1}-{mode}-intervention",
                    version=repair_parent.version + 1,
                    parent_ids=(repair_parent.policy_id,),
                    creation_step=int(state.get("step", state.get("global_step", 0)) or 0),
                    creation_mode=mode,
                    policy_spec=spec,
                )
                program = AdaptationProgram.from_dict(spec["program"])
                if candidate_applicability_reason(
                    state, dict(spec.get("conditions") or {}), program
                ) is not None:
                    raise ValueError(
                        "repair candidate is not executable at the current state"
                    )
                if not program.has_intervention():
                    raise ValueError("repair candidate must carry an intervention")
                replaced_id = next_generation[index].policy_id
                next_generation[index] = candidate
                audits.append(
                    {
                        "mode": mode,
                        "policy_id": candidate.policy_id,
                        "intervention_repair": True,
                        "replaced_keep": replaced_id,
                        "mutation": self._mutation_audit(
                            candidate, (repair_parent,)
                        ),
                        "errors": [],
                    }
                )
                return
            except Exception:
                # Best-effort round: try the next mode.
                continue

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "population_size": POLICY_POPULATION_SIZE,
            "generation": int(self.generation),
            "llm_initialized": bool(self.llm_initialized),
            "policies": [item.to_dict() for item in self.policies],
            "policy_archive": {
                policy_id: item.to_dict()
                for policy_id, item in self.policy_archive.items()
            },
            "shared_history": deepcopy(self.shared_history),
            "prior_context": deepcopy(self.prior_context),
            "last_evolution_evidence_sequence": int(self.last_evolution_evidence_sequence),
            "evolution_events": deepcopy(self.evolution_events),
            "capability_gap_events": deepcopy(self.capability_gap_events),
            "applicability_repair_pending": bool(self.applicability_repair_pending),
            "applicability_repair_count": int(self.applicability_repair_count),
        }

    def load_state_dict(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        candidates = [PolicyCandidate.from_dict(item) for item in payload.get("policies") or []]
        if len(candidates) != POLICY_POPULATION_SIZE:
            raise ValueError("Checkpoint policy population must contain exactly 5 candidates")
        self.policies = candidates
        self.policy_archive = {
            str(policy_id): PolicyCandidate.from_dict(item)
            for policy_id, item in dict(payload.get("policy_archive") or {}).items()
        }
        self.generation = int(payload.get("generation", 0))
        self.llm_initialized = bool(
            payload.get("llm_initialized", self.generation > 0)
        )
        self.shared_history = deepcopy(list(payload.get("shared_history") or []))
        self.prior_context = deepcopy(list(payload.get("prior_context") or []))[:8]
        self.last_evolution_evidence_sequence = int(
            payload.get("last_evolution_evidence_sequence", 0)
        )
        self.evolution_events = deepcopy(list(payload.get("evolution_events") or []))
        self.capability_gap_events = deepcopy(
            list(payload.get("capability_gap_events") or [])
        )
        self.applicability_repair_pending = bool(
            payload.get("applicability_repair_pending", False)
        )
        self.applicability_repair_count = int(
            payload.get("applicability_repair_count", 0)
        )



DEFAULT_ONLINE_MORPH_CONFIG: dict[str, Any] = {
    "enabled": False,
    "decision_mode": "deterministic",
    "allowed_primitives": list(PRIMITIVE_LIBRARY),
    "adaptation_freeze_step": 14000,
    "diagnostic_interval": 200,
    "gradient_diagnostics": {"persistence_windows": 2},
    "policy_set": {
        "enabled": True,
        "population_size": 5,
        "scoring": {
            "temperature": 0.0,
        },
        "evolution": {
            "enabled": False,
            "maximum_llm_calls_per_run": 0,
            "require_insufficient_progress": True,
            "temperature": 0.2,
            "offspring_per_generation": 5,
            "maximum_attempts_per_mode": 5,
            "maximum_transient_attempts_per_mode": 2,
            "transient_retry_initial_seconds": 1.0,
            "transient_retry_max_seconds": 4.0,
            "minimum_valid_offspring": 5,
            "maximum_applicability_repairs": 5,
        },
    },
    "minimum_iterations_before_morph": 400,
    "diagnostic_window": 100,
    "plateau_relative_improvement_threshold": 0.01,
    "plateau_min_windows": 2,
    # Legacy aliases remain readable by old AlgorithmSpecs.
    "cooldown_iterations": 400,
    "plateau_window": 40,
    "plateau_relative_improvement": 1e-3,
    "min_steps_between_morphs": 200,
    # Keep one action-outcome attribution window causally isolated. A new
    # intervention may begin only after the previous persistent probe closes.
    "block_new_morph_until_persistent_feedback": True,
    "min_remaining_iterations_for_morph": 200,
    "operator_retry_cooldown": 500,
    "max_emergency_recoveries_per_operator": 2,
    "minimum_loss": 0.0,
    "pde_dominance_ratio": 5.0,
    "residual_concentration_ratio": 8.0,
    "architecture_executor": {
        "recovery_window_steps": {
            "mild": 100,
            "moderate": 250,
            "strong": 500,
        }
    },
    # Deterministic parameters for registered additive mechanisms. The LLM may
    # select a capability name but cannot set raw frequencies or executable code.
    "fourier_frequency_scale": 1.0,
    "localized_overlap": 1.5,
    "localized_expert": {
        "center_strategy": "residual_farthest",
        "residual_power": 1.0,
        "distance_power": 1.0,
    },
    "residual_top_k_fraction": 0.05,
    "residual_top_k_mass_threshold": 0.5,
    "activation_only_iterations": 10,
    "new_parameters_only_iterations": 50,
    "joint_learning_rate_scale": 0.1,
    "utility_evaluation_enabled": True,
    "utility_window_iterations": 100,
    "minimum_relative_improvement": 0.01,
    "utility": {
        "enabled": True,
        "mode": "component_relative",
        "minimum_relative_improvement": 0.01,
        "max_component_degradation": 0.10,
        "catastrophic_rollback_enabled": True,
        "catastrophic_component_degradation_threshold": 3.0,
        "continuation_history_points": 7,
        "continuation_minimum_confidence": 0.35,
        "continuation_maximum_extrapolation_factor": 4.0,
        "continuation_reject_clipped_forecast": True,
        "minimum_trigger_observations": 3,
        "short_horizon_steps": 100,
        "medium_horizon_steps": 500,
        "persistent_horizon_steps": 1000,
        "retire_failed_additive_on_persistent": True,
        "weights": {
            "pde": 1.0,
            "boundary": 1.0,
            "initial": 1.0,
            "observation": 1.0,
        },
    },
    "optimizer_state_after_morph": {"mode": "preserve"},
    "physics_weight_growth_factor": 1.0,
    "function_preservation_tolerance": 1e-7,
    "derivative_preservation_tolerance": 1e-6,
    "derivative_check_dimensions": 2,
    "function_check_points": 128,
    "preservation": {
        "probe_size": 16,
        "function_tolerance": 1e-7,
        "first_derivative_tolerance": 1e-6,
        "second_derivative_tolerance": 1e-6,
        "pde_residual_tolerance": 1e-6,
        "adaptive_strict_morphism_tolerance": True,
        "adaptive_width_unit": 64,
        "adaptive_depth_unit": 2,
        "adaptive_first_derivative_tolerance_cap": 1e-3,
        "adaptive_second_derivative_tolerance_cap": 1e-2,
    },
}


RANDOM_MORPH_MODE = "random_morph"
RANDOM_SELECTION_MODE = "random_selection"
NO_MORPH_MODE = "no_morph"
PINNMORPH_MODE = "llm_evolution"


from morph.utils.llm_budget import BudgetedProvider, LLMBudgetExceeded, budget_reason, decision_signature, unchanged_decision


class OnlineMorphController:
    """Evidence-first selector for registry-backed architecture responses."""

    name = "online_network_morph"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        supplied = dict(config or {})
        merged = deepcopy(DEFAULT_ONLINE_MORPH_CONFIG)
        merged.update(supplied)
        for nested_name in (
            "localized_expert",
            "optimizer_state_after_morph",
            "gradient_diagnostics",
            "policy_set",
            "architecture_executor",
        ):
            nested = deepcopy(DEFAULT_ONLINE_MORPH_CONFIG[nested_name])
            nested.update(dict(supplied.get(nested_name) or {}))
            merged[nested_name] = nested
        supplied_policy = dict(supplied.get("policy_set") or {})
        policy_evolution = deepcopy(
            DEFAULT_ONLINE_MORPH_CONFIG["policy_set"]["evolution"]
        )
        policy_evolution.update(dict(supplied_policy.get("evolution") or {}))
        merged["policy_set"]["evolution"] = policy_evolution
        supplied_executor = dict(supplied.get("architecture_executor") or {})
        recovery_windows = deepcopy(
            DEFAULT_ONLINE_MORPH_CONFIG["architecture_executor"][
                "recovery_window_steps"
            ]
        )
        recovery_windows.update(
            dict(supplied_executor.get("recovery_window_steps") or {})
        )
        merged["architecture_executor"]["recovery_window_steps"] = recovery_windows
        preservation = deepcopy(DEFAULT_ONLINE_MORPH_CONFIG["preservation"])
        preservation.update(dict(supplied.get("preservation") or {}))
        if "preservation" not in supplied:
            preservation["probe_size"] = int(
                supplied.get("function_check_points", preservation["probe_size"])
            )
            preservation["function_tolerance"] = float(
                supplied.get(
                    "function_preservation_tolerance",
                    preservation["function_tolerance"],
                )
            )
            legacy_derivative = float(
                supplied.get(
                    "derivative_preservation_tolerance",
                    preservation["first_derivative_tolerance"],
                )
            )
            preservation["first_derivative_tolerance"] = legacy_derivative
            preservation["second_derivative_tolerance"] = legacy_derivative
        merged["preservation"] = preservation
        utility = deepcopy(DEFAULT_ONLINE_MORPH_CONFIG["utility"])
        supplied_utility = dict(supplied.get("utility") or {})
        utility.update(supplied_utility)
        utility["weights"] = {
            **deepcopy(DEFAULT_ONLINE_MORPH_CONFIG["utility"]["weights"]),
            **dict(supplied_utility.get("weights") or {}),
        }
        if "utility" not in supplied:
            utility["enabled"] = bool(
                supplied.get("utility_evaluation_enabled", utility["enabled"])
            )
            utility["minimum_relative_improvement"] = float(
                supplied.get(
                    "minimum_relative_improvement",
                    utility["minimum_relative_improvement"],
                )
            )
        merged["utility"] = utility
        # Flat aliases remain populated for old checkpoints and reporting.
        merged["utility_evaluation_enabled"] = bool(utility["enabled"])
        merged["minimum_relative_improvement"] = float(
            utility["minimum_relative_improvement"]
        )
        merged["function_check_points"] = int(preservation["probe_size"])
        merged["function_preservation_tolerance"] = float(
            preservation["function_tolerance"]
        )
        merged["derivative_preservation_tolerance"] = max(
            float(preservation["first_derivative_tolerance"]),
            float(preservation["second_derivative_tolerance"]),
        )
        if "diagnostic_window" not in supplied and "plateau_window" in supplied:
            merged["diagnostic_window"] = supplied["plateau_window"]
        if (
            "plateau_relative_improvement_threshold" not in supplied
            and "plateau_relative_improvement" in supplied
        ):
            merged["plateau_relative_improvement_threshold"] = supplied[
                "plateau_relative_improvement"
            ]
        if "min_steps_between_morphs" not in supplied and "cooldown_iterations" in supplied:
            merged["min_steps_between_morphs"] = supplied["cooldown_iterations"]
        allowed_source = supplied.get(
            "allowed_primitives", merged.get("allowed_primitives")
        )
        allowed = tuple(str(item).strip().upper() for item in allowed_source)
        if not allowed or not set(allowed).issubset(PRIMITIVE_LIBRARY):
            raise ValueError("online_morphing.allowed_primitives contains an unregistered primitive")
        merged["allowed_primitives"] = tuple(dict.fromkeys(allowed))
        policy_limits = dict(merged.get("policy_set") or {})
        evolution_limits = dict(policy_limits.get("evolution") or {})
        if int(policy_limits.get("population_size", 5)) != 5:
            raise ValueError("online policy population_size must be exactly 5")
        if int(evolution_limits.get("offspring_per_generation", 5)) != 5:
            raise ValueError("online policy offspring_per_generation must be exactly 5")
        if int(evolution_limits.get("minimum_valid_offspring", 5)) != 5:
            raise ValueError("online policy minimum_valid_offspring must be exactly 5")
        self.config = merged
        self.architecture_executor = DeterministicArchitectureExecutor(
            merged["architecture_executor"]
        )
        self.safe_execution_engine = SafeExecutionEngine(
            merged["architecture_executor"]
        )
        self.policy_population = PolicyPopulation()
        self.intervention_tracker = InterventionTracker()
        self.loss_history: list[float] = []
        self.last_successful_morph_step: int | None = None
        self.last_completed_intervention_step: int | None = None
        self.last_morph_iteration: int | None = None
        self.morph_count = 0
        self.diagnostic_count = 0
        self.rejected_decision_count = 0
        self.events: list[dict[str, Any]] = []
        self.diagnostic_events: list[dict[str, Any]] = []
        self.operator_failure_history: dict[str, dict[str, int]] = {}
        self.initial_parameter_count: int | None = None
        self.plateau_window_count = 0
        self.last_improvement_step: int | None = None
        self.physics_plateau_window_count = 0
        self.last_physics_improvement_step: int | None = None
        self.current_morph_phase = "idle"
        self.phase_remaining_iterations = 0
        self.active_trial: dict[str, Any] | None = None
        self.pending_persistent_feedback: list[dict[str, Any]] = []
        self.llm_budget_state = {"calls": 0, "wall_seconds": 0.0}
        self.last_decision_observation = {}
        self.intervention_sequence = 0
        self.warmup_until_iteration: int | None = None
        self.joint_transition_complete = True
        self.gradient_conflict_history: list[bool] = []
        self.physics_validation_history: list[dict[str, Any]] = []
        self.physics_audit_history: list[dict[str, Any]] = []
        self.registered_loss_intervention = False
        self.registered_optimization_intervention = False
        self.registered_sampling_intervention = False
        self.registered_curriculum_intervention = False
        policy_set_config = dict(self.config.get("policy_set") or {})
        self.policy_evolution_config = dict(
            policy_set_config.get("evolution") or {}
        )
        self.policy_evolution_calls = 0
        self.policy_evolution_cycles = 0
        self.policy_evolution_failures = 0
        self.policy_evolution_events: list[dict[str, Any]] = []
        self.policy_scoring_calls = 0
        self.policy_scoring_failures = 0
        self.policy_scoring_events: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def set_initial_parameter_count(self, count: int) -> None:
        if self.initial_parameter_count is None:
            self.initial_parameter_count = max(1, int(count))

    def set_registered_interventions(
        self,
        *,
        loss_intervention: bool,
        optimization_intervention: bool = False,
        sampling_intervention: bool = False,
        curriculum_intervention: bool = False,
    ) -> None:
        self.registered_loss_intervention = bool(loss_intervention)
        self.registered_optimization_intervention = bool(optimization_intervention)
        self.registered_sampling_intervention = bool(sampling_intervention)
        self.registered_curriculum_intervention = bool(curriculum_intervention)

    def record_physics_validation(self, step: int, metrics: Mapping[str, Any]) -> None:
        payload = deepcopy(dict(metrics))
        self.physics_validation_history.append(
            {"step": int(step), "metrics": payload}
        )
        maximum = max(
            16,
            int((self.config.get("utility") or {}).get("continuation_history_points", 7)) * 4,
        )
        if len(self.physics_validation_history) > maximum:
            del self.physics_validation_history[:-maximum]

    def record_physics_audit(self, step: int, metrics: Mapping[str, Any]) -> None:
        payload = deepcopy(dict(metrics))
        self.physics_audit_history.append(
            {"step": int(step), "metrics": payload}
        )
        maximum = max(
            16,
            int(
                (self.config.get("utility") or {}).get(
                    "continuation_history_points", 7
                )
            )
            * 4,
        )
        if len(self.physics_audit_history) > maximum:
            del self.physics_audit_history[:-maximum]

    def apply_physics_progress(
        self, diagnostic: OnlineMorphDiagnostics
    ) -> OnlineMorphDiagnostics:
        """Fuse fixed-probe physics progress into the hard adaptation gate.

        Training loss alone is not comparable across PINN pretraining and
        curriculum windows.  The fixed physics probe is, so a persistent
        plateau in its group-balanced score can open EOH even while the
        current training objective continues to decrease.
        """

        history_points = max(
            2,
            int(
                (self.config.get("utility") or {}).get(
                    "minimum_trigger_observations", 3
                )
            ),
            int(
                (self.config.get("utility") or {}).get(
                    "continuation_history_points", 7
                )
            ),
        )
        scores = [
            float(dict(item.get("metrics") or {}).get("total_score"))
            for item in self.physics_validation_history
            if isinstance(item, Mapping)
            and isinstance(
                dict(item.get("metrics") or {}).get("total_score"),
                (int, float),
            )
            and math.isfinite(
                float(dict(item.get("metrics") or {}).get("total_score"))
            )
        ]
        training_plateau = bool(
            diagnostic.optimization.training_plateau
            or diagnostic.optimization.plateau
        )
        physics_improvement: float | None = None
        physics_plateau = False
        if len(scores) >= history_points:
            recent = scores[-history_points:]
            split = max(1, len(recent) // 2)
            earlier = float(median(recent[:split]))
            later = float(median(recent[-split:]))
            physics_improvement = (earlier - later) / max(abs(earlier), 1e-12)
            threshold = max(
                0.0,
                float(self.config["plateau_relative_improvement_threshold"]),
            )
            low_progress = physics_improvement <= threshold
            self.physics_plateau_window_count = (
                self.physics_plateau_window_count + 1 if low_progress else 0
            )
            if not low_progress:
                self.last_physics_improvement_step = int(diagnostic.step)
            physics_plateau = self.physics_plateau_window_count >= max(
                1, int(self.config["plateau_min_windows"])
            )
        else:
            self.physics_plateau_window_count = 0

        insufficient = bool(training_plateau or physics_plateau)
        source = (
            "training_loss_and_physics_validation"
            if training_plateau and physics_plateau
            else "training_loss"
            if training_plateau
            else "physics_validation"
            if physics_plateau
            else "none"
        )
        physics_duration = self.physics_plateau_window_count * max(
            1, int(self.config["diagnostic_interval"])
        )
        physics_since = (
            max(0, int(diagnostic.step) - int(self.last_physics_improvement_step))
            if self.last_physics_improvement_step is not None
            else physics_duration
        )
        physics_plateau_score = (
            max(
                0.0,
                1.0
                - float(physics_improvement)
                / max(
                    float(self.config["plateau_relative_improvement_threshold"]),
                    1e-12,
                ),
            )
            if physics_improvement is not None
            and self.physics_plateau_window_count > 0
            else 0.0
        )
        generic_limit = bool(
            insufficient
            and diagnostic.gradient.available
            and not diagnostic.gradient.gradient_pathology
            and not diagnostic.spatial.localized_residual_concentration
            and not diagnostic.frequency.has_multiscale_or_oscillatory_evidence
            and diagnostic.budget.remaining_budget_sufficient
        )
        optimization = replace(
            diagnostic.optimization,
            plateau=insufficient,
            insufficient_progress=insufficient,
            generic_capacity_or_representation_limit=generic_limit,
            training_plateau=training_plateau,
            physics_plateau=physics_plateau,
            physics_relative_improvement=physics_improvement,
            insufficient_progress_source=source,
            plateau_duration=max(
                int(diagnostic.optimization.plateau_duration), physics_duration
            ),
            steps_since_last_improvement=max(
                int(diagnostic.optimization.steps_since_last_improvement),
                physics_since,
            ),
            plateau_score=max(
                float(diagnostic.optimization.plateau_score),
                physics_plateau_score,
            ),
        )
        return replace(diagnostic, optimization=optimization)

    def build_continuation_forecast(
        self,
        insertion_step: int,
    ) -> dict[str, Any]:
        targets = (
            int(insertion_step) + FEEDBACK_HORIZONS["short"],
            int(insertion_step) + FEEDBACK_HORIZONS["medium"],
            int(insertion_step) + FEEDBACK_HORIZONS["persistent"],
        )
        utility = dict(self.config.get("utility") or {})
        return fit_log_linear_continuation(
            self.physics_validation_history,
            target_steps=targets,
            history_points=int(utility["continuation_history_points"]),
            minimum_confidence=float(
                utility.get("continuation_minimum_confidence", 0.35)
            ),
            maximum_extrapolation_factor=float(
                utility.get("continuation_maximum_extrapolation_factor", 4.0)
            ),
            reject_clipped_forecast=bool(
                utility.get("continuation_reject_clipped_forecast", True)
            ),
        )

    def build_audit_continuation_forecast(
        self,
        insertion_step: int,
    ) -> dict[str, Any]:
        targets = (
            int(insertion_step) + FEEDBACK_HORIZONS["short"],
            int(insertion_step) + FEEDBACK_HORIZONS["medium"],
            int(insertion_step) + FEEDBACK_HORIZONS["persistent"],
        )
        utility = dict(self.config.get("utility") or {})
        return fit_log_linear_continuation(
            self.physics_audit_history,
            target_steps=targets,
            history_points=int(utility["continuation_history_points"]),
            minimum_confidence=float(
                utility.get("continuation_minimum_confidence", 0.35)
            ),
            maximum_extrapolation_factor=float(
                utility.get("continuation_maximum_extrapolation_factor", 4.0)
            ),
            reject_clipped_forecast=bool(
                utility.get("continuation_reject_clipped_forecast", True)
            ),
        )

    def record_loss(self, loss: float, *, iteration: int | None = None) -> None:
        if math.isfinite(float(loss)):
            self.loss_history.append(float(loss))
            maximum = max(
                8,
                int(self.config["diagnostic_window"]) * 4,
                int(self.config["utility_window_iterations"]) * 3,
            )
            if len(self.loss_history) > maximum:
                del self.loss_history[:-maximum]
            if self.current_morph_phase in {"short_utility", "medium_utility"} and self.active_trial is not None:
                self.active_trial.setdefault("utility_training_losses", []).append(
                    float(loss)
                )
                self.active_trial["utility_steps_observed"] = int(
                    self.active_trial.get("utility_steps_observed") or 0
                ) + 1
                target_step = int(
                    self.active_trial.get(
                        "short_utility_end_step"
                        if self.current_morph_phase == "short_utility"
                        else "medium_utility_end_step",
                        iteration or 0,
                    )
                )
                self.phase_remaining_iterations = max(
                    0, target_step - int(iteration or 0)
                )

    def diagnostic_due(
        self,
        *,
        iteration: int,
        first_order: bool,
        dynamic_objective_allowed: bool,
        remaining_stage_iterations: int,
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled"
        if not first_order or not dynamic_objective_allowed:
            return False, "first_order_dynamic_stage_required"
        interval = max(1, int(self.config["diagnostic_interval"]))
        if int(iteration) % interval:
            return False, "diagnostic_interval_not_reached"
        if int(iteration) < int(self.config["minimum_iterations_before_morph"]):
            return False, "minimum_training_not_reached"
        if self.active_trial is not None:
            return True, "morph_pending_diagnostic"
        return True, "diagnostic_due"

    def diagnose(
        self,
        *,
        iteration: int,
        current_total_loss: float,
        gradient_statistics: Mapping[str, Any] | None,
        component_categories: Mapping[str, str] | None,
        pointwise_residual_scores: torch.Tensor | None,
        pde_metadata: Mapping[str, Any] | None,
        current_architecture: Mapping[str, Any] | None,
        remaining_stage_iterations: int,
        current_parameter_count: int,
    ) -> OnlineMorphDiagnostics:
        optimization, self.plateau_window_count, self.last_improvement_step = (
            build_optimization_diagnostics(
                self.loss_history,
                window=int(self.config["diagnostic_window"]),
                relative_improvement_threshold=float(
                    self.config["plateau_relative_improvement_threshold"]
                ),
                plateau_window_count=self.plateau_window_count,
                plateau_min_windows=int(self.config["plateau_min_windows"]),
                diagnostic_interval=int(self.config["diagnostic_interval"]),
                step=int(iteration),
                last_improvement_step=self.last_improvement_step,
            )
        )
        gradient = build_gradient_diagnostics(
            gradient_statistics,
            component_categories=component_categories,
        )
        persistence = max(
            1,
            int((self.config.get("gradient_diagnostics") or {}).get("persistence_windows", 2)),
        )
        self.gradient_conflict_history.append(bool(gradient.gradient_conflict_detected))
        if len(self.gradient_conflict_history) > persistence:
            del self.gradient_conflict_history[:-persistence]
        persistent_conflict = bool(
            len(self.gradient_conflict_history) >= persistence
            and all(self.gradient_conflict_history[-persistence:])
        )
        transient_conflict = bool(
            gradient.gradient_conflict_detected and not persistent_conflict
        )
        gradient_status = (
            "magnitude_imbalance"
            if gradient.gradient_magnitude_imbalance
            else "persistent_directional_conflict"
            if persistent_conflict
            else "transient_directional_conflict"
            if transient_conflict
            else "normal"
        )
        gradient = replace(
            gradient,
            persistent_gradient_conflict=persistent_conflict,
            transient_gradient_conflict=transient_conflict,
            gradient_status=gradient_status,
        )
        spatial = build_spatial_residual_diagnostics(
            pointwise_residual_scores,
            top_k_fraction=float(self.config["residual_top_k_fraction"]),
            max_mean_threshold=float(self.config["residual_concentration_ratio"]),
            top_k_mass_threshold=float(self.config["residual_top_k_mass_threshold"]),
        )
        frequency = build_frequency_diagnostics(pde_metadata)
        initial = self.initial_parameter_count or max(1, int(current_parameter_count))
        required = max(
            int(self.config["min_remaining_iterations_for_morph"]),
            int(self.config["activation_only_iterations"])
            + int(self.config["new_parameters_only_iterations"])
            + (
                int((self.config.get("utility") or {}).get("medium_horizon_steps", 500))
                if self.config["utility_evaluation_enabled"]
                else 0
            )
            + 1,
        )
        budget = build_budget_diagnostics(
            step=int(iteration),
            remaining_iterations=int(remaining_stage_iterations),
            minimum_remaining_iterations=required,
            initial_parameter_count=initial,
            current_parameter_count=int(current_parameter_count),
            morph_count=self.morph_count,
            last_successful_morph_step=self.last_completed_intervention_step,
            min_steps_between_morphs=int(self.config["min_steps_between_morphs"]),
        )
        generic_limit = bool(
            optimization.plateau
            and gradient.available
            and not gradient.gradient_pathology
            and not spatial.localized_residual_concentration
            and not frequency.has_multiscale_or_oscillatory_evidence
            and budget.remaining_budget_sufficient
        )
        optimization = replace(
            optimization,
            generic_capacity_or_representation_limit=generic_limit,
        )
        diagnostic = OnlineMorphDiagnostics(
            step=int(iteration),
            optimization=optimization,
            gradient=gradient,
            spatial=spatial,
            frequency=frequency,
            budget=budget,
            pde=deepcopy(dict(pde_metadata or {})),
            current_architecture=deepcopy(dict(current_architecture or {})),
            previous_morph_events=self.recent_event_summaries(),
        )
        self.diagnostic_count += 1
        return diagnostic

    def recent_event_summaries(
        self, limit: int = 8
    ) -> tuple[dict[str, Any], ...]:
        """Bound history size and prevent events from containing prior events."""

        count = max(0, int(limit))
        if count == 0:
            return ()
        return tuple(compact_morph_event(item) for item in self.events[-count:])

    def propose_action(
        self,
        diagnostic: OnlineMorphDiagnostics,
        scoring_provider: OnlineMorphDecisionProvider | None = None,
        *,
        continuation_forecast: Mapping[str, Any] | None = None,
        audit_continuation_forecast: Mapping[str, Any] | None = None,
    ) -> OnlineMorphAction:
        """Ask all five peer policies for programs and let DUAL_LLM_2 score them.

        System hard constraints (adaptation freeze, morph pending, cooldown,
        remaining budget) are evaluated before any LLM call: while they hold
        the decision is KEEP and neither the generator nor the scorer is
        contacted. When every candidate expands to KEEP the scoring call is
        skipped. An applicability repair is scheduled only when every KEEP is
        caused by a condition mismatch; an explicit policy KEEP remains a
        legitimate decision and is never forcibly rewritten as intervention.
        """

        state = self.adaptation_state(diagnostic)
        efficiency = dict((self.config.get("adaptive_strategy") or {}).get("llm_efficiency") or {})
        if efficiency.get("enabled") and scoring_provider is not None:
            scoring_provider = BudgetedProvider(scoring_provider, self.llm_budget_state, efficiency)
        if continuation_forecast is not None:
            state["continuation_forecast"] = deepcopy(dict(continuation_forecast))
        if audit_continuation_forecast is not None:
            state["audit_continuation_forecast"] = deepcopy(
                dict(audit_continuation_forecast)
            )
        hard_reason = self._hard_no_morph_reason(diagnostic)
        adaptive = dict(self.config.get("adaptive_strategy") or {})
        confirmation_deadline = min(int(adaptive.get("freeze_step", 14000)), int(self.config.get("adaptation_freeze_step", 14000)))
        if adaptive.get("progressive_enabled") and int(diagnostic.step) + 1000 > confirmation_deadline:
            hard_reason = "insufficient_long_term_confirmation_budget"
        if str(self.config.get("decision_mode")) == RANDOM_MORPH_MODE:
            return self._propose_random_morph_action(
                diagnostic, hard_reason=hard_reason
            )
        proposals = self.policy_population.propose_all(state)
        if state.get("online_adaptation"):
            filtered = []
            for item in proposals:
                reason = program_rejection(item.program.to_dict(), state["online_adaptation"], filtered)
                filtered.append(replace(item, program=AdaptationProgram.keep(
                    diagnosis="unsupported_bottleneck_or_capability", reason=reason)) if reason else item)
            proposals = filtered
            if not any(item.program.has_intervention() for item in proposals):
                self.policy_population.request_applicability_repair()
        if hard_reason is not None:
            # Hard gate precedes every LLM call; no scoring happens.
            program = AdaptationProgram.keep(
                diagnosis=f"system_hard_constraint:{hard_reason}",
                reason="system safety constraint evaluated before any LLM call",
            )
            scoring_audit = {
                "scoring_source": "system_hard_gate",
                "status": "skipped_system_hard_gate",
                "hard_reason": hard_reason,
            }
            self.policy_scoring_events.append(
                {"step": int(diagnostic.step), **deepcopy(scoring_audit)}
            )
            return OnlineMorphAction(
                KEEP,
                0,
                0,
                {
                    "decision_reason": hard_reason,
                    "diagnostics": state,
                    "policy_selection": None,
                    "selected_policy_id": None,
                    "state_id": state_id(state),
                    "adaptation_program": program.to_dict(),
                    "program_proposals": [item.to_dict() for item in proposals],
                    "llm_scoring": scoring_audit,
                },
                "system_hard_constraint",
                InterventionClass.NO_ACTION.value,
                {},
                (),
            )
        proposals = self.policy_population.gate_uncredited_repeats(proposals)
        if all(not proposal.program.has_intervention() for proposal in proposals):
            all_trigger_miss = all(
                proposal.program.diagnosis == "policy_trigger_not_satisfied"
                for proposal in proposals
            )
            if all_trigger_miss:
                self.policy_population.request_applicability_repair()
            status = (
                "all_candidates_inapplicable"
                if all_trigger_miss
                else "all_candidates_explicit_keep"
            )
            scoring_audit = {
                "scoring_source": "DUAL_LLM_2_MODEL",
                "status": status,
                "skipped_scoring": True,
                "causal_repeat_gate": deepcopy(
                    self.policy_population.last_repeat_gate_audit
                ),
                "state_matches": {
                    item.policy_id: item.state_match for item in proposals
                },
            }
            self.policy_scoring_events.append(
                {"step": int(diagnostic.step), **deepcopy(scoring_audit)}
            )
            program = AdaptationProgram.keep(
                diagnosis=status,
                reason=(
                    "every candidate condition missed; applicability repair requested"
                    if all_trigger_miss
                    else "the policy population explicitly selected no structural change"
                ),
            )
            return OnlineMorphAction(
                KEEP,
                0,
                0,
                {
                    "decision_reason": status,
                    "diagnostics": state,
                    "policy_selection": None,
                    "selected_policy_id": None,
                    "state_id": state_id(state),
                    "adaptation_program": program.to_dict(),
                    "program_proposals": [item.to_dict() for item in proposals],
                    "llm_scoring": scoring_audit,
                },
                (
                    "eoh_all_candidates_inapplicable"
                    if all_trigger_miss
                    else "eoh_population_explicit_keep"
                ),
                InterventionClass.NO_ACTION.value,
                {},
                (),
            )
        try:
            if str(self.config.get("decision_mode")) == RANDOM_SELECTION_MODE:
                allowed_actions = set(self.effective_allowed_actions(diagnostic.step))
                allowed_primitives = set(self.effective_allowed_primitives(diagnostic.step))
                executable = []
                for item in proposals:
                    program = item.program
                    if not program.has_intervention():
                        executable.append(item)
                    elif program.action in allowed_actions:
                        if program.action in (*OPTIMIZATION_ACTIONS, *OBJECTIVE_ACTIONS, *SAMPLING_ACTIONS):
                            executable.append(item)
                        else:
                            plans = self.safe_execution_engine.compile(program, diagnostic)
                            if plans and all(plan.action in allowed_primitives for plan in plans):
                                executable.append(item)
                if not executable:
                    executable = [item for item in proposals if not item.program.has_intervention()]
                selected_proposal = random.choice(executable)
                scoring_audit = {
                    "scoring_source": RANDOM_SELECTION_MODE,
                    "status": "uniform_executable_candidate",
                    "executable_policy_ids": [item.policy_id for item in executable],
                }
            else:
                if efficiency.get("enabled"):
                    self.last_decision_observation = {"signature": decision_signature(state, self.policy_population.generation), "step": diagnostic.step, "loss": state["optimization"].get("current_total_loss")}
                selected_proposal, proposals = self.policy_population.select(
                    state, scoring_provider,
                    temperature=float(((self.config.get("policy_set") or {}).get("scoring") or {}).get("temperature", 0.0)),
                    keep_margin=float(((self.config.get("policy_set") or {}).get("scoring") or {}).get("keep_margin", DEFAULT_KEEP_BASELINE_MARGIN)),
                    proposals=proposals,
                )
                self.policy_scoring_calls += max(1, int(self.policy_population.last_selection_provider_calls))
                scoring_audit = deepcopy(self.policy_population.last_scoring_audit or {})
            scoring_audit["causal_repeat_gate"] = deepcopy(
                self.policy_population.last_repeat_gate_audit
            )
            self.policy_scoring_events.append(
                {
                    "step": int(diagnostic.step),
                    "status": "accepted",
                    **scoring_audit,
                }
            )
        except Exception as exc:
            if str(self.config.get("decision_mode")) != RANDOM_SELECTION_MODE:
                self.policy_scoring_calls += max(1, int(self.policy_population.last_selection_provider_calls))
            self.policy_scoring_failures += 1
            error = f"{type(exc).__name__}: {exc}"
            scoring_audit = {
                "scoring_source": "DUAL_LLM_2_MODEL",
                "status": "failed_closed_to_keep",
                "error": error,
            }
            self.policy_scoring_events.append(
                {"step": int(diagnostic.step), **deepcopy(scoring_audit)}
            )
            program = AdaptationProgram.keep(
                diagnosis="llm_candidate_scoring_unavailable",
                reason="independent candidate scoring failed; system failed closed",
            )
            return OnlineMorphAction(
                KEEP,
                0,
                0,
                {
                    "decision_reason": "llm_candidate_scoring_unavailable",
                    "diagnostics": state,
                    "policy_selection": None,
                    "selected_policy_id": None,
                    "state_id": state_id(state),
                    "adaptation_program": program.to_dict(),
                    "program_proposals": [item.to_dict() for item in proposals],
                    "llm_scoring": scoring_audit,
                },
                "eoh_llm_scoring_failure",
                InterventionClass.NO_ACTION.value,
                {},
                (),
            )
        program = selected_proposal.program
        allowed = set(self.effective_allowed_primitives(diagnostic.step))
        compiled = (
            ()
            if program.action in OPTIMIZATION_ACTIONS
            else self.safe_execution_engine.compile(program, diagnostic)
        )
        allowed_actions = set(self.effective_allowed_actions(diagnostic.step))
        if (
            program.has_intervention()
            and program.action not in allowed_actions
        ) or any(plan.action not in allowed for plan in compiled):
            program = AdaptationProgram.keep(
                diagnosis="program_uses_disabled_action_or_primitive",
                reason="system safety constraint",
            )
        operator = program.action
        action_parameters: dict[str, Any] = {}
        if program.has_intervention():
            strength = float(program.strength)
            action_parameters = {
                "target": str(program.target.get("region") or "middle_layers"),
                "selector": str(program.target.get("selector") or "all"),
                "mechanism": program.mechanism,
                "strength": (
                    "mild" if strength <= 0.15
                    else "moderate" if strength <= 0.35
                    else "strong"
                ),
            }
        selection = {
            "policy_id": selected_proposal.policy_id,
            "policy_version": selected_proposal.policy_version,
            "state_id": selected_proposal.state_id,
            "selection_score": selected_proposal.selection_score,
        }
        reason = {
            "decision_reason": program.diagnosis,
            "diagnostics": state,
            "policy_selection": selection,
            "selected_policy_id": selected_proposal.policy_id,
            "state_id": selected_proposal.state_id,
            "adaptation_program": program.to_dict(),
            "program_proposals": [item.to_dict() for item in proposals],
            "llm_scoring": scoring_audit,
        }
        return OnlineMorphAction(
            operator,
            0,
            0,
            reason,
            "eoh_policy_population",
            (
                InterventionClass.LOSS_INTERVENTION.value
                if program.action in OBJECTIVE_ACTIONS
                else InterventionClass.SAMPLING_INTERVENTION.value
                if program.action in SAMPLING_ACTIONS
                else InterventionClass.OPTIMIZATION_INTERVENTION.value
                if program.action in OPTIMIZATION_ACTIONS
                else InterventionClass.STRUCTURAL_INTERVENTION.value
                if program.has_intervention()
                else InterventionClass.NO_ACTION.value
            ),
            action_parameters,
            (),
        )

    def _propose_random_morph_action(
        self,
        diagnostic: OnlineMorphDiagnostics,
        *,
        hard_reason: str | None,
    ) -> OnlineMorphAction:
        """Random ablation: sample a legal action after system hard gates."""

        state = diagnostic.to_dict()
        sid = state_id(state)
        if hard_reason is not None:
            program = AdaptationProgram.keep(
                diagnosis=f"system_hard_constraint:{hard_reason}",
                reason="system safety constraint evaluated before random ablation",
            )
            random_audit = {
                "scoring_source": RANDOM_MORPH_MODE,
                "status": "skipped_system_hard_gate",
                "hard_reason": hard_reason,
                "valid_action_count": 0,
            }
            self.policy_scoring_events.append(
                {"step": int(diagnostic.step), **deepcopy(random_audit)}
            )
            return OnlineMorphAction(
                KEEP,
                0,
                0,
                {
                    "decision_reason": hard_reason,
                    "diagnostics": state,
                    "policy_selection": None,
                    "selected_policy_id": None,
                    "state_id": sid,
                    "adaptation_program": program.to_dict(),
                    "program_proposals": [],
                    "llm_scoring": random_audit,
                    "random_morph": random_audit,
                },
                "random_morph_hard_gate",
                InterventionClass.NO_ACTION.value,
                {},
                (),
            )
        allowed_actions = tuple(self.effective_allowed_actions(diagnostic.step))
        valid_programs: list[AdaptationProgram] = []
        rejected: list[dict[str, str]] = []
        for action in allowed_actions:
            try:
                program = self._random_ablation_program(action)
                if action in OPTIMIZATION_ACTIONS:
                    compiled = ()
                else:
                    compiled = self.safe_execution_engine.compile(
                        program, diagnostic
                    )
                    allowed_primitives = set(
                        self.effective_allowed_primitives(diagnostic.step)
                    )
                    if any(plan.action not in allowed_primitives for plan in compiled):
                        rejected.append(
                            {
                                "action": action,
                                "reason": "compiled_primitive_disabled",
                            }
                        )
                        continue
                valid_programs.append(program)
            except Exception as exc:
                rejected.append(
                    {"action": action, "reason": f"{type(exc).__name__}: {exc}"}
                )
        if not valid_programs:
            program = AdaptationProgram.keep(
                diagnosis="random_morph_no_valid_actions",
                reason="no valid intervention survived executor compilation",
            )
            random_audit = {
                "scoring_source": RANDOM_MORPH_MODE,
                "status": "no_valid_actions",
                "valid_action_count": 0,
                "rejected_actions": rejected,
            }
            self.policy_scoring_events.append(
                {"step": int(diagnostic.step), **deepcopy(random_audit)}
            )
            return OnlineMorphAction(
                KEEP,
                0,
                0,
                {
                    "decision_reason": "random_morph_no_valid_actions",
                    "diagnostics": state,
                    "policy_selection": None,
                    "selected_policy_id": None,
                    "state_id": sid,
                    "adaptation_program": program.to_dict(),
                    "program_proposals": [],
                    "llm_scoring": random_audit,
                    "random_morph": random_audit,
                },
                "random_morph_no_valid_actions",
                InterventionClass.NO_ACTION.value,
                {},
                (),
            )
        index = int(torch.randint(len(valid_programs), (1,)).item())
        program = valid_programs[index]
        selection = {
            "policy_id": "random_morph",
            "policy_version": 0,
            "state_id": sid,
            "selection_score": None,
            "random_index": index,
        }
        random_audit = {
            "scoring_source": RANDOM_MORPH_MODE,
            "status": "accepted",
            "valid_action_count": len(valid_programs),
            "valid_actions": [item.action for item in valid_programs],
            "selected_action": program.action,
            "selected_index": index,
            "rejected_actions": rejected,
        }
        self.policy_scoring_events.append(
            {"step": int(diagnostic.step), **deepcopy(random_audit)}
        )
        strength = float(program.strength)
        action_parameters = {
            "target": str(program.target.get("region") or "middle_layers"),
            "selector": str(program.target.get("selector") or "all"),
            "mechanism": program.mechanism,
            "strength": (
                "mild" if strength <= 0.15
                else "moderate" if strength <= 0.35
                else "strong"
            ),
        }
        return OnlineMorphAction(
            program.action,
            0,
            0,
            {
                "decision_reason": "random_valid_action",
                "diagnostics": state,
                "policy_selection": selection,
                "selected_policy_id": "random_morph",
                "state_id": sid,
                "adaptation_program": program.to_dict(),
                "program_proposals": [
                    {
                        "policy_id": f"random_morph_valid_{idx}",
                        "program": item.to_dict(),
                    }
                    for idx, item in enumerate(valid_programs)
                ],
                "llm_scoring": random_audit,
                "random_morph": random_audit,
            },
            "random_morph_uniform_valid_action",
            (
                InterventionClass.LOSS_INTERVENTION.value
                if program.action in OBJECTIVE_ACTIONS
                else InterventionClass.SAMPLING_INTERVENTION.value
                if program.action in SAMPLING_ACTIONS
                else InterventionClass.OPTIMIZATION_INTERVENTION.value
                if program.action in OPTIMIZATION_ACTIONS
                else InterventionClass.STRUCTURAL_INTERVENTION.value
            ),
            action_parameters,
            (),
        )

    @staticmethod
    def _default_random_target(action: str) -> dict[str, str]:
        preferred = (
            "middle_layers",
            "input_representation",
            "output_correction",
            "localized_regions",
            "conflicting_components",
            "dominant_components",
            "all_loss_components",
            "constraint_components",
            "high_residual_regions",
            "boundary_regions",
            "interface_regions",
            "multi_output_components",
            "global_domain",
        )
        allowed = ACTION_TARGETS[action]
        region = next((item for item in preferred if item in allowed), None)
        if region is None:
            region = sorted(allowed)[0]
        return {"region": region, "selector": "all"}

    def _random_ablation_program(self, action: str) -> AdaptationProgram:
        target = self._default_random_target(action)
        recovery = dict(
            (self.config.get("architecture_executor") or {}).get(
                "recovery_window_steps"
            )
            or {}
        )
        return AdaptationProgram(
            diagnosis="random_morph_ablation",
            trigger={},
            target=target,
            operations=(),
            recovery_steps=int(recovery.get("moderate", 250)),
            cooldown_steps=max(0, int(self.config.get("min_steps_between_morphs", 400))),
            fallback={},
            reason="uniform random draw from current legal online action space",
            action=action,
            strength=0.2,
            mechanism=AUTO_MECHANISM,
        )

    def maybe_evolve_policy(
        self,
        diagnostic: OnlineMorphDiagnostics,
        provider: Any | None,
    ) -> dict[str, Any] | None:
        """Evolve only after new medium/persistent execution evidence.

        Hard constraints are checked first: no generation call is spent while
        frozen, cooling down, morph pending, or out of budget.  An
        applicability repair (population expanded entirely to KEEP) evolves
        without waiting for new feedback, bounded by
        ``maximum_applicability_repairs``; the repair is consumed before the
        attempt so failed regenerations also count toward the cap.
        """

        config = self.policy_evolution_config
        random_direction = str(config.get("direction", "feedback")) == "random"
        initialization = not self.policy_population.llm_initialized
        if self._hard_no_morph_reason(diagnostic) is not None:
            return None
        repair_due = (
            not initialization
            and self.policy_population.applicability_repair_due(
                int(config.get("maximum_applicability_repairs", 5))
            )
        )
        if (
            not bool(config.get("enabled", False))
            or provider is None
            or self.active_trial is not None
            or (
                not initialization
                and not repair_due
                and not self.policy_population.can_evolve()
            )
        ):
            return None
        if repair_due:
            self.policy_population.consume_applicability_repair()
        before = [item.policy_id for item in self.policy_population.policies]
        gap_count_before = len(self.policy_population.capability_gap_events)
        self.policy_evolution_cycles += 1
        event: dict[str, Any] = {
            "generation_index": self.policy_population.generation + 1,
            "step": int(diagnostic.step),
            "generation_kind": (
                "initialization" if initialization else "evolution"
            ),
            "trigger": (
                "first_online_diagnostic"
                if initialization
                else "applicability_repair"
                if repair_due
                else "new_medium_or_persistent_physics_feedback"
            ),
            "policy_ids_before": before,
        }
        if repair_due:
            event["applicability_repair"] = True
        try:
            efficiency = dict((self.config.get("adaptive_strategy") or {}).get("llm_efficiency") or {})
            if efficiency.get("enabled"):
                provider = BudgetedProvider(provider, self.llm_budget_state, efficiency)
            candidates = self.policy_population.evolve(
                provider,
                state=self.adaptation_state(diagnostic),
                maximum_attempts_per_mode=int(
                    config.get("maximum_attempts_per_mode", 5)
                ),
                maximum_transient_attempts_per_mode=int(
                    config.get("maximum_transient_attempts_per_mode", 2)
                ),
                transient_retry_initial_seconds=float(
                    config.get("transient_retry_initial_seconds", 1.0)
                ),
                transient_retry_max_seconds=float(
                    config.get("transient_retry_max_seconds", 4.0)
                ),
                temperature=float(config.get("temperature", 0.35)),
                generation_kind=(
                    "initialization" if initialization else "evolution"
                ),
                force=repair_due,
                random_direction=random_direction,
            )
            self.policy_evolution_calls += int(
                self.policy_population.last_evolution_provider_calls
            )
            population_event = (
                self.policy_population.evolution_events[-1]
                if self.policy_population.evolution_events
                else {}
            )
            if initialization:
                # A fresh D-validated population can act at this state; any
                # stale repair request from a checkpoint is obsolete.
                self.policy_population.applicability_repair_pending = False
            event.update(
                {
                    "status": "accepted",
                    "evolution_modes": [item.creation_mode for item in candidates],
                    "policy_ids_after": [item.policy_id for item in candidates],
                    "candidate_count": len(candidates),
                    "provider_calls": int(
                        self.policy_population.last_evolution_provider_calls
                    ),
                    "transient_provider_failures": int(
                        self.policy_population.last_evolution_transient_failures
                    ),
                    "diversity_audit": deepcopy(
                        dict(population_event.get("diversity_audit") or {})
                    ),
                    "capability_gaps": deepcopy(
                        self.policy_population.capability_gap_events[
                            gap_count_before:
                        ]
                    ),
                }
            )
        except Exception as exc:
            self.policy_evolution_failures += 1
            event.update(
                {
                    "status": "rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                    "capability_gaps": deepcopy(
                        self.policy_population.capability_gap_events[
                            gap_count_before:
                        ]
                    ),
                }
            )
        self.policy_evolution_events.append(deepcopy(event))
        return event

    def adaptation_state(self, diagnostic):
        state = diagnostic.to_dict()
        config = dict(self.config.get("adaptive_strategy") or {})
        if config.get("bottleneck_enabled"):
            state["online_adaptation"] = bottleneck_contract(state, self.effective_allowed_primitives(diagnostic.step), config)
        else:
            state["online_adaptation"] = {"enabled": False,
                "allowed_primitives": list(self.effective_allowed_primitives(diagnostic.step)),
                "directive": "Use only allowed_primitives. For a restricted capability set, select an explicit permitted mechanism; never propose disabled operations."}
        return state

    def effective_allowed_primitives(self, iteration: int) -> tuple[str, ...]:
        del iteration
        return tuple(self.config["allowed_primitives"])

    def effective_allowed_actions(self, iteration: int) -> tuple[str, ...]:
        result: list[str] = []
        cooldown = max(0, int(self.config["operator_retry_cooldown"]))
        maximum = max(
            0, int(self.config["max_emergency_recoveries_per_operator"])
        )
        for action in (item for item in ONLINE_ADAPTATION_ACTIONS if item != KEEP):
            failure = self.operator_failure_history.get(action) or {}
            if maximum and int(failure.get("emergency_recoveries", 0)) >= maximum:
                continue
            last = failure.get("last_failure_step")
            if last is not None and int(iteration) - int(last) < cooldown:
                continue
            result.append(action)
        return tuple(result)

    def reject_policy_candidate(
        self, action: OnlineMorphAction, *, reason: str
    ) -> None:
        """Audit a program rejected before deployment; no credit is assigned."""

        self.events.append(
            {
                "event": "PROGRAM_REJECTED_PRE_EXECUTION",
                "policy_id": action.reason.get("selected_policy_id"),
                "state_id": action.reason.get("state_id"),
                "program": deepcopy(action.reason.get("adaptation_program") or {}),
                "reason": str(reason),
            }
        )

    def _hard_no_morph_reason(self, diagnostic: OnlineMorphDiagnostics) -> str | None:
        freeze_step = int(self.config.get("adaptation_freeze_step", 14000))
        if int(diagnostic.step) >= freeze_step:
            return "adaptation_frozen_for_final_convergence"
        if self.active_trial is not None:
            return "morph_pending"
        if (
            (bool((self.config.get("adaptive_strategy") or {}).get("progressive_enabled")) or bool(
                self.config.get(
                    "block_new_morph_until_persistent_feedback", True
                )
            ))
            and self.pending_persistent_feedback
        ):
            return "persistent_feedback_pending"
        if diagnostic.budget.cooldown_active:
            return "morph_cooldown"
        if not diagnostic.budget.remaining_budget_sufficient:
            return "insufficient_remaining_iterations"
        efficiency = dict((self.config.get("adaptive_strategy") or {}).get("llm_efficiency") or {})
        reason = budget_reason(self.llm_budget_state, efficiency)
        if reason:
            return reason
        if unchanged_decision(self.adaptation_state(diagnostic), self.policy_population.generation, self.last_decision_observation, efficiency):
            return "unchanged_bottleneck_state"
        evolution = dict((self.config.get("policy_set") or {}).get("evolution") or {})
        if bool(evolution.get("require_insufficient_progress", True)) and not bool(
            diagnostic.optimization.insufficient_progress
        ):
            return "sufficient_training_and_physics_progress"
        return None

    def record_diagnostic_decision(
        self, action: OnlineMorphAction, diagnostic: OnlineMorphDiagnostics
    ) -> dict[str, Any]:
        program_payload = dict(
            action.reason.get("adaptation_program") or {}
        )
        event = {
            "step": int(diagnostic.step),
            "loss": {"current_total_loss": diagnostic.optimization.current_total_loss},
            "optimization": asdict(diagnostic.optimization),
            "gradient": asdict(diagnostic.gradient),
            "spatial": asdict(diagnostic.spatial),
            "frequency": asdict(diagnostic.frequency),
            "budget": asdict(diagnostic.budget),
            "physics": asdict(diagnostic.physics),
            "decision": action.operator,
            "intervention_class": action.intervention_class,
            "intervention_event": (
                "NO_ACTION_DECISION"
                if action.operator == KEEP
                else "INTERVENTION_EVENT"
            ),
            "action": action.operator,
            "mechanism": program_payload.get("mechanism"),
            "reason": str(action.reason.get("decision_reason") or ""),
            "allowed_primitives": list(
                self.effective_allowed_primitives(diagnostic.step)
            ),
            "allowed_actions": list(
                self.effective_allowed_actions(diagnostic.step)
            ),
        }
        selection = action.reason.get("policy_selection")
        if isinstance(selection, Mapping):
            event["policy_selection"] = deepcopy(dict(selection))
            event["selected_policy_id"] = selection.get("policy_id")
            event["state_id"] = selection.get("state_id")
        self.diagnostic_events.append(event)
        return event

    def mark_applied(self, iteration: int, event: Mapping[str, Any]) -> dict[str, Any]:
        operator = str(event.get("operator") or "")
        if operator == KEEP:
            raise ValueError(
                "KEEP is a no-action decision and cannot be registered as an intervention"
            )
        program_payload = deepcopy(dict(event.get("adaptation_program") or {}))
        program = AdaptationProgram.from_dict(program_payload)
        if not program.has_intervention():
            raise ValueError(
                "Applied interventions require a non-empty AdaptationProgram"
            )
        selection_payload = deepcopy(dict(event.get("policy_selection") or {}))
        if not selection_payload.get("policy_id") or not selection_payload.get("state_id"):
            raise ValueError(
                "Applied interventions require policy_id and state_id attribution"
            )
        self.intervention_sequence += 1
        self.last_morph_iteration = int(iteration)
        self.morph_count += 1
        failure = self.operator_failure_history.setdefault(
            operator, {"attempts": 0, "emergency_recoveries": 0, "last_failure_step": -1}
        )
        failure["attempts"] = int(failure.get("attempts", 0)) + 1
        non_structural = bool(event.get("non_structural", False))
        activation = 0 if non_structural else max(
            0, int(self.config["activation_only_iterations"])
        )
        warmup = 0 if non_structural else max(
            0, int(self.config["new_parameters_only_iterations"])
        )
        # Feedback identities and horizons are protocol invariants.  Structural
        # activation/warmup and a program's recovery window must not shift the
        # causal observation points away from intervention step +100/+500/+1000.
        short_horizon = FEEDBACK_HORIZONS["short"]
        medium_horizon = FEEDBACK_HORIZONS["medium"]
        persistent_horizon = FEEDBACK_HORIZONS["persistent"]
        utility_count = short_horizon
        diagnostic_count = min(utility_count, max(1, len(self.loss_history)))
        baseline = list(self.loss_history[-diagnostic_count:])
        tracked = self.intervention_tracker.register(
            policy_id=str(selection_payload["policy_id"]),
            policy_version=int(selection_payload.get("policy_version") or 1),
            state_id=str(selection_payload["state_id"]),
            program=program,
            start_step=int(iteration),
            state_before=deepcopy(dict(event.get("state_before") or {})),
        )
        intervention_id = tracked.intervention_id
        policy_attribution = {
            "policy_id": selection_payload.get("policy_id"),
            "policy_version": selection_payload.get("policy_version"),
            "state_id": selection_payload.get("state_id"),
            "program": program_payload,
        }
        record = {
            "intervention_id": intervention_id,
            "policy_id": (
                str(event.get("selected_policy_id"))
                if event.get("selected_policy_id") else None
            ),
            "policy_version": int(policy_attribution.get("policy_version") or 0),
            "state_id": (
                str(event.get("state_id")) if event.get("state_id") else None
            ),
            "program": program_payload,
            "start_step": int(iteration),
            "short_due_step": int(iteration) + short_horizon,
            "medium_due_step": int(iteration) + medium_horizon,
            "persistent_due_step": int(iteration) + persistent_horizon,
            "short_feedback": None,
            "medium_feedback": None,
            "persistent_feedback": None,
        }
        self.active_trial = {
            "intervention_id": intervention_id,
            "operator": operator,
            "insertion_step": int(iteration),
            "activation_end_step": int(iteration) + activation,
            "warmup_end_step": int(iteration) + activation + warmup,
            "short_utility_end_step": int(iteration) + short_horizon,
            "medium_utility_end_step": int(iteration) + medium_horizon,
            "persistent_utility_end_step": int(iteration) + persistent_horizon,
            "intervention_record": deepcopy(record),
            "policy_attribution": policy_attribution,
            "adaptation_program": program_payload,
            "utility_target_count": utility_count,
            "baseline_training_losses": baseline,
            "utility_training_losses": [],
            "utility_steps_observed": 0,
            "morph_before_physics_metrics": deepcopy(
                event.get("morph_before_physics_metrics")
            ),
            "morph_before_audit_physics_metrics": deepcopy(
                event.get("morph_before_audit_physics_metrics")
            ),
            "continuation_forecast": deepcopy(event.get("continuation_forecast")),
            "audit_continuation_forecast": deepcopy(
                event.get("audit_continuation_forecast")
            ),
            "short_utility": None,
            "medium_utility": None,
            "start_wall_time": time.perf_counter(),
            "joint_lr_scaled": False,
            "selected_policy_id": event.get("selected_policy_id"),
            "state_id": event.get("state_id"),
            "policy_selection": deepcopy(event.get("policy_selection")),
            "parameter_growth_ratio": float(event.get("parameter_growth_ratio") or 0.0),
            "architecture_before": deepcopy(event.get("architecture_before") or {}),
            "architecture_after": deepcopy(event.get("architecture_after") or {}),
        }
        if activation > 0:
            self.current_morph_phase = "activation"
            self.phase_remaining_iterations = activation
        elif warmup > 0:
            self.current_morph_phase = "warmup"
            self.phase_remaining_iterations = warmup
        else:
            self.current_morph_phase = (
                "short_utility" if self.config["utility_evaluation_enabled"] else "joint"
            )
            self.phase_remaining_iterations = utility_count if self.current_morph_phase == "short_utility" else 0
        self.warmup_until_iteration = int(iteration) + activation + warmup
        self.joint_transition_complete = self.current_morph_phase in {"joint", "short_utility"}
        stored = deepcopy(dict(event))
        stored["iteration"] = int(iteration)
        stored["intervention_id"] = intervention_id
        stored["intervention_record"] = deepcopy(record)
        stored["feedback_status"] = "pending" if self.config["utility_evaluation_enabled"] else "success"
        self.events.append(stored)
        return deepcopy(stored)

    def _applied_event_for_intervention(
        self, intervention: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Resolve feedback by immutable intervention identity, never event order."""

        trial = intervention
        intervention_id = str(trial.get("intervention_id") or "")
        if not intervention_id:
            raise ValueError("Online feedback record is missing intervention_id")
        matches = [
            event
            for event in self.events
            if event.get("intervention_id") == intervention_id
            and event.get("iteration") is not None
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Online intervention {intervention_id} has {len(matches)} applied events"
            )
        applied = matches[0]
        expected = {
            "iteration": int(trial.get("insertion_step") or 0),
            "operator": str(trial.get("operator") or ""),
            "selected_policy_id": trial.get("selected_policy_id"),
            "state_id": trial.get("state_id"),
        }
        if any(applied.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Online intervention {intervention_id} does not match its applied action"
            )
        return applied

    def _restore_intervention_ids(self) -> None:
        """Migrate legacy checkpoints without inventing ambiguous feedback credit."""

        for event in self.events:
            if event.get("iteration") is None:
                continue
            existing = str(event.get("intervention_id") or event.pop("trial_id", "") or "")
            if existing.startswith("intervention_"):
                suffix = existing.removeprefix("intervention_")
                if suffix.isdigit():
                    self.intervention_sequence = max(self.intervention_sequence, int(suffix))
            elif not existing:
                self.intervention_sequence += 1
                existing = f"intervention_{self.intervention_sequence:04d}"
            elif existing.startswith("online_trial_"):
                suffix = existing.removeprefix("online_trial_")
                self.intervention_sequence = max(
                    self.intervention_sequence,
                    int(suffix) if suffix.isdigit() else self.intervention_sequence + 1,
                )
                existing = f"intervention_{self.intervention_sequence:04d}"
            event["intervention_id"] = existing

        for trial in [
            self.active_trial,
            *self.pending_persistent_feedback,
        ]:
            if trial is None:
                continue
            if not trial.get("intervention_id"):
                matches = [
                    event
                    for event in self.events
                    if event.get("iteration") == trial.get("insertion_step")
                    and event.get("operator") == trial.get("operator")
                    and event.get("selected_policy_id") == trial.get("selected_policy_id")
                    and event.get("state_id") == trial.get("state_id")
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Legacy online trial cannot be matched uniquely to its applied action"
                    )
                trial["intervention_id"] = matches[0]["intervention_id"]
            self._applied_event_for_intervention(trial)

    def record_emergency_recovery(
        self,
        operator: str,
        *,
        iteration: int,
        attempted_step: int,
        experience: Mapping[str, Any] | None = None,
        utility: float | None = None,
        physics_utility: Mapping[str, Any] | None = None,
        wall_time: float = 0.0,
    ) -> None:
        failure = self.operator_failure_history.setdefault(
            str(operator), {"attempts": 0, "emergency_recoveries": 0, "last_failure_step": -1}
        )
        failure["attempts"] = max(1, int(failure.get("attempts", 0)))
        failure["emergency_recoveries"] = int(
            failure.get("emergency_recoveries", failure.get("rollbacks", 0))
        ) + 1
        failure["last_failure_step"] = int(iteration)
        self.last_completed_intervention_step = int(iteration)
        self.last_morph_iteration = int(iteration)
        self.current_morph_phase = "joint"
        self.phase_remaining_iterations = 0
        self.active_trial = None
        self.joint_transition_complete = True
        self.warmup_until_iteration = None
        evidence = dict(experience or {})
        tracked = self.intervention_tracker.records.get(
            str(evidence.get("intervention_id") or "")
        )
        if utility is not None and tracked is not None and tracked.short_feedback is None:
            self._record_program_feedback(
                evidence,
                horizon="short",
                utility=min(0.0, float(utility)),
                physics_utility={
                    **deepcopy(dict(physics_utility or {})),
                    "emergency": True,
                },
            )

    def _record_program_feedback(
        self,
        trial: Mapping[str, Any],
        *,
        horizon: str,
        utility: float,
        physics_utility: Mapping[str, Any],
        confounded: bool = False,
    ) -> bool:
        intervention_id = str(trial.get("intervention_id") or "")
        policy_id = str(trial.get("selected_policy_id") or "")
        state_id_value = str(trial.get("state_id") or "")
        if not intervention_id or intervention_id not in self.intervention_tracker.records:
            return False
        payload = deepcopy(dict(physics_utility))
        payload["credit"] = float(utility)
        payload["confounded"] = bool(confounded)
        payload["feedback_status"] = (
            "confounded" if confounded else causal_feedback_status(payload)
        )
        result = self.intervention_tracker.record_feedback(
            intervention_id=intervention_id,
            horizon=horizon,
            feedback=payload,
        )
        if not bool(result.get("recorded", False)):
            return False
        shared = next(
            item for item in self.intervention_tracker.shared_history
            if item["intervention_id"] == intervention_id
        )
        self.policy_population.append_shared_evidence(shared)
        credit_eligible = bool(payload.get("credit_eligible", False))
        if not credit_eligible:
            shared["direct_credit_status"] = "withheld_low_confidence"
        if not confounded and credit_eligible and policy_id and state_id_value:
            try:
                self.policy_population.apply_direct_credit(
                    intervention_id=intervention_id,
                    policy_id=policy_id,
                    state_id_value=state_id_value,
                    horizon=horizon,
                    credit=float(utility),
                    feedback=payload,
                )
            except KeyError:
                # Legacy checkpoints may refer to a policy that predates the
                # archive schema.  Preserve shared causal evidence without
                # repeatedly retrying or crediting an unrelated policy.
                shared["direct_credit_status"] = "historical_policy_unavailable"
        return True

    def complete_medium_feedback(
        self,
        *,
        utility: float,
        wall_time: float,
        physics_utility: Mapping[str, Any] | None = None,
        iteration: int | None = None,
        schedule_persistent: bool = True,
        rolled_back: bool = False,
    ) -> dict[str, Any]:
        trial = deepcopy(dict(self.active_trial or {}))
        applied = self._applied_event_for_intervention(trial)
        payload = dict(physics_utility or {})
        if "accepted" not in payload:
            payload["accepted"] = bool(utility > 0.0)
        if rolled_back:
            payload["accepted"] = False
        outcome = causal_feedback_status(payload, rolled_back=rolled_back)
        success = outcome == "success"
        self._record_program_feedback(
            trial,
            horizon="medium",
            utility=float(utility),
            physics_utility=payload,
        )
        applied.update(
            {
                "utility": float(utility),
                "feedback_status": outcome,
                "utility_wall_time": float(wall_time),
                "physics_utility": deepcopy(payload),
                "rolled_back": bool(rolled_back),
            }
        )
        if (
            bool(self.config.get("utility_evaluation_enabled", True))
            and trial.get("persistent_utility_end_step") is not None
            and bool(schedule_persistent)
        ):
            trial["medium_utility"] = deepcopy(payload)
            if isinstance(trial.get("intervention_record"), dict):
                trial["intervention_record"]["medium_feedback"] = deepcopy(
                    payload
                )
            self.pending_persistent_feedback.append(trial)
        self.current_morph_phase = "joint"
        self.phase_remaining_iterations = 0
        self.active_trial = None
        self.joint_transition_complete = True
        self.warmup_until_iteration = None
        completed = iteration if iteration is not None else self.last_morph_iteration
        if completed is not None:
            self.last_completed_intervention_step = int(completed)
            if success and str(trial.get("operator")) != KEEP:
                self.last_successful_morph_step = int(completed)
        return deepcopy(applied)

    def mark_short_utility(self, utility: Mapping[str, Any], *, iteration: int) -> None:
        if self.active_trial is None:
            return
        applied = self._applied_event_for_intervention(self.active_trial)
        self.active_trial["short_utility"] = deepcopy(dict(utility))
        if isinstance(self.active_trial.get("intervention_record"), dict):
            self.active_trial["intervention_record"]["short_feedback"] = deepcopy(
                dict(utility)
            )
        self.active_trial["pending_morph"] = True
        self.current_morph_phase = "medium_utility"
        self.phase_remaining_iterations = max(
            0, int(self.active_trial["medium_utility_end_step"]) - int(iteration)
        )
        applied.update(
            {"feedback_status": "pending", "short_utility": deepcopy(dict(utility))}
        )
        self._record_program_feedback(
            self.active_trial,
            horizon="short",
            utility=float(dict(utility).get("aggregate_continuation_utility") or 0.0),
            physics_utility=utility,
        )

    def record_persistent_feedback(
        self,
        trial: Mapping[str, Any],
        physics_utility: Mapping[str, Any],
        *,
        iteration: int,
    ) -> dict[str, Any]:
        """Record persistent feedback, withholding credit after another intervention."""

        evidence = dict(trial)
        if isinstance(evidence.get("intervention_record"), dict):
            evidence["intervention_record"]["persistent_feedback"] = deepcopy(
                dict(physics_utility)
            )
        self._applied_event_for_intervention(evidence)
        intervening_trials = [
            event
            for event in self.events
            if event.get("iteration") is not None
            and event.get("intervention_id") != evidence["intervention_id"]
            and int(evidence["insertion_step"]) < int(event["iteration"]) <= int(iteration)
        ]
        utility = float(
            dict(physics_utility).get("aggregate_continuation_utility") or 0.0
        )
        outcome = causal_feedback_status(physics_utility)
        if intervening_trials:
            outcome = "confounded"
        recorded = self._record_program_feedback(
            evidence,
            horizon="persistent",
            utility=utility,
            physics_utility=physics_utility,
            confounded=bool(intervening_trials),
        )
        feedback_event = {
            "event": (
                "HEURISTIC_PERSISTENT_FEEDBACK"
                if recorded
                else "HEURISTIC_PERSISTENT_FEEDBACK_DUPLICATE_IGNORED"
            ),
            "intervention_id": evidence["intervention_id"],
            "step": int(iteration),
            "operator": evidence.get("operator"),
            "selected_policy_id": evidence.get("selected_policy_id"),
            "state_id": evidence.get("state_id"),
            "utility": utility,
            "feedback_status": outcome,
            "physics_utility": deepcopy(dict(physics_utility)),
            "confounded_by_intervention_ids": [
                event["intervention_id"] for event in intervening_trials
            ],
            "duplicate_ignored": not recorded,
        }
        if recorded:
            self.events.append(feedback_event)
        return deepcopy(feedback_event)

    def state_dict(self) -> dict[str, Any]:
        return {
            "llm_budget_state": deepcopy(self.llm_budget_state),
            "last_decision_observation": deepcopy(self.last_decision_observation),
            "loss_history": list(self.loss_history),
            "last_successful_morph_step": self.last_successful_morph_step,
            "last_completed_intervention_step": self.last_completed_intervention_step,
            "last_morph_iteration": self.last_morph_iteration,
            "morph_count": int(self.morph_count),
            "diagnostic_count": int(self.diagnostic_count),
            "rejected_decision_count": int(self.rejected_decision_count),
            "events": deepcopy(self.events),
            "diagnostic_events": deepcopy(self.diagnostic_events),
            "operator_failure_history": deepcopy(self.operator_failure_history),
            "initial_parameter_count": self.initial_parameter_count,
            "plateau_window_count": int(self.plateau_window_count),
            "last_improvement_step": self.last_improvement_step,
            "physics_plateau_window_count": int(
                self.physics_plateau_window_count
            ),
            "last_physics_improvement_step": self.last_physics_improvement_step,
            "current_morph_phase": self.current_morph_phase,
            "phase_remaining_iterations": int(self.phase_remaining_iterations),
            "active_trial": deepcopy(self.active_trial),
            "pending_persistent_feedback": deepcopy(self.pending_persistent_feedback),
            "intervention_sequence": int(self.intervention_sequence),
            "warmup_until_iteration": self.warmup_until_iteration,
            "joint_transition_complete": bool(self.joint_transition_complete),
            "gradient_conflict_history": list(self.gradient_conflict_history),
            "physics_validation_history": deepcopy(self.physics_validation_history),
            "physics_audit_history": deepcopy(self.physics_audit_history),
            "registered_loss_intervention": bool(self.registered_loss_intervention),
            "registered_optimization_intervention": bool(self.registered_optimization_intervention),
            "registered_sampling_intervention": bool(self.registered_sampling_intervention),
            "registered_curriculum_intervention": bool(self.registered_curriculum_intervention),
            "policy_population": self.policy_population.state_dict(),
            "intervention_tracker": self.intervention_tracker.state_dict(),
            "policy_evolution_calls": int(self.policy_evolution_calls),
            "policy_evolution_cycles": int(self.policy_evolution_cycles),
            "policy_evolution_failures": int(self.policy_evolution_failures),
            "policy_evolution_events": deepcopy(self.policy_evolution_events),
            "policy_scoring_calls": int(self.policy_scoring_calls),
            "policy_scoring_failures": int(self.policy_scoring_failures),
            "policy_scoring_events": deepcopy(self.policy_scoring_events),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        state = dict(state or {})
        if state.get("policy_population"):
            self.policy_population.load_state_dict(state.get("policy_population"))
        if state.get("intervention_tracker"):
            self.intervention_tracker.load_state_dict(state.get("intervention_tracker"))
        saved_budget = state.get("llm_budget_state") or {}
        for key in ("calls", "wall_seconds"):
            self.llm_budget_state[key] = max(self.llm_budget_state.get(key, 0), saved_budget.get(key, 0))
        self.last_decision_observation = deepcopy(state.get("last_decision_observation") or {})
        self.loss_history = [float(item) for item in state.get("loss_history") or []]
        last = state.get("last_morph_iteration")
        self.last_morph_iteration = int(last) if last is not None else None
        successful = state.get("last_successful_morph_step", last)
        self.last_successful_morph_step = int(successful) if successful is not None else None
        completed = state.get("last_completed_intervention_step", successful)
        self.last_completed_intervention_step = int(completed) if completed is not None else None
        self.morph_count = int(state.get("morph_count") or 0)
        self.diagnostic_count = int(state.get("diagnostic_count") or 0)
        self.rejected_decision_count = int(state.get("rejected_decision_count") or 0)
        self.events = [
            deepcopy(dict(item))
            for item in state.get("events") or []
            if isinstance(item, Mapping)
            and str(item.get("operator") or "") != KEEP
            # High-level actions were briefly rejected against the internal
            # primitive registry before deterministic lowering.  Those records
            # describe an obsolete wiring bug, not policy performance, so they
            # must not be replayed into future LLM decision context.
            and not (
                str(item.get("event") or "") == "PROGRAM_REJECTED_PRE_EXECUTION"
                and str(item.get("reason") or "")
                == "unregistered_adaptation_primitive"
            )
        ]
        self.diagnostic_events = deepcopy(state.get("diagnostic_events") or [])
        self.operator_failure_history = deepcopy(state.get("operator_failure_history") or {})
        self.operator_failure_history.pop(KEEP, None)
        initial = state.get("initial_parameter_count")
        self.initial_parameter_count = int(initial) if initial is not None else self.initial_parameter_count
        self.plateau_window_count = int(state.get("plateau_window_count") or 0)
        improved = state.get("last_improvement_step")
        self.last_improvement_step = int(improved) if improved is not None else None
        self.physics_plateau_window_count = int(
            state.get("physics_plateau_window_count") or 0
        )
        physics_improved = state.get("last_physics_improvement_step")
        self.last_physics_improvement_step = (
            int(physics_improved) if physics_improved is not None else None
        )
        self.current_morph_phase = str(state.get("current_morph_phase") or "idle")
        self.phase_remaining_iterations = int(state.get("phase_remaining_iterations") or 0)
        loaded_active_trial = deepcopy(state.get("active_trial"))
        discarded_keep_trial = bool(
            isinstance(loaded_active_trial, Mapping)
            and str(loaded_active_trial.get("operator") or "") == KEEP
        )
        self.active_trial = None if discarded_keep_trial else loaded_active_trial
        self.pending_persistent_feedback = deepcopy(
            [
                dict(item)
                for item in state.get("pending_persistent_feedback") or []
                if isinstance(item, Mapping)
                and str(item.get("operator") or "") != KEEP
            ]
        )
        if discarded_keep_trial:
            self.current_morph_phase = "joint"
            self.phase_remaining_iterations = 0
        self.intervention_sequence = max(
            0,
            int(state.get("intervention_sequence", state.get("trial_sequence", 0)) or 0),
        )
        self._restore_intervention_ids()
        until = state.get("warmup_until_iteration")
        self.warmup_until_iteration = int(until) if until is not None else None
        self.joint_transition_complete = bool(
            state.get("joint_transition_complete", True)
        )
        self.gradient_conflict_history = [
            bool(item) for item in state.get("gradient_conflict_history") or []
        ]
        self.physics_validation_history = deepcopy(
            state.get("physics_validation_history") or []
        )
        self.physics_audit_history = deepcopy(
            state.get("physics_audit_history") or []
        )
        self.registered_loss_intervention = bool(
            state.get("registered_loss_intervention", self.registered_loss_intervention)
        )
        self.registered_optimization_intervention = bool(
            state.get(
                "registered_optimization_intervention",
                self.registered_optimization_intervention,
            )
        )
        self.registered_sampling_intervention = bool(
            state.get(
                "registered_sampling_intervention",
                self.registered_sampling_intervention,
            )
        )
        self.registered_curriculum_intervention = bool(
            state.get(
                "registered_curriculum_intervention",
                self.registered_curriculum_intervention,
            )
        )
        self.policy_evolution_calls = int(state.get("policy_evolution_calls") or 0)
        self.policy_evolution_cycles = int(
            state.get("policy_evolution_cycles")
            or len(
                [
                    item
                    for item in state.get("policy_evolution_events") or []
                    if item.get("generation_kind") != "initialization"
                    and item.get("status") != "cache_hit_no_regeneration"
                ]
            )
        )
        self.policy_evolution_failures = int(
            state.get("policy_evolution_failures") or 0
        )
        self.policy_evolution_events = deepcopy(
            state.get("policy_evolution_events") or []
        )
        self.policy_scoring_calls = int(state.get("policy_scoring_calls") or 0)
        self.policy_scoring_failures = int(
            state.get("policy_scoring_failures") or 0
        )
        self.policy_scoring_events = deepcopy(
            state.get("policy_scoring_events") or []
        )

__all__ = [
    "DEFAULT_ONLINE_MORPH_CONFIG",
    "EVOLUTION_MODE_GUIDANCE",
    "EVOLUTION_MODES",
    "OnlineMorphController",
    "POLICY_POPULATION_SIZE",
    "PROMPT_HISTORY_MAX_CHARS",
    "PolicyPopulation",
    "causal_feedback_status",
    "adaptation_program_fingerprint",
    "adaptation_hypothesis_signature",
    "compact_morph_event",
    "compact_policy_candidate",
    "compact_policy_credit",
    "compact_policy_state",
    "compact_shared_evidence",
    "compact_shared_history",
    "default_policy_population",
    "state_id",
]
