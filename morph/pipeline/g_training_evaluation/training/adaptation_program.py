"""Declarative online adaptation programs for a single continuously trained PINN.

Policies may compose only the primitives declared here.  No object in this
module contains executable Python supplied by an LLM.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


WIDEN = "WIDEN"
ADD_LAYER = "ADD_LAYER"
PRUNE = "PRUNE"
REGROW = "REGROW"
REDISTRIBUTE = "REDISTRIBUTE"
LOCAL_REINIT = "LOCAL_REINIT"
ADD_RESIDUAL_BRANCH = "ADD_RESIDUAL_BRANCH"
ADD_CORRECTION_BRANCH = "ADD_CORRECTION_BRANCH"
ADD_FOURIER_BRANCH = "ADD_FOURIER_BRANCH"
ADD_GATED_BRANCH = "ADD_GATED_BRANCH"
ADD_LOCALIZED_EXPERT = "ADD_LOCALIZED_EXPERT"
ADD_ADAPTIVE_FOURIER_BRANCH = "ADD_ADAPTIVE_FOURIER_BRANCH"
ADD_COMPONENT_EXPERT = "ADD_COMPONENT_EXPERT"
REALLOCATE_COMPONENT_CAPACITY = "REALLOCATE_COMPONENT_CAPACITY"
SPLIT_CONFLICTING_TOPOLOGY = "SPLIT_CONFLICTING_TOPOLOGY"
REBALANCE_LOSS_WEIGHTS = "REBALANCE_LOSS_WEIGHTS"
BALANCE_COMPONENT_GRADIENTS = "BALANCE_COMPONENT_GRADIENTS"
REWEIGHT_RESIDUAL_REGIONS = "REWEIGHT_RESIDUAL_REGIONS"
PROJECT_CONFLICTING_GRADIENTS = "PROJECT_CONFLICTING_GRADIENTS"
EMPHASIZE_CONSTRAINTS = "EMPHASIZE_CONSTRAINTS"
MITIGATE_GRADIENT_CONFLICT = "MITIGATE_GRADIENT_CONFLICT"
ADAPT_SAMPLING_DISTRIBUTION = "ADAPT_SAMPLING_DISTRIBUTION"
ENTER_LBFGS_PHASE = "ENTER_LBFGS_PHASE"

AUTO_MECHANISM = "AUTO"
NO_MECHANISM = "NONE"

KEEP = "KEEP"
CONSERVATIVE_GROWTH = "CONSERVATIVE_GROWTH"
EXPLORATORY_MORPH = "EXPLORATORY_MORPH"
REALLOCATION = "REALLOCATION"
REPRESENTATION_AUGMENTATION = "REPRESENTATION_AUGMENTATION"
LOCAL_SPECIALIZATION = "LOCAL_SPECIALIZATION"
OBJECTIVE_REBALANCING = "OBJECTIVE_REBALANCING"
CONSTRAINT_EMPHASIS = "CONSTRAINT_EMPHASIS"
GRADIENT_CONFLICT_MITIGATION = "GRADIENT_CONFLICT_MITIGATION"
SPECTRAL_ADAPTATION = "SPECTRAL_ADAPTATION"
COMPONENT_CAPACITY_ADAPTATION = "COMPONENT_CAPACITY_ADAPTATION"
GRADIENT_TOPOLOGY_ADAPTATION = "GRADIENT_TOPOLOGY_ADAPTATION"
REGION_AWARE_REBALANCING = "REGION_AWARE_REBALANCING"
GRADIENT_SURGERY = "GRADIENT_SURGERY"
SAMPLING_ADAPTATION = "SAMPLING_ADAPTATION"
OPTIMIZATION_PHASE_SWITCH = "OPTIMIZATION_PHASE_SWITCH"

ARCHITECTURE_ACTIONS = (
    KEEP,
    CONSERVATIVE_GROWTH,
    EXPLORATORY_MORPH,
    REALLOCATION,
    REPRESENTATION_AUGMENTATION,
    LOCAL_SPECIALIZATION,
    SPECTRAL_ADAPTATION,
    COMPONENT_CAPACITY_ADAPTATION,
    GRADIENT_TOPOLOGY_ADAPTATION,
)

OBJECTIVE_ACTIONS = (
    OBJECTIVE_REBALANCING,
    CONSTRAINT_EMPHASIS,
    GRADIENT_CONFLICT_MITIGATION,
    REGION_AWARE_REBALANCING,
    GRADIENT_SURGERY,
)

SAMPLING_ACTIONS = (SAMPLING_ADAPTATION,)
OPTIMIZATION_ACTIONS = (OPTIMIZATION_PHASE_SWITCH,)

ONLINE_ADAPTATION_ACTIONS = (
    *ARCHITECTURE_ACTIONS,
    *OBJECTIVE_ACTIONS,
    *SAMPLING_ACTIONS,
    *OPTIMIZATION_ACTIONS,
)

PRIMITIVE_LIBRARY = (
    WIDEN,
    ADD_LAYER,
    PRUNE,
    REGROW,
    REDISTRIBUTE,
    LOCAL_REINIT,
    ADD_RESIDUAL_BRANCH,
    ADD_CORRECTION_BRANCH,
    ADD_FOURIER_BRANCH,
    ADD_GATED_BRANCH,
    ADD_LOCALIZED_EXPERT,
    ADD_ADAPTIVE_FOURIER_BRANCH,
    ADD_COMPONENT_EXPERT,
    REALLOCATE_COMPONENT_CAPACITY,
    SPLIT_CONFLICTING_TOPOLOGY,
    REBALANCE_LOSS_WEIGHTS,
    BALANCE_COMPONENT_GRADIENTS,
    REWEIGHT_RESIDUAL_REGIONS,
    PROJECT_CONFLICTING_GRADIENTS,
    EMPHASIZE_CONSTRAINTS,
    MITIGATE_GRADIENT_CONFLICT,
    ADAPT_SAMPLING_DISTRIBUTION,
    ENTER_LBFGS_PHASE,
)

PRIMITIVES_BY_ACTION = {
    KEEP: frozenset(),
    CONSERVATIVE_GROWTH: frozenset({WIDEN, REGROW, ADD_RESIDUAL_BRANCH}),
    # Exploratory morphing is intentionally the broad composition class.  It
    # may combine registered capabilities when a single response family is
    # insufficient; the executor still owns every concrete implementation.
    EXPLORATORY_MORPH: frozenset(
        {
            WIDEN,
            ADD_LAYER,
            PRUNE,
            REGROW,
            REDISTRIBUTE,
            LOCAL_REINIT,
            ADD_RESIDUAL_BRANCH,
            ADD_CORRECTION_BRANCH,
            ADD_FOURIER_BRANCH,
            ADD_GATED_BRANCH,
            ADD_LOCALIZED_EXPERT,
            ADD_ADAPTIVE_FOURIER_BRANCH,
            ADD_COMPONENT_EXPERT,
            SPLIT_CONFLICTING_TOPOLOGY,
        }
    ),
    REALLOCATION: frozenset({PRUNE, REGROW, REDISTRIBUTE}),
    REPRESENTATION_AUGMENTATION: frozenset(
        {ADD_FOURIER_BRANCH, ADD_GATED_BRANCH, ADD_CORRECTION_BRANCH}
    ),
    LOCAL_SPECIALIZATION: frozenset({ADD_LOCALIZED_EXPERT}),
    SPECTRAL_ADAPTATION: frozenset({ADD_ADAPTIVE_FOURIER_BRANCH}),
    COMPONENT_CAPACITY_ADAPTATION: frozenset(
        {ADD_COMPONENT_EXPERT, REALLOCATE_COMPONENT_CAPACITY}
    ),
    GRADIENT_TOPOLOGY_ADAPTATION: frozenset({SPLIT_CONFLICTING_TOPOLOGY}),
    OBJECTIVE_REBALANCING: frozenset({REBALANCE_LOSS_WEIGHTS}),
    REGION_AWARE_REBALANCING: frozenset(
        {BALANCE_COMPONENT_GRADIENTS, REWEIGHT_RESIDUAL_REGIONS}
    ),
    GRADIENT_SURGERY: frozenset({PROJECT_CONFLICTING_GRADIENTS}),
    CONSTRAINT_EMPHASIS: frozenset({EMPHASIZE_CONSTRAINTS}),
    GRADIENT_CONFLICT_MITIGATION: frozenset({MITIGATE_GRADIENT_CONFLICT}),
    SAMPLING_ADAPTATION: frozenset({ADAPT_SAMPLING_DISTRIBUTION}),
    OPTIMIZATION_PHASE_SWITCH: frozenset({ENTER_LBFGS_PHASE}),
}

# Policy-visible mechanisms are a constrained capability selector, not
# executable code. AUTO delegates the final choice to the deterministic
# executor using the current diagnostic state.
MECHANISMS_BY_ACTION = {
    KEEP: frozenset({NO_MECHANISM}),
    **{
        action: frozenset({AUTO_MECHANISM, *primitives})
        for action, primitives in PRIMITIVES_BY_ACTION.items()
        if action != KEEP
    },
}


def architecture_action_for_primitives(
    operations: Sequence["PrimitiveOperation"],
) -> str:
    """Migrate legacy primitive-authored programs to a high-level action."""

    names = {item.primitive for item in operations}
    if not names:
        return KEEP
    if names <= PRIMITIVES_BY_ACTION[CONSERVATIVE_GROWTH]:
        return CONSERVATIVE_GROWTH
    if names <= PRIMITIVES_BY_ACTION[REALLOCATION]:
        return REALLOCATION
    if names <= PRIMITIVES_BY_ACTION[REPRESENTATION_AUGMENTATION]:
        return REPRESENTATION_AUGMENTATION
    if names <= PRIMITIVES_BY_ACTION[LOCAL_SPECIALIZATION]:
        return LOCAL_SPECIALIZATION
    if names <= PRIMITIVES_BY_ACTION[SPECTRAL_ADAPTATION]:
        return SPECTRAL_ADAPTATION
    if names <= PRIMITIVES_BY_ACTION[COMPONENT_CAPACITY_ADAPTATION]:
        return COMPONENT_CAPACITY_ADAPTATION
    if names <= PRIMITIVES_BY_ACTION[GRADIENT_TOPOLOGY_ADAPTATION]:
        return GRADIENT_TOPOLOGY_ADAPTATION
    if names <= PRIMITIVES_BY_ACTION[OBJECTIVE_REBALANCING]:
        return OBJECTIVE_REBALANCING
    if names <= PRIMITIVES_BY_ACTION[CONSTRAINT_EMPHASIS]:
        return CONSTRAINT_EMPHASIS
    if names <= PRIMITIVES_BY_ACTION[GRADIENT_CONFLICT_MITIGATION]:
        return GRADIENT_CONFLICT_MITIGATION
    if names <= PRIMITIVES_BY_ACTION[REGION_AWARE_REBALANCING]:
        return REGION_AWARE_REBALANCING
    if names <= PRIMITIVES_BY_ACTION[GRADIENT_SURGERY]:
        return GRADIENT_SURGERY
    if names <= PRIMITIVES_BY_ACTION[SAMPLING_ADAPTATION]:
        return SAMPLING_ADAPTATION
    if names <= PRIMITIVES_BY_ACTION[OPTIMIZATION_PHASE_SWITCH]:
        return OPTIMIZATION_PHASE_SWITCH
    return EXPLORATORY_MORPH

LAYER_TARGET_REGIONS = (
    "early_layers",
    "middle_layers",
    "late_layers",
    "all_hidden_layers",
    "bottleneck_layers",
)
TARGET_REGIONS = (
    *LAYER_TARGET_REGIONS,
    "input_representation",
    "output_correction",
    "global_domain",
    "localized_regions",
    "all_loss_components",
    "constraint_components",
    "conflicting_components",
    "dominant_components",
    "interface_regions",
    "boundary_regions",
    "high_residual_regions",
    "multi_output_components",
    "optimizer_state",
)
TARGET_SELECTORS = (
    "highest_utility",
    "lowest_utility",
    "widest",
    "narrowest",
    "all",
)

MECHANISM_TARGETS = {
    WIDEN: frozenset(LAYER_TARGET_REGIONS),
    REGROW: frozenset(LAYER_TARGET_REGIONS),
    ADD_LAYER: frozenset(LAYER_TARGET_REGIONS),
    LOCAL_REINIT: frozenset(LAYER_TARGET_REGIONS),
    PRUNE: frozenset(LAYER_TARGET_REGIONS),
    REDISTRIBUTE: frozenset(LAYER_TARGET_REGIONS),
    ADD_RESIDUAL_BRANCH: frozenset({"output_correction", "global_domain"}),
    ADD_CORRECTION_BRANCH: frozenset({"output_correction", "global_domain"}),
    ADD_FOURIER_BRANCH: frozenset({"input_representation", "global_domain"}),
    ADD_GATED_BRANCH: frozenset({"input_representation", "global_domain"}),
    ADD_LOCALIZED_EXPERT: frozenset({"localized_regions"}),
    ADD_ADAPTIVE_FOURIER_BRANCH: frozenset(
        {"input_representation", "global_domain", "localized_regions"}
    ),
    ADD_COMPONENT_EXPERT: frozenset(
        {"multi_output_components", "dominant_components"}
    ),
    REALLOCATE_COMPONENT_CAPACITY: frozenset(
        {"multi_output_components", "dominant_components"}
    ),
    SPLIT_CONFLICTING_TOPOLOGY: frozenset(
        {"conflicting_components", "multi_output_components"}
    ),
    REBALANCE_LOSS_WEIGHTS: frozenset({"all_loss_components"}),
    BALANCE_COMPONENT_GRADIENTS: frozenset(
        {"all_loss_components", "dominant_components", "conflicting_components"}
    ),
    REWEIGHT_RESIDUAL_REGIONS: frozenset(
        {"high_residual_regions", "interface_regions", "boundary_regions"}
    ),
    PROJECT_CONFLICTING_GRADIENTS: frozenset({"conflicting_components"}),
    EMPHASIZE_CONSTRAINTS: frozenset({"constraint_components"}),
    MITIGATE_GRADIENT_CONFLICT: frozenset({"conflicting_components"}),
    ADAPT_SAMPLING_DISTRIBUTION: frozenset(
        {"high_residual_regions", "interface_regions", "boundary_regions"}
    ),
    ENTER_LBFGS_PHASE: frozenset({"optimizer_state", "global_domain"}),
}

ACTION_TARGETS = {
    KEEP: frozenset(),
    CONSERVATIVE_GROWTH: frozenset(
        {*LAYER_TARGET_REGIONS, "output_correction", "global_domain"}
    ),
    EXPLORATORY_MORPH: frozenset(
        {
            *LAYER_TARGET_REGIONS,
            "input_representation",
            "output_correction",
            "global_domain",
            "localized_regions",
        }
    ),
    REALLOCATION: frozenset(LAYER_TARGET_REGIONS),
    REPRESENTATION_AUGMENTATION: frozenset(
        {"input_representation", "output_correction", "global_domain"}
    ),
    LOCAL_SPECIALIZATION: frozenset({"localized_regions"}),
    SPECTRAL_ADAPTATION: frozenset(
        {"input_representation", "global_domain", "localized_regions"}
    ),
    COMPONENT_CAPACITY_ADAPTATION: frozenset(
        {"multi_output_components", "dominant_components"}
    ),
    GRADIENT_TOPOLOGY_ADAPTATION: frozenset(
        {"conflicting_components", "multi_output_components"}
    ),
    OBJECTIVE_REBALANCING: frozenset({"all_loss_components"}),
    REGION_AWARE_REBALANCING: frozenset(
        {
            "all_loss_components", "dominant_components",
            "conflicting_components", "high_residual_regions",
            "interface_regions", "boundary_regions",
        }
    ),
    GRADIENT_SURGERY: frozenset({"conflicting_components"}),
    CONSTRAINT_EMPHASIS: frozenset({"constraint_components"}),
    GRADIENT_CONFLICT_MITIGATION: frozenset({"conflicting_components"}),
    SAMPLING_ADAPTATION: frozenset(
        {"high_residual_regions", "interface_regions", "boundary_regions"}
    ),
    OPTIMIZATION_PHASE_SWITCH: frozenset({"optimizer_state", "global_domain"}),
}

PRIMITIVE_PARAMETER_CONTRACTS: dict[str, dict[str, Any]] = {
    WIDEN: {
        "width_fraction": "number in (0, 1]",
        "layer_indices": "non-empty list of non-negative hidden-layer indices",
    },
    REGROW: {
        "width_fraction": "number in (0, 1]",
        "layer_indices": "list of non-negative hidden-layer indices",
    },
    ADD_LAYER: {
        "layer_indices": "list containing the insertion source layer index",
    },
    LOCAL_REINIT: {
        "reset_fraction": "number in (0, 1]",
        "layer_indices": "list of non-negative hidden-layer indices",
    },
    PRUNE: {
        "width_fraction": "number in (0, 1]",
        "minimum_width": "integer >= 2",
        "layer_indices": "list of non-negative hidden-layer indices",
    },
    REDISTRIBUTE: {
        "move_fraction": "number in (0, 1]",
        "minimum_width": "integer >= 2",
        "layer_indices": "list of non-negative hidden-layer indices",
    },
    ADD_RESIDUAL_BRANCH: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
    },
    ADD_CORRECTION_BRANCH: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
    },
    ADD_FOURIER_BRANCH: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
        "frequency_scale": "number in [1e-4, 1e4]",
    },
    ADD_GATED_BRANCH: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
    },
    ADD_LOCALIZED_EXPERT: {
        "branch_width": "integer >= 4",
        "expert_count": "integer >= 1",
        "overlap": "number in [0.1, 10]",
        "residual_power": "number in [0.1, 8]",
        "distance_power": "number in [0.1, 8]",
        "center_strategy": "one of residual_farthest, residual_topk",
    },
    ADD_ADAPTIVE_FOURIER_BRANCH: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
        "frequency_scale": "number in [1e-4, 1e4]",
        "frequency_bandwidth": "number in [1, 16]",
        "localization_scale": "number in [0.1, 10]",
    },
    ADD_COMPONENT_EXPERT: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
        "component_index": "integer >= 0",
    },
    REALLOCATE_COMPONENT_CAPACITY: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
        "component_index": "integer >= 0",
        "parameter_tolerance": "number in (0, 1]",
    },
    SPLIT_CONFLICTING_TOPOLOGY: {
        "branch_width": "integer >= 4",
        "branch_depth": "integer >= 1",
        "component_index": "integer >= 0",
    },
    REBALANCE_LOSS_WEIGHTS: {
        "minimum_weight": "number in (0, 1e6]",
        "maximum_weight": "number in (0, 1e6]",
    },
    BALANCE_COMPONENT_GRADIENTS: {
        "minimum_weight": "number in (0, 1e6]",
        "maximum_weight": "number in (0, 1e6]",
        "adaptation_rate": "number in (0, 1]",
    },
    REWEIGHT_RESIDUAL_REGIONS: {
        "minimum_weight": "number in (0, 1e6]",
        "maximum_weight": "number in (0, 1e6]",
        "adaptation_rate": "number in (0, 1]",
    },
    PROJECT_CONFLICTING_GRADIENTS: {
        "projection_strength": "number in (0, 1]",
    },
    EMPHASIZE_CONSTRAINTS: {
        "constraint_multiplier": "number in [1, 4]",
        "pde_multiplier": "number in [0.25, 1]",
    },
    MITIGATE_GRADIENT_CONFLICT: {
        "constraint_multiplier": "number in [1, 4]",
        "pde_multiplier": "number in [0.25, 1]",
    },
    ADAPT_SAMPLING_DISTRIBUTION: {
        "local_fraction": "number in [0,1]; global fraction is its complement",
        "global_fraction": "number in [0,1]; local+global must equal 1",
        "replacement_fraction": "legacy alias for local_fraction, number in [0,1]",
        "exploration_fraction": "legacy alias for global_fraction, number in [0,1]",
        "candidate_multiplier": "integer >= 1",
        "point_count": "positive integer total PDE training points; may exceed configured sampling limits",
    },
    ENTER_LBFGS_PHASE: {
        "minimum_plateau_steps": "integer >= 1",
        "maximum_entry_step": "integer >= current step",
        "reason": "short string explaining why Adam plateau justifies second-order polish",
    },
}

CAPABILITY_REGISTRY = {
    mechanism: {
        "action_families": tuple(
            action
            for action, mechanisms in PRIMITIVES_BY_ACTION.items()
            if mechanism in mechanisms
        ),
        "targets": tuple(sorted(targets)),
        "deterministic_executor": True,
        "accepts_generated_code": False,
        "parameters": deepcopy(PRIMITIVE_PARAMETER_CONTRACTS[mechanism]),
    }
    for mechanism, targets in MECHANISM_TARGETS.items()
}
PROGRAM_TRIGGER_FIELDS = (
    "plateau_steps",
    "pde_dominant",
    "boundary_dominant",
    "initial_dominant",
    "gradient_stable",
    "gradient_conflict",
    "localized_residual",
    "multiscale",
    "interface_or_boundary_localization",
    "coupled_component_conflict",
    "minimum_remaining_steps",
)
# The only diagnostic state paths a generated policy may condition on.
# budget.cooldown_active is deliberately absent: cooldown is a system hard
# constraint, never a candidate-level trigger.
CONDITION_STATE_PATHS = (
    "optimization.plateau",
    "optimization.insufficient_progress",
    "gradient.gradient_pathology",
    "gradient.gradient_conflict_detected",
    "gradient.persistent_gradient_conflict",
    "gradient.gradient_imbalance_detected",
    "spatial.localized_residual_concentration",
    "frequency.has_multiscale_or_oscillatory_evidence",
    "physics.pde_residual_dominant",
    "physics.boundary_condition_dominant",
    "physics.initial_condition_dominant",
    "pde.has_interface",
    "pde.has_coupled_fields",
    "pde.has_piecewise_coefficient",
    "pde.has_shock",
    "budget.remaining_budget_sufficient",
    "budget.minimum_remaining_iterations",
)
MAXIMUM_CONDITION_COUNT = 3
MAXIMUM_DIAGNOSIS_CHARS = 384
MAXIMUM_REASON_CHARS = 512


@dataclass(frozen=True)
class PrimitiveOperation:
    primitive: str
    strength: float
    target: Mapping[str, str] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.primitive).upper()
        object.__setattr__(self, "primitive", name)
        if name not in PRIMITIVE_LIBRARY:
            raise ValueError(f"Unsupported adaptation primitive: {name}")
        value = float(self.strength)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("primitive strength must be finite and in (0, 1]")
        object.__setattr__(self, "strength", value)
        target = {str(key): str(item) for key, item in dict(self.target or {}).items()}
        if target:
            if set(target) != {"region", "selector"}:
                raise ValueError(
                    "operation target must contain exactly region and selector"
                )
            if target["region"] not in MECHANISM_TARGETS[name]:
                raise ValueError(
                    f"Target region {target['region']!r} is incompatible with "
                    f"primitive {name}"
                )
            if target["selector"] not in TARGET_SELECTORS:
                raise ValueError(
                    f"Unsupported operation target selector: {target['selector']}"
                )
        object.__setattr__(self, "target", target)
        parameters = deepcopy(dict(self.parameters or {}))
        allowed = set(PRIMITIVE_PARAMETER_CONTRACTS[name])
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {name}: {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}"
            )
        for key, item in parameters.items():
            if key == "layer_indices":
                if (
                    not isinstance(item, (list, tuple))
                    or not item
                    or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in item)
                ):
                    raise ValueError(
                        "layer_indices must be a non-empty list of non-negative integers"
                    )
                parameters[key] = list(dict.fromkeys(int(index) for index in item))
            elif key in {
                "branch_width", "minimum_width", "branch_depth", "expert_count",
                "component_index", "candidate_multiplier", "point_count",
            }:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(f"{key} must be an integer")
                lower = (
                    4 if key == "branch_width"
                    else 2 if key == "minimum_width"
                    else 0 if key == "component_index"
                    else 1
                )
                if item < lower:
                    raise ValueError(f"{key} must be >= {lower}")
            elif key == "center_strategy":
                if str(item) not in {"residual_farthest", "residual_topk"}:
                    raise ValueError(
                        "center_strategy must be residual_farthest or residual_topk"
                    )
                parameters[key] = str(item)
            else:
                if (name == ADAPT_SAMPLING_DISTRIBUTION and isinstance(item, bool)
                        and key in {"local_fraction", "global_fraction", "replacement_fraction", "exploration_fraction"}):
                    raise ValueError("Sampling fractions must be numbers, not booleans")
                numeric = float(item)
                if not math.isfinite(numeric):
                    raise ValueError(f"{key} must be finite")
                lower, upper = (
                    (1e-4, 1e4)
                    if key == "frequency_scale"
                    else (1.0, 16.0)
                    if key == "frequency_bandwidth"
                    else (1e-8, 1e6)
                    if key in {"minimum_weight", "maximum_weight"}
                    else (1.0, 4.0)
                    if key == "constraint_multiplier"
                    else (0.25, 1.0)
                    if key == "pde_multiplier"
                    else (0.1, 10.0)
                    if key == "overlap"
                    else (0.1, 8.0)
                    if key in {"residual_power", "distance_power"}
                    else (0.1, 10.0)
                    if key == "localization_scale"
                    else (0.0, 1.0)
                    if name == ADAPT_SAMPLING_DISTRIBUTION and key in {
                        "local_fraction", "global_fraction", "replacement_fraction", "exploration_fraction",
                    }
                    else (1e-6, 1.0)
                    if key in {
                        "adaptation_rate", "projection_strength",
                        "replacement_fraction", "exploration_fraction",
                    }
                    else (1e-6, 1.0)
                )
                if not lower <= numeric <= upper:
                    raise ValueError(f"{key} must be in [{lower}, {upper}]")
                parameters[key] = numeric
        if name == ADAPT_SAMPLING_DISTRIBUTION:
            from .sampling_runtime import sampling_policy_fractions
            sampling_policy_fractions(parameters)
        object.__setattr__(self, "parameters", parameters)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrimitiveOperation":
        unknown = set(value) - {"primitive", "strength", "target", "parameters"}
        if unknown:
            raise ValueError(
                f"Unsupported primitive operation fields: {sorted(unknown)}"
            )
        return cls(
            primitive=str(value.get("primitive") or ""),
            strength=float(value.get("strength", 0.0)),
            target=deepcopy(dict(value.get("target") or {})),
            parameters=deepcopy(dict(value.get("parameters") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationProgram:
    diagnosis: str
    trigger: Mapping[str, Any]
    target: Mapping[str, str]
    operations: tuple[PrimitiveOperation, ...]
    recovery_steps: int
    cooldown_steps: int
    fallback: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    # ``operations`` is the authoritative ordered execution request.  Empty
    # operations remain valid for historical checkpoints and are lowered from
    # ``action``/``mechanism`` by the deterministic executor.
    action: str = ""
    strength: float = 0.0
    mechanism: str = ""

    def __post_init__(self) -> None:
        diagnosis = str(self.diagnosis).strip()
        if not diagnosis:
            raise ValueError("program diagnosis must be non-empty")
        object.__setattr__(self, "diagnosis", diagnosis[:MAXIMUM_DIAGNOSIS_CHARS])
        object.__setattr__(
            self,
            "reason",
            str(self.reason or "").strip()[:MAXIMUM_REASON_CHARS],
        )
        unknown_trigger = set(self.trigger) - set(PROGRAM_TRIGGER_FIELDS)
        if unknown_trigger:
            raise ValueError(f"Unsupported program trigger fields: {sorted(unknown_trigger)}")
        action = str(self.action or "").strip().upper()
        if not action:
            action = architecture_action_for_primitives(self.operations)
        if action not in ONLINE_ADAPTATION_ACTIONS:
            raise ValueError(f"Unsupported online adaptation action: {action}")
        object.__setattr__(self, "action", action)
        mechanism = str(self.mechanism or "").strip().upper()
        if not mechanism:
            if action == KEEP:
                mechanism = NO_MECHANISM
            elif len(self.operations) == 1:
                mechanism = self.operations[0].primitive
            else:
                mechanism = AUTO_MECHANISM
        allowed_mechanisms = MECHANISMS_BY_ACTION[action]
        if mechanism not in allowed_mechanisms:
            raise ValueError(
                f"Mechanism {mechanism!r} is incompatible with architecture "
                f"action {action}; allowed: {sorted(allowed_mechanisms)}"
            )
        object.__setattr__(self, "mechanism", mechanism)
        if len(self.operations) > 1 and mechanism != AUTO_MECHANISM:
            raise ValueError(
                "A multi-operation program must use mechanism AUTO"
            )
        if (len(self.operations) == 1 and mechanism != AUTO_MECHANISM
                and mechanism != self.operations[0].primitive):
            raise ValueError("A single-operation mechanism must match its primitive or use AUTO")
        strength = float(self.strength or 0.0)
        if action != KEEP and strength <= 0.0 and self.operations:
            strength = float(self.operations[0].strength)
        if action != KEEP and strength <= 0.0:
            strength = 0.2
        if action == KEEP:
            strength = 0.0
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("program strength must be finite and in [0, 1]")
        object.__setattr__(self, "strength", strength)

        region = str(self.target.get("region") or "")
        selector = str(self.target.get("selector") or "")
        if action != KEEP:
            if region not in TARGET_REGIONS:
                raise ValueError(f"Unsupported target region: {region}")
            if selector not in TARGET_SELECTORS:
                raise ValueError(f"Unsupported target selector: {selector}")
            allowed_targets = (
                ACTION_TARGETS[action]
                if mechanism == AUTO_MECHANISM or self.operations
                else MECHANISM_TARGETS[mechanism]
            )
            if region not in allowed_targets:
                raise ValueError(
                    f"Target region {region!r} is incompatible with mechanism "
                    f"{mechanism}; allowed: {sorted(allowed_targets)}"
                )
        if action == KEEP and self.operations:
            raise ValueError("KEEP cannot carry executor primitives")
        if self.operations and any(
            item.primitive not in PRIMITIVES_BY_ACTION[action]
            for item in self.operations
        ):
            raise ValueError(
                f"Executor primitives are incompatible with architecture action {action}"
            )
        for item in self.operations:
            operation_region = str(item.target.get("region") or region)
            if operation_region not in MECHANISM_TARGETS[item.primitive]:
                raise ValueError(
                    f"Target region {operation_region!r} is incompatible with "
                    f"primitive {item.primitive}"
                )
        recovery = int(self.recovery_steps)
        cooldown = int(self.cooldown_steps)
        if recovery < 0 or recovery > 2000:
            raise ValueError("recovery_steps must be in [0, 2000]")
        if cooldown < 0 or cooldown > 5000:
            raise ValueError("cooldown_steps must be in [0, 5000]")
        if action == KEEP and recovery != 0:
            raise ValueError("KEEP program must use recovery_steps=0")
        fallback = dict(self.fallback or {})
        if fallback:
            keys = set(fallback)
            if keys == {"condition", "operation"}:  # legacy checkpoint
                fallback_operation = str(fallback["operation"]).upper()
                if fallback_operation not in PRIMITIVE_LIBRARY:
                    raise ValueError("fallback operation is outside the primitive library")
            elif keys == {"condition", "action"}:
                fallback_action = str(fallback["action"]).upper()
                if fallback_action not in ONLINE_ADAPTATION_ACTIONS:
                    raise ValueError("fallback action is outside the action library")
            else:
                raise ValueError(
                    "fallback must contain exactly condition and action"
                )

    @classmethod
    def keep(cls, *, diagnosis: str, reason: str = "") -> "AdaptationProgram":
        return cls(
            diagnosis=diagnosis,
            trigger={},
            target={},
            operations=(),
            recovery_steps=0,
            cooldown_steps=0,
            fallback={},
            reason=reason,
            action=KEEP,
            strength=0.0,
            mechanism=NO_MECHANISM,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdaptationProgram":
        raw = dict(value or {})
        allowed_fields = {
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
        unknown_fields = set(raw) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Unsupported adaptation program fields: {sorted(unknown_fields)}"
            )
        operations = tuple(
            item
            if isinstance(item, PrimitiveOperation)
            else PrimitiveOperation.from_dict(item)
            for item in raw.get("operations") or ()
        )
        return cls(
            diagnosis=str(raw.get("diagnosis") or "unspecified"),
            trigger=deepcopy(dict(raw.get("trigger") or {})),
            target={
                str(key): str(item)
                for key, item in dict(raw.get("target") or {}).items()
            },
            operations=operations,
            recovery_steps=int(raw.get("recovery_steps", 0)),
            cooldown_steps=int(raw.get("cooldown_steps", 0)),
            fallback=deepcopy(dict(raw.get("fallback") or {})),
            reason=str(raw.get("reason") or ""),
            action=str(raw.get("action") or ""),
            strength=float(raw.get("strength", 0.0) or 0.0),
            mechanism=str(raw.get("mechanism") or ""),
        )

    def has_intervention(self) -> bool:
        return self.action != KEEP

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "trigger": deepcopy(dict(self.trigger)),
            "target": deepcopy(dict(self.target)),
            "operations": [item.to_dict() for item in self.operations],
            "recovery_steps": int(self.recovery_steps),
            "cooldown_steps": int(self.cooldown_steps),
            "fallback": deepcopy(dict(self.fallback)),
            "reason": self.reason,
            "action": self.action,
            "strength": float(self.strength),
            "mechanism": self.mechanism,
        }


@dataclass
class PolicyCandidate:
    policy_id: str
    version: int
    parent_ids: tuple[str, ...]
    creation_step: int
    creation_mode: str
    policy_spec: Mapping[str, Any]
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    global_credit: float = 0.0
    state_conditioned_credit: dict[str, float] = field(default_factory=dict)
    last_selected_step: int | None = None
    recent_results: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.policy_id = str(self.policy_id)
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        self.version = max(1, int(self.version))
        self.parent_ids = tuple(str(item) for item in self.parent_ids)
        self.creation_step = max(0, int(self.creation_step))
        self.creation_mode = str(self.creation_mode)
        self.policy_spec = deepcopy(dict(self.policy_spec or {}))
        unknown_policy_fields = set(self.policy_spec) - {"conditions", "program"}
        if unknown_policy_fields:
            raise ValueError(
                f"Unsupported policy_spec fields: {sorted(unknown_policy_fields)}"
            )
        if "program" not in self.policy_spec:
            raise ValueError("policy_spec must contain a program template")
        # Normalize legacy policies immediately so prompts, checkpoints, and
        # evidence snapshots all expose the explicit action/mechanism schema.
        normalized_program = AdaptationProgram.from_dict(
            self.policy_spec["program"]
        )
        self.policy_spec["program"] = normalized_program.to_dict()

    def propose(
        self,
        *,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> AdaptationProgram:
        conditions = dict(self.policy_spec.get("conditions") or {})
        if not _conditions_match(state, conditions):
            return AdaptationProgram.keep(
                diagnosis="policy_trigger_not_satisfied",
                reason=f"{self.policy_id} did not match the current state",
            )
        program = AdaptationProgram.from_dict(self.policy_spec["program"])
        if history and not program.reason:
            payload = program.to_dict()
            payload["reason"] = "Policy matched current state and shared execution history"
            program = AdaptationProgram.from_dict(payload)
        return program

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyCandidate":
        raw = dict(value or {})
        return cls(
            policy_id=str(raw.get("policy_id") or ""),
            version=int(raw.get("version", 1)),
            parent_ids=tuple(raw.get("parent_ids") or ()),
            creation_step=int(raw.get("creation_step", 0)),
            creation_mode=str(raw.get("creation_mode") or "initialization"),
            policy_spec=deepcopy(dict(raw.get("policy_spec") or {})),
            execution_count=int(raw.get("execution_count", 0)),
            success_count=int(raw.get("success_count", 0)),
            failure_count=int(raw.get("failure_count", 0)),
            global_credit=float(raw.get("global_credit", 0.0)),
            state_conditioned_credit={
                str(key): float(item)
                for key, item in dict(raw.get("state_conditioned_credit") or {}).items()
            },
            last_selected_step=(
                int(raw["last_selected_step"])
                if raw.get("last_selected_step") is not None
                else None
            ),
            recent_results=deepcopy(list(raw.get("recent_results") or [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramProposal:
    policy_id: str
    policy_version: int
    state_id: str
    program: AdaptationProgram
    state_match: float
    selection_score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "state_id": self.state_id,
            "program": self.program.to_dict(),
            "state_match": float(self.state_match),
            "selection_score": float(self.selection_score),
            "reason": self.reason,
        }


def _nested_get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in str(path).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _conditions_match(state: Mapping[str, Any], conditions: Mapping[str, Any]) -> bool:
    for path, expected in conditions.items():
        actual = _nested_get(state, str(path))
        if isinstance(expected, Mapping):
            minimum = expected.get("min")
            maximum = expected.get("max")
            try:
                numeric = float(actual)
            except (TypeError, ValueError):
                return False
            if minimum is not None and numeric < float(minimum):
                return False
            if maximum is not None and numeric > float(maximum):
                return False
        elif actual != expected:
            return False
    return True


def conditions_rejection_reason(conditions: Mapping[str, Any]) -> str | None:
    """Return why these conditions cannot legally enter the population.

    Generated conditions must draw from the whitelisted diagnostic state
    paths (CONDITION_STATE_PATHS), at most MAXIMUM_CONDITION_COUNT of them,
    and use only the scalar or {"min"/"max"} syntax that ``_conditions_match``
    actually evaluates.  ``budget.cooldown_active`` is rejected because
    cooldown is a system hard constraint, never a candidate-level trigger.
    Legacy checkpoints are unaffected: PolicyCandidate.from_dict performs no
    conditions validation, so historical narrow candidates still load.
    """

    entries = dict(conditions or {})
    if len(entries) > MAXIMUM_CONDITION_COUNT:
        return (
            f"conditions may contain at most {MAXIMUM_CONDITION_COUNT} entries, "
            f"got {len(entries)}; prefer broad conditions that match the "
            f"current diagnostic moment"
        )
    for raw_path, expected in entries.items():
        path = str(raw_path)
        if path not in CONDITION_STATE_PATHS:
            return (
                f"condition path {path!r} is outside the allowed set: "
                f"{', '.join(CONDITION_STATE_PATHS)}"
            )
        if isinstance(expected, Mapping):
            if "min" not in expected and "max" not in expected:
                return (
                    f"condition {path!r} must use a scalar value or a min/max "
                    f"numeric range; unsupported keys {sorted(expected)}"
                )
            for bound in ("min", "max"):
                if bound in expected:
                    try:
                        float(expected[bound])
                    except (TypeError, ValueError):
                        return (
                            f"condition {path!r} bound {bound}="
                            f"{expected[bound]!r} is not numeric"
                        )
    return None


def candidate_applicability_reason(
    state: Mapping[str, Any],
    conditions: Mapping[str, Any],
    program: AdaptationProgram,
) -> str | None:
    """Return why this candidate cannot act at the current diagnostic moment.

    Candidates are generated to compete at the state they were generated for,
    so any candidate carrying an intervention must satisfy
    ``_conditions_match(current_state, conditions)`` at generation time.
    Explicit KEEP programs are exempt because they are
    executable in every state.  Returns None when the candidate is
    executable now.
    """

    if not program.has_intervention():
        return None
    if _conditions_match(state, dict(conditions or {})):
        return None
    return (
        "candidate carries an intervention but its conditions do not match "
        "the current state; every non-KEEP candidate must be executable at "
        "the diagnostic moment it is generated for"
    )


__all__ = [
    "ADD_CORRECTION_BRANCH",
    "ADD_FOURIER_BRANCH",
    "ADD_GATED_BRANCH",
    "ADD_LAYER",
    "ADD_LOCALIZED_EXPERT",
    "ADD_RESIDUAL_BRANCH",
    "REBALANCE_LOSS_WEIGHTS",
    "EMPHASIZE_CONSTRAINTS",
    "ENTER_LBFGS_PHASE",
    "MITIGATE_GRADIENT_CONFLICT",
    "ARCHITECTURE_ACTIONS",
    "OBJECTIVE_ACTIONS",
    "OPTIMIZATION_ACTIONS",
    "OPTIMIZATION_PHASE_SWITCH",
    "ONLINE_ADAPTATION_ACTIONS",
    "ACTION_TARGETS",
    "AUTO_MECHANISM",
    "AdaptationProgram",
    "CONSERVATIVE_GROWTH",
    "CAPABILITY_REGISTRY",
    "CONDITION_STATE_PATHS",
    "EXPLORATORY_MORPH",
    "KEEP",
    "LOCAL_REINIT",
    "LOCAL_SPECIALIZATION",
    "OBJECTIVE_REBALANCING",
    "CONSTRAINT_EMPHASIS",
    "GRADIENT_CONFLICT_MITIGATION",
    "LAYER_TARGET_REGIONS",
    "MAXIMUM_CONDITION_COUNT",
    "MAXIMUM_DIAGNOSIS_CHARS",
    "MAXIMUM_REASON_CHARS",
    "MECHANISMS_BY_ACTION",
    "MECHANISM_TARGETS",
    "NO_MECHANISM",
    "PRIMITIVE_LIBRARY",
    "PRIMITIVES_BY_ACTION",
    "PROGRAM_TRIGGER_FIELDS",
    "PRUNE",
    "PolicyCandidate",
    "PrimitiveOperation",
    "ProgramProposal",
    "REDISTRIBUTE",
    "REALLOCATION",
    "REPRESENTATION_AUGMENTATION",
    "REGROW",
    "TARGET_REGIONS",
    "TARGET_SELECTORS",
    "WIDEN",
    "architecture_action_for_primitives",
    "candidate_applicability_reason",
    "conditions_rejection_reason",
]
