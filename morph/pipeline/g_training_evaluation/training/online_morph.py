"""Safe execution of declarative adaptation programs on one live PINN.

Policy generation, selection, credit and evolution live in
``policy_evolution.py``.  This module owns only deterministic network mutation,
parameter/optimizer migration metadata and numerical safety checks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
from typing import Any, Callable, Mapping, Protocol

import torch
from torch import nn

from .online_diagnostics import (
    OnlineMorphDiagnostics,
)
from .adaptation_program import (
    ADD_CORRECTION_BRANCH,
    ADD_FOURIER_BRANCH,
    ADD_GATED_BRANCH,
    ADD_LAYER,
    ADD_LOCALIZED_EXPERT,
    ADD_ADAPTIVE_FOURIER_BRANCH,
    ADD_COMPONENT_EXPERT,
    REALLOCATE_COMPONENT_CAPACITY,
    SPLIT_CONFLICTING_TOPOLOGY,
    ADD_RESIDUAL_BRANCH,
    CONSTRAINT_EMPHASIS,
    EMPHASIZE_CONSTRAINTS,
    ACTION_TARGETS,
    ARCHITECTURE_ACTIONS,
    AUTO_MECHANISM,
    CONSERVATIVE_GROWTH,
    EXPLORATORY_MORPH,
    KEEP,
    GRADIENT_CONFLICT_MITIGATION,
    GRADIENT_SURGERY,
    GRADIENT_TOPOLOGY_ADAPTATION,
    LOCAL_REINIT,
    LOCAL_SPECIALIZATION,
    SPECTRAL_ADAPTATION,
    COMPONENT_CAPACITY_ADAPTATION,
    MECHANISMS_BY_ACTION,
    MECHANISM_TARGETS,
    MITIGATE_GRADIENT_CONFLICT,
    BALANCE_COMPONENT_GRADIENTS,
    REWEIGHT_RESIDUAL_REGIONS,
    PROJECT_CONFLICTING_GRADIENTS,
    ADAPT_SAMPLING_DISTRIBUTION,
    OBJECTIVE_ACTIONS,
    OBJECTIVE_REBALANCING,
    REGION_AWARE_REBALANCING,
    SAMPLING_ADAPTATION,
    ONLINE_ADAPTATION_ACTIONS,
    PRIMITIVE_PARAMETER_CONTRACTS,
    PRIMITIVE_LIBRARY,
    PRUNE,
    REBALANCE_LOSS_WEIGHTS,
    REDISTRIBUTE,
    REALLOCATION,
    REPRESENTATION_AUGMENTATION,
    REGROW,
    TARGET_REGIONS,
    WIDEN,
    AdaptationProgram,
)


# Policy-visible KEEP is also the only no-change action.
# Historical checkpoints used ADD_RESIDUAL_BLOCK for the capability now
# exposed as ADD_RESIDUAL_BRANCH. Keep the old token readable on disk.
ADD_RESIDUAL_BLOCK = "ADD_RESIDUAL_BLOCK"
WIDEN_LAYER = "WIDEN_LAYER"


class InterventionClass(str, Enum):
    NO_ACTION = "NO_ACTION"
    LOSS_INTERVENTION = "LOSS_INTERVENTION"
    SAMPLING_INTERVENTION = "SAMPLING_INTERVENTION"
    OPTIMIZATION_INTERVENTION = "OPTIMIZATION_INTERVENTION"
    STRUCTURAL_INTERVENTION = "STRUCTURAL_INTERVENTION"
INTERNAL_MORPH_PRIMITIVES = (
    ADD_RESIDUAL_BLOCK,
    WIDEN_LAYER,
    ADD_CORRECTION_BRANCH,
    ADD_FOURIER_BRANCH,
    ADD_GATED_BRANCH,
    ADD_LOCALIZED_EXPERT,
    ADD_ADAPTIVE_FOURIER_BRANCH,
    ADD_COMPONENT_EXPERT,
    REALLOCATE_COMPONENT_CAPACITY,
    SPLIT_CONFLICTING_TOPOLOGY,
)
REGISTERED_MORPH_OPERATORS = (
    *PRIMITIVE_LIBRARY,
)
ONLINE_MORPH_ACTIONS = (
    *ONLINE_ADAPTATION_ACTIONS,
)


class OnlineMorphDecisionProvider(Protocol):
    """Optional LLM/policy boundary used only after the cheap trigger fires."""

    def __call__(
        self,
        diagnostic: Mapping[str, Any],
        allowed_primitives: tuple[str, ...],
    ) -> str | Mapping[str, Any]: ...


@dataclass(frozen=True)
class OnlineMorphAction:
    operator: str
    width: int
    depth: int
    reason: Mapping[str, Any]
    decision_source: str = "deterministic_policy"
    intervention_class: str = ""
    action_parameters: Mapping[str, Any] = field(default_factory=dict)
    design_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in ONLINE_MORPH_ACTIONS:
            raise ValueError(f"Unregistered online morph operator: {self.operator}")
        if self.operator == KEEP:
            if int(self.width) != 0 or int(self.depth) != 0:
                raise ValueError("KEEP must not carry structural dimensions")
            if self.action_parameters:
                raise ValueError("KEEP must not carry architecture action parameters")
        elif int(self.width) != 0 or int(self.depth) != 0:
            raise ValueError(
                "High-level architecture actions carry target/strength, not width/depth"
            )
        if self.operator != KEEP:
            parameters = dict(self.action_parameters or {})
            if not parameters:
                parameters = {
                    "target": (
                        "bottleneck_layers" if self.operator == REALLOCATION
                        else "input_representation"
                        if self.operator == REPRESENTATION_AUGMENTATION
                        else "localized_regions"
                        if self.operator == LOCAL_SPECIALIZATION
                        else "input_representation"
                        if self.operator == SPECTRAL_ADAPTATION
                        else "multi_output_components"
                        if self.operator == COMPONENT_CAPACITY_ADAPTATION
                        else "conflicting_components"
                        if self.operator == GRADIENT_TOPOLOGY_ADAPTATION
                        else "all_loss_components"
                        if self.operator == OBJECTIVE_REBALANCING
                        else "dominant_components"
                        if self.operator == REGION_AWARE_REBALANCING
                        else "conflicting_components"
                        if self.operator == GRADIENT_SURGERY
                        else "high_residual_regions"
                        if self.operator == SAMPLING_ADAPTATION
                        else "constraint_components"
                        if self.operator == CONSTRAINT_EMPHASIS
                        else "conflicting_components"
                        if self.operator == GRADIENT_CONFLICT_MITIGATION
                        else "middle_layers"
                    ),
                    "strength": "moderate",
                }
                object.__setattr__(self, "action_parameters", parameters)
            if not {"target", "strength"}.issubset(parameters) or set(parameters) - {
                "target",
                "selector",
                "strength",
                "mechanism",
            }:
                raise ValueError(
                    "Architecture action parameters require target and strength, "
                    "with optional selector and registered mechanism"
                )
            if str(parameters["target"]) not in TARGET_REGIONS:
                raise ValueError("Unsupported architecture target")
            if str(parameters["strength"]) not in {
                "mild",
                "moderate",
                "strong",
            }:
                raise ValueError("Unsupported architecture strength")
            mechanism = str(
                parameters.get("mechanism") or AUTO_MECHANISM
            ).upper()
            if mechanism not in MECHANISMS_BY_ACTION[self.operator]:
                raise ValueError(
                    f"Unsupported mechanism {mechanism!r} for {self.operator}"
                )
            allowed_targets = (
                ACTION_TARGETS[self.operator]
                if mechanism == AUTO_MECHANISM
                else MECHANISM_TARGETS[mechanism]
            )
            if str(parameters["target"]) not in allowed_targets:
                raise ValueError(
                    f"Unsupported target {parameters['target']!r} for mechanism "
                    f"{mechanism}"
                )
        if not self.intervention_class:
            object.__setattr__(
                self,
                "intervention_class",
                InterventionClass.NO_ACTION.value
                if self.operator == KEEP
                else InterventionClass.LOSS_INTERVENTION.value
                if self.operator in OBJECTIVE_ACTIONS
                else InterventionClass.SAMPLING_INTERVENTION.value
                if self.operator == SAMPLING_ADAPTATION
                else InterventionClass.STRUCTURAL_INTERVENTION.value,
            )
        if str(self.intervention_class) not in {item.value for item in InterventionClass}:
            raise ValueError("Unknown intervention class")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterReplacement:
    old_parameter: nn.Parameter
    new_parameter: nn.Parameter
    row_indices: tuple[int, ...] = ()
    column_indices: tuple[int, ...] = ()
    column_divisors: tuple[int, ...] = ()
    reset_row_indices: tuple[int, ...] = ()

    def migrate_state_tensor(self, value: torch.Tensor, state_name: str) -> torch.Tensor:
        if tuple(value.shape) != tuple(self.old_parameter.shape):
            return value.detach().clone()
        result = value
        if self.row_indices:
            rows = torch.tensor(self.row_indices, device=value.device)
            result = result.index_select(0, rows)
        if self.column_indices and result.ndim >= 2:
            columns = torch.tensor(self.column_indices, device=value.device)
            result = result.index_select(1, columns)
            divisors = torch.tensor(
                self.column_divisors, device=value.device, dtype=value.dtype
            )
            power = 2 if "sq" in str(state_name).casefold() else 1
            result = result / divisors.pow(power).unsqueeze(0)
        if self.reset_row_indices and result.ndim >= 1:
            reset_rows = torch.tensor(
                self.reset_row_indices, device=result.device
            )
            result = result.clone()
            result.index_fill_(0, reset_rows, 0)
        return result.detach().clone()


@dataclass(frozen=True)
class MorphResult:
    module: nn.Module
    new_parameters: tuple[nn.Parameter, ...]
    activation_parameters: tuple[nn.Parameter, ...]
    warmup_parameters: tuple[nn.Parameter, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parameter_replacements: tuple[ParameterReplacement, ...] = ()

    def __iter__(self):
        """Preserve the historical ``module, parameters = result`` contract."""

        yield self.module
        yield self.new_parameters


@dataclass(frozen=True)
class ArchitectureExecutionPlan:
    """Deterministic lowering of a policy action into a safe topology change."""

    action: str
    target: str
    strength: str
    primitive: str
    strict_preservation: bool
    recovery_steps: int
    # Exact policy strength. ``strength`` remains the coarse compatibility
    # label used by old checkpoints/configuration, while execution dimensions
    # are derived from this numeric value when it is positive.
    strength_value: float = 0.0
    # Validated operation-specific controls authored by the policy.  Only the
    # registered keys in adaptation_program.py can reach this boundary.
    parameters: Mapping[str, Any] = field(default_factory=dict)
    operation_index: int = 0
    selector: str = "all"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicArchitectureExecutor:
    """Compatibility lowering used internally by :class:`SafeExecutionEngine`."""

    _TARGETS = set(TARGET_REGIONS)
    _STRENGTHS = {"mild", "moderate", "strong"}

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def plan(
        self,
        action: OnlineMorphAction,
        diagnostic: OnlineMorphDiagnostics,
    ) -> ArchitectureExecutionPlan:
        if action.operator == KEEP:
            return ArchitectureExecutionPlan(KEEP, "none", "none", "KEEP", True, 0)
        parameters = dict(action.action_parameters or {})
        target = str(parameters.get("target") or "middle_layers")
        selector = str(parameters.get("selector") or "all")
        strength = str(parameters.get("strength") or "moderate")
        if target not in self._TARGETS:
            raise ValueError(f"Unsupported architecture target: {target}")
        if strength not in self._STRENGTHS:
            raise ValueError(f"Unsupported architecture strength: {strength}")
        if action.operator == CONSERVATIVE_GROWTH:
            selected_primitive = (
                ADD_RESIDUAL_BRANCH
                if target in {"output_correction", "global_domain"}
                else WIDEN
            )
        elif action.operator == EXPLORATORY_MORPH:
            if target == "input_representation":
                selected_primitive = ADD_GATED_BRANCH
            elif target in {"output_correction", "global_domain"}:
                selected_primitive = ADD_CORRECTION_BRANCH
            else:
                selected_primitive = (
                    LOCAL_REINIT
                    if diagnostic.gradient.persistent_gradient_conflict
                    else ADD_LAYER
                )
        elif action.operator == REALLOCATION:
            selected_primitive = REDISTRIBUTE
        elif action.operator == REPRESENTATION_AUGMENTATION:
            if target == "output_correction":
                selected_primitive = ADD_CORRECTION_BRANCH
            else:
                selected_primitive = (
                    ADD_FOURIER_BRANCH
                    if diagnostic.frequency.has_multiscale_or_oscillatory_evidence
                    else ADD_GATED_BRANCH
                )
        elif action.operator == LOCAL_SPECIALIZATION:
            selected_primitive = ADD_LOCALIZED_EXPERT
        elif action.operator == SPECTRAL_ADAPTATION:
            selected_primitive = ADD_ADAPTIVE_FOURIER_BRANCH
        elif action.operator == COMPONENT_CAPACITY_ADAPTATION:
            selected_primitive = REALLOCATE_COMPONENT_CAPACITY
        elif action.operator == GRADIENT_TOPOLOGY_ADAPTATION:
            selected_primitive = SPLIT_CONFLICTING_TOPOLOGY
        elif action.operator == OBJECTIVE_REBALANCING:
            selected_primitive = REBALANCE_LOSS_WEIGHTS
        elif action.operator == CONSTRAINT_EMPHASIS:
            selected_primitive = EMPHASIZE_CONSTRAINTS
        elif action.operator == GRADIENT_CONFLICT_MITIGATION:
            selected_primitive = MITIGATE_GRADIENT_CONFLICT
        elif action.operator == REGION_AWARE_REBALANCING:
            selected_primitive = (
                REWEIGHT_RESIDUAL_REGIONS
                if target in {"high_residual_regions", "interface_regions", "boundary_regions"}
                else BALANCE_COMPONENT_GRADIENTS
            )
        elif action.operator == GRADIENT_SURGERY:
            selected_primitive = PROJECT_CONFLICTING_GRADIENTS
        elif action.operator == SAMPLING_ADAPTATION:
            selected_primitive = ADAPT_SAMPLING_DISTRIBUTION
        else:
            raise ValueError(f"Unsupported architecture action: {action.operator}")
        requested_mechanism = str(
            parameters.get("mechanism") or AUTO_MECHANISM
        ).upper()
        if requested_mechanism != AUTO_MECHANISM:
            if requested_mechanism not in MECHANISMS_BY_ACTION[action.operator]:
                raise ValueError(
                    f"Mechanism {requested_mechanism!r} is incompatible with "
                    f"architecture action {action.operator}"
                )
            selected_primitive = requested_mechanism
        plan = self.plan_primitive(
            selected_primitive,
            target=target,
            strength=strength,
            selector=selector,
            diagnostic=diagnostic,
        )
        if selected_primitive in {
            REALLOCATE_COMPONENT_CAPACITY, SPLIT_CONFLICTING_TOPOLOGY
        }:
            norms = dict(diagnostic.gradient.component_gradient_norms or {})
            indexed: list[tuple[float, int]] = []

            def pde_component_index(name: object) -> int | None:
                rendered = str(name)
                token = rendered.rsplit("_", 1)[-1]
                if rendered.startswith("pde/") and token.isdigit():
                    return int(token)
                return None

            # A topology split must follow the measured conflict direction.
            # Prefer the PDE output in the most negative pair; only fall back
            # to the weakest PDE gradient when the audit contains no pairwise
            # identity (for example an older checkpoint).
            conflict_candidates: list[tuple[float, float, int]] = []
            for pair in diagnostic.gradient.conflicting_component_pairs:
                cosine = float(pair.get("cosine_similarity", 0.0))
                for key in ("component_a", "component_b"):
                    name = str(pair.get(key, ""))
                    index = pde_component_index(name)
                    if index is not None:
                        conflict_candidates.append(
                            (cosine, float(norms.get(name, float("inf"))), index)
                        )
            for name, value in norms.items():
                index = pde_component_index(name)
                if index is not None:
                    indexed.append((float(value), index))
            if conflict_candidates or indexed:
                component_index = (
                    min(conflict_candidates)[2]
                    if conflict_candidates
                    else min(indexed)[1]
                )
                plan = replace(
                    plan,
                    parameters={**dict(plan.parameters), "component_index": component_index},
                )
        return plan

    def plan_primitive(
        self,
        primitive_action: str,
        *,
        target: str,
        strength: str,
        selector: str = "all",
        strength_value: float | None = None,
        diagnostic: OnlineMorphDiagnostics,
    ) -> ArchitectureExecutionPlan:
        """Lower one executor-owned primitive into a concrete safe plan."""

        allowed_targets = MECHANISM_TARGETS.get(primitive_action)
        if allowed_targets is None:
            raise ValueError(f"Unsupported adaptation primitive: {primitive_action}")
        if target not in allowed_targets:
            raise ValueError(
                f"Target {target!r} is incompatible with adaptation primitive "
                f"{primitive_action}; allowed: {sorted(allowed_targets)}"
            )

        if primitive_action in {WIDEN, REGROW}:
            primitive = "NET2WIDER"
            strict = True
        elif primitive_action in {ADD_LAYER, LOCAL_REINIT}:
            primitive = (
                "LOCALIZED_TOPOLOGY_RESHAPE"
                if diagnostic.spatial.localized_residual_concentration
                else "SPECTRAL_TOPOLOGY_RESHAPE"
                if diagnostic.frequency.has_multiscale_or_oscillatory_evidence
                else "TOPOLOGY_RESHAPE"
            )
            strict = False
        elif primitive_action in {PRUNE, REDISTRIBUTE}:
            primitive = "STRUCTURED_PRUNE_REGROW"
            strict = False
        elif primitive_action in {
            ADD_RESIDUAL_BRANCH,
            ADD_CORRECTION_BRANCH,
            ADD_FOURIER_BRANCH,
            ADD_GATED_BRANCH,
            ADD_LOCALIZED_EXPERT,
            ADD_ADAPTIVE_FOURIER_BRANCH,
            ADD_COMPONENT_EXPERT,
            SPLIT_CONFLICTING_TOPOLOGY,
        }:
            # Every additive branch is initialized with an exactly-zero output
            # map, so the live PINN function and its derivatives are preserved
            # at the mutation instant. The branch becomes active only through
            # the normal post-morph optimizer warmup.
            primitive = primitive_action
            strict = True
        elif primitive_action == REALLOCATE_COMPONENT_CAPACITY:
            primitive = primitive_action
            strict = False
        elif primitive_action in {
            REBALANCE_LOSS_WEIGHTS,
            EMPHASIZE_CONSTRAINTS,
            MITIGATE_GRADIENT_CONFLICT,
            BALANCE_COMPONENT_GRADIENTS,
            REWEIGHT_RESIDUAL_REGIONS,
            PROJECT_CONFLICTING_GRADIENTS,
            ADAPT_SAMPLING_DISTRIBUTION,
        }:
            # Objective plans are executed by the trainer's loss boundary,
            # never by the topology mutator. The plan remains declarative so
            # the same schema, scoring, audit, and causal-credit path applies.
            primitive = primitive_action
            strict = True
        else:
            raise ValueError(f"Unsupported adaptation primitive: {primitive_action}")
        recovery = int(
            (self.config.get("recovery_window_steps") or {}).get(strength, 0)
            if isinstance(self.config.get("recovery_window_steps"), Mapping)
            else 0
        )
        if recovery <= 0:
            recovery = {"mild": 100, "moderate": 250, "strong": 500}[strength]
        return ArchitectureExecutionPlan(
            primitive_action,
            target,
            strength,
            primitive,
            strict,
            recovery,
            (
                float(strength_value)
                if strength_value is not None
                else {"mild": 0.125, "moderate": 0.25, "strong": 0.5}[strength]
            ),
            selector=selector,
        )


@dataclass(frozen=True)
class DerivativeSnapshot:
    prediction: torch.Tensor
    first_derivative: torch.Tensor
    second_derivative: torch.Tensor


@dataclass(frozen=True)
class PreservationCheck:
    function_error: float
    first_derivative_error: float
    second_derivative_error: float


@dataclass(frozen=True)
class PDEResidualPreservationCheck:
    per_residual_errors: dict[str, float]
    maximum_error: float
    residual_names_match: bool


def compare_pde_residual_snapshots(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> PDEResidualPreservationCheck:
    """Compare every registered residual independently."""

    names_match = set(before) == set(after)
    errors: dict[str, float] = {}
    for name in sorted(set(before).intersection(after)):
        left = before[name]
        right = after[name]
        if tuple(left.shape) != tuple(right.shape):
            errors[str(name)] = float("inf")
        else:
            errors[str(name)] = float(
                (right.detach().cpu() - left.detach().cpu()).abs().max()
            )
    maximum = (
        max(errors.values())
        if errors and names_match
        else float("inf")
    )
    return PDEResidualPreservationCheck(
        per_residual_errors=errors,
        maximum_error=maximum,
        residual_names_match=names_match,
    )


def capture_function_and_derivatives(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    maximum_dimensions: int | None = None,
) -> DerivativeSnapshot:
    """Capture output plus diagonal first/second input derivatives."""

    points = inputs.detach().clone().requires_grad_(True)
    prediction = model(points)
    dimension_count = min(
        int(points.shape[1]),
        int(maximum_dimensions) if maximum_dimensions is not None else int(points.shape[1]),
    )
    first_by_output: list[torch.Tensor] = []
    second_by_output: list[torch.Tensor] = []
    for output_index in range(int(prediction.shape[1])):
        first = torch.autograd.grad(
            prediction[:, output_index].sum(),
            points,
            create_graph=True,
            retain_graph=True,
        )[0][:, :dimension_count]
        second_parts = []
        for dimension in range(dimension_count):
            scalar = first[:, dimension].sum()
            if scalar.requires_grad:
                second = torch.autograd.grad(
                    scalar,
                    points,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                second_parts.append(
                    second[:, dimension]
                    if second is not None
                    else torch.zeros_like(first[:, dimension])
                )
            else:
                second_parts.append(torch.zeros_like(first[:, dimension]))
        first_by_output.append(first)
        second_by_output.append(torch.stack(second_parts, dim=-1))
    return DerivativeSnapshot(
        prediction=prediction.detach(),
        first_derivative=torch.stack(first_by_output, dim=1).detach(),
        second_derivative=torch.stack(second_by_output, dim=1).detach(),
    )


def check_function_and_derivative_preservation(
    before: DerivativeSnapshot,
    after: DerivativeSnapshot,
) -> PreservationCheck:
    def maximum(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).abs().max().detach().cpu())

    return PreservationCheck(
        function_error=maximum(before.prediction, after.prediction),
        first_derivative_error=maximum(before.first_derivative, after.first_derivative),
        second_derivative_error=maximum(before.second_derivative, after.second_derivative),
    )


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().casefold().replace("-", "_")
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    return nn.Tanh()


def _hidden_stack(
    input_dim: int,
    output_dim: int,
    *,
    width: int,
    depth: int,
    activation: str,
    zero_output: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(max(1, int(depth))):
        layers.extend((nn.Linear(current, int(width)), _activation(activation)))
        current = int(width)
    output = nn.Linear(current, int(output_dim))
    if zero_output:
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class _ResidualOutputBlock(nn.Module):
    """Output-space residual depth growth with a zero gate at insertion."""

    def __init__(
        self, output_dim: int, *, width: int, depth: int, activation: str
    ) -> None:
        super().__init__()
        self.branch = _hidden_stack(
            output_dim,
            output_dim,
            width=width,
            depth=depth,
            activation=activation,
            zero_output=False,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        del inputs
        return prediction + self.gate * self.branch(prediction)


class _WidthExpansion(nn.Module):
    """Add parallel hidden units with a zero initialized outgoing map."""

    def __init__(self, output_dim: int, *, width: int, activation: str) -> None:
        super().__init__()
        self.branch = _hidden_stack(
            output_dim,
            output_dim,
            width=width,
            depth=1,
            activation=activation,
            zero_output=True,
        )

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        del inputs
        return prediction + self.branch(prediction)


class _InputCorrectionBranch(nn.Module):
    """Parallel input-to-output corrector with a zero initialized output map."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        depth: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.branch = _hidden_stack(
            input_dim,
            output_dim,
            width=width,
            depth=depth,
            activation=activation,
            zero_output=True,
        )

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return prediction + self.branch(inputs)


class _FourierCorrectionBranch(nn.Module):
    """Multi-scale Fourier corrector with a zero initialized output map."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        feature_count: int,
        depth: int,
        activation: str,
        frequency_scale: float,
        axis_frequencies=None,
    ) -> None:
        super().__init__()
        feature_count = max(1, int(feature_count))
        # A deterministic, axis-balanced log-frequency bank makes checkpoint
        # reconstruction independent of the process RNG state.
        frequencies = torch.zeros(feature_count, int(input_dim))
        bands_per_axis = max(1, math.ceil(feature_count / int(input_dim)))
        for index in range(feature_count):
            axis = index % int(input_dim)
            band = index // int(input_dim)
            relative_band = band / max(1, bands_per_axis - 1)
            frequencies[index, axis] = float(frequency_scale) * (
                2.0 ** (4.0 * relative_band)
            )
        if axis_frequencies:
            from .physical_features import directional_frequency_bank
            frequencies = directional_frequency_bank(input_dim, feature_count, axis_frequencies)
        self.register_buffer("frequencies", frequencies)
        self.include_raw_input = bool(axis_frequencies)
        self.branch = _hidden_stack(
            2 * feature_count + (input_dim if self.include_raw_input else 0),
            output_dim,
            width=max(8, 2 * feature_count),
            depth=depth,
            activation=activation,
            zero_output=True,
        )

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        phase = (2.0 * math.pi) * inputs @ self.frequencies.transpose(0, 1)
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        if self.include_raw_input:
            features = torch.cat((features, inputs), dim=-1)
        return prediction + self.branch(features)


class _AdaptiveFourierCorrectionBranch(nn.Module):
    """Trainable multi-band Fourier correction, optionally localized.

    The output map starts at exactly zero, so adding the module preserves the
    current PINN and all input derivatives. Frequencies become trainable as
    soon as the zero-output warm-up releases the rest of the branch.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        feature_count: int,
        depth: int,
        activation: str,
        frequency_scale: float,
        frequency_bandwidth: float,
        localization_scale: float | None,
        context_points: torch.Tensor | None,
        residual_scores: torch.Tensor | None,
        axis_frequencies=None,
    ) -> None:
        super().__init__()
        feature_count = max(1, int(feature_count))
        frequencies = torch.zeros(feature_count, int(input_dim))
        bands_per_axis = max(1, math.ceil(feature_count / int(input_dim)))
        bandwidth = max(1.0, float(frequency_bandwidth))
        for index in range(feature_count):
            axis = index % int(input_dim)
            band = index // int(input_dim)
            fraction = band / max(1, bands_per_axis - 1)
            frequencies[index, axis] = float(frequency_scale) * (
                bandwidth ** fraction
            )
        self.directional = bool(axis_frequencies)
        if axis_frequencies:
            from .physical_features import directional_frequency_bank
            frequencies = directional_frequency_bank(input_dim, feature_count, axis_frequencies)
        self.frequencies = nn.Parameter(frequencies)
        if self.directional:
            self.register_buffer("frequency_lower", frequencies * 0.5)
            self.register_buffer("frequency_upper", frequencies * 1.5)
        center = torch.zeros(int(input_dim))
        scale = torch.ones(int(input_dim))
        self.localized = localization_scale is not None
        if self.localized and isinstance(context_points, torch.Tensor) and context_points.ndim == 2:
            points = context_points.detach().to(device="cpu", dtype=torch.float32)
            if points.shape[0] > 0 and points.shape[1] == int(input_dim):
                if isinstance(residual_scores, torch.Tensor) and residual_scores.numel() == points.shape[0]:
                    center = points[int(torch.argmax(residual_scores.detach().abs().cpu()))]
                else:
                    center = points.mean(dim=0)
                extent = (points.amax(dim=0) - points.amin(dim=0)).clamp_min(1e-4)
                scale = extent / max(0.1, float(localization_scale))
        self.register_buffer("center", center)
        self.register_buffer("local_scale", scale.clamp_min(1e-4))
        self.branch = _hidden_stack(
            2 * feature_count + (input_dim if self.directional else 0),
            output_dim,
            width=max(8, 2 * feature_count),
            depth=depth,
            activation=activation,
            zero_output=True,
        )

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequencies.clamp(min=self.frequency_lower, max=self.frequency_upper) if self.directional else self.frequencies
        phase = (2.0 * math.pi) * inputs @ frequencies.transpose(0, 1)
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        if self.directional:
            features = torch.cat((features, inputs), dim=-1)
        correction = self.branch(features)
        if self.localized:
            distance = ((inputs - self.center) / self.local_scale).square().sum(dim=-1, keepdim=True)
            correction = correction * torch.exp(-0.5 * distance)
        return prediction + correction


class _ComponentCorrectionBranch(nn.Module):
    """Zero-output expert restricted to one state component."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        depth: int,
        activation: str,
        component_index: int,
        gated: bool,
    ) -> None:
        super().__init__()
        self.component_index = min(max(0, int(component_index)), int(output_dim) - 1)
        self.branch = _hidden_stack(
            input_dim, 1, width=width, depth=depth,
            activation=activation, zero_output=True,
        )
        # The branch output is exactly zero at insertion, so a non-zero gate
        # preserves the function while avoiding a bilinear zero-gradient
        # deadlock between the gate and the zero output layer.
        self.gate = nn.Parameter(torch.tensor(1.0)) if gated else None

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        value = self.branch(inputs)
        if self.gate is not None:
            value = torch.tanh(self.gate) * value
        mask = torch.zeros_like(prediction)
        mask[..., self.component_index : self.component_index + 1] = value
        return prediction + mask


class _GatedCorrectionBranch(nn.Module):
    """Modified-MLP-style feature gate with a zero initialized output map."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        depth: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.encoder_u = nn.Linear(input_dim, width)
        self.encoder_v = nn.Linear(input_dim, width)
        self.input_layer = nn.Linear(input_dim, width)
        self.gates = nn.ModuleList(
            nn.Linear(width, width) for _ in range(max(1, int(depth)))
        )
        self.activation = _activation(activation)
        self.output_layer = nn.Linear(width, output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        encoded_u = self.activation(self.encoder_u(inputs))
        encoded_v = self.activation(self.encoder_v(inputs))
        hidden = self.activation(self.input_layer(inputs))
        for layer in self.gates:
            gate = torch.sigmoid(layer(hidden))
            hidden = (1.0 - gate) * encoded_u + gate * encoded_v
        return prediction + self.output_layer(hidden)


class _LocalizedExpertBank(nn.Module):
    """Overlapping localized experts inspired by finite-basis PINNs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        expert_count: int,
        activation: str,
        overlap: float,
        context_points: torch.Tensor | None,
        residual_scores: torch.Tensor | None,
        center_strategy: str,
        residual_power: float,
        distance_power: float,
    ) -> None:
        super().__init__()
        expert_count = max(1, int(expert_count))
        centers, scales, selected_indices = self._initial_windows(
            input_dim=int(input_dim),
            expert_count=expert_count,
            overlap=float(overlap),
            context_points=context_points,
            residual_scores=residual_scores,
            center_strategy=str(center_strategy),
            residual_power=float(residual_power),
            distance_power=float(distance_power),
        )
        self.register_buffer("centers", centers)
        self.register_buffer("scales", scales)
        self.register_buffer("selected_point_indices", selected_indices)
        self.experts = nn.ModuleList(
            _hidden_stack(
                input_dim,
                output_dim,
                width=width,
                depth=1,
                activation=activation,
                zero_output=True,
            )
            for _ in range(expert_count)
        )

    @staticmethod
    def _initial_windows(
        *,
        input_dim: int,
        expert_count: int,
        overlap: float,
        context_points: torch.Tensor | None,
        residual_scores: torch.Tensor | None,
        center_strategy: str,
        residual_power: float,
        distance_power: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            isinstance(context_points, torch.Tensor)
            and context_points.ndim == 2
            and context_points.shape[0] > 0
            and context_points.shape[1] == input_dim
        ):
            points = context_points.detach().to(device="cpu", dtype=torch.float32)
            extent = (points.amax(dim=0) - points.amin(dim=0)).clamp_min(1e-6)
            normalized = (points - points.amin(dim=0)) / extent
            scores = (
                residual_scores.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                if isinstance(residual_scores, torch.Tensor)
                and residual_scores.numel() == points.shape[0]
                else None
            )
            use_residual = str(center_strategy) == "residual_farthest" and scores is not None
            if use_residual:
                scores = scores.abs()
                chosen = [int(torch.argmax(scores))]
                normalized_residual = scores / scores.max().clamp_min(1e-12)
            else:
                chosen = [int(torch.argmin((normalized - normalized.mean(dim=0)).square().sum(dim=1)))]
                normalized_residual = torch.ones(points.shape[0])
            minimum_distance = (normalized - normalized[chosen[0]]).square().sum(dim=1)
            while len(chosen) < expert_count:
                distance = minimum_distance.sqrt()
                selection_score = (
                    (normalized_residual + 1e-12).pow(max(0.0, residual_power))
                    * (distance + 1e-12).pow(max(0.0, distance_power))
                )
                selection_score[torch.as_tensor(chosen, dtype=torch.long)] = -1.0
                next_index = int(torch.argmax(selection_score))
                chosen.append(next_index)
                distance = (normalized - normalized[next_index]).square().sum(dim=1)
                minimum_distance = torch.minimum(minimum_distance, distance)
            centers = points[chosen].clone()
            partitions_per_axis = max(1.0, expert_count ** (1.0 / input_dim))
            scale = extent * max(overlap, 1e-6) / partitions_per_axis
            scales = scale.expand_as(centers).clone()
        else:
            centers = torch.zeros(expert_count, input_dim)
            scales = torch.ones(expert_count, input_dim)
            chosen = list(range(expert_count))
        return centers, scales, torch.as_tensor(chosen, dtype=torch.long)

    def forward(self, inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        normalized = (inputs[:, None, :] - self.centers[None, :, :]) / self.scales[None, :, :]
        log_windows = -0.5 * normalized.square().sum(dim=-1)
        weights = torch.softmax(log_windows, dim=1)
        corrections = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        return prediction + (weights.unsqueeze(-1) * corrections).sum(dim=1)


class OnlineArchitectureModel(nn.Module):
    """Stable wrapper whose base MLP topology can change during training."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        input_dim: int,
        output_dim: int,
        activation: str = "tanh",
        morph_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_name = str(activation)
        self.morph_config = dict(morph_config or {})
        self.branch_input_transform = nn.Identity()
        if self.morph_config.get("branch_input_transform"):
            from .physical_features import SpatialPeriodicTransform
            self.branch_input_transform = SpatialPeriodicTransform(self.input_dim, **self.morph_config["branch_input_transform"])
            self.input_dim = self.branch_input_transform.output_dim
        self.interventions = nn.ModuleList()
        self._morph_specs: list[dict[str, Any]] = []
        self._initial_hidden_widths = self._direct_hidden_widths()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prediction = self.base_model(inputs)
        additive_index = -1
        for spec in self._morph_specs:
            if spec.get("direct_topology"):
                continue
            additive_index += 1
            if bool(spec.get("disabled", False)):
                continue
            influence = float(spec.get("adaptive_influence", 1.0))
            if influence <= 0.0:
                continue
            modified = self.interventions[additive_index](self.branch_input_transform(inputs), prediction)
            prediction = prediction + influence * (modified - prediction)
        return prediction

    @property
    def morph_count(self) -> int:
        return len(self._morph_specs)

    def architecture_state(self) -> list[dict[str, Any]]:
        return deepcopy(self._morph_specs)

    def current_hidden_widths(self) -> tuple[int, ...]:
        """Return the live direct-MLP width profile used by safety checks."""

        widths = self._direct_hidden_widths()
        return tuple(widths or ())

    def _direct_hidden_widths(self) -> list[int] | None:
        hidden = getattr(self.base_model, "hidden", None)
        output = getattr(self.base_model, "output", None)
        if not isinstance(hidden, nn.ModuleList) or not isinstance(output, nn.Linear):
            return None
        if not hidden or not all(isinstance(layer, nn.Linear) for layer in hidden):
            return None
        if bool(getattr(self.base_model, "residual_enabled", False)):
            return None
        return [int(layer.out_features) for layer in hidden]

    @staticmethod
    def _target_layer_indices(target: str, layer_count: int) -> tuple[int, ...]:
        if layer_count < 1:
            return ()
        if target == "early_layers":
            return tuple(range(max(1, layer_count // 3)))
        if target == "late_layers":
            start = min(layer_count - 1, max(0, (2 * layer_count) // 3))
            return tuple(range(start, layer_count))
        if target == "middle_layers":
            if layer_count <= 2:
                return tuple(range(layer_count))
            return tuple(range(1, layer_count - 1))
        if target == "bottleneck_layers":
            return ()  # Resolved from the live width profile below.
        return tuple(range(layer_count))

    @staticmethod
    def _parameter_count_for_widths(
        input_dim: int, output_dim: int, widths: list[int], bias: bool
    ) -> int:
        dimensions = [int(input_dim), *map(int, widths), int(output_dim)]
        return sum(
            left * right + (right if bias else 0)
            for left, right in zip(dimensions, dimensions[1:])
        )

    def _selected_layer_indices(
        self, plan: ArchitectureExecutionPlan, widths: list[int]
    ) -> list[int]:
        """Resolve a policy selector against the live network, not a proxy."""

        requested = list((plan.parameters or {}).get("layer_indices") or ())
        if requested:
            selected = [int(index) for index in requested if 0 <= int(index) < len(widths)]
            if len(selected) != len(requested):
                raise ValueError("operation layer_indices exceed the live hidden depth")
            return selected
        candidates = list(self._target_layer_indices(plan.target, len(widths)))
        if plan.target == "bottleneck_layers":
            minimum = min(widths)
            candidates = [
                index for index, width in enumerate(widths) if width == minimum
            ]
        if not candidates:
            candidates = list(range(len(widths)))
        selector = str(plan.selector or "all")
        if selector == "all":
            return candidates
        if selector in {"widest", "narrowest"}:
            chooser = max if selector == "widest" else min
            return [chooser(candidates, key=lambda index: widths[index])]
        if selector in {"highest_utility", "lowest_utility"}:
            hidden = list(self.base_model.hidden)
            output = self.base_model.output
            scores: dict[int, float] = {}
            for index in candidates:
                following = hidden[index + 1] if index + 1 < len(hidden) else output
                scores[index] = float(
                    hidden[index].weight.detach().square().mean().sqrt()
                    + following.weight.detach().square().mean().sqrt()
                )
            chooser = max if selector == "highest_utility" else min
            return [chooser(candidates, key=scores.__getitem__)]
        raise ValueError(f"Unsupported architecture selector: {selector}")

    @staticmethod
    def _execution_strength_fraction(plan: ArchitectureExecutionPlan) -> float:
        """Resolve the actual continuous structural fraction used by execution."""

        parameters = dict(plan.parameters or {})
        numeric = float(
            parameters.get("width_fraction")
            or parameters.get("reset_fraction")
            or parameters.get("move_fraction")
            or plan.strength_value
            or 0.0
        )
        if numeric <= 0.0:
            numeric = {"mild": 0.125, "moderate": 0.25, "strong": 0.5}[
                plan.strength
            ]
        return min(1.0, max(1e-6, numeric))

    def _planned_hidden_widths(
        self, plan: ArchitectureExecutionPlan
    ) -> tuple[list[int], dict[str, Any]]:
        widths = self._direct_hidden_widths()
        if widths is None:
            raise RuntimeError(
                "Direct topology morphing requires a non-residual MLP with Linear hidden layers"
            )
        strength_fraction = self._execution_strength_fraction(plan)
        targets = self._selected_layer_indices(plan, widths)
        updated = list(widths)
        metadata: dict[str, Any] = {"target_layer_indices": targets}
        if plan.action in {WIDEN, REGROW}:
            for index in targets:
                increment = max(1, int(math.ceil(widths[index] * strength_fraction)))
                updated[index] += increment
        elif plan.action in {ADD_LAYER, LOCAL_REINIT}:
            if not targets:
                targets = list(range(len(widths)))
            for position, index in enumerate(targets):
                delta = max(1, int(math.ceil(widths[index] * strength_fraction)))
                updated[index] = (
                    widths[index] + delta
                    if position % 2 == 0
                    else max(4, widths[index] - delta)
                )
            if updated == widths:
                updated[targets[0]] += 1
        elif plan.action == REDISTRIBUTE:
            recipients = targets or [min(range(len(widths)), key=widths.__getitem__)]
            recipient = min(recipients, key=widths.__getitem__)
            donors = [index for index in range(len(widths)) if index != recipient]
            if not donors:
                raise RuntimeError("REDISTRIBUTE requires at least two hidden layers")
            donor = max(donors, key=widths.__getitem__)
            moved = max(1, int(math.ceil(widths[donor] * strength_fraction)))
            minimum_width = int((plan.parameters or {}).get("minimum_width", 4))
            moved = min(moved, max(1, widths[donor] - minimum_width))
            updated[donor] = max(minimum_width, widths[donor] - moved)
            updated[recipient] += moved
            first = self.base_model.hidden[0]
            output = self.base_model.output
            old_count = self._parameter_count_for_widths(
                first.in_features, output.out_features, widths, first.bias is not None
            )
            best = list(updated)
            best_error = abs(
                self._parameter_count_for_widths(
                    first.in_features, output.out_features, best, first.bias is not None
                )
                - old_count
            )
            for adjustment in range(-moved, moved + 1):
                candidate = list(updated)
                candidate[recipient] = max(4, updated[recipient] + adjustment)
                error = abs(
                    self._parameter_count_for_widths(
                        first.in_features,
                        output.out_features,
                        candidate,
                        first.bias is not None,
                    )
                    - old_count
                )
                if error < best_error:
                    best, best_error = candidate, error
            updated = best
            metadata.update(
                {
                    "donor_layer": donor,
                    "recipient_layer": recipient,
                    "parameter_count_delta_estimate": (
                        self._parameter_count_for_widths(
                            first.in_features,
                            output.out_features,
                            updated,
                            first.bias is not None,
                        )
                        - old_count
                    ),
                }
            )
        elif plan.action == PRUNE:
            minimum_width = int((plan.parameters or {}).get("minimum_width", 4))
            for index in targets:
                decrement = max(1, int(math.ceil(widths[index] * strength_fraction)))
                updated[index] = max(minimum_width, widths[index] - decrement)
        else:
            raise ValueError(f"Cannot plan hidden widths for {plan.action}")
        return updated, metadata

    @staticmethod
    def _unit_map(
        current_width: int,
        requested_width: int,
        utility: torch.Tensor | None,
    ) -> list[int]:
        if utility is None:
            ranked = list(range(current_width))
        else:
            ranked = [
                int(index)
                for index in torch.argsort(utility.detach().cpu(), descending=True)
            ]
        selected = ranked[: min(current_width, requested_width)]
        if not selected:
            selected = [0]
        seed = list(selected)
        while len(selected) < requested_width:
            selected.append(seed[(len(selected) - len(seed)) % len(seed)])
        return selected

    def apply_soft_prune_preview(
        self, plan: ArchitectureExecutionPlan
    ) -> dict[str, Any] | None:
        """Temporarily attenuate proposed donor neurons before hard reallocation."""

        if plan.action != REDISTRIBUTE:
            return None
        requested, metadata = self._planned_hidden_widths(plan)
        donor = int(metadata["donor_layer"])
        hidden = list(self.base_model.hidden)
        current_width = int(hidden[donor].out_features)
        prune_count = max(0, current_width - int(requested[donor]))
        if prune_count < 1:
            return None
        following = hidden[donor + 1] if donor + 1 < len(hidden) else self.base_model.output
        utility = hidden[donor].weight.detach().norm(dim=1) + following.weight.detach().norm(dim=0)
        indices = torch.argsort(utility)[:prune_count].to(following.weight.device)
        original = following.weight.detach().index_select(1, indices).clone()
        with torch.no_grad():
            following.weight[:, indices] *= 0.5
        return {
            "following_layer": following,
            "indices": indices,
            "original_columns": original,
            "donor_layer": donor,
            "soft_scale": 0.5,
        }

    @staticmethod
    def restore_soft_prune_preview(preview: Mapping[str, Any] | None) -> None:
        if not preview:
            return
        layer = preview["following_layer"]
        indices = preview["indices"]
        with torch.no_grad():
            layer.weight[:, indices] = preview["original_columns"]

    def execute_architecture_plan(
        self,
        plan: ArchitectureExecutionPlan,
        *,
        context_points: torch.Tensor | None = None,
        residual_scores: torch.Tensor | None = None,
    ) -> MorphResult:
        if plan.action == KEEP:
            return MorphResult(self, (), (), (), {"execution_plan": plan.to_dict()})
        if plan.action == REALLOCATE_COMPONENT_CAPACITY:
            widths = list(self.current_hidden_widths())
            if not widths:
                raise RuntimeError("component reallocation requires a direct MLP")
            parameters = dict(plan.parameters or {})
            fraction = self._execution_strength_fraction(plan)
            branch_width = int(
                parameters.get(
                    "branch_width", max(4, math.ceil(max(widths) * fraction))
                )
            )
            branch_depth = int(parameters.get("branch_depth", 1))
            before_count = sum(parameter.numel() for parameter in self.parameters())
            estimated_branch = self.estimate_intervention_parameter_count(
                ADD_COMPONENT_EXPERT,
                width=branch_width,
                depth=branch_depth,
            )
            donor = max(range(len(widths)), key=widths.__getitem__)
            first = self.base_model.hidden[0]
            output = self.base_model.output
            baseline_direct = self._parameter_count_for_widths(
                first.in_features, output.out_features, widths, first.bias is not None
            )
            requested = list(widths)
            while requested[donor] > 4:
                requested[donor] -= 1
                reduced = baseline_direct - self._parameter_count_for_widths(
                    first.in_features, output.out_features, requested,
                    first.bias is not None,
                )
                if reduced >= estimated_branch:
                    break
            if requested[donor] == widths[donor]:
                raise RuntimeError("no shared capacity is available for reallocation")
            prune_fraction = (widths[donor] - requested[donor]) / widths[donor]
            prune_plan = ArchitectureExecutionPlan(
                action=PRUNE,
                target="all_hidden_layers",
                strength=plan.strength,
                primitive="STRUCTURED_PRUNE_REGROW",
                strict_preservation=False,
                recovery_steps=plan.recovery_steps,
                strength_value=prune_fraction,
                parameters={
                    "layer_indices": [donor],
                    "width_fraction": prune_fraction,
                    "minimum_width": 4,
                },
                operation_index=plan.operation_index,
                selector="widest",
            )
            pruned = self.execute_architecture_plan(prune_plan)
            branch = self.add_intervention(
                ADD_COMPONENT_EXPERT,
                width=branch_width,
                depth=branch_depth,
                operation_parameters=parameters,
            )
            self._morph_specs[-1]["execution_plan"] = plan.to_dict()
            self._morph_specs[-1]["capability"] = REALLOCATE_COMPONENT_CAPACITY
            after_count = sum(parameter.numel() for parameter in self.parameters())
            tolerance = float(parameters.get("parameter_tolerance", 0.05))
            if after_count > before_count * (1.0 + tolerance):
                raise RuntimeError(
                    "component capacity reallocation exceeded its fixed-budget tolerance"
                )
            return MorphResult(
                module=self,
                new_parameters=tuple(
                    dict.fromkeys((*pruned.new_parameters, *branch.new_parameters))
                ),
                activation_parameters=branch.activation_parameters,
                warmup_parameters=tuple(
                    dict.fromkeys((*pruned.warmup_parameters, *branch.warmup_parameters))
                ),
                metadata={
                    "execution_plan": plan.to_dict(),
                    "capability": REALLOCATE_COMPONENT_CAPACITY,
                    "donor_layer": donor,
                    "shared_width_before": widths[donor],
                    "shared_width_after": requested[donor],
                    "component_index": int(parameters.get("component_index", 0)),
                    "parameter_count_before": before_count,
                    "parameter_count_after": after_count,
                    "parameter_delta": after_count - before_count,
                },
                parameter_replacements=pruned.parameter_replacements,
            )
        if plan.action in {
            ADD_RESIDUAL_BRANCH,
            ADD_CORRECTION_BRANCH,
            ADD_FOURIER_BRANCH,
            ADD_GATED_BRANCH,
            ADD_LOCALIZED_EXPERT,
            ADD_ADAPTIVE_FOURIER_BRANCH,
            ADD_COMPONENT_EXPERT,
            SPLIT_CONFLICTING_TOPOLOGY,
        }:
            widths = self.current_hidden_widths()
            reference_width = max(widths) if widths else 32
            fraction = self._execution_strength_fraction(plan)
            parameters = dict(plan.parameters or {})
            if (
                plan.action == ADD_ADAPTIVE_FOURIER_BRANCH
                and plan.target == "localized_regions"
            ):
                parameters.setdefault("localization_scale", 2.0)
                plan = replace(plan, parameters=parameters)
            width = int(
                parameters.get(
                    "branch_width", max(4, int(math.ceil(reference_width * fraction)))
                )
            )
            depth = int(
                parameters.get(
                    "expert_count" if plan.action == ADD_LOCALIZED_EXPERT else "branch_depth",
                    1 if fraction <= 0.35 else 2 if fraction <= 0.70 else 3,
                )
            )
            operator = (
                ADD_RESIDUAL_BLOCK
                if plan.action == ADD_RESIDUAL_BRANCH
                else plan.action
            )
            result = self.add_intervention(
                operator,
                width=width,
                depth=depth,
                context_points=context_points,
                residual_scores=residual_scores,
                operation_parameters=parameters,
            )
            self._morph_specs[-1]["execution_plan"] = plan.to_dict()
            self._morph_specs[-1]["capability"] = plan.action
            return MorphResult(
                module=result.module,
                new_parameters=result.new_parameters,
                activation_parameters=result.activation_parameters,
                warmup_parameters=result.warmup_parameters,
                metadata={
                    **dict(result.metadata),
                    "execution_plan": plan.to_dict(),
                    "capability": plan.action,
                    "width": width,
                    "depth": depth,
                },
                parameter_replacements=result.parameter_replacements,
            )
        if plan.action == ADD_LAYER:
            return self._execute_add_layer(plan)
        if plan.action == LOCAL_REINIT:
            return self._execute_local_reinit(plan)
        old_hidden = list(getattr(self.base_model, "hidden", ()))
        old_output = getattr(self.base_model, "output", None)
        if not old_hidden or not isinstance(old_output, nn.Linear):
            raise RuntimeError("The current PINN architecture does not support direct topology morphing")
        requested, plan_metadata = self._planned_hidden_widths(plan)
        old_widths = [int(layer.out_features) for layer in old_hidden]
        utilities: list[torch.Tensor] = []
        for index, layer in enumerate(old_hidden):
            following = old_hidden[index + 1] if index + 1 < len(old_hidden) else old_output
            utilities.append(
                layer.weight.detach().norm(dim=1)
                + following.weight.detach().norm(dim=0)
            )
        maps = [
            self._unit_map(old, new, utility)
            for old, new, utility in zip(old_widths, requested, utilities)
        ]
        reference = old_hidden[0].weight
        bias = old_hidden[0].bias is not None
        new_hidden = nn.ModuleList()
        replacements: list[ParameterReplacement] = []
        previous_map: list[int] | None = None
        for index, (old_layer, output_map) in enumerate(zip(old_hidden, maps)):
            input_map = previous_map or list(range(old_layer.in_features))
            counts = [input_map.count(item) for item in input_map]
            layer = nn.Linear(len(input_map), len(output_map), bias=bias).to(
                device=reference.device, dtype=reference.dtype
            )
            with torch.no_grad():
                rows = torch.tensor(output_map, device=reference.device)
                columns = torch.tensor(input_map, device=reference.device)
                weight = old_layer.weight.index_select(0, rows).index_select(1, columns)
                if plan.strict_preservation:
                    weight = weight / torch.tensor(
                        counts, device=reference.device, dtype=reference.dtype
                    ).unsqueeze(0)
                layer.weight.copy_(weight)
                if layer.bias is not None and old_layer.bias is not None:
                    layer.bias.copy_(old_layer.bias.index_select(0, rows))
            replacements.append(
                ParameterReplacement(
                    old_layer.weight,
                    layer.weight,
                    tuple(output_map),
                    tuple(input_map),
                    tuple(counts if plan.strict_preservation else [1] * len(counts)),
                )
            )
            if layer.bias is not None and old_layer.bias is not None:
                replacements.append(
                    ParameterReplacement(
                        old_layer.bias, layer.bias, tuple(output_map)
                    )
                )
            new_hidden.append(layer)
            previous_map = output_map
        final_map = previous_map or list(range(old_output.in_features))
        final_counts = [final_map.count(item) for item in final_map]
        output = nn.Linear(len(final_map), old_output.out_features, bias=old_output.bias is not None).to(
            device=reference.device, dtype=reference.dtype
        )
        with torch.no_grad():
            columns = torch.tensor(final_map, device=reference.device)
            weight = old_output.weight.index_select(1, columns)
            if plan.strict_preservation:
                weight = weight / torch.tensor(
                    final_counts, device=reference.device, dtype=reference.dtype
                ).unsqueeze(0)
            output.weight.copy_(weight)
            if output.bias is not None and old_output.bias is not None:
                output.bias.copy_(old_output.bias)
        replacements.append(
            ParameterReplacement(
                old_output.weight,
                output.weight,
                (),
                tuple(final_map),
                tuple(
                    final_counts
                    if plan.strict_preservation
                    else [1] * len(final_counts)
                ),
            )
        )
        if output.bias is not None and old_output.bias is not None:
            replacements.append(ParameterReplacement(old_output.bias, output.bias))
        self.base_model.hidden = new_hidden
        self.base_model.output = output
        if hasattr(self.base_model, "projections"):
            self.base_model.projections = nn.ModuleList(
                nn.Identity() for _ in requested
            )
        spec = {
            "operator": plan.action,
            "execution_plan": plan.to_dict(),
            "hidden_layers_before": old_widths,
            "hidden_layers_after": requested,
            "direct_topology": True,
            **plan_metadata,
        }
        self._morph_specs.append(spec)
        parameters = tuple(
            parameter
            for layer in [*new_hidden, output]
            for parameter in layer.parameters()
        )
        return MorphResult(
            self.base_model,
            parameters,
            parameters,
            parameters,
            {"execution_plan": plan.to_dict(), **plan_metadata},
            tuple(replacements),
        )

    def _execute_add_layer(self, plan: ArchitectureExecutionPlan) -> MorphResult:
        hidden = list(getattr(self.base_model, "hidden", ()))
        if not hidden or not all(isinstance(layer, nn.Linear) for layer in hidden):
            raise RuntimeError("ADD_LAYER requires a direct MLP with Linear hidden layers")
        targets = self._selected_layer_indices(
            plan, [int(item.out_features) for item in hidden]
        )
        source_index = targets[len(targets) // 2] if targets else len(hidden) // 2
        insertion_index = min(len(hidden), source_index + 1)
        width = int(hidden[source_index].out_features)
        reference = hidden[source_index].weight
        layer = nn.Linear(width, width, bias=hidden[source_index].bias is not None).to(
            device=reference.device, dtype=reference.dtype
        )
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        before_widths = [int(item.out_features) for item in hidden]
        hidden.insert(insertion_index, layer)
        self.base_model.hidden = nn.ModuleList(hidden)
        if hasattr(self.base_model, "projections"):
            self.base_model.projections = nn.ModuleList(
                nn.Identity() for _ in hidden
            )
        after_widths = [int(item.out_features) for item in hidden]
        spec = {
            "operator": ADD_LAYER,
            "execution_plan": plan.to_dict(),
            "hidden_layers_before": before_widths,
            "hidden_layers_after": after_widths,
            "inserted_layer_index": insertion_index,
            "direct_topology": True,
        }
        self._morph_specs.append(spec)
        parameters = tuple(layer.parameters())
        return MorphResult(
            self.base_model,
            parameters,
            parameters,
            parameters,
            deepcopy(spec),
        )

    def _execute_local_reinit(self, plan: ArchitectureExecutionPlan) -> MorphResult:
        hidden = list(getattr(self.base_model, "hidden", ()))
        if not hidden or not all(isinstance(layer, nn.Linear) for layer in hidden):
            raise RuntimeError("LOCAL_REINIT requires a direct MLP with Linear hidden layers")
        fraction = self._execution_strength_fraction(plan)
        targets = self._selected_layer_indices(
            plan, [int(item.out_features) for item in hidden]
        )
        if not targets:
            targets = [len(hidden) // 2]
        replacements: list[ParameterReplacement] = []
        new_parameters: list[nn.Parameter] = []
        reset_by_layer: dict[str, list[int]] = {}
        for index in targets:
            old = hidden[index]
            following = hidden[index + 1] if index + 1 < len(hidden) else self.base_model.output
            utility = old.weight.detach().norm(dim=1) + following.weight.detach().norm(dim=0)
            count = max(1, int(math.ceil(old.out_features * fraction)))
            reset = torch.argsort(utility)[:count]
            new = nn.Linear(
                old.in_features, old.out_features, bias=old.bias is not None
            ).to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(old.weight)
                fresh = torch.empty(
                    count, old.in_features,
                    device=old.weight.device,
                    dtype=old.weight.dtype,
                )
                nn.init.xavier_uniform_(fresh)
                new.weight[reset] = fresh
                if new.bias is not None and old.bias is not None:
                    new.bias.copy_(old.bias)
                    new.bias[reset] = 0
            reset_rows = tuple(int(item) for item in reset.cpu())
            replacements.append(
                ParameterReplacement(
                    old.weight, new.weight,
                    reset_row_indices=reset_rows,
                )
            )
            if old.bias is not None and new.bias is not None:
                replacements.append(
                    ParameterReplacement(
                        old.bias, new.bias,
                        reset_row_indices=reset_rows,
                    )
                )
            hidden[index] = new
            new_parameters.extend(new.parameters())
            reset_by_layer[str(index)] = list(reset_rows)
        self.base_model.hidden = nn.ModuleList(hidden)
        spec = {
            "operator": LOCAL_REINIT,
            "execution_plan": plan.to_dict(),
            "hidden_layers_before": [int(item.out_features) for item in hidden],
            "hidden_layers_after": [int(item.out_features) for item in hidden],
            "reset_units_by_layer": reset_by_layer,
            "direct_topology": True,
        }
        self._morph_specs.append(spec)
        parameters = tuple(new_parameters)
        return MorphResult(
            self.base_model,
            parameters,
            parameters,
            parameters,
            deepcopy(spec),
            tuple(replacements),
        )

    def add_intervention(
        self,
        operator: str,
        *,
        width: int,
        depth: int = 1,
        context_points: torch.Tensor | None = None,
        residual_scores: torch.Tensor | None = None,
        operation_parameters: Mapping[str, Any] | None = None,
    ) -> MorphResult:
        operator = str(operator).strip().upper()
        width = int(width)
        depth = int(depth)
        module, activation_parameters, metadata = self._create_intervention(
            operator,
            width=width,
            depth=depth,
            context_points=context_points,
            residual_scores=residual_scores,
            operation_parameters=operation_parameters,
        )
        reference = next(self.base_model.parameters())
        module.to(device=reference.device, dtype=reference.dtype)
        self.interventions.append(module)
        spec = {"operator": operator, "width": width, "depth": depth}
        self._morph_specs.append(spec)
        parameters = tuple(module.parameters())
        activation_ids = {id(parameter) for parameter in activation_parameters}
        # ``module.to`` preserves Parameter identities.
        activation = tuple(
            parameter for parameter in parameters if id(parameter) in activation_ids
        )
        return MorphResult(
            module=module,
            new_parameters=parameters,
            activation_parameters=activation,
            warmup_parameters=parameters,
            metadata=metadata,
        )

    def latest_morph_result(self) -> MorphResult | None:
        if not self._morph_specs:
            return None
        latest = self._morph_specs[-1]
        if latest.get("direct_topology"):
            parameters = tuple(
                parameter
                for module in [*self.base_model.hidden, self.base_model.output]
                for parameter in module.parameters()
            )
            return MorphResult(
                self.base_model,
                parameters,
                parameters,
                parameters,
                {"execution_plan": deepcopy(latest.get("execution_plan") or {})},
            )
        if not self.interventions:
            return None
        module = self.interventions[-1]
        operator = str(self._morph_specs[-1]["operator"])
        parameters = tuple(module.parameters())
        activation = self._activation_parameters_for(module, operator)
        metadata: dict[str, Any] = {}
        if operator == ADD_LOCALIZED_EXPERT:
            metadata = {
                "selected_point_indices": module.selected_point_indices.tolist(),
            }
        return MorphResult(module, parameters, activation, parameters, metadata)

    @staticmethod
    def _activation_parameters_for(
        module: nn.Module, operator: str
    ) -> tuple[nn.Parameter, ...]:
        if operator == ADD_RESIDUAL_BLOCK:
            return (module.gate,)
        if operator in {
            WIDEN_LAYER, ADD_CORRECTION_BRANCH, ADD_FOURIER_BRANCH,
            ADD_ADAPTIVE_FOURIER_BRANCH, ADD_COMPONENT_EXPERT,
            SPLIT_CONFLICTING_TOPOLOGY,
        }:
            return tuple(module.branch[-1].parameters())
        if operator == ADD_GATED_BRANCH:
            return tuple(module.output_layer.parameters())
        if operator == ADD_LOCALIZED_EXPERT:
            return tuple(
                parameter
                for expert in module.experts
                for parameter in expert[-1].parameters()
            )
        raise ValueError(f"Unregistered online morph operator: {operator}")

    def estimate_intervention_parameter_count(
        self,
        operator: str,
        *,
        width: int,
        depth: int = 1,
        context_points: torch.Tensor | None = None,
        residual_scores: torch.Tensor | None = None,
    ) -> int:
        with torch.random.fork_rng(devices=[]):
            module, _, _ = self._create_intervention(
                str(operator).strip().upper(),
                width=int(width),
                depth=int(depth),
                context_points=context_points,
                residual_scores=residual_scores,
            )
        return int(sum(parameter.numel() for parameter in module.parameters()))

    def _create_intervention(
        self,
        operator: str,
        *,
        width: int,
        depth: int,
        context_points: torch.Tensor | None,
        residual_scores: torch.Tensor | None,
        operation_parameters: Mapping[str, Any] | None = None,
    ) -> tuple[nn.Module, tuple[nn.Parameter, ...], dict[str, Any]]:
        metadata: dict[str, Any] = {}
        operation_parameters = dict(operation_parameters or {})
        if context_points is not None:
            context_points = self.branch_input_transform(context_points).detach()
        if operator == ADD_RESIDUAL_BLOCK:
            module: nn.Module = _ResidualOutputBlock(
                self.output_dim,
                width=width,
                depth=depth,
                activation=self.activation_name,
            )
            activation_parameters = (module.gate,)
        elif operator == WIDEN_LAYER:
            module = _WidthExpansion(
                self.output_dim,
                width=width,
                activation=self.activation_name,
            )
            activation_parameters = tuple(module.branch[-1].parameters())
        elif operator == ADD_CORRECTION_BRANCH:
            module = _InputCorrectionBranch(
                self.input_dim,
                self.output_dim,
                width=width,
                depth=depth,
                activation=self.activation_name,
            )
            activation_parameters = tuple(module.branch[-1].parameters())
        elif operator == ADD_FOURIER_BRANCH:
            module = _FourierCorrectionBranch(
                self.input_dim,
                self.output_dim,
                feature_count=width,
                depth=depth,
                activation=self.activation_name,
                frequency_scale=float(
                    operation_parameters.get(
                        "frequency_scale",
                        self.morph_config.get("fourier_frequency_scale", 1.0),
                    )
                ),
                axis_frequencies=self.morph_config.get("fourier_axis_frequencies"),
            )
            activation_parameters = tuple(module.branch[-1].parameters())
        elif operator == ADD_ADAPTIVE_FOURIER_BRANCH:
            localized = operation_parameters.get("localization_scale")
            module = _AdaptiveFourierCorrectionBranch(
                self.input_dim,
                self.output_dim,
                feature_count=width,
                depth=depth,
                activation=self.activation_name,
                frequency_scale=float(
                    operation_parameters.get(
                        "frequency_scale",
                        self.morph_config.get("fourier_frequency_scale", 1.0),
                    )
                ),
                frequency_bandwidth=float(
                    operation_parameters.get("frequency_bandwidth", 8.0)
                ),
                localization_scale=(
                    float(localized) if localized is not None else None
                ),
                context_points=context_points,
                residual_scores=residual_scores,
                axis_frequencies=self.morph_config.get("fourier_axis_frequencies"),
            )
            activation_parameters = tuple(module.branch[-1].parameters())
            metadata = {
                "trainable_frequencies": True,
                "localized": bool(localized is not None),
                "frequency_bandwidth": float(
                    operation_parameters.get("frequency_bandwidth", 8.0)
                ),
            }
        elif operator in {ADD_COMPONENT_EXPERT, SPLIT_CONFLICTING_TOPOLOGY}:
            component_index = int(operation_parameters.get("component_index", 0))
            module = _ComponentCorrectionBranch(
                self.input_dim,
                self.output_dim,
                width=width,
                depth=depth,
                activation=self.activation_name,
                component_index=component_index,
                gated=operator == SPLIT_CONFLICTING_TOPOLOGY,
            )
            activation_parameters = tuple(module.branch[-1].parameters())
            metadata = {
                "component_index": int(module.component_index),
                "gradient_conflict_gate": operator == SPLIT_CONFLICTING_TOPOLOGY,
            }
        elif operator == ADD_GATED_BRANCH:
            module = _GatedCorrectionBranch(
                self.input_dim,
                self.output_dim,
                width=width,
                depth=depth,
                activation=self.activation_name,
            )
            activation_parameters = tuple(module.output_layer.parameters())
        elif operator == ADD_LOCALIZED_EXPERT:
            localized = dict(self.morph_config.get("localized_expert") or {})
            module = _LocalizedExpertBank(
                self.input_dim,
                self.output_dim,
                width=width,
                expert_count=depth,
                activation=self.activation_name,
                overlap=float(
                    operation_parameters.get(
                        "overlap", self.morph_config.get("localized_overlap", 1.5)
                    )
                ),
                context_points=context_points,
                residual_scores=residual_scores,
                center_strategy=str(
                    operation_parameters.get(
                        "center_strategy",
                        localized.get(
                        "center_strategy",
                        self.morph_config.get(
                            "localized_center_strategy", "residual_farthest"
                        ),
                    ),
                    )
                ),
                residual_power=float(
                    operation_parameters.get(
                        "residual_power",
                        localized.get(
                        "residual_power",
                        self.morph_config.get("localized_residual_power", 1.0),
                    ),
                    )
                ),
                distance_power=float(
                    operation_parameters.get(
                        "distance_power",
                        localized.get(
                        "distance_power",
                        self.morph_config.get("localized_distance_power", 1.0),
                    ),
                    )
                ),
            )
            activation_parameters = tuple(
                parameter
                for expert in module.experts
                for parameter in expert[-1].parameters()
            )
            metadata = {
                "center_strategy": str(
                    operation_parameters.get(
                        "center_strategy",
                        localized.get(
                        "center_strategy",
                        self.morph_config.get("localized_center_strategy", "residual_farthest"),
                    ),
                    )
                ),
                "selected_point_indices": module.selected_point_indices.tolist(),
            }
        else:
            raise ValueError(f"Unregistered online morph operator: {operator}")
        return module, tuple(activation_parameters), metadata

    def remove_last_intervention(self) -> None:
        if not self._morph_specs:
            return
        removed = self._morph_specs.pop()
        if removed.get("direct_topology"):
            previous = next(
                (
                    spec.get("hidden_layers_after")
                    for spec in reversed(self._morph_specs)
                    if spec.get("direct_topology")
                ),
                self._initial_hidden_widths,
            )
            if previous is None:
                raise RuntimeError("Cannot restore an unsupported base topology")
            self._reshape_for_state_load([int(value) for value in previous])
        elif self.interventions:
            del self.interventions[-1]

    def bind_latest_intervention_id(self, intervention_id: str) -> None:
        """Bind causal identity to the latest additive architecture module."""

        if not self._morph_specs or not self.interventions:
            return
        latest = self._morph_specs[-1]
        if latest.get("direct_topology"):
            return
        latest["intervention_id"] = str(intervention_id)

    def bind_trial_modules(self, intervention_id, first_index):
        for spec in self._morph_specs[first_index:]:
            spec["intervention_id"] = str(intervention_id)

    def trial_is_reversible(self, intervention_id):
        return not any(s.get("direct_topology") for s in self._morph_specs
                       if s.get("intervention_id") == intervention_id)

    def retire_trial_modules(self, intervention_id):
        retired = 0
        index = -1
        for spec in self._morph_specs:
            if spec.get("direct_topology"):
                continue
            index += 1
            if spec.get("intervention_id") != intervention_id:
                continue
            spec.update(disabled=True, disabled_reason="bounded_trial_retirement", adaptive_influence=0.0)
            for parameter in self.interventions[index].parameters():
                parameter.requires_grad_(False)
            retired += 1
        return retired

    def set_intervention_influence(self, intervention_id: str, influence: float) -> bool:
        matched = False
        for spec in self._morph_specs:
            if not spec.get("direct_topology") and spec.get("intervention_id") == intervention_id:
                spec["adaptive_influence"] = min(1.0, max(0.0, float(influence)))
                matched = True
        return matched

    def soft_disable_intervention(self, intervention_id: str) -> dict[str, Any]:
        """Zero and freeze a failed additive branch without rewinding training."""

        additive_index = -1
        for spec in self._morph_specs:
            if spec.get("direct_topology"):
                continue
            additive_index += 1
            if str(spec.get("intervention_id") or "") != str(intervention_id):
                continue
            if additive_index >= len(self.interventions):
                raise RuntimeError("Architecture intervention index is inconsistent")
            if bool(spec.get("disabled", False)):
                return {
                    "status": "already_disabled",
                    "intervention_id": str(intervention_id),
                    "architecture_index": additive_index,
                }
            module = self.interventions[additive_index]
            operator = str(spec.get("operator") or "")
            activation = self._activation_parameters_for(module, operator)
            with torch.no_grad():
                for parameter in activation:
                    parameter.zero_()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            spec["disabled"] = True
            spec["disabled_reason"] = "persistent_physics_feedback_failed"
            return {
                "status": "disabled",
                "intervention_id": str(intervention_id),
                "architecture_index": additive_index,
                "operator": operator,
                "disabled_parameter_count": int(
                    sum(parameter.numel() for parameter in module.parameters())
                ),
            }
        return {
            "status": "not_additive_or_not_found",
            "intervention_id": str(intervention_id),
        }

    def enforce_disabled_interventions(self) -> None:
        """Keep retired additive branches inert after global trainability changes."""

        additive_index = -1
        for spec in self._morph_specs:
            if spec.get("direct_topology"):
                continue
            additive_index += 1
            if not bool(spec.get("disabled", False)):
                continue
            if additive_index >= len(self.interventions):
                raise RuntimeError("Architecture intervention index is inconsistent")
            module = self.interventions[additive_index]
            operator = str(spec.get("operator") or "")
            with torch.no_grad():
                for parameter in self._activation_parameters_for(module, operator):
                    parameter.zero_()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _reshape_for_state_load(self, widths: list[int]) -> None:
        current = self._direct_hidden_widths()
        if current is None:
            raise RuntimeError("Checkpoint topology restore requires a direct MLP")
        old_hidden = list(self.base_model.hidden)
        old_output = self.base_model.output
        reference = old_hidden[0].weight
        bias = old_hidden[0].bias is not None
        input_dim = old_hidden[0].in_features
        dimensions = [input_dim, *map(int, widths)]
        self.base_model.hidden = nn.ModuleList(
            nn.Linear(left, right, bias=bias).to(
                device=reference.device, dtype=reference.dtype
            )
            for left, right in zip(dimensions, dimensions[1:])
        )
        self.base_model.output = nn.Linear(
            widths[-1], old_output.out_features, bias=old_output.bias is not None
        ).to(device=reference.device, dtype=reference.dtype)
        if hasattr(self.base_model, "projections"):
            self.base_model.projections = nn.ModuleList(
                nn.Identity() for _ in widths
            )

    def restore_architecture(self, specs: list[Mapping[str, Any]] | None) -> None:
        requested = [dict(item) for item in (specs or [])]
        desired_widths = next(
            (
                item.get("hidden_layers_after")
                for item in reversed(requested)
                if item.get("direct_topology")
            ),
            self._initial_hidden_widths,
        )
        if desired_widths is not None and self._direct_hidden_widths() != list(desired_widths):
            self._reshape_for_state_load([int(value) for value in desired_widths])
        self.interventions = nn.ModuleList()
        self._morph_specs = []
        for item in requested:
            if item.get("direct_topology"):
                self._morph_specs.append(deepcopy(item))
                continue
            self.add_intervention(
                str(item["operator"]),
                width=int(item["width"]),
                depth=int(item.get("depth", 1)),
                operation_parameters=dict(
                    (item.get("execution_plan") or {}).get("parameters") or {}
                ),
            )
            self._morph_specs[-1].update(
                {
                    str(key): deepcopy(value)
                    for key, value in item.items()
                    if key not in {"operator", "width", "depth"}
                }
            )
        self.enforce_disabled_interventions()


class SafeExecutionEngine:
    """Validate and execute a selected program; never accepts generated code."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._lowering = DeterministicArchitectureExecutor(self.config)

    @staticmethod
    def _strength_label(value: float) -> str:
        numeric = float(value)
        return "mild" if numeric <= 0.15 else "moderate" if numeric <= 0.35 else "strong"

    def validate(self, program: AdaptationProgram) -> None:
        # Reparse the serialized form so callers cannot bypass schema checks by
        # mutating nested mappings after dataclass construction.
        AdaptationProgram.from_dict(program.to_dict())

    def compile(
        self,
        program: AdaptationProgram,
        diagnostic: OnlineMorphDiagnostics,
    ) -> tuple[ArchitectureExecutionPlan, ...]:
        plans, _ = self.compile_with_audit(program, diagnostic)
        return plans

    def compile_with_audit(
        self,
        program: AdaptationProgram,
        diagnostic: OnlineMorphDiagnostics,
    ) -> tuple[tuple[ArchitectureExecutionPlan, ...], dict[str, Any]]:
        """Compile the requested sequence and expose every semantic change."""

        self.validate(program)
        plans: list[ArchitectureExecutionPlan] = []
        target = str(program.target.get("region") or "middle_layers")
        operations = list(program.operations)
        audit: dict[str, Any] = {
            "request_source": "llm_operations" if operations else "legacy_action_lowering",
            "requested_operations": [item.to_dict() for item in operations],
            "compiled_operations": [],
            "executed_operations": [],
            "skipped_operations": [],
            "parameter_adjustments": [],
            "substitutions": [],
            "semantic_fidelity": 1.0,
        }
        allowed = {
            str(item).upper()
            for item in (self.config.get("allowed_primitives") or PRIMITIVE_LIBRARY)
        }
        if not operations and program.has_intervention():
            label = self._strength_label(program.strength)
            action = OnlineMorphAction(
                program.action,
                0,
                0,
                {"adaptation_program": program.to_dict()},
                "policy_program",
                action_parameters={
                    "target": target,
                    "selector": str(program.target.get("selector") or "all"),
                    "strength": label,
                    "mechanism": program.mechanism,
                },
            )
            lowered = self._lowering.plan(action, diagnostic)
            if lowered.action not in allowed:
                raise ValueError(
                    f"Lowered primitive {lowered.action} is disabled by executor capabilities"
                )
            plans = [
                replace(
                    lowered,
                    recovery_steps=int(program.recovery_steps),
                    strength_value=float(program.strength),
                    operation_index=0,
                )
            ]
            audit["compiled_operations"] = [item.to_dict() for item in plans]
            return tuple(plans), audit
        equivalents = {
            WIDEN: (REGROW,),
            REGROW: (WIDEN,),
            ADD_RESIDUAL_BRANCH: (ADD_CORRECTION_BRANCH,),
            ADD_CORRECTION_BRANCH: (ADD_RESIDUAL_BRANCH,),
        }
        for index, operation in enumerate(operations):
            requested_primitive = operation.primitive
            selected_primitive = requested_primitive
            operation_target = str(operation.target.get("region") or target)
            operation_selector = str(
                operation.target.get("selector")
                or program.target.get("selector")
                or "all"
            )
            parameters = deepcopy(dict(operation.parameters or {}))
            if selected_primitive not in allowed:
                selected_primitive = next(
                    (
                        candidate
                        for candidate in equivalents.get(requested_primitive, ())
                        if candidate in allowed
                        and operation_target in MECHANISM_TARGETS[candidate]
                    ),
                    "",
                )
                if not selected_primitive:
                    raise ValueError(
                        f"Requested primitive {requested_primitive} is unavailable and "
                        "has no target-compatible equivalent"
                    )
                accepted_parameters = set(
                    PRIMITIVE_PARAMETER_CONTRACTS[selected_primitive]
                )
                removed = sorted(set(parameters) - accepted_parameters)
                if removed:
                    parameters = {
                        key: value
                        for key, value in parameters.items()
                        if key in accepted_parameters
                    }
                    audit["parameter_adjustments"].append(
                        {
                            "operation_index": index,
                            "removed_parameters": removed,
                            "reason": "equivalent_primitive_parameter_contract",
                        }
                    )
                audit["substitutions"].append(
                    {
                        "operation_index": index,
                        "requested": requested_primitive,
                        "compiled": selected_primitive,
                        "reason": "requested_capability_unavailable",
                    }
                )
                audit["semantic_fidelity"] = min(
                    float(audit["semantic_fidelity"]), 0.8
                )
            label = self._strength_label(operation.strength)
            lowered = self._lowering.plan_primitive(
                selected_primitive,
                target=operation_target,
                strength=label,
                selector=operation_selector,
                strength_value=float(operation.strength),
                diagnostic=diagnostic,
            )
            plans.append(
                replace(
                    lowered,
                    recovery_steps=int(program.recovery_steps),
                    strength_value=float(operation.strength),
                    parameters=parameters,
                    operation_index=index,
                )
            )
        audit["compiled_operations"] = [item.to_dict() for item in plans]
        return tuple(plans), audit

    def execute_plans(
        self,
        model: OnlineArchitectureModel,
        plans: tuple[ArchitectureExecutionPlan, ...],
        *,
        program: AdaptationProgram | None = None,
        compilation_audit: Mapping[str, Any] | None = None,
        context_points: torch.Tensor | None = None,
        residual_scores: torch.Tensor | None = None,
    ) -> MorphResult:
        """Execute one precompiled sequence as an atomic model transaction.

        The trainer additionally snapshots optimizer/scheduler state because
        those objects live outside this executor.
        """

        architecture_before = deepcopy(model.architecture_state())
        state_before = deepcopy(model.state_dict())
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        results: list[MorphResult] = []
        executed: list[dict[str, Any]] = []
        for plan in plans:
            try:
                result = model.execute_architecture_plan(
                    plan,
                    context_points=context_points,
                    residual_scores=residual_scores,
                )
            except Exception as exc:
                model.restore_architecture(architecture_before)
                model.load_state_dict(state_before)
                torch.set_rng_state(cpu_rng_before)
                if cuda_rng_before is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_before)
                failed_audit = deepcopy(dict(compilation_audit or {}))
                failed_audit["executed_operations"] = executed
                failed_audit["skipped_operations"] = [
                    {
                        **item.to_dict(),
                        "reason": (
                            "transaction_failed"
                            if item.operation_index == plan.operation_index
                            else "transaction_aborted_after_prior_failure"
                        ),
                    }
                    for item in plans
                    if item.operation_index >= plan.operation_index
                ]
                failed_audit["transaction_status"] = "rolled_back"
                error = RuntimeError(
                    f"operation[{plan.operation_index}] {plan.action} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                setattr(error, "execution_audit", failed_audit)
                raise error from exc
            results.append(result)
            executed.append(plan.to_dict())
        audit = deepcopy(dict(compilation_audit or {}))
        audit["executed_operations"] = executed
        audit["transaction_status"] = "committed"
        return MorphResult(
            results[-1].module if results else model,
            tuple(parameter for result in results for parameter in result.new_parameters),
            tuple(parameter for result in results for parameter in result.activation_parameters),
            tuple(parameter for result in results for parameter in result.warmup_parameters),
            {
                "adaptation_program": program.to_dict() if program is not None else None,
                "execution_plans": [plan.to_dict() for plan in plans],
                "operation_results": [
                    deepcopy(dict(result.metadata)) for result in results
                ],
                "compilation_audit": audit,
            },
            tuple(
                replacement
                for result in results
                for replacement in result.parameter_replacements
            ),
        )

    def execute(
        self,
        model: OnlineArchitectureModel,
        program: AdaptationProgram,
        diagnostic: OnlineMorphDiagnostics,
        *,
        context_points: torch.Tensor | None = None,
        residual_scores: torch.Tensor | None = None,
    ) -> MorphResult:
        plans, audit = self.compile_with_audit(program, diagnostic)
        if not plans:
            return MorphResult(
                model,
                (),
                (),
                (),
                {"adaptation_program": program.to_dict(), "execution_plans": []},
            )
        return self.execute_plans(
            model,
            plans,
            program=program,
            compilation_audit=audit,
            context_points=context_points,
            residual_scores=residual_scores,
        )


# Compatibility alias for historical checkpoints and third-party imports. New
# construction uses the semantically accurate name above.
FunctionPreservingMorphableNetwork = OnlineArchitectureModel


__all__ = [
    "KEEP",
    "ONLINE_MORPH_ACTIONS",
    "ArchitectureExecutionPlan",
    "DeterministicArchitectureExecutor",
    "FunctionPreservingMorphableNetwork",
    "OnlineArchitectureModel",
    "InterventionClass",
    "MorphResult",
    "OnlineMorphAction",
    "OnlineMorphDecisionProvider",
    "PreservationCheck",
    "PDEResidualPreservationCheck",
    "ParameterReplacement",
    "REGISTERED_MORPH_OPERATORS",
    "SafeExecutionEngine",
    "capture_function_and_derivatives",
    "compare_pde_residual_snapshots",
    "check_function_and_derivative_preservation",
]
