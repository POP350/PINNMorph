"""Formal phase-based trainer for BuiltPINN objects."""

from __future__ import annotations

from .adaptive_runtime import AdaptiveOnlineMixin

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import torch

from morph.pipeline.f_pinn_construction.builders import BuiltPINN
from morph.pipeline.f_pinn_construction.builders.pinn_builder import (
    OptimizationPhase,
)
from .controllers import (
    AdaptiveAction,
    AdvanceCurriculumStateAction,
    ReplaceSamplingSubsetAction,
    TrainerObservableState,
)
from .rollback_whitelist import (
    ROLLBACK_ACTION_WHITELIST,
    is_rollback_whitelisted,
)
from .adaptive_reporting import (
    AdaptiveComputeBudgetLedger,
    build_adaptive_control_summary,
)
from .online_morph import (
    KEEP,
    OnlineMorphAction,
    capture_function_and_derivatives,
    check_function_and_derivative_preservation,
    compare_pde_residual_snapshots,
)
from .adaptation_program import (
    ADD_CORRECTION_BRANCH,
    ADD_FOURIER_BRANCH,
    ADD_GATED_BRANCH,
    ADD_LOCALIZED_EXPERT,
    ADD_RESIDUAL_BRANCH,
    BALANCE_COMPONENT_GRADIENTS,
    REWEIGHT_RESIDUAL_REGIONS,
    PROJECT_CONFLICTING_GRADIENTS,
    ADAPT_SAMPLING_DISTRIBUTION,
    AdaptationProgram,
    OBJECTIVE_ACTIONS,
    OPTIMIZATION_ACTIONS,
    OPTIMIZATION_PHASE_SWITCH,
    SAMPLING_ACTIONS,
    REBALANCE_LOSS_WEIGHTS,
    EMPHASIZE_CONSTRAINTS,
    MITIGATE_GRADIENT_CONFLICT,
    REGROW,
    WIDEN,
)
from .validation_probe import (
    PhysicsValidationMetrics,
    combine_cross_probe_feedback,
    compare_continuation_aware_metrics,
    compare_physics_validation_metrics,
    continuation_forecast_supports_targets,
    missing_continuation_feedback,
)
from .online_diagnostics import build_physics_health_diagnostics
from .intervention_tracker import FEEDBACK_HORIZONS
from .sampling_runtime import (
    SamplingBatch,
    fixed_pde_cloud_counts,
    sampling_policy_fractions,
)


def _bounded_event_log_payload(value: Any) -> Any:
    """Strip recursive policy-history fields before JSONL serialization."""

    if isinstance(value, Mapping):
        return {
            str(key): _bounded_event_log_payload(item)
            for key, item in value.items()
            if str(key) != "previous_morph_events"
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_event_log_payload(item) for item in value]
    return value


class _StopLBFGS(RuntimeError):
    """Internal control flow used to stop a numerically invalid L-BFGS step."""


@dataclass
class TrainingReport:
    success: bool
    actual_iterations: int
    phase_iterations: dict[str, int]
    training_time: float
    final_loss: float | None
    nan_detected: bool
    stopped_early: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    peak_gpu_memory: int = 0
    adaptive_refinement_events: list[dict[str, Any]] = field(default_factory=list)
    actual_peak_sampling_points: int = 0
    final_sampling_snapshot: dict[str, Any] = field(default_factory=dict)
    optimizer_audit: dict[str, Any] = field(default_factory=dict)
    time_strategy_audit: dict[str, Any] = field(default_factory=dict)
    component_gradient_audits: list[dict[str, Any]] = field(default_factory=list)
    adaptive_control_history: list[dict[str, Any]] = field(default_factory=list)
    adaptive_control_summary: dict[str, Any] = field(default_factory=dict)
    physics_validation_history: list[dict[str, Any]] = field(default_factory=list)
    physics_audit_history: list[dict[str, Any]] = field(default_factory=list)
    best_training_loss_checkpoint: dict[str, Any] | None = None
    best_physics_validation_checkpoint: dict[str, Any] | None = None
    trainer_capabilities: dict[str, Any] = field(default_factory=dict)
    registry_schema_version: str | None = None
    final_model_policy: str = "last"
    sampling_control_summary: dict[str, Any] = field(default_factory=dict)
    curriculum_control_summary: dict[str, Any] = field(default_factory=dict)
    adaptive_compute_budget_summary: dict[str, Any] = field(default_factory=dict)
    runtime_scaled_policy_summary: dict[str, Any] = field(default_factory=dict)
    selected_checkpoint_source: str = "last"
    selected_checkpoint_step: int | None = None
    selected_checkpoint_metric: dict[str, Any] | None = None
    pre_lbfgs_physics_checkpoint: dict[str, Any] | None = None
    best_lbfgs_physics_checkpoint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _minimum_physics_probe_history_entry(
    history: list[dict[str, Any]],
    *,
    lbfgs: bool,
) -> dict[str, Any] | None:
    eligible = [
        dict(item)
        for item in history
        if isinstance(item, dict)
        and ("lbfgs" in str(item.get("stage_id") or "").casefold())
        is lbfgs
        and isinstance(item.get("aggregate_score"), (int, float))
        and math.isfinite(float(item["aggregate_score"]))
        and item.get("normalization_calibrated", True)
    ]
    if not eligible:
        return None
    if lbfgs:
        return min(eligible, key=lambda item: float(item["aggregate_score"]))
    pre_second_order = [
        item
        for item in eligible
        if str(item.get("stage_id") or "") == "pre_second_order"
    ]
    return (pre_second_order or eligible)[-1]


def _has_first_order_recovery_phase(phases: list[Any]) -> bool:
    """Return whether a completed schedule contains a usable Adam-family phase.

    Non-finite L-BFGS recovery is a numerical safety boundary, so it must not
    depend on the optional stagnation controller being enabled.
    """

    return any(
        str(getattr(phase, "optimizer_name", "")).casefold()
        in {"adam", "adamw", "radam", "nadam"}
        for phase in phases
    )


def _physics_group_degradations(
    baseline_scores: Mapping[str, float] | None,
    current_scores: Mapping[str, float] | None,
    *,
    maximum_ratio: float,
) -> dict[str, dict[str, float]]:
    """Return physics groups that exceed a relative degradation boundary.

    The aggregate physics-validation score is intentionally not sufficient
    here: a large improvement in one group (for example the initial
    condition) must not hide a material deterioration in another group (for
    example the PDE residual). Scores are already normalized by the fixed
    validation probe, so the same configured ratio is meaningful for PDE,
    boundary, and initial-condition groups.
    """

    baseline = dict(baseline_scores or {})
    current = dict(current_scores or {})
    ratio = max(1.0, float(maximum_ratio))
    degraded: dict[str, dict[str, float]] = {}
    for group in sorted(set(baseline).intersection(current)):
        baseline_value = float(baseline[group])
        current_value = float(current[group])
        if not (
            math.isfinite(baseline_value)
            and math.isfinite(current_value)
        ):
            continue
        limit = baseline_value + (ratio - 1.0) * max(
            abs(baseline_value), 1.0e-12
        )
        if current_value > limit:
            degraded[str(group)] = {
                "baseline": baseline_value,
                "current": current_value,
                "limit": float(limit),
                "ratio": current_value / max(abs(baseline_value), 1.0e-12),
            }
    return degraded


def collect_component_gradient_diagnostics(
    *,
    total_loss: torch.Tensor,
    component_losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    audit_iteration: int,
    gradient_imbalance_threshold: float = 100.0,
    gradient_conflict_threshold: float = -0.1,
    maximum_components: int = 12,
    maximum_conflicting_pairs: int = 5,
) -> dict[str, Any]:
    """Measure registered loss gradients without touching ``parameter.grad``."""

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    base = {
        "available": False,
        "reason": None,
        "audit_iteration": int(audit_iteration),
        "total_grad_norm": None,
        "component_grad_norms": {},
        "component_to_total_ratios": {},
        "unavailable_components": {},
        "gradient_norm_noise_floor": None,
        "negligible_gradient_components": [],
        "max_to_min_nonzero_component_ratio": None,
        "dominant_component": None,
        "weakest_nonzero_component": None,
        "gradient_imbalance_threshold": float(gradient_imbalance_threshold),
        "gradient_imbalance_detected": None,
        "gradient_conflict_threshold": float(gradient_conflict_threshold),
        "gradient_conflict_detected": None,
        "pairwise_cosine_summary": {
            "available": False,
            "reason": "fewer_than_two_valid_nonzero_component_gradients",
            "minimum": None,
            "median": None,
            "mean": None,
            "conflicting_pairs": [],
            "evaluated_pair_count": 0,
        },
    }
    if not trainable:
        base["reason"] = "no_trainable_parameters"
        return base
    selected = sorted(
        component_losses.items(), key=lambda item: str(item[0])
    )[: max(2, int(maximum_components))]
    if not selected:
        base["reason"] = "no_registered_major_loss_components"
        return base
    total_vector, total_reason = _safe_autograd_vector(total_loss, trainable)
    if total_vector is not None:
        base["total_grad_norm"] = float(torch.linalg.vector_norm(total_vector).cpu())
    elif total_reason:
        base["total_gradient_unavailable_reason"] = total_reason
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    for name, component_loss in selected:
        vector, reason = _safe_autograd_vector(component_loss, trainable)
        if vector is None:
            unavailable[str(name)] = reason or "component_gradient_unavailable"
            continue
        norm = float(torch.linalg.vector_norm(vector).cpu())
        vectors[str(name)] = vector
        norms[str(name)] = norm
    base["component_grad_norms"] = norms
    base["unavailable_components"] = unavailable
    total_norm = base.get("total_grad_norm")
    if isinstance(total_norm, float) and total_norm > 0.0:
        base["component_to_total_ratios"] = {
            name: value / total_norm for name, value in norms.items()
        }
    gradient_dtype = (
        total_vector.dtype
        if isinstance(total_vector, torch.Tensor) and total_vector.is_floating_point()
        else next(
            (
                vector.dtype
                for vector in vectors.values()
                if vector.is_floating_point()
            ),
            torch.float32,
        )
    )
    gradient_scale = max(
        1.0,
        float(total_norm) if isinstance(total_norm, float) else 0.0,
        max(norms.values(), default=0.0),
    )
    noise_floor = 64.0 * float(torch.finfo(gradient_dtype).eps) * gradient_scale
    base["gradient_norm_noise_floor"] = noise_floor
    base["negligible_gradient_components"] = sorted(
        name for name, value in norms.items() if value <= noise_floor
    )
    # Numerically satisfied constraints remain visible in component_grad_norms,
    # but do not create enormous ratios or unstable cosine signs by acting as
    # a near-zero denominator.
    nonzero = {
        name: value for name, value in norms.items() if value > noise_floor
    }
    if nonzero:
        base["dominant_component"] = max(nonzero, key=nonzero.get)
        base["weakest_nonzero_component"] = min(nonzero, key=nonzero.get)
    if len(nonzero) >= 2:
        ratio = max(nonzero.values()) / min(nonzero.values())
        base["max_to_min_nonzero_component_ratio"] = ratio
        base["gradient_imbalance_detected"] = bool(
            ratio > float(gradient_imbalance_threshold)
        )
    else:
        base["gradient_imbalance_unavailable_reason"] = (
            "fewer_than_two_valid_nonzero_component_gradients"
        )
    pairwise: list[dict[str, Any]] = []
    names = list(nonzero)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = vectors[left_name]
            right = vectors[right_name]
            denominator = nonzero[left_name] * nonzero[right_name]
            cosine = float(torch.dot(left, right).cpu()) / denominator
            cosine = max(-1.0, min(1.0, cosine))
            pairwise.append(
                {
                    "component_a": left_name,
                    "component_b": right_name,
                    "cosine_similarity": cosine,
                }
            )
    if pairwise:
        cosine_values = sorted(
            float(item["cosine_similarity"]) for item in pairwise
        )
        midpoint = len(cosine_values) // 2
        median = (
            cosine_values[midpoint]
            if len(cosine_values) % 2
            else 0.5
            * (cosine_values[midpoint - 1] + cosine_values[midpoint])
        )
        conflicting = sorted(
            [
                item
                for item in pairwise
                if float(item["cosine_similarity"])
                < float(gradient_conflict_threshold)
            ],
            key=lambda item: float(item["cosine_similarity"]),
        )[: max(1, int(maximum_conflicting_pairs))]
        base["pairwise_cosine_summary"] = {
            "available": True,
            "reason": None,
            "minimum": cosine_values[0],
            "median": median,
            "mean": sum(cosine_values) / len(cosine_values),
            "conflicting_pairs": conflicting,
            "evaluated_pair_count": len(pairwise),
        }
        base["gradient_conflict_detected"] = bool(conflicting)
    base["available"] = bool(norms)
    base["reason"] = None if norms else "no_component_gradients_available"
    return base


def _safe_autograd_vector(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, str | None]:
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return None, "loss_does_not_require_grad"
    try:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except (RuntimeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not any(gradient is not None for gradient in gradients):
        return None, "all_parameter_gradients_unused"
    chunks: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            chunks.append(
                torch.zeros(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device="cpu",
                )
            )
            continue
        detached = gradient.detach().reshape(-1)
        if not torch.isfinite(detached).all():
            return None, "non_finite_component_gradient"
        chunks.append(detached.to(device="cpu"))
    return torch.cat(chunks), None


def online_morph_preservation_tolerances(
    preservation_config: Mapping[str, Any],
    parameter_dtype: torch.dtype,
    *,
    primitive: str | Sequence[str] | None = None,
    hidden_widths: Sequence[int] = (),
) -> tuple[float, float, float, float]:
    """Preservation-check tolerances, each floored by the dtype's noise scale.

    The function and PDE-residual floors absorb 64 eps of float32
    summation-order noise. First and second derivatives of a deep tanh chain
    amplify that noise further. For mathematically strict Net2Net changes and
    zero-output additive branches, both derivative floors therefore scale by
    coarse depth and width bands. Non-strict topology changes and all other
    checks retain fixed floors. Explicit config values can only raise a
    tolerance.
    """

    epsilon = float(torch.finfo(parameter_dtype).eps)
    floor = 64.0 * epsilon
    first_derivative_floor = floor
    second_derivative_floor = 1024.0 * epsilon
    widths = tuple(max(1, int(item)) for item in hidden_widths)
    adaptive_primitives = {
        WIDEN,
        REGROW,
        ADD_RESIDUAL_BRANCH,
        ADD_CORRECTION_BRANCH,
        ADD_FOURIER_BRANCH,
        ADD_GATED_BRANCH,
        ADD_LOCALIZED_EXPERT,
    }
    requested_primitives = (
        tuple(str(item) for item in primitive)
        if isinstance(primitive, Sequence) and not isinstance(primitive, str)
        else (str(primitive or ""),)
    )
    adaptive_strict_sequence = bool(requested_primitives) and all(
        item in adaptive_primitives for item in requested_primitives
    )
    if (
        bool(
            preservation_config.get(
                "adaptive_strict_morphism_tolerance", True
            )
        )
        and adaptive_strict_sequence
        and widths
    ):
        width_reference = max(
            1, int(preservation_config.get("adaptive_width_unit", 64))
        )
        depth_reference = max(
            1, int(preservation_config.get("adaptive_depth_unit", 2))
        )
        width_bands = max(1, math.ceil(max(widths) / width_reference))
        depth_bands = max(1, math.ceil(len(widths) / depth_reference))
        structure_bands = width_bands * depth_bands
        adaptive_first_floor = first_derivative_floor * structure_bands
        adaptive_first_cap = float(
            preservation_config.get(
                "adaptive_first_derivative_tolerance_cap", 1e-3
            )
        )
        first_derivative_floor = min(
            adaptive_first_floor, adaptive_first_cap
        )
        adaptive_second_floor = second_derivative_floor * structure_bands
        adaptive_second_cap = float(
            preservation_config.get(
                "adaptive_second_derivative_tolerance_cap", 1e-2
            )
        )
        second_derivative_floor = min(
            adaptive_second_floor, adaptive_second_cap
        )
    return (
        max(float(preservation_config.get("function_tolerance", 1e-7)), floor),
        max(
            float(preservation_config.get("first_derivative_tolerance", 1e-6)),
            first_derivative_floor,
        ),
        max(
            float(preservation_config.get("second_derivative_tolerance", 1e-6)),
            second_derivative_floor,
        ),
        max(
            float(preservation_config.get("pde_residual_tolerance", 1e-6)),
            floor,
        ),
    )


class OpenSpecTrainer(AdaptiveOnlineMixin):
    def __init__(
        self,
        built: BuiltPINN,
        *,
        checkpoint_dir: str | Path | None = None,
        history_interval: int = 1,
        component_gradient_diagnostics_enabled: bool = True,
        gradient_imbalance_threshold: float = 100.0,
        gradient_conflict_threshold: float = -0.1,
        maximum_gradient_components: int = 12,
        maximum_conflicting_pairs: int = 5,
        runtime_scaled_policy_summary: dict[str, Any] | None = None,
        online_policy_generation_provider: Any | None = None,
        online_candidate_scoring_provider: Any | None = None,
    ) -> None:
        self.built = built
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history_interval = max(1, int(history_interval))
        self.history: list[dict[str, Any]] = []
        self._history_phase_names_recorded: set[str] = set()
        self._history_boundary_iterations: set[int] = set()
        self.completed_phase_iterations: dict[str, int] = {}
        self._train_started = 0.0
        self.runtime_scaled_policy_summary = deepcopy(
            runtime_scaled_policy_summary or {}
        )
        self.online_policy_generation_provider = online_policy_generation_provider
        self.online_candidate_scoring_provider = online_candidate_scoring_provider
        self.adaptive_compute_budget_ledger = AdaptiveComputeBudgetLedger()
        self._actual_iterations = 0
        self.adaptive_refinement_events: list[dict[str, Any]] = []
        self.optimizer_audit: dict[str, Any] = {}
        self._selected_final_model_policy: str | None = None
        self.time_strategy_audit: dict[str, Any] = {}
        self.component_gradient_diagnostics_enabled = bool(
            component_gradient_diagnostics_enabled
        )
        self.gradient_imbalance_threshold = max(
            1.0, float(gradient_imbalance_threshold)
        )
        self.gradient_conflict_threshold = float(gradient_conflict_threshold)
        self.maximum_gradient_components = max(2, int(maximum_gradient_components))
        self.maximum_conflicting_pairs = max(1, int(maximum_conflicting_pairs))
        self.component_gradient_audits: list[dict[str, Any]] = []
        self._gradient_audit_targets: list[int] = []
        self._gradient_audit_targets_completed: set[int] = set()
        self._last_training_batch: Any | None = None
        self._last_training_phase: Any | None = None
        self.gradient_balance_controller = getattr(
            built, "gradient_balance_controller", None
        )
        self.lbfgs_stall_controller = getattr(built, "lbfgs_stall_controller", None)
        self.rollback_manager = getattr(built, "rollback_manager", None)
        self.adaptive_audit_logger = getattr(built, "adaptive_audit_logger", None)
        if self.adaptive_audit_logger is not None:
            audit_path = (
                self.checkpoint_dir / "adaptive_control_history.jsonl"
                if self.checkpoint_dir is not None
                else None
            )
            self.adaptive_audit_logger.bind(audit_path)
        self.adaptive_control_summary: dict[str, Any] = {
            "lbfgs_stall_detected": False,
            "lbfgs_stall_iteration": None,
            "planned_lbfgs_iterations": 0,
            "executed_lbfgs_iterations": 0,
            "recovered_iteration_budget": 0,
            "unused_iteration_budget": 0,
            "recovery_action": None,
            "gradient_balance_actions": 0,
            "rollbacks": 0,
        }
        self._pending_recovery: dict[str, Any] | None = None
        self.registry = getattr(built, "training_component_registry", None)
        self.action_boundary_validator = getattr(
            built, "action_boundary_validator", None
        )
        self.controller_coordinator = getattr(
            built, "controller_coordinator", None
        )
        self.validation_probe_manager = getattr(
            built, "validation_probe_manager", None
        )
        self.best_training_loss_checkpoint_manager = getattr(
            built, "best_training_loss_checkpoint_manager", None
        )
        self.fixed_budget_sampling_controller = getattr(
            built, "fixed_budget_sampling_controller", None
        )
        self.curriculum_runtimes: dict[str, Any] = dict(
            getattr(built, "curriculum_runtimes", {}) or {}
        )
        self._initial_curriculum_runtime_states = {
            curriculum_id: deepcopy(runtime.snapshot_state())
            for curriculum_id, runtime in self.curriculum_runtimes.items()
        }
        self.curriculum_advance_controller = getattr(
            built, "curriculum_advance_controller", None
        )
        self.morphable_model = getattr(built, "morphable_model", None)
        self.online_morph_controller = getattr(
            built, "online_morph_controller", None
        )
        self._pending_online_morph_snapshot: dict[str, Any] | None = None
        self._active_morph_result: Any | None = None
        self._online_morph_phase_wall_started: float | None = None
        self.physics_validation_probe: Any | None = None
        self.physics_validation_probe_metadata: dict[str, Any] = {}
        if self.online_morph_controller is not None:
            if self.validation_probe_manager is None:
                raise ValueError(
                    "Online morphing requires physics_validation.enabled=true"
                )
            self.online_morph_controller.set_registered_interventions(
                loss_intervention=True,
                optimization_intervention=True,
                sampling_intervention=bool(
                    self.fixed_budget_sampling_controller is not None
                    or any(
                        item.replacement_supported
                        and item.pointwise_scoring_supported
                        and item.candidate_generation_supported
                        for item in getattr(
                            self.registry, "sampling_components", ()
                        )
                    )
                ),
                curriculum_intervention=self.curriculum_advance_controller is not None,
            )
            self.online_morph_controller.set_initial_parameter_count(
                sum(parameter.numel() for parameter in self.built.model.parameters())
            )
        self.sampling_control_summary: dict[str, Any] = {
            "post_action_validation": "skipped_sampling_whitelist",
            "rollback_allowed": False,
            "sampling_diagnostic_evaluations": 0,
            "sampling_points_diagnosed": 0,
            "candidate_points_evaluated": 0,
            "sampling_updates_executed": 0,
            "sampling_updates_rejected": 0,
            "sampling_updates_rolled_back": 0,
            "sampling_points_replaced": 0,
            "sampling_compute_time": 0.0,
        }
        self._pending_sampling_observation: dict[str, Any] | None = None
        self._sampling_score_cache: dict[str, Any] = {}
        self.curriculum_control_summary: dict[str, Any] = {
            "curriculum_diagnostic_checks": 0,
            "curriculum_advances_proposed": 0,
            "curriculum_advances_executed": 0,
            "curriculum_advances_forced": 0,
            "curriculum_advances_rejected": 0,
            "curriculum_advances_rolled_back": 0,
            "curriculum_runtime_refresh_cost": 0.0,
            "curriculum_validation_cost": 0.0,
            "iterations_per_level": {},
            "unused_levels": {},
        }
        self._pending_curriculum_observation: dict[str, Any] | None = None
        self._curriculum_resume_disabled = False
        self._curriculum_termination_requested = False
        self._algorithm_spec_id = hashlib.sha256(
            json.dumps(
                getattr(self.built, "normalized_spec", {}),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        if self.online_morph_controller is not None:
            self.adaptive_control_summary.update(
                {
                    "online_morph_diagnostics": 0,
                    "online_morph_triggers": 0,
                    "online_morphs_applied": 0,
                    "online_morphs_retained": 0,
                    "online_morphs_rolled_back": 0,
                    "online_morphs_rejected": 0,
                    "online_heuristic_emergency_recoveries": 0,
                    "online_morph_events": [],
                    "online_morph_diagnostic_events": [],
                    "online_morph_final_architecture": [],
                    "online_morph_max_function_error": 0.0,
                    "online_morph_max_first_derivative_error": 0.0,
                    "online_morph_max_second_derivative_error": 0.0,
                    "online_morph_max_pde_residual_preservation_error": 0.0,
                "online_morph_wall_time": 0.0,
                    "online_morph_timing": {
                        name: 0.0
                        for name in (
                            "base_forward_backward",
                            "gradient_diagnostics",
                            "heldout_probe",
                            "continuation_fit",
                            "preservation_check",
                            "snapshot",
                            "activation",
                            "warmup",
                            "short_utility",
                            "medium_utility",
                            "rollback",
                        )
                    },
                }
            )

    def train(self, maximum_iterations: int | None = None) -> TrainingReport:
        started = time.perf_counter()
        self._train_started = started
        self._actual_iterations = sum(self.completed_phase_iterations.values())
        device = next(self.built.model.parameters()).device
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        remaining = None if maximum_iterations is None else max(0, int(maximum_iterations))
        available_iterations = sum(
            max(
                0,
                int(phase.iterations)
                - int(self.completed_phase_iterations.get(phase.name, 0)),
            )
            for phase in self.built.optimization_phases
        )
        planned_iterations = (
            available_iterations
            if remaining is None
            else min(available_iterations, remaining)
        )
        self._configure_gradient_audit_targets(
            start_iteration=self._actual_iterations,
            planned_iterations=planned_iterations,
        )
        self._configure_history_boundary_iterations(
            start_iteration=self._actual_iterations,
            maximum_iterations=planned_iterations,
        )
        phase_counts: dict[str, int] = {}

        def _merge_lbfgs_audit(
            accumulated: dict[str, Any] | None,
            current: dict[str, Any],
        ) -> dict[str, Any]:
            if accumulated is None:
                return dict(current)
            merged = dict(current)
            for key in (
                "completed_internal_iterations",
                "outer_step_calls",
                "closure_evaluations",
                "accepted_loss_evaluations",
                "recovery_adam_iterations",
                "skipped_stalled_iterations",
            ):
                merged[key] = int(accumulated.get(key, 0) or 0) + int(
                    current.get(key, 0) or 0
                )
            merged["configured_phase_iterations"] = int(
                accumulated.get("configured_phase_iterations", 0) or 0
            ) + int(current.get("configured_phase_iterations", 0) or 0)
            merged["budget_fulfilled"] = bool(current.get("budget_fulfilled"))
            merged["controlled_stall_detected"] = bool(
                accumulated.get("controlled_stall_detected", False)
                or current.get("controlled_stall_detected", False)
            )
            merged["converged_early"] = bool(
                accumulated.get("converged_early", False)
                or current.get("converged_early", False)
            )
            merged["chunk_count"] = int(accumulated.get("chunk_count", 1) or 1) + 1
            merged["chunk_audits"] = [
                *list(accumulated.get("chunk_audits", [accumulated]) or []),
                dict(current),
            ]
            return merged

        nan_detected = False
        stopped_early = False
        failure_reason = None
        final_loss: float | None = None
        training = self.built.normalized_spec["training"]
        clipping = training["gradient_clipping"]
        clipping_parameters = clipping.get("parameters") or {}
        clipping_norm = float(clipping_parameters.get("max_norm", 1.0)) if clipping.get("enabled") else None
        clipping_norm_type = float(clipping_parameters.get("norm_type", 2.0))
        early = training["early_stopping"]
        early_parameters = early.get("parameters") or {}
        threshold = float(early_parameters.get("threshold", -float("inf"))) if early.get("enabled") else -float("inf")
        patience = max(1, int(early_parameters.get("patience", 1))) if early.get("enabled") else 1
        below_threshold = 0
        adaptive_parameters = self.built.normalized_spec["sampling"]["adaptive_refinement"].get("parameters") or {}
        adaptive_interval = max(1, int(adaptive_parameters.get("update_every", 100)))
        optimizer_switch_config = self._optimizer_phase_switch_config()
        self._optimizer_phase_switch_requested = None
        non_lbfgs_completed = sum(
            int(self.completed_phase_iterations.get(phase.name, 0))
            for phase in self.built.optimization_phases
            if self._stage_descriptor(phase).first_order
        )
        self._initialize_physics_validation()
        self._bind_online_morph_physics_probe()

        try:
            for phase in self.built.optimization_phases:
                completed_before = int(self.completed_phase_iterations.get(phase.name, 0))
                available = max(0, int(phase.iterations) - completed_before)
                phase_budget = available if remaining is None else min(available, remaining)
                phase_counts[phase.name] = 0
                if phase_budget <= 0:
                    continue
                # Online morphs may replace Parameter objects while later
                # optimization phases are still dormant. Reconcile every
                # optimizer at the phase boundary and retain PyTorch's lazy
                # state allocation before the optimizer executes its first
                # step (especially L-BFGS).
                self._synchronize_optimizer_parameters_exact()
                stage_descriptor = self._stage_descriptor(phase)
                phase_extra_budget_consumed = 0
                if stage_descriptor.second_order:
                    lbfgs_audit_accumulated: dict[str, Any] | None = None
                    recovered_this_phase = 0
                    first_lbfgs_chunk = True
                    while (
                        phase_counts[phase.name] + recovered_this_phase
                        < int(phase_budget)
                    ):
                        chunk_budget = int(phase_budget) - int(
                            phase_counts[phase.name] + recovered_this_phase
                        )
                        if chunk_budget <= 0:
                            break
                        self._maybe_run_physics_validation(
                            phase=SimpleNamespace(name="pre_second_order"),
                            force=True,
                            training_loss=final_loss,
                        )
                        if first_lbfgs_chunk and (
                            self._pending_recovery is not None
                            or (
                                self.lbfgs_stall_controller is not None
                                and self.lbfgs_stall_controller.triggered
                            )
                        ):
                            pending = dict(self._pending_recovery or {})
                            recovery_budget = min(
                                int(pending.get("iterations") or 0),
                                chunk_budget,
                            )
                            if recovery_budget > 0:
                                (
                                    final_loss,
                                    recovered,
                                    recovery_failure,
                                ) = self._run_recovery_adam(
                                    recovery_budget,
                                    clipping_norm,
                                    clipping_norm_type,
                                    adaptive_interval,
                                )
                                phase_counts["adaptive_recovery_adam"] = int(
                                    phase_counts.get("adaptive_recovery_adam", 0)
                                ) + recovered
                                self.completed_phase_iterations[
                                    "adaptive_recovery_adam"
                                ] = int(
                                    self.completed_phase_iterations.get(
                                        "adaptive_recovery_adam", 0
                                    )
                                ) + recovered
                                non_lbfgs_completed += recovered
                                recovered_this_phase += recovered
                                if recovery_failure is not None:
                                    failure_reason = recovery_failure
                                    nan_detected = recovery_failure.startswith(
                                        "non_finite"
                                    )
                                    break
                            self._pending_recovery = None
                            self._emit_adaptive_audit(
                                iteration=self._actual_iterations,
                                stage="lbfgs",
                                controller="lbfgs_stall",
                                event_type="stalled_phase_resumed_after_recovery",
                                trigger={"already_triggered": True},
                                after={
                                    "remaining_lbfgs_budget": max(
                                        0,
                                        int(phase_budget)
                                        - int(
                                            phase_counts[phase.name]
                                            + recovered_this_phase
                                        ),
                                    )
                                },
                            )
                            first_lbfgs_chunk = False
                            continue
                        first_lbfgs_chunk = False
                        self._configure_training_stage(
                            phase=phase,
                            non_lbfgs_iteration=non_lbfgs_completed,
                        )
                        previous_training_mode = bool(self.built.model.training)
                        self.built.model.eval()
                        try:
                            (
                                final_loss,
                                iterations,
                                lbfgs_failure,
                                lbfgs_audit,
                            ) = self._run_lbfgs(
                                phase,
                                chunk_budget,
                                clipping_norm,
                                clipping_norm_type,
                            )
                        finally:
                            self.built.model.train(previous_training_mode)
                            unfreeze = getattr(
                                self.built.loss_function,
                                "unfreeze_dynamic_weighting",
                                None,
                            )
                            if callable(unfreeze):
                                unfreeze()
                        phase_counts[phase.name] += iterations
                        lbfgs_audit_accumulated = _merge_lbfgs_audit(
                            lbfgs_audit_accumulated, lbfgs_audit
                        )
                        self.optimizer_audit[phase.name] = lbfgs_audit_accumulated
                        self.adaptive_control_summary.update(
                            {
                                "planned_lbfgs_iterations": int(phase_budget),
                                "executed_lbfgs_iterations": int(
                                    phase_counts[phase.name]
                                ),
                                "lbfgs_stall_detected": bool(
                                    lbfgs_audit_accumulated.get(
                                        "controlled_stall_detected", False
                                    )
                                ),
                                "lbfgs_stall_iteration": lbfgs_audit.get(
                                    "stall_iteration"
                                ),
                                "recovery_action": lbfgs_audit.get(
                                    "recovery_action"
                                ),
                            }
                        )
                        if lbfgs_failure is not None:
                            nan_detected = lbfgs_failure.startswith("non_finite")
                            failure_reason = lbfgs_failure
                            break
                        recovery_budget = int(
                            lbfgs_audit.get("recovery_adam_iterations") or 0
                        )
                        self._pending_recovery = (
                            {
                                "phase": phase.name,
                                "iterations": recovery_budget,
                                "action": lbfgs_audit.get("recovery_action"),
                            }
                            if lbfgs_audit.get("controlled_stall_detected")
                            and recovery_budget > 0
                            else None
                        )
                        self.adaptive_control_summary[
                            "unused_iteration_budget"
                        ] = max(
                            0,
                            int(phase_budget)
                            - int(phase_counts[phase.name])
                            - int(recovered_this_phase)
                            - int(recovery_budget),
                        )
                        if (
                            self._pending_recovery is not None
                            and self.checkpoint_dir is not None
                        ):
                            self.completed_phase_iterations[phase.name] = (
                                completed_before + phase_counts[phase.name]
                            )
                            self.save_checkpoint(
                                self.checkpoint_dir
                                / "adaptive_control"
                                / "recovery_resume_checkpoint.pt"
                            )
                        if recovery_budget > 0:
                            (
                                final_loss,
                                recovered,
                                recovery_failure,
                            ) = self._run_recovery_adam(
                                recovery_budget,
                                clipping_norm,
                                clipping_norm_type,
                                adaptive_interval,
                            )
                            phase_counts["adaptive_recovery_adam"] = int(
                                phase_counts.get("adaptive_recovery_adam", 0)
                            ) + recovered
                            self.completed_phase_iterations[
                                "adaptive_recovery_adam"
                            ] = int(
                                self.completed_phase_iterations.get(
                                    "adaptive_recovery_adam", 0
                                )
                            ) + recovered
                            non_lbfgs_completed += recovered
                            recovered_this_phase += recovered
                            self.adaptive_control_summary[
                                "recovered_iteration_budget"
                            ] = int(
                                self.adaptive_control_summary.get(
                                    "recovered_iteration_budget", 0
                                )
                                or 0
                            ) + recovered
                            self.adaptive_control_summary[
                                "unused_iteration_budget"
                            ] = max(
                                0,
                                int(phase_budget)
                                - int(phase_counts[phase.name])
                                - int(recovered_this_phase),
                            )
                            self._pending_recovery = None
                            if recovery_failure is not None:
                                failure_reason = recovery_failure
                                nan_detected = recovery_failure.startswith(
                                    "non_finite"
                                )
                                break
                            continue
                        self._pending_recovery = None
                        if int(iterations) <= 0:
                            break
                    phase_extra_budget_consumed = int(recovered_this_phase)
                    if lbfgs_audit_accumulated is not None:
                        lbfgs_audit_accumulated[
                            "configured_phase_iterations"
                        ] = int(phase_budget)
                        lbfgs_audit_accumulated[
                            "completed_internal_iterations"
                        ] = int(phase_counts[phase.name])
                        lbfgs_audit_accumulated[
                            "recovery_adam_iterations"
                        ] = int(recovered_this_phase)
                        lbfgs_audit_accumulated["budget_fulfilled"] = (
                            int(phase_counts[phase.name])
                            + int(recovered_this_phase)
                        ) >= int(phase_budget)
                        self.optimizer_audit[phase.name] = lbfgs_audit_accumulated
                        self.adaptive_control_summary.update(
                            {
                                "planned_lbfgs_iterations": int(phase_budget),
                                "executed_lbfgs_iterations": int(
                                    phase_counts[phase.name]
                                ),
                                "recovered_iteration_budget": int(
                                    recovered_this_phase
                                ),
                                "unused_iteration_budget": max(
                                    0,
                                    int(phase_budget)
                                    - int(phase_counts[phase.name])
                                    - int(recovered_this_phase),
                                ),
                            }
                        )
                else:
                    for _phase_iteration_offset in range(phase_budget):
                        self._advance_online_morph_stage(phase)
                        stage = self._configure_training_stage(
                            phase=phase,
                            non_lbfgs_iteration=non_lbfgs_completed + 1,
                        )
                        if self.online_morph_controller is not None:
                            stage["online_morph_phase"] = str(
                                self.online_morph_controller.current_morph_phase
                            )
                        batch = self.built.sampler.sample()
                        self._last_training_batch = batch
                        self._last_training_phase = phase
                        phase.optimizer.zero_grad(set_to_none=True)
                        projected_iteration = self._actual_iterations + 1
                        online_diagnostic_checkpoint = bool(
                            self.online_morph_controller is not None
                            and self.online_morph_controller.enabled
                            and (
                                projected_iteration % max(1, int(
                                    self.online_morph_controller.config["diagnostic_interval"]
                                )) == 0
                                or self.online_morph_controller.forced_initial_sampling_due(projected_iteration)
                            )
                        )
                        self.built.loss_function.capture_residual_statistics = bool(
                            online_diagnostic_checkpoint
                            or self._should_record_history(
                                projected_iteration,
                                phase.name,
                            )
                        )
                        loss, components = self.built.loss_function(self.built.model, batch)
                        if not torch.isfinite(loss):
                            nan_detected = True
                            failure_reason = "non_finite_loss"
                            self._release_iteration_graphs()
                            break
                        gradient_statistics = None
                        controller_diagnostic_due = bool(
                            self.gradient_balance_controller is not None
                            and stage_descriptor.first_order
                            and stage_descriptor.dynamic_objective_allowed
                            and projected_iteration
                            % int(
                                self.gradient_balance_controller.config[
                                    "diagnostic_interval"
                                ]
                            )
                            == 0
                        )
                        if (
                            controller_diagnostic_due
                            or online_diagnostic_checkpoint
                            or (
                                self._should_record_history(
                                    projected_iteration,
                                    phase.name,
                                )
                                and self._gradient_diagnostic_due(projected_iteration)
                            )
                        ):
                            gradient_timing_started = (
                                self._online_morph_timing_start()
                                if online_diagnostic_checkpoint
                                else None
                            )
                            gradient_statistics = (
                                self._collect_component_gradient_diagnostic(
                                    total_loss=loss,
                                    iteration=projected_iteration,
                                    phase=phase,
                                    checkpoint_kind=self._gradient_checkpoint_kind(
                                        projected_iteration
                                    ),
                                )
                            )
                            if gradient_timing_started is not None:
                                self._record_online_morph_timing(
                                    "gradient_diagnostics", gradient_timing_started
                                )
                        if phase.optimizer_name == "multiadam":
                            multiadam_losses = (
                                self.built.loss_function.last_term_values
                            )
                            if getattr(phase.optimizer, "grouping", None) is not None:
                                effective_weights = (
                                    self.built.loss_function.last_effective_loss_weights
                                )
                                multiadam_losses = [
                                    value
                                    * float(effective_weights.get(name, 1.0))
                                    for name, value in zip(
                                        self.built.loss_function.last_term_names,
                                        self.built.loss_function.last_term_values,
                                    )
                                ]
                            phase.optimizer.step_losses(
                                multiadam_losses,
                                self.built.loss_function.last_term_names,
                            )
                            self.optimizer_audit[phase.name] = {
                                "optimizer": "multiadam",
                                "grouping": getattr(
                                    phase.optimizer, "grouping", None
                                ),
                                "loss_groups": dict(
                                    getattr(
                                        phase.optimizer,
                                        "last_loss_groups",
                                        {},
                                    )
                                ),
                            }
                            if not self._parameters_finite():
                                nan_detected = True
                                failure_reason = "non_finite_parameter"
                                self._release_iteration_graphs()
                                break
                        else:
                            projection_audit = (
                                self._backward_with_runtime_gradient_projection(loss)
                            )
                            if projection_audit is not None:
                                self.optimizer_audit[phase.name] = {
                                    **dict(self.optimizer_audit.get(phase.name) or {}),
                                    "gradient_projection": projection_audit,
                                }
                            if clipping_norm is not None:
                                torch.nn.utils.clip_grad_norm_(self.built.model.parameters(), clipping_norm, norm_type=clipping_norm_type)
                            if not self._gradients_finite():
                                nan_detected = True
                                failure_reason = "non_finite_gradient"
                                self._release_iteration_graphs()
                                break
                            gradient_norm = self._gradient_norm()
                            phase.optimizer.step()
                        if phase.optimizer_name == "multiadam":
                            gradient_norm = None
                        final_loss = float(loss.detach().cpu())
                        phase_counts[phase.name] += 1
                        self._actual_iterations += 1
                        non_lbfgs_completed += 1
                        self._note_curriculum_training_iteration()
                        if self.gradient_balance_controller is not None:
                            self._observe_gradient_balance(
                                phase=phase,
                                loss=final_loss,
                                components=components,
                                gradient_norm=gradient_norm,
                                diagnostic=gradient_statistics,
                            )
                        if (
                            self.rollback_manager is not None
                            and self._pending_curriculum_observation is None
                            and self.rollback_manager.observe_loss(final_loss)
                        ):
                            self._rollback_active_action(
                                phase=phase,
                                reason="post_action_loss_degradation_or_non_finite",
                            )
                        self._maybe_run_physics_validation(
                            phase=phase,
                            force=False,
                            training_loss=final_loss,
                        )
                        self._observe_sampling_control(phase=phase)
                        self._observe_curriculum_control(phase=phase)
                        online_morph_event = self._observe_online_morph(
                            phase=phase,
                            batch=batch,
                            loss=final_loss,
                            components=components,
                            gradient_statistics=gradient_statistics,
                            remaining_stage_iterations=max(
                                0,
                                int(phase.iterations)
                                - completed_before
                                - int(phase_counts[phase.name]),
                            ),
                        )
                        if self._curriculum_termination_requested:
                            stopped_early = True
                            failure_reason = "curriculum_forced_termination"
                            self._release_iteration_graphs()
                            break
                        adaptive_event = self._actual_iterations % adaptive_interval == 0
                        dynamic_update = bool(
                            (
                                getattr(
                                    self.built.loss_function,
                                    "last_dynamic_weighting",
                                    None,
                                )
                                or {}
                            ).get("updated")
                        )
                        should_record = self._should_record_history(
                            self._actual_iterations,
                            phase.name,
                            event=bool(
                                adaptive_event
                                or dynamic_update
                                or online_morph_event
                            ),
                        )
                        if should_record:
                            self.history.append(
                                self._history_entry(
                                    iteration=self._actual_iterations,
                                    phase=phase,
                                    total_loss=final_loss,
                                    components=components,
                                    batch=batch,
                                    gradient_norm=gradient_norm,
                                    gradient_statistics=gradient_statistics,
                                    adaptive_refinement_event=adaptive_event,
                                    training_stage=stage,
                                )
                            )
                            self._history_phase_names_recorded.add(str(phase.name))
                            if online_morph_event:
                                self.history[-1]["online_morph_event"] = deepcopy(
                                    online_morph_event
                                )
                        self._scheduler_step(phase.scheduler, final_loss)
                        if adaptive_event:
                            self.built.sampler.adaptive_update(self.built.model)
                            sampler_events = getattr(
                                self.built.sampler, "adaptive_refinement_events", []
                            )
                            sampler_event = dict(sampler_events[-1]) if sampler_events else {}
                            self.adaptive_refinement_events.append(
                                {
                                    **sampler_event,
                                    "iteration": self._actual_iterations,
                                    "phase": phase.name,
                                    "elapsed_seconds": time.perf_counter() - self._train_started,
                                }
                            )
                            if (
                                self.history
                                and self.history[-1].get("iteration")
                                == self._actual_iterations
                            ):
                                self.history[-1][
                                    "adaptive_refinement_event"
                                ] = dict(self.adaptive_refinement_events[-1])
                        forced_lbfgs_entry = bool(
                            optimizer_switch_config["enabled"]
                            and self._next_second_order_phase(phase) is not None
                            and int(self._actual_iterations)
                            >= int(
                                optimizer_switch_config[
                                    "latest_lbfgs_entry_step"
                                ]
                            )
                        )
                        llm_lbfgs_entry = bool(
                            isinstance(
                                getattr(
                                    self,
                                    "_optimizer_phase_switch_requested",
                                    None,
                                ),
                                Mapping,
                            )
                            and (
                                self._optimizer_phase_switch_requested or {}
                            ).get("requested")
                        )
                        phase_ready = not optimizer_switch_config.get("require_trial_completion") or not self._adaptive_trial_pending()
                        if (forced_lbfgs_entry or llm_lbfgs_entry) and phase_ready:
                            reason = (
                                "llm_optimizer_phase_switch"
                                if llm_lbfgs_entry
                                else "latest_lbfgs_entry_step_reached"
                            )
                            unused = max(
                                0,
                                int(phase.iterations)
                                - completed_before
                                - int(phase_counts[phase.name]),
                            )
                            transferred = (
                                self._transfer_unused_first_order_budget_to_lbfgs(
                                    current_phase=phase,
                                    unused_iterations=unused,
                                    reason=reason,
                                )
                            )
                            phase.iterations = completed_before + int(
                                phase_counts[phase.name]
                            )
                            event = {
                                "event": "OPTIMIZER_PHASE_SWITCH_EXECUTED",
                                "step": int(self._actual_iterations),
                                "phase": str(phase.name),
                                "reason": reason,
                                "forced": bool(
                                    forced_lbfgs_entry and not llm_lbfgs_entry
                                ),
                                "transferred_iterations": int(transferred),
                                "llm_request": deepcopy(
                                    getattr(
                                        self,
                                        "_optimizer_phase_switch_requested",
                                        None,
                                    )
                                    or {}
                                ),
                            }
                            self._append_morph_event_record(event)
                            self._emit_adaptive_audit(
                                iteration=self._actual_iterations,
                                stage=str(phase.name),
                                controller="optimizer_phase_switch",
                                event_type="optimizer_phase_switch_executed",
                                trigger=event,
                            )
                            self._optimizer_phase_switch_requested = None
                            self._release_iteration_graphs()
                            break
                        below_threshold = below_threshold + 1 if final_loss <= threshold else 0
                        if below_threshold >= patience:
                            stopped_early = True
                            if not should_record:
                                self.history.append(
                                    self._history_entry(
                                        iteration=self._actual_iterations,
                                        phase=phase,
                                        total_loss=final_loss,
                                        components=components,
                                        batch=batch,
                                        gradient_norm=gradient_norm,
                                        gradient_statistics=gradient_statistics,
                                        adaptive_refinement_event=adaptive_event,
                                        training_stage=stage,
                                    )
                                )
                                self._history_phase_names_recorded.add(
                                    str(phase.name)
                                )
                            self.history[-1]["early_stopping"] = True
                            self._release_iteration_graphs()
                            break
                        self._release_iteration_graphs()
                    if remaining is not None:
                        remaining = max(
                            0,
                            remaining
                            - phase_counts[phase.name]
                            - phase_extra_budget_consumed,
                        )
                self.completed_phase_iterations[phase.name] = completed_before + phase_counts[phase.name]
                if nan_detected or stopped_early or remaining == 0:
                    break
            if failure_reason is None and not nan_detected:
                if self._last_training_phase is not None:
                    self._advance_online_morph_stage(self._last_training_phase)
                self._collect_final_component_gradient_diagnostic()
                self._maybe_run_physics_validation(
                    phase=self._last_training_phase,
                    force=True,
                    training_loss=final_loss,
                )
                if self.checkpoint_dir is not None:
                    terminal_path = (
                        self.checkpoint_dir / "last_optimizer_checkpoint.pt"
                    )
                    terminal_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": {
                                name: value.detach().clone().cpu()
                                for name, value in self.built.model.state_dict().items()
                            },
                            "iteration": int(self._actual_iterations),
                            "online_morph_architecture": (
                                self.morphable_model.architecture_state()
                                if self.morphable_model is not None
                                else []
                            ),
                            "physics_validation": (
                                self.validation_probe_manager.last_result.to_dict()
                                if self.validation_probe_manager is not None
                                and self.validation_probe_manager.last_result is not None
                                else None
                            ),
                            "physics_audit_validation": (
                                self.validation_probe_manager.audit_last_result.to_dict()
                                if self.validation_probe_manager is not None
                                and self.validation_probe_manager.audit_last_result is not None
                                else None
                            ),
                            "reference_metrics_used_for_selection": False,
                        },
                        terminal_path,
                    )
                if (
                    self.best_training_loss_checkpoint_manager is not None
                    and self.best_training_loss_checkpoint_manager.final_model_policy
                    in {"best_train_loss", "best_physics_validation"}
                    and self._pending_curriculum_observation is None
                ):
                    restored_training_loss = self._restore_final_best_checkpoint()
                    if restored_training_loss is not None:
                        final_loss = restored_training_loss
                if self.online_morph_controller is not None:
                    self._sync_online_morph_summary()
                    self.adaptive_control_summary[
                        "online_morph_final_architecture"
                    ] = (
                        self.morphable_model.architecture_state()
                        if self.morphable_model is not None
                        else []
                    )
            if self.checkpoint_dir is not None:
                self.save_checkpoint(self.checkpoint_dir / "final_checkpoint.pt")
                if self.online_morph_controller is not None:
                    timing = self.adaptive_control_summary.setdefault(
                        "online_morph_timing", {}
                    )
                    explicit = sum(
                        float(value)
                        for name, value in timing.items()
                        if name != "base_forward_backward"
                    )
                    timing["base_forward_backward"] = max(
                        0.0, time.perf_counter() - started - explicit
                    )
                    (self.checkpoint_dir / "intervention_timing.json").write_text(
                        json.dumps(timing, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                if self.history:
                    self.history[-1]["checkpoint_iteration"] = True
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"

        actual = sum(phase_counts.values())
        training_wall_time = time.perf_counter() - started
        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        sampler = getattr(self.built, "sampler", None)
        adaptive_history = list(
            getattr(self.adaptive_audit_logger, "records", []) or []
        )
        physics_history = list(
            getattr(self.validation_probe_manager, "history", []) or []
        )
        physics_audit_history = list(
            getattr(self.validation_probe_manager, "audit_history", []) or []
        )
        best_training_loss_metadata = deepcopy(
            getattr(
                self.best_training_loss_checkpoint_manager,
                "best_metadata",
                None,
            )
        )
        best_checkpoint_metadata = deepcopy(
            getattr(
                self.best_training_loss_checkpoint_manager,
                "best_metadata",
                None,
            )
        )
        checkpoint_physics_history = physics_audit_history or physics_history
        pre_lbfgs_physics_checkpoint = _minimum_physics_probe_history_entry(
            checkpoint_physics_history, lbfgs=False
        )
        best_lbfgs_physics_checkpoint = _minimum_physics_probe_history_entry(
            checkpoint_physics_history, lbfgs=True
        )
        curriculum_summary = deepcopy(
            self._finalize_curriculum_control_summary()
        )
        probe_budgets = {}
        if (
            self.validation_probe_manager is not None
            and self.validation_probe_manager.probe_set is not None
        ):
            probe_budgets = dict(
                self.validation_probe_manager.probe_set.component_budgets
            )
            if self.validation_probe_manager.audit_probe_set is not None:
                probe_budgets.update(
                    {
                        f"audit::{name}": int(value)
                        for name, value in self.validation_probe_manager.audit_probe_set.component_budgets.items()
                    }
                )
        compute_budget = self.adaptive_compute_budget_ledger.reconcile(
            actual_iterations=self._actual_iterations,
            phase_iterations=phase_counts,
            optimizer_audit=self.optimizer_audit,
            component_gradient_audits=self.component_gradient_audits,
            sampling_summary=self.sampling_control_summary,
            curriculum_summary=curriculum_summary,
            physics_history=physics_history,
            probe_component_budgets=probe_budgets,
            adaptive_summary=self.adaptive_control_summary,
            adaptive_history=adaptive_history,
            training_wall_time=training_wall_time,
        )
        runtime_weights = {}
        get_weights = getattr(
            self.built.loss_function, "get_runtime_loss_weights", None
        )
        if callable(get_weights):
            runtime_weights = dict(get_weights())
        final_model_policy = str(
            self._selected_final_model_policy
            or (
                self.best_training_loss_checkpoint_manager.final_model_policy
                if self.best_training_loss_checkpoint_manager is not None
                else "last"
            )
        )
        compact_control_summary = build_adaptive_control_summary(
            spec=self.built.normalized_spec,
            runtime_loss_weights=runtime_weights,
            component_gradient_audits=self.component_gradient_audits,
            adaptive_history=adaptive_history,
            adaptive_summary=self.adaptive_control_summary,
            sampling_summary=self.sampling_control_summary,
            curriculum_summary=curriculum_summary,
            physics_history=physics_history,
            best_training_loss=deepcopy(
                getattr(
                    self.best_training_loss_checkpoint_manager,
                    "best_metadata",
                    None,
                )
            ),
            final_model_policy=final_model_policy,
            compute_budget=compute_budget,
        )
        report_adaptive_summary = {
            **deepcopy(self.adaptive_control_summary),
            **compact_control_summary,
        }
        return TrainingReport(
            success=failure_reason is None and not nan_detected,
            actual_iterations=actual,
            phase_iterations=phase_counts,
            training_time=training_wall_time,
            final_loss=final_loss,
            nan_detected=nan_detected,
            stopped_early=stopped_early,
            history=self.history,
            failure_reason=failure_reason,
            peak_gpu_memory=peak_gpu_memory,
            adaptive_refinement_events=self.adaptive_refinement_events,
            actual_peak_sampling_points=int(
                getattr(sampler, "peak_total_sampling_points", 0)
            ),
            final_sampling_snapshot=dict(
                getattr(sampler, "last_sampling_snapshot", {}) or {}
            ),
            optimizer_audit=dict(self.optimizer_audit),
            time_strategy_audit=dict(self.time_strategy_audit),
            component_gradient_audits=list(self.component_gradient_audits),
            adaptive_control_history=adaptive_history,
            adaptive_control_summary=report_adaptive_summary,
            physics_validation_history=physics_history,
            physics_audit_history=physics_audit_history,
            best_training_loss_checkpoint=deepcopy(
                best_checkpoint_metadata
                if final_model_policy == "best_train_loss"
                else None
            ),
            best_physics_validation_checkpoint=deepcopy(
                best_checkpoint_metadata
                if final_model_policy == "best_physics_validation"
                else None
            ),
            trainer_capabilities=(
                self.registry.capabilities.to_dict()
                if self.registry is not None
                else {}
            ),
            registry_schema_version=(
                self.registry.schema_version if self.registry is not None else None
            ),
            final_model_policy=final_model_policy,
            sampling_control_summary=deepcopy(self.sampling_control_summary),
            curriculum_control_summary=curriculum_summary,
            adaptive_compute_budget_summary=compute_budget,
            runtime_scaled_policy_summary=deepcopy(
                self.runtime_scaled_policy_summary
            ),
            selected_checkpoint_source=final_model_policy,
            selected_checkpoint_step=(
                int(best_checkpoint_metadata["iteration"])
                if final_model_policy
                in {"best_train_loss", "best_physics_validation"}
                and best_checkpoint_metadata is not None
                else int(self._actual_iterations)
            ),
            selected_checkpoint_metric=(
                {
                    "name": str(
                            best_checkpoint_metadata.get("selection_metric", "training_loss")
                    ),
                    "value": float(
                        best_checkpoint_metadata["selection_score"]
                    ),
                }
                if final_model_policy
                in {"best_train_loss", "best_physics_validation"}
                and best_checkpoint_metadata is not None
                else {
                    "name": "training_loss",
                    "value": final_loss,
                }
            ),
            pre_lbfgs_physics_checkpoint=pre_lbfgs_physics_checkpoint,
            best_lbfgs_physics_checkpoint=best_lbfgs_physics_checkpoint,
        )

    def _emit_adaptive_audit(
        self,
        *,
        iteration: int,
        stage: str,
        controller: str,
        event_type: str,
        trigger: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        rollback_status: str = "not_required",
    ) -> None:
        if self.adaptive_audit_logger is None:
            return
        self.adaptive_audit_logger.emit(
            {
                "iteration": int(iteration),
                "stage": str(stage),
                "controller": str(controller),
                "event_type": str(event_type),
                "trigger": deepcopy(trigger or {}),
                "before": deepcopy(before or {}),
                "after": deepcopy(after or {}),
                "limits": deepcopy(limits or {}),
                "checkpoint_id": checkpoint_id,
                "rollback_status": rollback_status,
            }
        )

    def _attach_missing_morph_parameters(self) -> None:
        """Attach dynamically registered parameters without resetting optimizers."""

        self._synchronize_optimizer_parameters_exact()

    @staticmethod
    def _refresh_optimizer_parameter_cache(optimizer: Any) -> None:
        """Refresh optimizer-specific caches after replacing a param list."""

        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer._params = optimizer.param_groups[0]["params"]
            optimizer._numel_cache = None

    @staticmethod
    def _replace_optimizer_state(
        optimizer: Any,
        entries: Mapping[Any, Any],
    ) -> None:
        """Replace state entries without disabling lazy per-parameter state.

        ``torch.optim.Optimizer.state`` is a ``defaultdict(dict)``. Converting
        it to a plain ``dict`` is usually invisible to Adam after its state has
        already been initialized, but a dormant L-BFGS phase indexes the state
        of its first (possibly morph-replaced) Parameter on phase entry. A
        plain dictionary turns that legitimate lazy initialization into a
        ``KeyError``.
        """

        current = getattr(optimizer, "state", None)
        default_factory = getattr(current, "default_factory", None)
        if default_factory is None and isinstance(optimizer, torch.optim.Optimizer):
            default_factory = dict
        optimizer.state = (
            defaultdict(default_factory, entries)
            if default_factory is not None
            else dict(entries)
        )

    def _migrate_morph_parameter_replacements(self, result: Any) -> None:
        """Replace resized Parameters while retaining compatible optimizer moments."""

        replacements = tuple(getattr(result, "parameter_replacements", ()) or ())
        if not replacements:
            return
        by_old_id = {id(item.old_parameter): item for item in replacements}

        def final_parameter(parameter: torch.nn.Parameter) -> torch.nn.Parameter:
            current = parameter
            visited: set[int] = set()
            while id(current) in by_old_id and id(current) not in visited:
                visited.add(id(current))
                current = by_old_id[id(current)].new_parameter
            return current

        def migrate_state(
            parameter: torch.nn.Parameter,
            state: Mapping[str, Any],
        ) -> tuple[torch.nn.Parameter, dict[str, Any]]:
            current = parameter
            migrated = dict(state)
            visited: set[int] = set()
            while id(current) in by_old_id and id(current) not in visited:
                visited.add(id(current))
                replacement = by_old_id[id(current)]
                migrated = {
                    name: (
                        replacement.migrate_state_tensor(value, str(name))
                        if isinstance(value, torch.Tensor)
                        else deepcopy(value)
                    )
                    for name, value in migrated.items()
                }
                current = replacement.new_parameter
            return current, migrated

        for phase in self.built.optimization_phases:
            optimizer = phase.optimizer
            for group in optimizer.param_groups:
                group["params"] = [
                    final_parameter(parameter)
                    for parameter in group.get("params", ())
                ]
            migrated_state: dict[Any, Any] = {}
            for parameter, state in list(optimizer.state.items()):
                if isinstance(parameter, torch.Tensor):
                    final, migrated = migrate_state(parameter, state)
                    migrated_state[final] = migrated
                else:
                    migrated_state[parameter] = deepcopy(state)
            self._replace_optimizer_state(optimizer, migrated_state)
            self._refresh_optimizer_parameter_cache(optimizer)
        self._synchronize_optimizer_parameters_exact()

    def _synchronize_optimizer_parameters_exact(self) -> None:
        """Remove rolled-back Parameters and attach all live model Parameters."""

        live = tuple(self.built.model.parameters())
        live_ids = {id(parameter) for parameter in live}
        for phase in self.built.optimization_phases:
            groups = list(getattr(phase.optimizer, "param_groups", ()) or ())
            if not groups:
                continue
            retained_ids: set[int] = set()
            for group in groups:
                retained: list[torch.nn.Parameter] = []
                for parameter in group.get("params", ()):
                    parameter_id = id(parameter)
                    if parameter_id not in live_ids or parameter_id in retained_ids:
                        continue
                    retained.append(parameter)
                    retained_ids.add(parameter_id)
                group["params"] = retained
            groups[0]["params"].extend(
                parameter for parameter in live if id(parameter) not in retained_ids
            )
            retained_ids.update(id(parameter) for parameter in groups[0]["params"])
            retained_state = {
                parameter: state
                for parameter, state in phase.optimizer.state.items()
                if not isinstance(parameter, torch.Tensor)
                or id(parameter) in retained_ids
            }
            self._replace_optimizer_state(phase.optimizer, retained_state)
            self._refresh_optimizer_parameter_cache(phase.optimizer)

    def _optimizer_param_names(self, optimizer: Any) -> list[str | None]:
        """Name every optimizer parameter so restores can remap by identity."""

        name_by_id = {
            id(parameter): name
            for name, parameter in self.built.model.named_parameters()
        }
        return [
            name_by_id.get(id(parameter))
            for group in getattr(optimizer, "param_groups", ()) or ()
            for parameter in group.get("params", ())
        ]

    def _reorder_optimizer_params_for_state_load(
        self,
        optimizer: Any,
        state: Mapping[str, Any] | None,
        param_names: Sequence[str | None] | None,
    ) -> None:
        """Align live parameter order with a snapshot before a positional load.

        ``Optimizer.load_state_dict`` attaches saved moments by position, but
        morphs such as ADD_LAYER append new parameters at the end of a group
        while the model orders them mid-network. A restore that rebuilds the
        group in model order would then load another parameter's moments onto
        each tensor (run3 crashed with a [162, 162] Adam moment on the
        [1, 162] output weight this way). The snapshot records every
        parameter's model name; reorder the live groups to that order first.
        Any mismatch falls back to the plain positional load.
        """

        if not param_names:
            return
        groups = list(getattr(optimizer, "param_groups", ()) or ())
        saved_groups = list((state or {}).get("param_groups") or [])
        if not groups or len(groups) != len(saved_groups):
            return
        saved_lengths = [len(group.get("params", ())) for group in saved_groups]
        if sum(saved_lengths) != len(param_names):
            return
        name_to_parameter: dict[str, torch.nn.Parameter] = {}
        for name, parameter in self.built.model.named_parameters():
            name_to_parameter.setdefault(name, parameter)
        if len(name_to_parameter) != len(param_names):
            return
        ordered: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for name in param_names:
            parameter = name_to_parameter.get(name) if name is not None else None
            if parameter is None or id(parameter) in seen:
                return
            seen.add(id(parameter))
            ordered.append(parameter)
        position = 0
        for group, count in zip(groups, saved_lengths):
            group["params"] = ordered[position : position + count]
            position += count
        self._refresh_optimizer_parameter_cache(optimizer)

    def _load_optimizer_state_by_param_names(
        self,
        phases: Sequence[Any],
        states: Sequence[Mapping[str, Any] | None],
        param_name_rows: Sequence[Sequence[str | None] | None] | None,
    ) -> None:
        """Reorder live groups to the snapshot order, then load positionally."""

        rows = list(param_name_rows or [])
        for index, (phase, state) in enumerate(zip(phases, states)):
            if state is None:
                continue
            self._reorder_optimizer_params_for_state_load(
                phase.optimizer,
                state,
                rows[index] if index < len(rows) else None,
            )
            phase.optimizer.load_state_dict(state)

    def _capture_online_morph_snapshot(self) -> dict[str, Any]:
        device = next(self.built.model.parameters()).device
        return {
            "online_morph_architecture": self.morphable_model.architecture_state(),
            "model_state_dict": deepcopy(self.built.model.state_dict()),
            "optimizer_states": [
                deepcopy(item.optimizer.state_dict())
                for item in self.built.optimization_phases
            ],
            "optimizer_param_names": [
                self._optimizer_param_names(item.optimizer)
                for item in self.built.optimization_phases
            ],
            "scheduler_states": [
                deepcopy(item.scheduler.state_dict()) if item.scheduler else None
                for item in self.built.optimization_phases
            ],
            "controller_state": self.online_morph_controller.state_dict(),
            "loss_state": self.built.loss_function.state_dict(),
            "sampler_state": self.built.sampler.state_dict(),
            "physics_validation_probe_metadata": deepcopy(
                self.physics_validation_probe_metadata
            ),
            "best_training_loss_checkpoint_manager_state": (
                self.best_training_loss_checkpoint_manager.state_dict()
                if self.best_training_loss_checkpoint_manager is not None
                else None
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        }

    def _restore_online_morph_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        restore_controller: bool = True,
    ) -> None:
        self.morphable_model.restore_architecture(
            snapshot.get("online_morph_architecture") or []
        )
        self._synchronize_optimizer_parameters_exact()
        self.built.model.load_state_dict(snapshot["model_state_dict"])
        self._load_optimizer_state_by_param_names(
            self.built.optimization_phases,
            snapshot.get("optimizer_states") or [],
            snapshot.get("optimizer_param_names"),
        )
        for phase, state in zip(
            self.built.optimization_phases, snapshot.get("scheduler_states") or []
        ):
            if phase.scheduler is not None and state is not None:
                phase.scheduler.load_state_dict(state)
        self.built.loss_function.load_state_dict(snapshot.get("loss_state"))
        self.built.sampler.load_state_dict(snapshot.get("sampler_state"))
        self.physics_validation_probe_metadata = deepcopy(
            snapshot.get("physics_validation_probe_metadata") or {}
        )
        if self.best_training_loss_checkpoint_manager is not None:
            self.best_training_loss_checkpoint_manager.load_state_dict(
                snapshot.get("best_training_loss_checkpoint_manager_state")
            )
        if restore_controller:
            self.online_morph_controller.load_state_dict(
                snapshot.get("controller_state")
            )
        random.setstate(snapshot["python_random_state"])
        np.random.set_state(snapshot["numpy_random_state"])
        torch.set_rng_state(snapshot["torch_random_state"].cpu())
        cuda_states = snapshot.get("cuda_random_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        self._active_morph_result = self.morphable_model.latest_morph_result()
        self._restore_online_morph_trainability()

    def _append_morph_event_record(self, event: dict[str, Any]) -> None:
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / "morph_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _bounded_event_log_payload(event)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            stream.write("\n")

    def _append_capability_gap_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> None:
        """Persist audit-only LLM requests that are absent from the executor."""

        if self.checkpoint_dir is None or not records:
            return
        path = self.checkpoint_dir / "capability_gaps.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        _bounded_event_log_payload(dict(record)),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                stream.write("\n")

    def _sync_online_morph_summary(self) -> None:
        controller = self.online_morph_controller
        if controller is None:
            return
        applied = int(controller.morph_count)
        rolled_back = int(
            self.adaptive_control_summary.get(
                "online_morph_catastrophic_rollbacks", 0
            )
        ) + int(self.adaptive_control_summary.get("online_morph_trial_rollbacks", 0)) + int(
            self.adaptive_control_summary.get("online_morph_persistent_rollbacks", 0))
        current_parameter_count = int(
            sum(parameter.numel() for parameter in self.built.model.parameters())
        )
        self.adaptive_control_summary.update(
            {
                "online_morphs_applied": applied,
                "online_morphs_rolled_back": rolled_back,
                "online_morphs_retained": max(0, applied - rolled_back),
                "online_morph_current_parameter_count": current_parameter_count,
                "online_morph_current_phase": controller.current_morph_phase,
                "online_morph_phase_remaining_iterations": int(
                    controller.phase_remaining_iterations
                ),
                "online_morph_operator_failure_history": deepcopy(
                    controller.operator_failure_history
                ),
                "online_morph_diagnostic_events": deepcopy(
                    controller.diagnostic_events
                ),
                "online_morph_events": deepcopy(controller.events),
                "online_llm_budget": deepcopy(controller.llm_budget_state),
                "online_policy_population": (
                    controller.policy_population.state_dict()
                    if getattr(controller, "policy_population", None) is not None
                    else None
                ),
                "online_policy_evolution_calls": int(
                    getattr(controller, "policy_evolution_calls", 0)
                ),
                "online_policy_evolution_cycles": int(
                    getattr(controller, "policy_evolution_cycles", 0)
                ),
                "online_policy_evolution_failures": int(
                    getattr(controller, "policy_evolution_failures", 0)
                ),
                "online_policy_evolution_events": deepcopy(
                    getattr(controller, "policy_evolution_events", [])
                ),
                "online_policy_scoring_calls": int(
                    getattr(controller, "policy_scoring_calls", 0)
                ),
                "online_policy_scoring_failures": int(
                    getattr(controller, "policy_scoring_failures", 0)
                ),
                "online_policy_scoring_events": deepcopy(
                    getattr(controller, "policy_scoring_events", [])
                ),
            }
        )

    def _complete_online_joint_transition(self, phase: Any) -> None:
        controller = self.online_morph_controller
        if controller is None or controller.active_trial is None:
            return
        for parameter in self.built.model.parameters():
            parameter.requires_grad_(True)
        morphable = getattr(self, "morphable_model", None)
        if morphable is not None:
            morphable.enforce_disabled_interventions()
        if not bool(controller.active_trial.get("joint_lr_scaled", False)):
            scale = float(controller.config["joint_learning_rate_scale"])
            base_rows = controller.active_trial.get(
                "optimizer_learning_rates_before_morph"
            ) or self._capture_online_morph_learning_rates()
            controller.active_trial[
                "optimizer_learning_rates_before_morph"
            ] = deepcopy(base_rows)
            for item, rows in zip(self.built.optimization_phases, base_rows):
                for group, saved in zip(item.optimizer.param_groups, rows):
                    group["lr"] = float(saved["lr"]) * scale
                    if bool(saved.get("has_initial_lr")):
                        group["initial_lr"] = float(saved["initial_lr"]) * scale
            controller.active_trial["joint_lr_scaled"] = True
        else:
            scale = float(controller.config["joint_learning_rate_scale"])
        controller.joint_transition_complete = True
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(getattr(phase, "name", "unknown")),
            controller=controller.name,
            event_type="online_morph_joint_finetune_started",
            trigger={
                "warmup_until_iteration": controller.warmup_until_iteration,
                "joint_learning_rate_scale": scale,
            },
        )

    def _capture_online_morph_learning_rates(self) -> list[list[dict[str, Any]]]:
        return [
            [
                {
                    "lr": float(group.get("lr", 0.0)),
                    "has_initial_lr": "initial_lr" in group,
                    "initial_lr": float(
                        group.get("initial_lr", group.get("lr", 0.0))
                    ),
                }
                for group in phase.optimizer.param_groups
            ]
            for phase in self.built.optimization_phases
        ]

    def _restore_online_morph_learning_rates(
        self, trial: Mapping[str, Any] | None
    ) -> None:
        rows_by_phase = list(
            dict(trial or {}).get("optimizer_learning_rates_before_morph") or []
        )
        for phase, rows in zip(self.built.optimization_phases, rows_by_phase):
            for group, saved in zip(phase.optimizer.param_groups, rows):
                group["lr"] = float(saved["lr"])
                if bool(saved.get("has_initial_lr")):
                    group["initial_lr"] = float(saved["initial_lr"])

    def _advance_online_morph_stage(self, phase: Any) -> None:
        controller = self.online_morph_controller
        if controller is None:
            return
        for pending in controller.pending_persistent_feedback:
            self._ramp_adaptive_trial(pending)
        if controller.active_trial is None:
            return
        trial = controller.active_trial
        current = int(self._actual_iterations)
        self._ramp_adaptive_trial(trial)
        if controller.current_morph_phase == "activation":
            end = int(trial["activation_end_step"])
            controller.phase_remaining_iterations = max(0, end - current)
            if current >= end:
                if self._online_morph_phase_wall_started is not None:
                    self._record_online_morph_timing(
                        "activation", self._online_morph_phase_wall_started
                    )
                warmup_end = int(trial["warmup_end_step"])
                if current < warmup_end:
                    controller.current_morph_phase = "warmup"
                    controller.phase_remaining_iterations = warmup_end - current
                    self._online_morph_phase_wall_started = self._online_morph_timing_start()
                    self._set_online_morph_trainability("warmup")
                    self._emit_adaptive_audit(
                        iteration=current,
                        stage=str(getattr(phase, "name", "training")),
                        controller=controller.name,
                        event_type="online_morph_module_warmup_started",
                    )
                else:
                    self._enter_online_joint_phase(phase)
        if controller.current_morph_phase == "warmup":
            end = int(trial["warmup_end_step"])
            controller.phase_remaining_iterations = max(0, end - current)
            if current >= end:
                if self._online_morph_phase_wall_started is not None:
                    self._record_online_morph_timing(
                        "warmup", self._online_morph_phase_wall_started
                    )
                self._enter_online_joint_phase(phase)
        if controller.current_morph_phase == "short_utility":
            end = int(trial["short_utility_end_step"])
            controller.phase_remaining_iterations = max(0, end - current)
            if current >= end:
                if self._online_morph_phase_wall_started is not None:
                    self._record_online_morph_timing(
                        "short_utility", self._online_morph_phase_wall_started
                    )
                self._evaluate_online_morph_utility(phase, horizon="short")
        if (
            controller.active_trial is not None
            and controller.current_morph_phase == "medium_utility"
        ):
            end = int(controller.active_trial["medium_utility_end_step"])
            controller.phase_remaining_iterations = max(0, end - current)
            if current >= end:
                if self._online_morph_phase_wall_started is not None:
                    self._record_online_morph_timing(
                        "medium_utility", self._online_morph_phase_wall_started
                    )
                self._evaluate_online_morph_utility(phase, horizon="medium")
        self._sync_online_morph_summary()

    def _enter_online_joint_phase(self, phase: Any) -> None:
        controller = self.online_morph_controller
        if controller is None or controller.active_trial is None:
            return
        self._complete_online_joint_transition(phase)
        if bool(controller.config["utility_evaluation_enabled"]):
            controller.current_morph_phase = "short_utility"
            controller.phase_remaining_iterations = int(
                controller.active_trial["short_utility_end_step"]
                - int(self._actual_iterations)
            )
            self._online_morph_phase_wall_started = self._online_morph_timing_start()
        else:
            wall = max(
                0.0,
                time.perf_counter() - float(controller.active_trial["start_wall_time"]),
            )
            self._restore_online_morph_learning_rates(controller.active_trial)
            controller.complete_medium_feedback(
                utility=0.0, wall_time=wall, iteration=self._actual_iterations
            )
            self._pending_online_morph_snapshot = None
            self._active_morph_result = None

    def _set_online_morph_trainability(self, morph_phase: str) -> None:
        result = self._active_morph_result or (
            self.morphable_model.latest_morph_result()
            if self.morphable_model is not None
            else None
        )
        if result is None:
            return
        allowed = (
            result.activation_parameters
            if morph_phase == "activation"
            else result.warmup_parameters
            if morph_phase == "warmup"
            else tuple(self.built.model.parameters())
        )
        allowed_ids = {id(parameter) for parameter in allowed}
        for parameter in self.built.model.parameters():
            parameter.requires_grad_(id(parameter) in allowed_ids)
        morphable = getattr(self, "morphable_model", None)
        if morphable is not None:
            morphable.enforce_disabled_interventions()

    def _evaluate_online_morph_utility(self, phase: Any, *, horizon: str) -> None:
        controller = self.online_morph_controller
        snapshot = self._pending_online_morph_snapshot
        if controller is None or controller.active_trial is None or snapshot is None:
            return
        trial = deepcopy(controller.active_trial)
        baseline = [
            float(value)
            for value in trial.get("baseline_training_losses") or []
        ]
        target = int(
            (controller.config.get("utility") or {}).get(
                "short_horizon_steps" if horizon == "short" else "medium_horizon_steps",
                100 if horizon == "short" else 500,
            )
        )
        post = [
            float(value)
            for value in (trial.get("utility_training_losses") or [])[-target:]
        ]
        baseline_mean = (
            sum(baseline) / len(baseline) if baseline else None
        )
        post_mean = sum(post) / len(post) if post else None
        before_payload = trial.get("morph_before_physics_metrics")
        if not isinstance(before_payload, dict):
            raise RuntimeError(
                "Online morph trial is missing its held-out physics baseline"
            )
        before_metrics = PhysicsValidationMetrics.from_dict(before_payload)
        before_audit_payload = trial.get("morph_before_audit_physics_metrics")
        if not isinstance(before_audit_payload, dict):
            raise RuntimeError(
                "Online morph trial is missing its independent audit baseline"
            )
        before_audit_metrics = PhysicsValidationMetrics.from_dict(
            before_audit_payload
        )
        after_metrics = self._evaluate_online_morph_physics_metrics()
        after_audit_metrics = self._evaluate_online_morph_physics_metrics(
            channel="audit"
        )
        utility_config = dict(controller.config.get("utility") or {})
        raw_decision = compare_physics_validation_metrics(
            before_metrics,
            after_metrics,
            weights=dict(utility_config.get("weights") or {}),
            minimum_relative_improvement=float(
                utility_config.get("minimum_relative_improvement", 0.01)
            ),
            maximum_component_degradation=float(
                utility_config.get("max_component_degradation", 0.10)
            ),
        )
        raw_audit_decision = compare_physics_validation_metrics(
            before_audit_metrics,
            after_audit_metrics,
            weights=dict(utility_config.get("weights") or {}),
            minimum_relative_improvement=float(
                utility_config.get("minimum_relative_improvement", 0.01)
            ),
            maximum_component_degradation=float(
                utility_config.get("max_component_degradation", 0.10)
            ),
        )
        forecast = dict(trial.get("continuation_forecast") or {})
        predictions = dict(forecast.get("predictions") or {})
        forecast_target_step = int(
            trial[
                "short_utility_end_step"
                if horizon == "short"
                else "medium_utility_end_step"
            ]
        )
        prediction = predictions.get(str(forecast_target_step))
        audit_forecast = dict(trial.get("audit_continuation_forecast") or {})
        audit_prediction = dict(audit_forecast.get("predictions") or {}).get(
            str(forecast_target_step)
        )
        physics_utility = missing_continuation_feedback(
            prediction, audit_prediction, after_metrics, after_audit_metrics,
            target_step=forecast_target_step,
        )
        if physics_utility is not None:
            physics_utility["max_component_degradation"] = max(
                float(raw_decision.max_component_degradation),
                float(raw_audit_decision.max_component_degradation),
            )
        else:
            decision_utility = compare_continuation_aware_metrics(
                prediction,
                after_metrics,
                weights=dict(utility_config.get("weights") or {}),
                minimum_relative_improvement=float(
                    utility_config.get("minimum_relative_improvement", 0.01)
                ),
                maximum_component_degradation=float(
                    utility_config.get("max_component_degradation", 0.10)
                ),
            )
            audit_utility = compare_continuation_aware_metrics(
                audit_prediction,
                after_audit_metrics,
                weights=dict(utility_config.get("weights") or {}),
                minimum_relative_improvement=float(
                    utility_config.get("minimum_relative_improvement", 0.01)
                ),
                maximum_component_degradation=float(
                    utility_config.get("max_component_degradation", 0.10)
                ),
            )
            physics_utility = combine_cross_probe_feedback(
                decision_utility, audit_utility
            )
        physics_utility["horizon"] = horizon
        physics_utility["raw_before_after_telemetry"] = raw_decision.to_dict()
        physics_utility["raw_audit_before_after_telemetry"] = (
            raw_audit_decision.to_dict()
        )
        physics_utility["raw_before_after_improvement"] = float(
            raw_decision.aggregate_relative_improvement
        )
        physics_utility["module_telemetry"] = self._online_morph_module_telemetry()
        utility = float(physics_utility["aggregate_continuation_utility"])
        wall = max(
            0.0,
            time.perf_counter()
            - float(trial.get("start_wall_time") or time.perf_counter()),
        )
        operator = str(trial.get("operator") or "")
        finite_feedback = math.isfinite(float(after_metrics.total_score)) and all(
            math.isfinite(float(value))
            for value in after_metrics.component_metrics.values()
        ) and math.isfinite(float(after_audit_metrics.total_score)) and all(
            math.isfinite(float(value))
            for value in after_audit_metrics.component_metrics.values()
        )
        if not finite_feedback and is_rollback_whitelisted(trial):
            physics_utility.update(
                accepted=False,
                credit_eligible=False,
                reason="non_finite_physics_feedback",
                rollback_allowed=False,
            )
            utility = 0.0
        if not finite_feedback and not is_rollback_whitelisted(trial):
            attempted_step = int(trial.get("insertion_step") or self._actual_iterations)
            self._restore_online_morph_snapshot(snapshot)
            controller = self.online_morph_controller
            controller.record_emergency_recovery(
                operator,
                iteration=self._actual_iterations,
                attempted_step=attempted_step,
                experience=trial,
                utility=min(0.0, utility),
                physics_utility={**physics_utility, "emergency": True},
                wall_time=wall,
            )
            self.adaptive_control_summary[
                "online_heuristic_emergency_recoveries"
            ] = int(
                self.adaptive_control_summary.get(
                    "online_heuristic_emergency_recoveries", 0
                )
            ) + 1
            emergency_event = {
                "event": "HEURISTIC_EMERGENCY_RECOVERY",
                "intervention_id": trial["intervention_id"],
                "operator": operator,
                "reason": "non_finite_physics_feedback",
                "step": int(self._actual_iterations),
                "selected_policy_id": trial.get("selected_policy_id"),
                "state_id": trial.get("state_id"),
            }
            self._append_morph_event_record(emergency_event)
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(getattr(phase, "name", "training")),
                controller=controller.name,
                event_type="online_heuristic_emergency_recovery",
                trigger=emergency_event,
                rollback_status="emergency_only",
            )
        elif horizon == "short":
            self._update_adaptive_trial(controller.active_trial, physics_utility, "short")
            controller.mark_short_utility(
                physics_utility, iteration=self._actual_iterations
            )
            self._online_morph_phase_wall_started = self._online_morph_timing_start()
            pending_event = {
                "event": "HEURISTIC_SHORT_FEEDBACK",
                "rollback_allowed": not is_rollback_whitelisted(trial),
                "intervention_id": trial["intervention_id"],
                "selected_policy_id": trial.get("selected_policy_id"),
                "state_id": trial.get("state_id"),
                "operator": operator,
                "step": int(self._actual_iterations),
                "short_utility": deepcopy(physics_utility),
                "module_telemetry": deepcopy(physics_utility["module_telemetry"]),
                "feedback_status": (
                    "credit_withheld"
                    if not bool(physics_utility.get("credit_eligible", True))
                    else "success"
                    if bool(physics_utility["accepted"])
                    else "failure"
                ),
            }
            self._append_morph_event_record(pending_event)
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(getattr(phase, "name", "training")),
                controller=controller.name,
                event_type="online_heuristic_short_feedback",
                trigger=pending_event,
            )
            self._sync_online_morph_summary()
            return
        else:
            short_utility = dict(trial.get("short_utility") or {})
            physics_utility["short_horizon"] = short_utility
            raw_degradation = max(
                float(raw_decision.max_component_degradation),
                float(raw_audit_decision.max_component_degradation),
            )
            catastrophic_threshold = float(
                utility_config.get(
                    "catastrophic_component_degradation_threshold", 3.0
                )
            )
            catastrophic = bool(
                not is_rollback_whitelisted(trial)
                and utility_config.get("catastrophic_rollback_enabled", True)
                and math.isfinite(raw_degradation)
                and raw_degradation > catastrophic_threshold
            )
            layered_trial = bool(self._adaptive_config().get("trial_control", {}).get("enabled"))
            if layered_trial:
                from .trial_control import physical_observation
                adaptive_rejection = bool(trial.get("adaptive_trial") and not physical_observation(
                    physics_utility, self._adaptive_config()["trial_control"].get("safety_degradation", 0.1)).safe)
            else:
                adaptive_rejection = bool(trial.get("adaptive_trial") and physics_utility.get("credit_eligible") and not physics_utility.get("accepted"))
            if is_rollback_whitelisted(trial):
                adaptive_rejection = False
            physical_catastrophic = catastrophic
            if not catastrophic and not adaptive_rejection:
                self._update_adaptive_trial(controller.active_trial, physics_utility, "medium")
            catastrophic = catastrophic or adaptive_rejection
            if adaptive_rejection:
                physics_utility["adaptive_trial_status"] = "rejected_at_medium"
            self._restore_online_morph_learning_rates(trial)
            if catastrophic:
                # Restore only model/optimizer/runtime state.  The live policy
                # controller and intervention tracker must retain the exact
                # action identity so the failed action receives its own credit.
                self._restore_online_morph_snapshot(
                    snapshot, restore_controller=False
                )
                controller = self.online_morph_controller
            applied_feedback = controller.complete_medium_feedback(
                utility=utility,
                wall_time=wall,
                physics_utility=physics_utility,
                iteration=self._actual_iterations,
                schedule_persistent=not catastrophic,
                rolled_back=catastrophic,
            )
            feedback_event = deepcopy(applied_feedback)
            intervention_start_iteration = feedback_event.get("iteration")
            feedback_event.update(
                {
                    "event": (
                        "HEURISTIC_CATASTROPHIC_ROLLBACK"
                        if physical_catastrophic
                        else "ADAPTIVE_TRIAL_MEDIUM_ROLLBACK"
                        if adaptive_rejection
                        else "HEURISTIC_MEDIUM_FEEDBACK"
                    ),
                    "baseline_loss_mean": baseline_mean,
                    "post_loss_mean": post_mean,
                    "physics_utility": physics_utility,
                    "step": int(self._actual_iterations),
                    "iteration": int(self._actual_iterations),
                    "intervention_start_iteration": intervention_start_iteration,
                    "catastrophic_component_degradation_threshold": (
                        catastrophic_threshold
                    ),
                }
            )
            if catastrophic:
                modified_parameter_count = feedback_event.get(
                    "parameter_count_after"
                )
                restored_parameter_count = int(
                    sum(
                        parameter.numel()
                        for parameter in self.built.model.parameters()
                    )
                )
                feedback_event["rolled_back_parameter_count"] = (
                    modified_parameter_count
                )
                feedback_event["parameter_count_before"] = (
                    modified_parameter_count
                )
                feedback_event["parameter_count_after"] = (
                    restored_parameter_count
                )
                failure = controller.operator_failure_history.setdefault(
                    operator,
                    {
                        "attempts": 0,
                        "emergency_recoveries": 0,
                        "catastrophic_rollbacks": 0,
                        "last_failure_step": -1,
                    },
                )
                failure_key = "catastrophic_rollbacks" if physical_catastrophic else "trial_rollbacks"
                failure[failure_key] = int(failure.get(failure_key, 0)) + 1
                failure["last_failure_step"] = int(self._actual_iterations)
                summary_key = "online_morph_catastrophic_rollbacks" if physical_catastrophic else "online_morph_trial_rollbacks"
                self.adaptive_control_summary[summary_key] = int(
                    self.adaptive_control_summary.get(summary_key, 0)) + 1
                self.adaptive_compute_budget_ledger.rollback_count += 1
                self.adaptive_control_summary["rollbacks"] = int(
                    self.adaptive_control_summary.get("rollbacks", 0)
                ) + 1
                for parameter in self.built.model.parameters():
                    parameter.requires_grad_(True)
                morphable = getattr(self, "morphable_model", None)
                if morphable is not None:
                    morphable.enforce_disabled_interventions()
            self._append_morph_event_record(feedback_event)
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(getattr(phase, "name", "training")),
                controller=controller.name,
                event_type=(
                    "online_heuristic_catastrophic_rollback"
                    if physical_catastrophic
                    else "online_adaptive_trial_medium_rollback"
                    if adaptive_rejection
                    else "online_heuristic_medium_feedback"
                ),
                trigger=feedback_event,
                rollback_status=(
                    "disabled_by_whitelist" if is_rollback_whitelisted(trial)
                    else "completed" if catastrophic else "not_required"
                ),
            )
        if not self._adaptive_trial_pending():
            self._pending_online_morph_snapshot = None
            self._active_morph_result = None
        self.adaptive_control_summary["online_morph_events"] = deepcopy(
            controller.events
        )
        self.adaptive_control_summary["online_morph_final_architecture"] = (
            self.morphable_model.architecture_state()
        )
        self._sync_online_morph_summary()

    def _restore_online_morph_trainability(self) -> None:
        controller = self.online_morph_controller
        if controller is None or self.morphable_model is None:
            return
        for parameter in self.built.model.parameters():
            parameter.requires_grad_(True)
        self.morphable_model.enforce_disabled_interventions()
        if controller.active_trial is None or self.morphable_model.morph_count < 1:
            return
        self._active_morph_result = self.morphable_model.latest_morph_result()
        self._set_online_morph_trainability(controller.current_morph_phase)
        self._online_morph_phase_wall_started = self._online_morph_timing_start()

    def _record_due_persistent_heuristic_feedback(
        self,
        metrics: PhysicsValidationMetrics,
        audit_metrics: PhysicsValidationMetrics | None = None,
    ) -> bool:
        controller = self.online_morph_controller
        if controller is None or not controller.pending_persistent_feedback:
            return False
        state_restored = False
        remaining: list[dict[str, Any]] = []
        seen_intervention_ids: set[str] = set()
        utility_config = dict(controller.config.get("utility") or {})
        for trial in controller.pending_persistent_feedback:
            intervention_id = str(trial.get("intervention_id") or "")
            if intervention_id in seen_intervention_ids:
                continue
            seen_intervention_ids.add(intervention_id)
            extended = dict(trial.get("adaptive_trial") or {})
            target = int(extended.get("observation_end_step") or trial.get("persistent_utility_end_step") or 0)
            if int(self._actual_iterations) < target:
                remaining.append(trial)
                continue
            try:
                prediction = dict(
                    dict(trial.get("continuation_forecast") or {}).get("predictions") or {}
                ).get(str(target))
                audit_prediction = dict(
                    dict(trial.get("audit_continuation_forecast") or {}).get("predictions") or {}
                ).get(str(target))
                feedback = missing_continuation_feedback(
                    prediction, audit_prediction, metrics, audit_metrics,
                    target_step=target,
                )
                if feedback is None:
                    decision_feedback = compare_continuation_aware_metrics(
                        prediction, metrics,
                        weights=dict(utility_config.get("weights") or {}),
                        minimum_relative_improvement=float(utility_config.get("minimum_relative_improvement", 0.01)),
                        maximum_component_degradation=float(utility_config.get("max_component_degradation", 0.10)),
                    )
                    audit_feedback = compare_continuation_aware_metrics(
                        audit_prediction, audit_metrics,
                        weights=dict(utility_config.get("weights") or {}),
                        minimum_relative_improvement=float(utility_config.get("minimum_relative_improvement", 0.01)),
                        maximum_component_degradation=float(utility_config.get("max_component_degradation", 0.10)),
                    )
                    feedback = combine_cross_probe_feedback(decision_feedback, audit_feedback)
                if extended.get("observation_end_step"):
                    feedback = {"credit_eligible": False, "accepted": False, "reason": "bounded_observation_without_counterfactual_credit"}
                if trial.get("adaptive_trial"):
                    for baseline_key, current, telemetry_key in (
                        ("morph_before_physics_metrics", metrics, "raw_before_after_telemetry"),
                        ("morph_before_audit_physics_metrics", audit_metrics, "raw_audit_before_after_telemetry")):
                        baseline = trial.get(baseline_key)
                        if baseline and current is not None:
                            feedback[telemetry_key] = compare_physics_validation_metrics(
                                PhysicsValidationMetrics.from_dict(baseline), current,
                                minimum_relative_improvement=0.01, maximum_component_degradation=0.1).to_dict()
                feedback["horizon"] = "persistent"
                feedback["raw_after_physics"] = metrics.to_dict()
                feedback["raw_after_audit_physics"] = audit_metrics.to_dict() if audit_metrics is not None else None
                if extended.get("observation_end_step"):
                    persistent_event = {"event": "ADAPTIVE_EXTENDED_OBSERVATION", "intervention_id": intervention_id,
                        "step": self._actual_iterations, "feedback": feedback, "policy_credit_assigned": False}
                else:
                    persistent_event = controller.record_persistent_feedback(
                        trial, feedback, iteration=self._actual_iterations
                    )
                restored_persistent = self._rollback_degraded_persistent_trial(
                    trial, feedback, persistent_event, metrics, audit_metrics
                )
                state_restored = state_restored or restored_persistent
                if trial.get("adaptive_trial") and not persistent_event.get("duplicate_ignored") and not persistent_event.get("confounded_by_intervention_ids"):
                    finished = self._finish_adaptive_trial(trial, feedback, persistent_event)
                    if not finished:
                        remaining.append(trial)
                applied_program = dict(trial.get("adaptation_program") or {})
                objective_failed = bool(
                    not restored_persistent
                    and not is_rollback_whitelisted(trial)
                    and
                    not trial.get("adaptive_trial")
                    and applied_program.get("action") in OBJECTIVE_ACTIONS
                    and not bool(feedback.get("accepted", False))
                    and bool(feedback.get("credit_eligible", False))
                    and not bool(persistent_event.get("confounded_by_intervention_ids"))
                    and not bool(persistent_event.get("duplicate_ignored", False))
                )
                if objective_failed:
                    applied_event = controller._applied_event_for_intervention(trial)
                    previous_weights = dict(
                        dict(applied_event.get("architecture_before") or {}).get(
                            "loss_weights"
                        )
                        or {}
                    )
                    if previous_weights:
                        self.built.loss_function.set_runtime_loss_weights(
                            previous_weights
                        )
                        persistent_event["objective_retirement"] = {
                            "status": "restored_previous_loss_weights",
                            "loss_weights": previous_weights,
                        }
                if (
                    not restored_persistent
                    and not is_rollback_whitelisted(trial)
                    and not trial.get("adaptive_trial")
                    and not bool(
                        persistent_event.get("confounded_by_intervention_ids")
                    )
                    and not bool(persistent_event.get("duplicate_ignored", False))
                    and not bool(feedback.get("accepted", False))
                    and bool(feedback.get("credit_eligible", False))
                    and bool(
                        utility_config.get(
                            "retire_failed_additive_on_persistent", True
                        )
                    )
                    and self.morphable_model is not None
                    and applied_program.get("action")
                    not in (
                        *OBJECTIVE_ACTIONS,
                        *SAMPLING_ACTIONS,
                        *OPTIMIZATION_ACTIONS,
                    )
                ):
                    retirement = self.morphable_model.soft_disable_intervention(
                        intervention_id
                    )
                    if retirement.get("status") in {"disabled", "already_disabled"}:
                        persistent_event["architecture_retirement"] = retirement
                        for controller_event in reversed(controller.events):
                            if (
                                controller_event.get("event")
                                == "HEURISTIC_PERSISTENT_FEEDBACK"
                                and controller_event.get("intervention_id")
                                == intervention_id
                            ):
                                controller_event["architecture_retirement"] = deepcopy(
                                    retirement
                                )
                                break
                        self.adaptive_control_summary[
                            "online_morph_final_architecture"
                        ] = self.morphable_model.architecture_state()
                self._append_morph_event_record(persistent_event)
            except Exception as exc:
                if trial.get("adaptive_trial") and self._pending_online_morph_snapshot is not None:
                    # Never release checkpoint eligibility with an unresolved failed trial.
                    failure_event = {
                        "event": ("ADAPTIVE_OBSERVATION_ERROR_RETAINED"
                                  if is_rollback_whitelisted(trial)
                                  else "ADAPTIVE_OBSERVATION_ERROR_ROLLBACK"),
                        "rollback_allowed": not is_rollback_whitelisted(trial),
                        "step": self._actual_iterations,
                        "intervention_id": intervention_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self._finish_adaptive_trial(trial, {}, failure_event)
                    self._append_morph_event_record(failure_event)
                self._append_morph_event_record(
                    {
                        "event": "HEURISTIC_PERSISTENT_FEEDBACK_ERROR",
                        "intervention_id": intervention_id,
                        "step": int(self._actual_iterations),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        controller.pending_persistent_feedback = remaining
        if not getattr(controller, "active_trial", None) and not remaining:
            self._pending_online_morph_snapshot = None
            self._active_morph_result = None
        summary = getattr(self, "adaptive_control_summary", None)
        if isinstance(summary, dict):
            summary["online_morph_events"] = deepcopy(controller.events)
            morphable = getattr(self, "morphable_model", None)
            if morphable is not None:
                summary["online_morph_final_architecture"] = (
                    morphable.architecture_state()
                )
            self._sync_online_morph_summary()
        return state_restored

    def _rollback_degraded_persistent_trial(self, trial, feedback, event, metrics, audit_metrics):
        """Restore an isolated trial only after trusted, measured degradation.

        A weak counterfactual benefit alone is insufficient: both raw probes
        must also have worsened relative to the actual pre-intervention model.
        Optimizer steps already spent remain charged to the trajectory budget.
        """
        controller = self.online_morph_controller
        snapshot = getattr(self, "_pending_online_morph_snapshot", None)
        if (is_rollback_whitelisted(trial) or trial.get("adaptive_trial") or snapshot is None
                or not controller.config.get("block_new_morph_until_persistent_feedback", False)
                or not feedback.get("credit_eligible") or feedback.get("accepted")
                or event.get("duplicate_ignored") or event.get("confounded_by_intervention_ids")
                or audit_metrics is None):
            return False
        threshold = float((controller.config.get("utility") or {}).get(
            "max_component_degradation", 0.10))
        telemetry = []
        for key, current in (("morph_before_physics_metrics", metrics),
                             ("morph_before_audit_physics_metrics", audit_metrics)):
            baseline = trial.get(key)
            if not baseline:
                return False
            raw = compare_physics_validation_metrics(
                PhysicsValidationMetrics.from_dict(baseline), current,
                minimum_relative_improvement=0.01,
                maximum_component_degradation=threshold).to_dict()
            telemetry.append(raw)
            if float(raw["aggregate_relative_improvement"]) >= -threshold:
                return False
        self._restore_online_morph_snapshot(snapshot, restore_controller=False)
        event["persistent_rollback"] = {"status": "restored_pre_intervention_state",
            "optimizer_steps_preserved": int(self._actual_iterations),
            "raw_probe_telemetry": telemetry}
        # Refresh cached measurements after restoring the function and objective.
        restored_decision = self._evaluate_online_morph_physics_metrics()
        restored_audit = self._evaluate_online_morph_physics_metrics(channel="audit")
        manager = getattr(self, "validation_probe_manager", None)
        if manager is not None:
            for channel in ("decision", "audit"):
                manager.evaluate(self.built.model, self.registry,
                                 self.built.loss_function, channel=channel)
        if hasattr(controller, "record_physics_validation"):
            # A restored model is a discontinuity, not a continuation of the
            # rejected trajectory. Do not forecast through that jump.
            controller.physics_validation_history.clear()
            controller.physics_audit_history.clear()
            controller.record_physics_validation(self._actual_iterations, restored_decision.to_dict())
            controller.record_physics_audit(self._actual_iterations, restored_audit.to_dict())
            controller.physics_plateau_window_count = 0
            controller.last_physics_improvement_step = int(self._actual_iterations)
        if self.best_training_loss_checkpoint_manager is not None and self.checkpoint_dir is not None:
            self.best_training_loss_checkpoint_manager.save_checkpoint(self.checkpoint_dir)
        self.adaptive_control_summary["online_morph_persistent_rollbacks"] = int(
            self.adaptive_control_summary.get("online_morph_persistent_rollbacks", 0)) + 1
        self.adaptive_control_summary["rollbacks"] = int(self.adaptive_control_summary.get("rollbacks", 0)) + 1
        self.adaptive_compute_budget_ledger.rollback_count += 1
        for recorded in reversed(controller.events):
            if (recorded.get("event") == "HEURISTIC_PERSISTENT_FEEDBACK"
                    and recorded.get("intervention_id") == event.get("intervention_id")):
                recorded["persistent_rollback"] = deepcopy(event["persistent_rollback"])
                break
        return True

    def _execute_online_optimization_action(
        self,
        *,
        action: OnlineMorphAction,
        diagnostic: Any,
        phase: Any,
        started_at: float,
    ) -> dict[str, Any] | None:
        controller = self.online_morph_controller
        if controller is None:
            return None
        if self._next_second_order_phase(phase) is None:
            controller.reject_policy_candidate(
                action, reason="no_later_lbfgs_phase_available"
            )
            return None
        if action.operator != OPTIMIZATION_PHASE_SWITCH:
            controller.reject_policy_candidate(
                action, reason="unsupported_optimization_action"
            )
            return None
        self._optimizer_phase_switch_requested = {
            "requested": True,
            "operator": action.operator,
            "step": int(self._actual_iterations),
            "phase": str(phase.name),
            "reason": str(action.reason.get("decision_reason") or ""),
            "policy_selection": deepcopy(action.reason.get("policy_selection")),
            "adaptation_program": deepcopy(
                action.reason.get("adaptation_program") or {}
            ),
        }
        event = {
            "event": "OPTIMIZATION_PHASE_SWITCH_REQUESTED",
            "operator": action.operator,
            "intervention_class": action.intervention_class,
            "non_structural": True,
            "step": int(self._actual_iterations),
            "phase": str(phase.name),
            "target_phase": str(self._next_second_order_phase(phase).name),
            "reason": self._optimizer_phase_switch_requested["reason"],
            "policy_selection": deepcopy(action.reason.get("policy_selection")),
            "adaptation_program": deepcopy(
                action.reason.get("adaptation_program") or {}
            ),
        }
        diagnostic_event = controller.mark_applied(self._actual_iterations, event)
        event["intervention_id"] = diagnostic_event["intervention_id"]
        self._append_morph_event_record(event)
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="online_optimizer_phase_switch_requested",
            trigger=dict(action.reason),
            after=deepcopy(event),
        )
        self.adaptive_control_summary["online_morph_wall_time"] = float(
            self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
        ) + (time.perf_counter() - started_at)
        self._sync_online_morph_summary()
        return event

    def _execute_online_objective_action(
        self,
        *,
        action: OnlineMorphAction,
        diagnostic: Any,
        phase: Any,
        before_physics_metrics: PhysicsValidationMetrics | None,
        before_audit_metrics: PhysicsValidationMetrics | None,
        continuation_forecast: Mapping[str, Any],
        audit_continuation_forecast: Mapping[str, Any],
        started_at: float,
    ) -> dict[str, Any] | None:
        """Execute a bounded non-structural objective intervention.

        The LLM selects only a registered high-level response. Exact weights
        are derived from registered component gradients/categories and pass
        through the loss runtime boundary; generated numeric tensors or code
        never reach training.
        """

        controller = self.online_morph_controller
        if controller is None:
            return None
        if before_physics_metrics is None or before_audit_metrics is None:
            controller.reject_policy_candidate(
                action, reason="held_out_physics_baseline_unavailable"
            )
            return None
        program_payload = action.reason.get("adaptation_program")
        if not isinstance(program_payload, Mapping):
            controller.reject_policy_candidate(
                action, reason="objective_program_unavailable"
            )
            return None
        program = AdaptationProgram.from_dict(program_payload)
        plans, compilation_audit = controller.safe_execution_engine.compile_with_audit(
            program, diagnostic
        )
        if not plans:
            controller.reject_policy_candidate(
                action, reason="objective_program_compiled_to_no_operations"
            )
            return None

        old_weights = self.built.loss_function.get_runtime_loss_weights()
        new_weights = dict(old_weights)
        categories = {
            item.component_id: item.category
            for item in getattr(self.registry, "loss_components", ())
        }
        norms = {
            str(name): max(float(value), 1.0e-12)
            for name, value in dict(
                getattr(diagnostic.gradient, "component_gradient_norms", {}) or {}
            ).items()
            if name in old_weights and math.isfinite(float(value))
        }
        strength = max(0.01, min(1.0, float(program.strength)))
        maximum_ratio = min(4.0, 1.0 + 3.0 * strength)
        minimum_ratio = 1.0 / maximum_ratio
        old_projection_strength = float(
            getattr(
                self.built.loss_function,
                "get_gradient_projection_strength",
                lambda: 0.0,
            )()
        )
        new_projection_strength = old_projection_strength
        point_weighting_request: tuple[str, float] | None = None
        gradnorm_request: tuple[float, int, float, float] | None = None

        for plan in plans:
            parameters = dict(plan.parameters or {})
            minimum_weight = max(1.0e-8, float(parameters.get("minimum_weight", 0.01)))
            maximum_weight = max(
                minimum_weight, float(parameters.get("maximum_weight", 100.0))
            )
            if plan.action in {
                REBALANCE_LOSS_WEIGHTS, BALANCE_COMPONENT_GRADIENTS
            } and len(norms) >= 2:
                ordered = sorted(norms.values())
                target = ordered[len(ordered) // 2]
                alpha = float(
                    parameters.get("adaptation_rate", 0.5 * strength + 0.25)
                )
                for name, old in old_weights.items():
                    if name not in norms:
                        continue
                    ratio = (target / norms[name]) ** alpha
                    ratio = min(max(ratio, minimum_ratio), maximum_ratio)
                    new_weights[name] = min(
                        maximum_weight, max(minimum_weight, old * ratio)
                    )
                if plan.action == BALANCE_COMPONENT_GRADIENTS:
                    gradnorm_request = (
                        float(parameters.get("adaptation_rate", strength)),
                        10,
                        minimum_weight,
                        maximum_weight,
                    )
            elif plan.action == REWEIGHT_RESIDUAL_REGIONS:
                # Region probes identify where the failure is concentrated;
                # combine a bounded component-level correction with genuine
                # pointwise weighting inside the differentiable PDE residual.
                target_region = str(plan.target)
                eligible = [
                    name
                    for name in old_weights
                    if (
                        target_region == "boundary_regions"
                        and str(categories.get(name) or "")
                        in {"boundary_constraint", "boundary"}
                    )
                    or (
                        target_region != "boundary_regions"
                        and str(categories.get(name) or "") == "equation_residual"
                    )
                ]
                if eligible:
                    chosen = max(
                        eligible,
                        key=lambda name: float(norms.get(name, 0.0)),
                    )
                    rate = float(parameters.get("adaptation_rate", strength))
                    new_weights[chosen] = min(
                        maximum_weight,
                        max(minimum_weight, old_weights[chosen] * (1.0 + 3.0 * rate)),
                    )
                point_weighting_request = (target_region, strength)
            elif plan.action == PROJECT_CONFLICTING_GRADIENTS:
                new_projection_strength = float(
                    parameters.get("projection_strength", strength)
                )
            else:
                constraint_multiplier = float(
                    parameters.get("constraint_multiplier", maximum_ratio)
                )
                pde_multiplier = float(
                    parameters.get("pde_multiplier", minimum_ratio)
                )
                constraint_multiplier = min(
                    maximum_ratio, max(1.0, constraint_multiplier)
                )
                pde_multiplier = min(1.0, max(minimum_ratio, pde_multiplier))
                for name, old in old_weights.items():
                    category = str(categories.get(name) or "")
                    if category == "equation_residual":
                        value = old * pde_multiplier
                    elif category in {
                        "boundary_constraint",
                        "initial_constraint",
                        "boundary",
                        "initial",
                    }:
                        value = old * constraint_multiplier
                    else:
                        value = old
                    new_weights[name] = min(
                        maximum_weight, max(minimum_weight, value)
                    )

        pre_action_snapshot = self._capture_online_morph_snapshot()
        self.built.loss_function.set_runtime_loss_weights(new_weights)
        set_projection = getattr(
            self.built.loss_function, "set_gradient_projection_strength", None
        )
        if callable(set_projection):
            set_projection(new_projection_strength)
        if point_weighting_request is not None:
            configure = getattr(
                self.built.loss_function, "configure_online_point_weighting", None
            )
            if callable(configure):
                configure(
                    target=point_weighting_request[0],
                    strength=point_weighting_request[1],
                )
        if gradnorm_request is not None:
            configure = getattr(
                self.built.loss_function, "configure_online_gradnorm", None
            )
            if callable(configure):
                configure(
                    strength=gradnorm_request[0],
                    update_every=gradnorm_request[1],
                    minimum_weight=gradnorm_request[2],
                    maximum_weight=gradnorm_request[3],
                )
        self._pending_online_morph_snapshot = pre_action_snapshot
        self._active_morph_result = None
        parameter_count = sum(
            parameter.numel() for parameter in self.built.model.parameters()
        )
        compilation_audit["executed_operations"] = [
            item.to_dict() for item in plans
        ]
        compilation_audit["transaction_status"] = "committed"
        event = {
            "operator": action.operator,
            "intervention_class": action.intervention_class,
            "non_structural": True,
            "adaptation_program": program.to_dict(),
            "policy_selection": deepcopy(action.reason.get("policy_selection") or {}),
            "selected_policy_id": action.reason.get("selected_policy_id"),
            "state_id": action.reason.get("state_id"),
            "state_before": diagnostic.to_dict(),
            "morph_before_physics_metrics": before_physics_metrics.to_dict(),
            "morph_before_audit_physics_metrics": before_audit_metrics.to_dict(),
            "continuation_forecast": deepcopy(dict(continuation_forecast or {})),
            "audit_continuation_forecast": deepcopy(
                dict(audit_continuation_forecast or {})
            ),
            "architecture_before": {
                "loss_weights": dict(old_weights),
                "gradient_projection_strength": old_projection_strength,
            },
            "architecture_after": {
                "loss_weights": dict(new_weights),
                "gradient_projection_strength": new_projection_strength,
            },
            "old_parameter_count": parameter_count,
            "new_parameter_count": parameter_count,
            "parameter_count_before": parameter_count,
            "parameter_count_after": parameter_count,
            "parameter_growth_ratio": 0.0,
            "objective_instruction_audit": compilation_audit,
            "optimizer_state_preserved": True,
            "continued_without_restart": True,
        }
        applied = controller.mark_applied(self._actual_iterations, event)
        self._initialize_adaptive_trial(applied)
        self.adaptive_control_summary["online_morphs_applied"] = int(
            self.adaptive_control_summary.get("online_morphs_applied", 0)
        ) + 1
        self.adaptive_control_summary.setdefault(
            "online_objective_interventions", 0
        )
        self.adaptive_control_summary["online_objective_interventions"] += 1
        self._append_morph_event_record(
            {"event": "OBJECTIVE_ACTION_APPLIED", **applied}
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="online_objective_action_applied",
            before={
                "loss_weights": old_weights,
                "gradient_projection_strength": old_projection_strength,
            },
            after={
                "loss_weights": new_weights,
                "gradient_projection_strength": new_projection_strength,
                "program": program.to_dict(),
            },
        )
        self.adaptive_control_summary["online_morph_wall_time"] = float(
            self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
        ) + (time.perf_counter() - started_at)
        self._sync_online_morph_summary()
        return applied

    def _execute_online_sampling_policy_action(
        self,
        *,
        action: OnlineMorphAction,
        diagnostic: Any,
        phase: Any,
        before_physics_metrics: PhysicsValidationMetrics | None,
        before_audit_metrics: PhysicsValidationMetrics | None,
        continuation_forecast: Mapping[str, Any],
        audit_continuation_forecast: Mapping[str, Any],
        started_at: float,
    ) -> dict[str, Any] | None:
        """Commit the LLM's local/global allocation as a fixed cloud, without a trial."""

        controller = self.online_morph_controller
        forced = bool(action.reason.get("forced_initial_sampling"))
        sampler = getattr(self.built, "sampler", None)
        if controller is None or sampler is None or self.registry is None:
            if forced:
                raise RuntimeError("Mandatory sampling runtime is unavailable")
            if controller is not None:
                controller.reject_policy_candidate(
                    action, reason="sampling_runtime_unavailable"
                )
            return None
        rollback_manager = getattr(self, "rollback_manager", None)
        if (
            getattr(controller, "active_trial", None)
            or getattr(controller, "pending_persistent_feedback", ())
            or (rollback_manager is not None and rollback_manager.active_checkpoint_id is not None)
        ):
            if forced:
                raise RuntimeError("Mandatory sampling cannot run while another execution transaction is pending")
            controller.reject_policy_candidate(
                action, reason="another_intervention_observation_pending"
            )
            return None
        program_payload = action.reason.get("adaptation_program")
        if not isinstance(program_payload, Mapping):
            controller.reject_policy_candidate(
                action, reason="sampling_program_unavailable"
            )
            return None
        try:
            program = AdaptationProgram.from_dict(program_payload)
            is_current = getattr(controller, "sampling_policy_is_current", None)
            if not forced and callable(is_current) and is_current(program):
                self._append_morph_event_record({
                    "event": "SAMPLING_ACTION_SKIPPED_CURRENT_POLICY",
                    "step": self._actual_iterations,
                    "operator": KEEP,
                    "reason": "sampling_policy_already_active",
                })
                return None
            plans, compilation_audit = (
                controller.safe_execution_engine.compile_with_audit(
                    program, diagnostic
                )
            )
            if len(plans) != 1 or plans[0].action != ADAPT_SAMPLING_DISTRIBUTION:
                raise ValueError("sampling policy must compile to one sampling primitive")
            plan = plans[0]
            parameters = dict(plan.parameters or {})
            descriptors = [
                item for item in self.registry.sampling_components
                if item.domain_role == "interior"
                and item.replacement_supported
                and item.pointwise_scoring_supported
                and item.candidate_generation_supported
            ]
            if not descriptors:
                raise RuntimeError("no registered mutable PDE pointwise sampler")
            descriptor = sorted(descriptors, key=lambda item: item.sampler_id)[0]
            current = sampler.get_training_points(descriptor.sampler_id)
            scored = sampler.score_points(
                self.built.model,
                descriptor.associated_loss_component_ids,
                current,
                None,
            )
            self._sampling_score_cache[descriptor.sampler_id] = (current, scored)
            active_policy = dict((getattr(controller, "current_sampling_policy", None) or {}).get("policy") or {})
            local_fraction, global_fraction = sampling_policy_fractions(
                parameters, default_local_fraction=float(active_policy.get("local_fraction", 0.7))
            )
            target_point_count = int(parameters.get("point_count", current.point_count))
            if target_point_count <= 0:
                raise ValueError("point_count must be a positive integer")
            local_count, global_count = fixed_pde_cloud_counts(target_point_count, local_fraction)
            candidate_multiplier = max(1, int(parameters.get("candidate_multiplier", active_policy.get("candidate_multiplier", 4))))
            candidate_count = max(1, local_count * candidate_multiplier)
            sampling_action = ReplaceSamplingSubsetAction(
                action_type="replace_sampling_subset",
                controller="eoh_online_policy",
                sampler_id=descriptor.sampler_id,
                replacement_fraction=local_fraction,
                hard_point_retention_fraction=0.0,
                exploration_fraction=global_fraction,
                scoring_component_ids=tuple(descriptor.associated_loss_component_ids),
                score_statistics=dict(scored.statistics),
                candidate_count=candidate_count,
                requested_compute_budget=candidate_count,
                sampler_state_version=int(current.state_version),
                before={
                    "point_count": current.point_count,
                    "sampler_state_version": current.state_version,
                },
                after={
                    "target": plan.target,
                    "point_count": target_point_count,
                    "local_points": local_count,
                    "global_points": global_count,
                },
                reason={
                    "source": "eoh_online_policy",
                    "region_target": plan.target,
                    "local_fraction": local_fraction,
                    "global_fraction": global_fraction,
                    "point_count": target_point_count,
                },
                limits={"fixed_total_point_budget": False, "llm_point_count_override": True},
            )
            mutation = self._execute_sampling_action(sampling_action)
        except Exception as exc:
            if forced:
                raise RuntimeError(f"Mandatory LLM sampling execution failed: {exc}") from exc
            controller.reject_policy_candidate(
                action,
                reason=f"sampling_execution_failed:{type(exc).__name__}:{exc}"[:500],
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            return None

        self._pending_online_morph_snapshot = None
        self._active_morph_result = None
        parameter_count = sum(
            parameter.numel() for parameter in self.built.model.parameters()
        )
        compilation_audit["executed_operations"] = [item.to_dict() for item in plans]
        compilation_audit["effective_sampling_rule"] = {
            "local_fraction": local_fraction,
            "global_fraction": global_fraction,
            "candidate_multiplier": candidate_multiplier,
            "point_count": target_point_count,
            "local_points": local_count,
            "global_points": global_count,
            "fixed_between_actions": True,
        }
        compilation_audit["transaction_status"] = "committed"
        event = {
            "operator": action.operator,
            "intervention_class": action.intervention_class,
            "non_structural": True,
            "post_action_validation": "skipped_sampling_whitelist",
            "rollback_allowed": False,
            "adaptation_program": program.to_dict(),
            "forced_initial_sampling": forced,
            "policy_selection": deepcopy(action.reason.get("policy_selection") or {}),
            "selected_policy_id": action.reason.get("selected_policy_id"),
            "state_id": action.reason.get("state_id"),
            "state_before": diagnostic.to_dict(),
            "architecture_before": dict(sampling_action.before),
            "architecture_after": {
                "sampler_id": mutation.sampler_id,
                "points_replaced": mutation.points_replaced,
                "point_count": mutation.point_count_after,
                "state_version": mutation.state_version_after,
                "local_points": local_count,
                "global_points": global_count,
                "region_target": plan.target,
            },
            "old_parameter_count": parameter_count,
            "new_parameter_count": parameter_count,
            "parameter_count_before": parameter_count,
            "parameter_count_after": parameter_count,
            "parameter_growth_ratio": 0.0,
            "sampling_instruction_audit": compilation_audit,
            "optimizer_state_preserved": True,
            "continued_without_restart": True,
        }
        applied = controller.mark_applied(self._actual_iterations, event)
        self.adaptive_control_summary["online_morphs_applied"] = int(
            self.adaptive_control_summary.get("online_morphs_applied", 0)
        ) + 1
        self.sampling_control_summary["sampling_updates_executed"] += 1
        self.sampling_control_summary["sampling_points_replaced"] += int(
            mutation.points_replaced
        )
        self._append_morph_event_record({"event": "SAMPLING_ACTION_APPLIED", **applied})
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="online_sampling_action_applied",
            before=dict(sampling_action.before),
            after=deepcopy(event["architecture_after"]),
        )
        self.adaptive_control_summary["online_morph_wall_time"] = float(
            self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
        ) + (time.perf_counter() - started_at)
        self._sync_online_morph_summary()
        return applied

    def _observe_online_morph(
        self,
        *,
        phase: Any,
        batch: Any,
        loss: float,
        components: dict[str, float],
        gradient_statistics: dict[str, Any] | None,
        remaining_stage_iterations: int,
    ) -> dict[str, Any] | None:
        controller = self.online_morph_controller
        morphable = self.morphable_model
        if controller is None or morphable is None:
            return None
        controller.record_loss(loss, iteration=self._actual_iterations)
        descriptor = self._stage_descriptor(phase)
        due, reason = controller.diagnostic_due(
            iteration=self._actual_iterations,
            first_order=bool(getattr(descriptor, "first_order", False)),
            dynamic_objective_allowed=bool(
                getattr(descriptor, "dynamic_objective_allowed", False)
            ),
            remaining_stage_iterations=int(remaining_stage_iterations),
        )
        if not due:
            return None
        forced_initial_sampling = controller.forced_initial_sampling_due(self._actual_iterations)
        morph_control_started = time.perf_counter()
        pde_component_ids = tuple(
            item.component_id
            for item in getattr(self.registry, "loss_components", ())
            if item.category == "equation_residual"
        )
        points = getattr(batch, "domain_samples", None)
        pointwise_residual_scores: torch.Tensor | None = None
        pointwise_error: str | None = None
        if isinstance(points, torch.Tensor) and points.ndim == 2 and pde_component_ids:
            try:
                pointwise = self.built.loss_function.compute_pointwise_component_scores(
                    self.built.model,
                    pde_component_ids,
                    points,
                )
                if pointwise:
                    ordered = [pointwise[name] for name in sorted(pointwise)]
                    pointwise_residual_scores = torch.stack(ordered, dim=0).sum(dim=0).sqrt()
            except Exception as exc:
                pointwise_error = f"{type(exc).__name__}: {exc}"
        problem_spec = self.built.loss_function.problem.get_spec()
        metadata = deepcopy(dict(problem_spec.metadata or {}))
        metadata.setdefault("problem_id", str(problem_spec.problem_id))
        metadata["residual_frequency_axes"] = deepcopy(self.adaptive_control_summary.get("residual_frequency_axes", {}))
        time_state = getattr(self.built.loss_function, "_online_time_progress", {})
        if time_state:
            metadata["online_time_progress"] = {k: deepcopy(time_state.get(k)) for k in ("level", "bins", "bin_scores", "last_advance", "initial_errors", "initial_ready", "junction_score", "coverage_released", "stable_count")}
        categories = {
            item.component_id: item.category
            for item in getattr(self.registry, "loss_components", ())
        }
        architecture = {
            "initial_network_spec": deepcopy(
                self.built.normalized_spec.get("network") or {}
            ),
            "online_architecture_state": morphable.architecture_state(),
        }
        diagnostic = controller.diagnose(
            iteration=self._actual_iterations,
            current_total_loss=float(loss),
            gradient_statistics=gradient_statistics,
            component_categories=categories,
            pointwise_residual_scores=pointwise_residual_scores,
            pde_metadata=metadata,
            current_architecture=architecture,
            remaining_stage_iterations=int(remaining_stage_iterations),
            current_parameter_count=sum(
                parameter.numel() for parameter in self.built.model.parameters()
            ),
        )
        diagnostic_physics_metrics: PhysicsValidationMetrics | None = None
        diagnostic_audit_metrics: PhysicsValidationMetrics | None = None
        diagnostic_physics_error: str | None = None
        try:
            diagnostic_physics_metrics = self._evaluate_online_morph_physics_metrics()
            diagnostic_audit_metrics = self._evaluate_online_morph_physics_metrics(
                channel="audit"
            )
            diagnostic = replace(
                diagnostic,
                physics=build_physics_health_diagnostics(
                    diagnostic_physics_metrics.to_dict()
                ),
            )
            controller.record_physics_validation(
                self._actual_iterations, diagnostic_physics_metrics.to_dict()
            )
            controller.record_physics_audit(
                self._actual_iterations, diagnostic_audit_metrics.to_dict()
            )
            diagnostic = controller.apply_physics_progress(diagnostic)
            state_restored = self._record_due_persistent_heuristic_feedback(
                diagnostic_physics_metrics,
                diagnostic_audit_metrics,
            )
            if state_restored:
                # The diagnostic was computed before rollback; do not select
                # another action from that obsolete model state.
                return None
        except Exception as exc:
            diagnostic_physics_error = f"{type(exc).__name__}: {exc}"
        self.adaptive_control_summary["online_morph_diagnostics"] = int(
            controller.diagnostic_count
        )
        if not forced_initial_sampling and str(controller.config.get("decision_mode")) in {"llm_evolution", "random_selection"}:
            evolution_event = controller.maybe_evolve_policy(
                diagnostic, self.online_policy_generation_provider
            )
            if evolution_event is not None:
                self.adaptive_control_summary.setdefault(
                    "online_policy_evolution_events", []
                ).append(deepcopy(evolution_event))
                self._append_capability_gap_records(
                    list(evolution_event.get("capability_gaps") or ())
                )
        if forced_initial_sampling:
            continuation_forecast = {}
            audit_continuation_forecast = {}
            action = controller.propose_forced_initial_sampling(
                diagnostic, self.online_policy_generation_provider
            )
        else:
            continuation_started = self._online_morph_timing_start()
            continuation_forecast = controller.build_continuation_forecast(self._actual_iterations)
            audit_continuation_forecast = controller.build_audit_continuation_forecast(self._actual_iterations)
            self._record_online_morph_timing("continuation_fit", continuation_started)
            action = controller.propose_action(
                diagnostic, self.online_candidate_scoring_provider,
                continuation_forecast=continuation_forecast,
                audit_continuation_forecast=audit_continuation_forecast,
            )
        required_forecast_steps = (
            self._actual_iterations + FEEDBACK_HORIZONS["short"],
            self._actual_iterations + FEEDBACK_HORIZONS["medium"],
        )

        # Forecast confidence gates later policy credit, not safe execution.
        # Requiring credit eligibility here caused a deadlock in which no
        # experiment could run to produce better causal outcome evidence.
        decision_forecast_ready = continuation_forecast_supports_targets(
            continuation_forecast, required_forecast_steps
        )
        audit_forecast_ready = continuation_forecast_supports_targets(
            audit_continuation_forecast, required_forecast_steps
        )
        forecast_ready = bool(decision_forecast_ready and audit_forecast_ready)
        forecast_failure_reason = str(
            (
                continuation_forecast.get("reason")
                if not decision_forecast_ready
                else audit_continuation_forecast.get("reason")
            )
            or "continuation_fit_unavailable"
        )
        if not forecast_ready:
            action = replace(
                action,
                reason={
                    **dict(action.reason),
                    "policy_credit_eligible": False,
                    "policy_credit_withheld_reason": forecast_failure_reason,
                    "continuation_forecast": continuation_forecast,
                    "audit_continuation_forecast": audit_continuation_forecast,
                },
            )
        diagnostic_event = controller.record_diagnostic_decision(action, diagnostic)
        self._append_morph_event_record(
            {
                "event": (
                    "NO_ACTION_DECISION"
                    if action.operator == KEEP
                    else "INTERVENTION_EVENT"
                ),
                "intervention_class": action.intervention_class,
                "action": action.operator,
                "mechanism": dict(
                    action.reason.get("adaptation_program") or {}
                ).get("mechanism"),
                "reason": str(action.reason.get("decision_reason") or ""),
                "step": int(self._actual_iterations),
            }
        )
        diagnostic_event["heldout_physics_metrics"] = (
            diagnostic_physics_metrics.to_dict()
            if diagnostic_physics_metrics is not None
            else None
        )
        diagnostic_event["audit_physics_metrics"] = (
            diagnostic_audit_metrics.to_dict()
            if diagnostic_audit_metrics is not None
            else None
        )
        if diagnostic_physics_error is not None:
            diagnostic_event["heldout_physics_error"] = diagnostic_physics_error
        if pointwise_error is not None:
            diagnostic_event["spatial_residual_error"] = pointwise_error
        self.adaptive_control_summary["online_morph_diagnostic_events"] = deepcopy(
            controller.diagnostic_events
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="online_morph_diagnostic",
            trigger={**diagnostic_event, "diagnostic_due_reason": reason},
        )
        if action.operator == KEEP:
            self._sync_online_morph_summary()
            self.adaptive_control_summary["online_morph_wall_time"] = float(
                self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
            ) + (time.perf_counter() - morph_control_started)
            return diagnostic_event
        self.adaptive_control_summary["online_morph_triggers"] = int(
            self.adaptive_control_summary.get("online_morph_triggers", 0)
        ) + 1
        if action.operator in OPTIMIZATION_ACTIONS:
            return self._execute_online_optimization_action(
                action=action,
                diagnostic=diagnostic,
                phase=phase,
                started_at=morph_control_started,
            )
        if action.operator in OBJECTIVE_ACTIONS:
            return self._execute_online_objective_action(
                action=action,
                diagnostic=diagnostic,
                phase=phase,
                before_physics_metrics=diagnostic_physics_metrics,
                before_audit_metrics=diagnostic_audit_metrics,
                continuation_forecast=continuation_forecast,
                audit_continuation_forecast=audit_continuation_forecast,
                started_at=morph_control_started,
            )
        if action.operator in SAMPLING_ACTIONS:
            return self._execute_online_sampling_policy_action(
                action=action,
                diagnostic=diagnostic,
                phase=phase,
                before_physics_metrics=diagnostic_physics_metrics,
                before_audit_metrics=diagnostic_audit_metrics,
                continuation_forecast=continuation_forecast,
                audit_continuation_forecast=audit_continuation_forecast,
                started_at=morph_control_started,
            )
        if not isinstance(points, torch.Tensor) or points.ndim != 2:
            controller.reject_policy_candidate(
                action, reason="function_check_inputs_unavailable"
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="online_morph_rejected",
                trigger={"reason": "function_check_inputs_unavailable"},
                after=action.to_dict(),
            )
            self.adaptive_control_summary["online_morph_wall_time"] = float(
                self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
            ) + (time.perf_counter() - morph_control_started)
            return None
        try:
            before_physics_metrics = diagnostic_physics_metrics
            before_audit_metrics = diagnostic_audit_metrics
            if before_physics_metrics is None or before_audit_metrics is None:
                raise RuntimeError(
                    diagnostic_physics_error or "diagnostic held-out physics unavailable"
                )
        except Exception as exc:
            controller.reject_policy_candidate(
                action, reason="held_out_physics_baseline_unavailable"
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="online_morph_rejected",
                trigger={
                    "reason": "held_out_physics_baseline_unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                after=action.to_dict(),
            )
            return None
        preservation_config = dict(
            controller.config.get("preservation") or {}
        )
        check_count = max(1, int(preservation_config.get("probe_size", 16)))
        validation_batch = getattr(
            self.physics_validation_probe, "batch", None
        )
        validation_interior = getattr(
            validation_batch, "domain_samples", None
        )
        if not isinstance(validation_interior, torch.Tensor):
            raise RuntimeError(
                "Online morph preservation requires held-out interior points"
            )
        probe = validation_interior[:check_count].detach().clone()
        residual_evaluator = getattr(
            self.built.loss_function,
            "evaluate_registered_pde_residuals",
            None,
        )
        if not callable(residual_evaluator):
            raise RuntimeError(
                "Registered PDE residual evaluator is unavailable"
            )
        was_training = self.built.model.training
        self.built.model.eval()
        before = capture_function_and_derivatives(
            self.built.model,
            probe,
            maximum_dimensions=int(controller.config["derivative_check_dimensions"]),
        )
        before_pde_residuals = residual_evaluator(
            self.built.model, probe
        )
        snapshot_started = self._online_morph_timing_start()
        pre_morph_snapshot = self._capture_online_morph_snapshot()
        self._record_online_morph_timing("snapshot", snapshot_started)
        preservation_started = self._online_morph_timing_start()
        old_parameters = tuple(self.built.model.parameters())
        program_payload = action.reason.get("adaptation_program")
        program = (
            AdaptationProgram.from_dict(program_payload)
            if isinstance(program_payload, dict)
            else None
        )
        try:
            if program is not None:
                (
                    compiled_plans,
                    compilation_audit,
                ) = controller.safe_execution_engine.compile_with_audit(
                    program, diagnostic
                )
            else:
                compiled_plans = (
                    controller.architecture_executor.plan(action, diagnostic),
                )
                compilation_audit = {
                    "request_source": "legacy_action",
                    "requested_operations": [],
                    "compiled_operations": [
                        item.to_dict() for item in compiled_plans
                    ],
                    "executed_operations": [],
                    "skipped_operations": [],
                    "parameter_adjustments": [],
                    "substitutions": [],
                    "semantic_fidelity": 1.0,
                }
            if not compiled_plans:
                raise RuntimeError(
                    "Structural adaptation program compiled to no operations"
                )
        except Exception as exc:
            requested_operations = (
                [item.to_dict() for item in program.operations]
                if program is not None
                else []
            )
            compile_failure_audit = {
                "request_source": (
                    "llm_operations"
                    if requested_operations
                    else "legacy_action_lowering"
                ),
                "requested_operations": requested_operations,
                "compiled_operations": [],
                "executed_operations": [],
                "skipped_operations": [
                    {**item, "reason": "compilation_failed"}
                    for item in requested_operations
                ],
                "parameter_adjustments": [],
                "substitutions": [],
                "semantic_fidelity": 0.0,
                "transaction_status": "rejected_before_execution",
                "error": f"{type(exc).__name__}: {exc}",
            }
            controller.reject_policy_candidate(
                action,
                reason=(
                    f"architecture_compile_failed:{type(exc).__name__}: {exc}"
                )[:500],
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            self._append_morph_event_record(
                {
                    "event": "ARCHITECTURE_COMPILE_REJECTED",
                    "operator": action.operator,
                    "iteration": int(self._actual_iterations),
                    "error": f"{type(exc).__name__}: {exc}",
                    "requested_program": (
                        program.to_dict() if program is not None else None
                    ),
                    "architecture_instruction_audit": compile_failure_audit,
                }
            )
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="online_architecture_compile_rejected",
                trigger=compile_failure_audit,
                after=action.to_dict(),
            )
            self.built.model.train(was_training)
            return None
        execution_plan = compiled_plans[0]
        sequence_strict_preservation = all(
            item.strict_preservation for item in compiled_plans
        )
        preservation_hidden_widths = morphable.current_hidden_widths()
        soft_prune_validation: dict[str, Any] | None = None
        soft_prune_bad = False
        preview = morphable.apply_soft_prune_preview(execution_plan)
        if preview is not None:
            try:
                soft_after = capture_function_and_derivatives(
                    self.built.model,
                    probe,
                    maximum_dimensions=int(controller.config["derivative_check_dimensions"]),
                )
                soft_residuals = residual_evaluator(self.built.model, probe)
                soft_check = check_function_and_derivative_preservation(
                    before, soft_after
                )
                soft_pde = compare_pde_residual_snapshots(
                    before_pde_residuals, soft_residuals
                )
                soft_prune_bad = (
                    not math.isfinite(soft_check.function_error)
                    or not math.isfinite(soft_check.first_derivative_error)
                    or not math.isfinite(soft_check.second_derivative_error)
                    or not soft_pde.residual_names_match
                    or not math.isfinite(soft_pde.maximum_error)
                )
                soft_prune_validation = {
                    "donor_layer": int(preview["donor_layer"]),
                    "soft_scale": float(preview["soft_scale"]),
                    "function_error": soft_check.function_error,
                    "first_derivative_error": soft_check.first_derivative_error,
                    "second_derivative_error": soft_check.second_derivative_error,
                    "maximum_pde_residual_error": soft_pde.maximum_error,
                    "finite_and_differentiable": not soft_prune_bad,
                }
            finally:
                morphable.restore_soft_prune_preview(preview)
        try:
            morph_result = (
                controller.safe_execution_engine.execute_plans(
                    morphable,
                    compiled_plans,
                    program=program,
                    compilation_audit=compilation_audit,
                    context_points=points,
                    residual_scores=pointwise_residual_scores,
                )
                if program is not None
                else morphable.execute_architecture_plan(execution_plan)
            )
        except Exception as exc:
            self._restore_online_morph_snapshot(pre_morph_snapshot)
            self.built.model.train(was_training)
            execution_audit = deepcopy(
                dict(
                    getattr(exc, "execution_audit", None)
                    or compilation_audit
                )
            )
            execution_audit.setdefault("transaction_status", "rolled_back")
            controller.reject_policy_candidate(
                action,
                reason=(
                    f"architecture_execution_failed:{type(exc).__name__}: {exc}"
                )[:500],
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="online_architecture_action_rejected",
                trigger={
                    "reason": "architecture_executor_unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "architecture_instruction_audit": execution_audit,
                },
                after=action.to_dict(),
            )
            self._append_morph_event_record(
                {
                    "event": "ARCHITECTURE_ACTION_REJECTED",
                    "operator": action.operator,
                    "iteration": int(self._actual_iterations),
                    "error": f"{type(exc).__name__}: {exc}",
                    "architecture_instruction_audit": execution_audit,
                }
            )
            return None
        after = capture_function_and_derivatives(
            self.built.model,
            probe,
            maximum_dimensions=int(controller.config["derivative_check_dimensions"]),
        )
        after_pde_residuals = residual_evaluator(
            self.built.model, probe
        )
        self.built.model.train(was_training)
        preservation = check_function_and_derivative_preservation(before, after)
        pde_preservation = compare_pde_residual_snapshots(
            before_pde_residuals, after_pde_residuals
        )
        self._record_online_morph_timing("preservation_check", preservation_started)
        function_error = preservation.function_error
        parameter_dtype = next(self.built.model.parameters()).dtype
        (
            tolerance,
            first_derivative_tolerance,
            second_derivative_tolerance,
            pde_residual_tolerance,
        ) = online_morph_preservation_tolerances(
            preservation_config,
            parameter_dtype,
            primitive=[item.action for item in compiled_plans],
            hidden_widths=preservation_hidden_widths,
        )
        adaptive_strict_primitives = {
            WIDEN,
            REGROW,
            ADD_RESIDUAL_BRANCH,
            ADD_CORRECTION_BRANCH,
            ADD_FOURIER_BRANCH,
            ADD_GATED_BRANCH,
            ADD_LOCALIZED_EXPERT,
        }
        adaptive_preservation_context = {
            "mode": (
                "structure_adaptive_strict_morphism"
                if sequence_strict_preservation
                and all(
                    item.action in adaptive_strict_primitives
                    for item in compiled_plans
                )
                and bool(
                    preservation_config.get(
                        "adaptive_strict_morphism_tolerance", True
                    )
                )
                else "fixed_dtype_floor"
            ),
            "primitive": execution_plan.action,
            "operation_sequence": [item.action for item in compiled_plans],
            "sequence_strict_preservation": sequence_strict_preservation,
            "hidden_widths": list(preservation_hidden_widths),
            "hidden_depth": len(preservation_hidden_widths),
            "maximum_hidden_width": (
                max(preservation_hidden_widths)
                if preservation_hidden_widths
                else None
            ),
        }
        derivative_bad = (
            not math.isfinite(preservation.first_derivative_error)
            or not math.isfinite(preservation.second_derivative_error)
            or (
                sequence_strict_preservation
                and (
                    preservation.first_derivative_error > first_derivative_tolerance
                    or preservation.second_derivative_error > second_derivative_tolerance
                )
            )
        )
        pde_bad = (
            soft_prune_bad
            or
            not pde_preservation.residual_names_match
            or not math.isfinite(pde_preservation.maximum_error)
            or (
                sequence_strict_preservation
                and pde_preservation.maximum_error > pde_residual_tolerance
            )
        )
        if (
            not math.isfinite(function_error)
            or (sequence_strict_preservation and function_error > tolerance)
            or derivative_bad
            or pde_bad
        ):
            self._restore_online_morph_snapshot(pre_morph_snapshot)
            controller.reject_policy_candidate(
                action,
                reason=(
                    "pde_residual_preservation_failed"
                    if pde_bad
                    else "function_or_derivative_preservation_failed"
                ),
            )
            self.adaptive_control_summary["online_morphs_rejected"] = int(
                self.adaptive_control_summary.get("online_morphs_rejected", 0)
            ) + 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="online_morph_rejected_pretrain",
                trigger={
                    **dict(action.reason),
                    "reason": (
                        "pde_residual_preservation_failed"
                        if pde_bad
                        else "function_or_derivative_preservation_failed"
                    ),
                    "function_error": function_error,
                    "first_derivative_error": preservation.first_derivative_error,
                    "second_derivative_error": preservation.second_derivative_error,
                    "per_residual_preservation_error": dict(
                        pde_preservation.per_residual_errors
                    ),
                    "max_residual_preservation_error": (
                        pde_preservation.maximum_error
                    ),
                    "preservation_tolerance_context": (
                        adaptive_preservation_context
                    ),
                    "architecture_instruction_audit": deepcopy(
                        dict(
                            morph_result.metadata.get("compilation_audit")
                            or compilation_audit
                        )
                    ),
                },
                after=action.to_dict(),
                limits={
                    "function_preservation_tolerance": tolerance,
                    "first_derivative_preservation_tolerance": (
                        first_derivative_tolerance
                    ),
                    "second_derivative_preservation_tolerance": (
                        second_derivative_tolerance
                    ),
                    "pde_residual_preservation_tolerance": (
                        pde_residual_tolerance
                    ),
                    "preservation_tolerance_context": (
                        adaptive_preservation_context
                    ),
                    "architecture_instruction_audit": deepcopy(
                        dict(
                            morph_result.metadata.get("compilation_audit")
                            or compilation_audit
                        )
                    ),
                },
            )
            self._append_morph_event_record(
                {
                    "event": "MORPH_REJECTED_PRETRAIN",
                    "reason": (
                        "pde_residual_preservation_failed"
                        if pde_bad
                        else "function_or_derivative_preservation_failed"
                    ),
                    "operator": action.operator,
                    "iteration": int(self._actual_iterations),
                    "function_error": function_error,
                    "first_derivative_preservation_error": (
                        preservation.first_derivative_error
                    ),
                    "second_derivative_preservation_error": (
                        preservation.second_derivative_error
                    ),
                    "per_residual_preservation_error": dict(
                        pde_preservation.per_residual_errors
                    ),
                    "max_residual_preservation_error": (
                        pde_preservation.maximum_error
                    ),
                    "function_preservation_tolerance": tolerance,
                    "first_derivative_preservation_tolerance": (
                        first_derivative_tolerance
                    ),
                    "second_derivative_preservation_tolerance": (
                        second_derivative_tolerance
                    ),
                    "pde_residual_preservation_tolerance": (
                        pde_residual_tolerance
                    ),
                    "preservation_tolerance_context": (
                        adaptive_preservation_context
                    ),
                    "architecture_instruction_audit": deepcopy(
                        dict(
                            morph_result.metadata.get("compilation_audit")
                            or compilation_audit
                        )
                    ),
                }
            )
            self.adaptive_control_summary["online_morph_wall_time"] = float(
                self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
            ) + (time.perf_counter() - morph_control_started)
            return None
        self._migrate_morph_parameter_replacements(morph_result)
        self._attach_missing_morph_parameters()
        optimizer_mode = str(
            (controller.config.get("optimizer_state_after_morph") or {}).get(
                "mode", "preserve"
            )
        )
        if optimizer_mode == "reset":
            for item in self.built.optimization_phases:
                item.optimizer.state.clear()
        physics_weight_update: dict[str, dict[str, float]] = {}
        physics_growth = float(
            controller.config.get("physics_weight_growth_factor", 1.0)
        )
        if physics_growth > 1.0:
            get_weights = getattr(
                self.built.loss_function, "get_runtime_loss_weights", None
            )
            set_weights = getattr(
                self.built.loss_function, "set_runtime_loss_weights", None
            )
            if callable(get_weights) and callable(set_weights):
                current_weights = dict(get_weights())
                targets = [
                    name for name in pde_component_ids if name in current_weights
                ]
                if not targets:
                    targets = [
                        name
                        for name in current_weights
                        if "pde" in name.casefold()
                        or "residual" in name.casefold()
                    ]
                updated_weights = dict(current_weights)
                for name in targets:
                    before_weight = float(current_weights[name])
                    after_weight = before_weight * physics_growth
                    updated_weights[name] = after_weight
                    physics_weight_update[name] = {
                        "before": before_weight,
                        "after": after_weight,
                    }
                if physics_weight_update:
                    set_weights(updated_weights)
        new_parameters = morph_result.new_parameters
        new_ids = {id(parameter) for parameter in new_parameters}
        parameter_count_before = int(sum(parameter.numel() for parameter in old_parameters))
        parameter_count_after = int(
            sum(parameter.numel() for parameter in self.built.model.parameters())
        )
        state_before = deepcopy(action.reason.get("diagnostics") or {})
        state_before.pop("previous_morph_events", None)
        event = {
            "iteration": int(self._actual_iterations),
            "stage": str(phase.name),
            "operator": action.operator,
            "intervention_class": action.intervention_class,
            "target": execution_plan.target,
            "strength": float(execution_plan.strength_value),
            "strength_label": execution_plan.strength,
            "action_parameters": deepcopy(dict(action.action_parameters)),
            "decision_source": action.decision_source,
            "function_error": function_error,
            "first_derivative_preservation_error": preservation.first_derivative_error,
            "second_derivative_preservation_error": preservation.second_derivative_error,
            "per_residual_preservation_error": dict(
                pde_preservation.per_residual_errors
            ),
            "max_residual_preservation_error": pde_preservation.maximum_error,
            "function_tolerance": tolerance,
            "first_derivative_tolerance": first_derivative_tolerance,
            "second_derivative_tolerance": second_derivative_tolerance,
            "pde_residual_tolerance": pde_residual_tolerance,
            "preservation_tolerance_context": adaptive_preservation_context,
            "morph_before_physics_metrics": before_physics_metrics.to_dict(),
            "morph_before_audit_physics_metrics": before_audit_metrics.to_dict(),
            "continuation_forecast": deepcopy(continuation_forecast),
            "audit_continuation_forecast": deepcopy(
                audit_continuation_forecast
            ),
            "new_parameter_count": int(
                sum(parameter.numel() for parameter in new_parameters)
            ),
            "old_parameter_count": parameter_count_before,
            "parameter_count_before": parameter_count_before,
            "parameter_count_after": parameter_count_after,
            "parameter_growth_ratio": (
                (parameter_count_after - parameter_count_before)
                / max(1, parameter_count_before)
            ),
            "activation_phase_steps": int(controller.config["activation_only_iterations"]),
            "warmup_steps": int(controller.config["new_parameters_only_iterations"]),
            "utility_window_iterations": int(controller.config["utility_window_iterations"]),
            "continued_without_restart": True,
            "optimizer_state_preserved": optimizer_mode == "preserve",
            "optimizer_state_mode": optimizer_mode,
            "physics_weight_continuation": physics_weight_update,
            "operator_metadata": deepcopy(dict(morph_result.metadata)),
            "adaptation_program": (
                program.to_dict() if program is not None else None
            ),
            "state_before": state_before,
            "execution_plans": [item.to_dict() for item in compiled_plans],
            "architecture_instruction_audit": deepcopy(
                dict(morph_result.metadata.get("compilation_audit") or compilation_audit)
            ),
            "recovery_window_steps": max(
                int(item.recovery_steps) for item in compiled_plans
            ),
            "soft_prune_validation": deepcopy(soft_prune_validation),
            "architecture_before": deepcopy(diagnostic.current_architecture),
            "architecture_after": {
                "initial_network_spec": deepcopy(
                    self.built.normalized_spec.get("network") or {}
                ),
                "online_architecture_state": morphable.architecture_state(),
            },
            "wall_time": time.perf_counter() - morph_control_started,
        }
        policy_selection = action.reason.get("policy_selection")
        if isinstance(policy_selection, dict):
            event["policy_selection"] = deepcopy(policy_selection)
            event["selected_policy_id"] = policy_selection.get("policy_id")
            event["state_id"] = policy_selection.get("state_id")
        if isinstance(action.reason.get("llm_scoring"), dict):
            event["llm_scoring"] = deepcopy(action.reason["llm_scoring"])
        if not new_ids:
            raise RuntimeError("Online morph registered no trainable parameters")
        applied = controller.mark_applied(self._actual_iterations, event)
        event.update({key: applied[key] for key in (
            "rollback_allowed", "rollback_whitelist_groups", "executed_primitives",
        )})
        morphable.bind_latest_intervention_id(str(applied["intervention_id"]))
        if controller.active_trial is not None:
            controller.active_trial[
                "optimizer_learning_rates_before_morph"
            ] = self._capture_online_morph_learning_rates()
        event["intervention_id"] = applied["intervention_id"]
        self._online_morph_phase_wall_started = self._online_morph_timing_start()
        self._pending_online_morph_snapshot = pre_morph_snapshot
        self._initialize_adaptive_trial(applied)
        self._active_morph_result = morph_result
        self._set_online_morph_trainability(controller.current_morph_phase)
        if controller.current_morph_phase in {"joint", "short_utility"}:
            controller.joint_transition_complete = False
            self._complete_online_joint_transition(phase)
        self.adaptive_control_summary["online_morphs_applied"] = int(
            controller.morph_count
        )
        self.adaptive_control_summary["online_morph_events"] = deepcopy(
            controller.events
        )
        self.adaptive_control_summary["online_morph_final_architecture"] = (
            morphable.architecture_state()
        )
        self._sync_online_morph_summary()
        self.adaptive_control_summary["online_morph_max_function_error"] = max(
            float(
                self.adaptive_control_summary.get(
                    "online_morph_max_function_error", 0.0
                )
            ),
            function_error,
        )
        self.adaptive_control_summary["online_morph_max_first_derivative_error"] = max(
            float(self.adaptive_control_summary.get("online_morph_max_first_derivative_error", 0.0)),
            preservation.first_derivative_error,
        )
        self.adaptive_control_summary["online_morph_max_second_derivative_error"] = max(
            float(self.adaptive_control_summary.get("online_morph_max_second_derivative_error", 0.0)),
            preservation.second_derivative_error,
        )
        self.adaptive_control_summary[
            "online_morph_max_pde_residual_preservation_error"
        ] = max(
            float(
                self.adaptive_control_summary.get(
                    "online_morph_max_pde_residual_preservation_error", 0.0
                )
            ),
            pde_preservation.maximum_error,
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="online_morph_applied",
            trigger=dict(action.reason),
            before={
                "morph_count": controller.morph_count - 1,
                "parameter_count": event["old_parameter_count"],
            },
            after=event,
            limits={
                "allowed_primitives": list(controller.config["allowed_primitives"]),
                "function_preservation_tolerance": tolerance,
                "first_derivative_preservation_tolerance": (
                    first_derivative_tolerance
                ),
                "second_derivative_preservation_tolerance": (
                    second_derivative_tolerance
                ),
                "pde_residual_preservation_tolerance": (
                    pde_residual_tolerance
                ),
            },
        )
        self._append_morph_event_record({"event": "MORPH_APPLIED", **event})
        self.adaptive_control_summary["online_morph_wall_time"] = float(
            self.adaptive_control_summary.get("online_morph_wall_time", 0.0)
        ) + (time.perf_counter() - morph_control_started)
        return event

    def _observe_gradient_balance(
        self,
        *,
        phase: Any,
        loss: float,
        components: dict[str, float],
        gradient_norm: float | None,
        diagnostic: dict[str, Any] | None,
    ) -> None:
        controller = self.gradient_balance_controller
        if controller is None:
            return
        online_controller = self.online_morph_controller
        if online_controller is not None:
            # Preserve one-to-one policy credit: deterministic weight control
            # cannot mutate the objective during an EOH intervention's
            # short/medium/persistent observation horizon, or on the same step
            # that an EOH decision is made.
            if (
                online_controller.active_trial is not None
                or online_controller.pending_persistent_feedback
                or self._actual_iterations
                % max(1, int(online_controller.config["diagnostic_interval"]))
                == 0
            ):
                return
        diagnostic = dict(diagnostic or {})
        pairwise = dict(diagnostic.get("pairwise_cosine_summary") or {})
        stage_descriptor = self._stage_descriptor(phase)
        eligible_component_ids = (
            self.registry.active_weightable_component_ids(phase.name)
            if self.registry is not None
            else tuple(sorted(diagnostic.get("component_grad_norms") or {}))
        )
        state = TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(phase.optimizer_name),
            current_loss=float(loss),
            loss_components=dict(components),
            loss_weights=dict(
                getattr(
                    self.built.loss_function,
                    "get_runtime_loss_weights",
                    lambda: {},
                )()
            ),
            component_gradient_norms=dict(
                diagnostic.get("component_grad_norms") or {}
            ),
            minimum_gradient_cosine=pairwise.get("minimum"),
            learning_rates=tuple(
                float(group.get("lr", 0.0))
                for group in phase.optimizer.param_groups
            ),
            gradient_norm=gradient_norm,
            completed_stage_iterations=int(
                self.completed_phase_iterations.get(phase.name, 0)
            ),
            sampling_state=dict(
                getattr(self.built.sampler, "last_sampling_snapshot", {}) or {}
            ),
            time_window_fraction=float(
                getattr(self.built.sampler, "time_window_fraction", 1.0)
            ),
            numerical_finite=math.isfinite(float(loss)),
            stage_descriptor=stage_descriptor,
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            eligible_component_ids=eligible_component_ids,
            component_gradient_ema=dict(controller.ema_gradient_norms),
            optimizer_diagnostics=dict(diagnostic),
            budget_state={
                "completed_stage_iterations": int(
                    self.completed_phase_iterations.get(phase.name, 0)
                ),
                "remaining_stage_iterations": max(
                    0,
                    int(getattr(phase, "iterations", 0))
                    - int(self.completed_phase_iterations.get(phase.name, 0)),
                ),
            },
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
        )
        if self.controller_coordinator is not None:
            observation = self.controller_coordinator.observe_controller(
                controller, state
            )
        else:
            observation = controller.observe(state)
        if observation.get("reason") != "diagnostic_interval_not_reached":
            event_type = "diagnostic"
            if observation.get("gradient_conflict"):
                event_type = (
                    "conflict_aware_weight_update_proposed"
                    if controller.should_act()
                    else "gradient_conflict_observed"
                )
            elif observation.get("in_cooldown"):
                event_type = "action_rejected_cooldown"
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type=event_type,
                trigger=observation,
            )
        proposals = (
            self.controller_coordinator.collect_proposals()
            if self.controller_coordinator is not None
            else tuple(
                item
                for item in (controller.propose_action(),)
                if item is not None
            )
        )
        if self.controller_coordinator is not None:
            action, rejected = self.controller_coordinator.select_action(
                proposals, iteration=self._actual_iterations
            )
            for action_id, reason in rejected.items():
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller="controller_coordinator",
                    event_type="action_rejected_by_coordinator",
                    trigger={"action_id": action_id, "reason": reason},
                )
        else:
            action = proposals[0] if proposals else None
        if action is None:
            return
        action_signature = self._action_signature(action)
        if (
            self.rollback_manager is not None
            and action.action_type not in ROLLBACK_ACTION_WHITELIST
            and action_signature
            in self.rollback_manager.failed_action_signatures
        ):
            controller.mark_action_applied(self._actual_iterations)
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=action.controller,
                event_type="action_rejected_previous_rollback",
                trigger=dict(action.reason),
                before=dict(action.before),
                after=dict(action.after),
                limits=dict(action.limits),
            )
            return
        checkpoint_id = self._save_pre_action_checkpoint(action, loss)
        try:
            self.apply_validated_action(action, observable_state=state)
            if action.action_type == "terminate_candidate":
                controller.pending_action = None
                if self.controller_coordinator is not None:
                    self.controller_coordinator.record_outcome(
                        action,
                        iteration=self._actual_iterations,
                        outcome="applied",
                    )
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="action_applied",
                    trigger=dict(action.reason),
                    after=dict(action.after),
                )
        except Exception as exc:
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=action.controller,
                event_type="action_failed",
                trigger={**dict(action.reason), "error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
                limits=dict(action.limits),
                checkpoint_id=checkpoint_id,
            )
            if self.rollback_manager is not None:
                self.rollback_manager.clear()
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action,
                    iteration=self._actual_iterations,
                    outcome="failed",
                )
            return
        controller.mark_action_applied(self._actual_iterations)
        self.adaptive_control_summary["gradient_balance_actions"] = int(
            self.adaptive_control_summary.get("gradient_balance_actions", 0)
        ) + 1
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action,
                iteration=self._actual_iterations,
                outcome="applied",
            )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=action.controller,
            event_type="action_applied",
            trigger=dict(action.reason),
            before=dict(action.before),
            after=dict(action.after),
            limits=dict(action.limits),
            checkpoint_id=checkpoint_id,
        )

    def apply_validated_action(
        self,
        action: AdaptiveAction,
        *,
        observable_state: TrainerObservableState | None = None,
    ) -> Any:
        """Enforce the finite action space at the only mutation boundary."""

        if self._last_training_phase is None:
            raise ValueError("No active optimization stage")
        stage_descriptor = self._stage_descriptor(self._last_training_phase)
        if observable_state is None:
            observable_state = TrainerObservableState(
                iteration=self._actual_iterations,
                stage=stage_descriptor.stage_id,
                optimizer_name=str(self._last_training_phase.optimizer_name),
                loss_weights=self.built.loss_function.get_runtime_loss_weights(),
                remaining_iteration_budget=max(
                    0,
                    int(getattr(self._last_training_phase, "iterations", 0))
                    - int(
                        self.completed_phase_iterations.get(
                            self._last_training_phase.name, 0
                        )
                    ),
                ),
                stage_descriptor=stage_descriptor,
                capability_snapshot=(
                    self.registry.capabilities if self.registry is not None else None
                ),
                eligible_component_ids=(
                    self.registry.active_weightable_component_ids(
                        stage_descriptor.stage_id
                    )
                    if self.registry is not None
                    else tuple(self.built.loss_function.get_runtime_loss_weights())
                ),
            )
        if self.action_boundary_validator is not None and self.registry is not None:
            self.action_boundary_validator.validate(
                action=action,
                registry=self.registry,
                stage=stage_descriptor,
                observable_state=observable_state,
            )
        if action.action_type == "replace_sampling_subset":
            if not isinstance(action, ReplaceSamplingSubsetAction):
                raise ValueError("Sampling replacement requires its structured action type")
            return self._execute_sampling_action(action)
        if action.action_type == "advance_curriculum_state":
            if not isinstance(action, AdvanceCurriculumStateAction):
                raise ValueError("Curriculum advance requires its structured action type")
            return self._execute_curriculum_advance_action(
                action, observable_state
            )
        if action.action_type == "terminate_candidate":
            self._curriculum_termination_requested = True
            return None
        if action.action_type != "update_loss_weights":
            raise ValueError(f"No executor is enabled for action: {action.action_type}")
        old_all = self.built.loss_function.get_runtime_loss_weights()
        updated = dict(old_all)
        minimum = float(action.limits["minimum_weight"])
        maximum = float(action.limits["maximum_weight"])
        minimum_ratio = float(action.limits["minimum_update_ratio"])
        maximum_ratio = float(action.limits["maximum_update_ratio"])
        if set(action.after) - set(old_all):
            raise ValueError("An adaptive action cannot introduce loss terms")
        for name, value in action.after.items():
            before = float(old_all[name])
            after = float(value)
            ratio = after / before
            if not math.isfinite(after) or not minimum <= after <= maximum:
                raise ValueError("Adaptive loss weight violates its global bounds")
            if not minimum_ratio - 1e-12 <= ratio <= maximum_ratio + 1e-12:
                raise ValueError("Adaptive loss weight violates its per-action ratio bound")
            updated[name] = after
        self.built.loss_function.set_runtime_loss_weights(updated)
        return None

    def _runtime_state_payload(
        self, *, current_loss: float | None = None
    ) -> dict[str, Any]:
        # Optimizer.state_dict() requires every tensor state key to occur in a
        # parameter group.  Online morphing replaces Parameter objects, so
        # enforce that invariant immediately before every runtime snapshot.
        self._synchronize_optimizer_parameters_exact()
        device = next(self.built.model.parameters()).device
        return {
            "model_state_dict": deepcopy(self.built.model.state_dict()),
            "online_morph_architecture": (
                self.morphable_model.architecture_state()
                if self.morphable_model is not None
                else []
            ),
            "online_morph_controller_state": (
                self.online_morph_controller.state_dict()
                if self.online_morph_controller is not None
                else None
            ),
            "pending_online_morph_snapshot": deepcopy(
                self._pending_online_morph_snapshot
            ),
            "optimizer_states": [
                deepcopy(phase.optimizer.state_dict())
                for phase in self.built.optimization_phases
            ],
            "optimizer_param_names": [
                self._optimizer_param_names(phase.optimizer)
                for phase in self.built.optimization_phases
            ],
            "scheduler_states": [
                deepcopy(phase.scheduler.state_dict()) if phase.scheduler else None
                for phase in self.built.optimization_phases
            ],
            "active_optimizer_state": (
                deepcopy(self._last_training_phase.optimizer.state_dict())
                if self._last_training_phase is not None
                else None
            ),
            "active_phase_name": (
                str(self._last_training_phase.name)
                if self._last_training_phase is not None
                else None
            ),
            "history": deepcopy(self.history),
            "completed_phase_iterations": dict(self.completed_phase_iterations),
            "runtime_phase_iterations": {
                str(phase.name): int(phase.iterations)
                for phase in self.built.optimization_phases
            },
            "actual_iterations": int(self._actual_iterations),
            "adaptive_refinement_events": deepcopy(self.adaptive_refinement_events),
            "optimizer_audit": deepcopy(self.optimizer_audit),
            "component_gradient_audits": deepcopy(self.component_gradient_audits),
            "loss_state": self.built.loss_function.state_dict(),
            "sampler_state": self.built.sampler.state_dict(),
            "gradient_balance_controller_state": (
                self.gradient_balance_controller.state_dict()
                if self.gradient_balance_controller is not None
                else None
            ),
            "lbfgs_stall_controller_state": (
                self.lbfgs_stall_controller.state_dict()
                if self.lbfgs_stall_controller is not None
                else None
            ),
            "rollback_manager_state": (
                self.rollback_manager.state_dict()
                if self.rollback_manager is not None
                else None
            ),
            "adaptive_control_summary": deepcopy(self.adaptive_control_summary),
            "adaptive_compute_budget_ledger_state": (
                self.adaptive_compute_budget_ledger.state_dict()
            ),
            "runtime_scaled_policy_summary": deepcopy(
                self.runtime_scaled_policy_summary
            ),
            "pending_recovery": deepcopy(self._pending_recovery),
            "training_component_registry": (
                self.registry.to_dict() if self.registry is not None else None
            ),
            "controller_coordinator_state": (
                self.controller_coordinator.state_dict()
                if self.controller_coordinator is not None
                else None
            ),
            "validation_probe_manager_state": (
                self.validation_probe_manager.state_dict()
                if self.validation_probe_manager is not None
                else None
            ),
            "physics_validation_probe_metadata": deepcopy(
                self.physics_validation_probe_metadata
            ),
            "best_training_loss_checkpoint_manager_state": (
                self.best_training_loss_checkpoint_manager.state_dict()
                if self.best_training_loss_checkpoint_manager is not None
                else None
            ),
            "selected_final_model_policy": self._selected_final_model_policy,
            "fixed_budget_sampling_controller_state": (
                self.fixed_budget_sampling_controller.state_dict()
                if self.fixed_budget_sampling_controller is not None
                else None
            ),
            "pending_sampling_observation": deepcopy(
                self._pending_sampling_observation
            ),
            "sampling_control_summary": deepcopy(
                self.sampling_control_summary
            ),
            "curriculum_runtime_states": {
                curriculum_id: deepcopy(runtime.snapshot_state())
                for curriculum_id, runtime in self.curriculum_runtimes.items()
            },
            "curriculum_runtime_state_headers": {
                curriculum_id: runtime.get_current_state().to_dict()
                for curriculum_id, runtime in self.curriculum_runtimes.items()
            },
            "curriculum_advance_controller_state": (
                self.curriculum_advance_controller.state_dict()
                if self.curriculum_advance_controller is not None
                else None
            ),
            "pending_curriculum_observation": deepcopy(
                self._pending_curriculum_observation
            ),
            "curriculum_control_summary": deepcopy(
                self.curriculum_control_summary
            ),
            "curriculum_resume_disabled": bool(
                self._curriculum_resume_disabled
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "current_loss": current_loss,
        }

    def _restore_runtime_state(
        self, payload: dict[str, Any], *, preserve_consumed_budget: bool = False
    ) -> None:
        consumed = int(self._actual_iterations)
        completed = dict(self.completed_phase_iterations)
        consumed_compute = self.adaptive_compute_budget_ledger.state_dict()
        if self.morphable_model is not None:
            self.morphable_model.restore_architecture(
                payload.get("online_morph_architecture") or []
            )
            self._synchronize_optimizer_parameters_exact()
        self.built.model.load_state_dict(payload["model_state_dict"])
        self._load_optimizer_state_by_param_names(
            self.built.optimization_phases,
            payload.get("optimizer_states") or [],
            payload.get("optimizer_param_names"),
        )
        for phase, state in zip(
            self.built.optimization_phases, payload.get("scheduler_states") or []
        ):
            if phase.scheduler is not None and state is not None:
                phase.scheduler.load_state_dict(state)
        active_state = payload.get("active_optimizer_state")
        active_phase_name = payload.get("active_phase_name")
        self._restore_matching_active_optimizer_state(
            self._last_training_phase,
            active_state,
            active_phase_name,
        )
        self.history = list(payload.get("history") or [])
        self.adaptive_refinement_events = list(
            payload.get("adaptive_refinement_events") or []
        )
        self.optimizer_audit = dict(payload.get("optimizer_audit") or {})
        self.component_gradient_audits = list(
            payload.get("component_gradient_audits") or []
        )
        self.completed_phase_iterations = {
            str(name): int(value)
            for name, value in (
                payload.get("completed_phase_iterations") or {}
            ).items()
        }
        runtime_phase_iterations = dict(payload.get("runtime_phase_iterations") or {})
        if runtime_phase_iterations:
            for phase in self.built.optimization_phases:
                if phase.name in runtime_phase_iterations:
                    phase.iterations = int(runtime_phase_iterations[phase.name])
        self._actual_iterations = int(payload.get("actual_iterations") or 0)
        self.built.loss_function.load_state_dict(payload.get("loss_state"))
        self.built.sampler.load_state_dict(payload.get("sampler_state"))
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.load_state_dict(
                payload.get("gradient_balance_controller_state")
            )
        if self.lbfgs_stall_controller is not None:
            self.lbfgs_stall_controller.load_state_dict(
                payload.get("lbfgs_stall_controller_state")
            )
        if self.online_morph_controller is not None:
            self.online_morph_controller.load_state_dict(
                payload.get("online_morph_controller_state")
            )
            self._pending_online_morph_snapshot = (
                deepcopy(payload.get("pending_online_morph_snapshot"))
                if (self.online_morph_controller.active_trial
                    or self.online_morph_controller.pending_persistent_feedback)
                else None
            )
            self._active_morph_result = self.morphable_model.latest_morph_result()
            self._restore_online_morph_trainability()
        if self.rollback_manager is not None:
            self.rollback_manager.load_state_dict(
                payload.get("rollback_manager_state")
            )
        self.adaptive_control_summary = dict(
            payload.get("adaptive_control_summary")
            or self.adaptive_control_summary
        )
        self.adaptive_compute_budget_ledger.load_state_dict(
            payload.get("adaptive_compute_budget_ledger_state"),
            preserve_consumed=preserve_consumed_budget,
        )
        if preserve_consumed_budget:
            for name, value in consumed_compute.items():
                current = getattr(self.adaptive_compute_budget_ledger, name)
                setattr(
                    self.adaptive_compute_budget_ledger,
                    name,
                    type(current)(max(current, value)),
                )
        stored_runtime_policy = payload.get("runtime_scaled_policy_summary")
        if stored_runtime_policy is not None:
            if (
                self.runtime_scaled_policy_summary
                and dict(stored_runtime_policy) != self.runtime_scaled_policy_summary
            ):
                raise ValueError("Checkpoint runtime-scaled policy is incompatible")
            self.runtime_scaled_policy_summary = deepcopy(stored_runtime_policy)
        self._pending_recovery = deepcopy(payload.get("pending_recovery"))
        stored_registry = payload.get("training_component_registry")
        if stored_registry is not None and self.registry is not None:
            current = self.registry.to_dict()
            legacy_registry = int(payload.get("checkpoint_schema_version") or 0) < 5
            if (
                (
                    not legacy_registry
                    and stored_registry.get("schema_version") != current.get("schema_version")
                )
                or {
                    item["component_id"]
                    for item in stored_registry.get("loss_components") or []
                }
                != self.registry.component_ids
                or {
                    item["sampler_id"]
                    for item in stored_registry.get("sampling_components") or []
                }
                != {
                    item.sampler_id for item in self.registry.sampling_components
                }
                or (
                    not legacy_registry
                    and {
                        item["curriculum_id"]
                        for item in stored_registry.get("curricula") or []
                    }
                    != self.registry.curriculum_ids
                )
            ):
                raise ValueError("Checkpoint training-component registry is incompatible")
        if self.controller_coordinator is not None:
            self.controller_coordinator.load_state_dict(
                payload.get("controller_coordinator_state")
            )
        if self.validation_probe_manager is not None:
            self.validation_probe_manager.load_state_dict(
                payload.get("validation_probe_manager_state")
            )
            # A fresh trainer has no probes yet. Restore their saved seed and
            # tensor contract before binding, without recalibrating resumed scores.
            if self.validation_probe_manager.probe_set is None:
                self.validation_probe_manager.build_probes(self.registry, self.built.sampler)
            self._bind_online_morph_physics_probe()
        if self.best_training_loss_checkpoint_manager is not None:
            self.best_training_loss_checkpoint_manager.load_state_dict(
                payload.get("best_training_loss_checkpoint_manager_state")
            )
        self._selected_final_model_policy = payload.get(
            "selected_final_model_policy"
        )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.load_state_dict(
                payload.get("fixed_budget_sampling_controller_state")
            )
        self._pending_sampling_observation = None
        if (
            self.rollback_manager is not None
            and str(self.rollback_manager.active_checkpoint_id or "").startswith("pre_sampling_action")
        ):
            self.rollback_manager.clear()
        self.sampling_control_summary = dict(
            payload.get("sampling_control_summary")
            or self.sampling_control_summary
        )
        self.sampling_control_summary.update(
            post_action_validation="skipped_sampling_whitelist",
            rollback_allowed=False,
        )
        runtime_states = payload.get("curriculum_runtime_states")
        runtime_headers = payload.get("curriculum_runtime_state_headers") or {}
        if runtime_states is None:
            if self.curriculum_advance_controller is not None:
                self._curriculum_resume_disabled = True
                for curriculum_id, runtime in self.curriculum_runtimes.items():
                    runtime.restore_state(
                        self._initial_curriculum_runtime_states[curriculum_id]
                    )
                    descriptor = self.registry.curriculum(curriculum_id)
                    refresh = getattr(
                        self.built.sampler, "refresh_curriculum_samplers", None
                    )
                    if callable(refresh):
                        refresh(descriptor.associated_sampler_ids)
        else:
            if set(runtime_states) != set(self.curriculum_runtimes):
                raise ValueError("Checkpoint Curriculum Runtime IDs are incompatible")
            for curriculum_id, runtime in self.curriculum_runtimes.items():
                runtime.restore_state(runtime_states[curriculum_id])
                restored = runtime.get_current_state()
                expected = dict(runtime_headers.get(curriculum_id) or {})
                if expected and (
                    restored.current_level_index
                    != int(expected.get("current_level_index", -1))
                    or restored.runtime_state_version
                    != int(expected.get("runtime_state_version", -1))
                    or restored.state_digest != str(expected.get("state_digest", ""))
                ):
                    raise ValueError(
                        f"Checkpoint Curriculum Runtime state is inconsistent: {curriculum_id}"
                    )
            self._curriculum_resume_disabled = bool(
                payload.get("curriculum_resume_disabled", False)
            )
        if self.curriculum_advance_controller is not None:
            self.curriculum_advance_controller.load_state_dict(
                payload.get("curriculum_advance_controller_state")
            )
        self._pending_curriculum_observation = deepcopy(
            payload.get("pending_curriculum_observation")
        )
        self.curriculum_control_summary = dict(
            payload.get("curriculum_control_summary")
            or self.curriculum_control_summary
        )
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            # ``torch.load(..., map_location=cuda)`` also relocates the CPU
            # RNG ByteTensor.  The CPU generator API requires it on CPU.
            torch.set_rng_state(payload["torch_random_state"].cpu())
        if payload.get("cuda_random_states") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_random_states"]]
            )
        if preserve_consumed_budget:
            self._actual_iterations = consumed
            self.completed_phase_iterations = completed

    def _save_pre_action_checkpoint(
        self, action: AdaptiveAction, loss: float | None
    ) -> str | None:
        if (
            self.rollback_manager is None
            or action.action_type in ROLLBACK_ACTION_WHITELIST
        ):
            return None
        prefix = (
            "pre_curriculum_advance"
            if action.action_type == "advance_curriculum_state"
            else "pre_action"
        )
        checkpoint_id = f"{prefix}_iter_{self._actual_iterations}_{action.controller}"
        self.adaptive_compute_budget_ledger.checkpoint_count += 1
        payload = self._runtime_state_payload(current_loss=loss)
        payload["action_signature"] = self._action_signature(action)
        payload["active_action_type"] = str(action.action_type)
        payload["active_action_metadata"] = {
            "curriculum_id": str(getattr(action, "curriculum_id", "")),
            "from_level_index": int(getattr(action, "from_level_index", -1)),
            "to_level_index": int(getattr(action, "to_level_index", -1)),
            "forced": bool(getattr(action, "forced", False)),
        }
        payload["coordinator_action_signature"] = (
            self.controller_coordinator.action_signature(action)
            if self.controller_coordinator is not None
            else None
        )
        self.rollback_manager.save(
            checkpoint_id,
            payload,
            self.checkpoint_dir,
        )
        return checkpoint_id

    @staticmethod
    def _restore_matching_active_optimizer_state(
        phase: Any | None,
        state: dict[str, Any] | None,
        saved_phase_name: str | None,
    ) -> bool:
        """Restore the duplicate active state only into the phase that saved it.

        Loading an Adam state into the current L-BFGS phase can succeed in
        PyTorch while silently replacing the L-BFGS parameter-group defaults.
        Its next step then fails with ``KeyError: 'tolerance_grad'``.
        """

        if phase is None or state is None or saved_phase_name is None:
            return False
        if str(getattr(phase, "name", "")) != str(saved_phase_name):
            return False
        try:
            phase.optimizer.load_state_dict(state)
        except (ValueError, KeyError):
            return False
        return True

    @staticmethod
    def _action_signature(action: AdaptiveAction) -> str:
        return repr(
            (
                str(action.controller),
                str(action.action_type),
                tuple(
                    sorted(
                        (str(name), repr(value))
                        for name, value in action.after.items()
                    )
                ),
            )
        )

    def _rollback_active_action(self, *, phase: Any, reason: str) -> None:
        manager = self.rollback_manager
        if manager is None or manager.active_checkpoint_id is None:
            return
        checkpoint_id = manager.active_checkpoint_id
        payload = manager.load_payload(next(self.built.model.parameters()).device)
        if payload is None:
            return
        active_action_type = str(payload.get("active_action_type") or "")
        if active_action_type in ROLLBACK_ACTION_WHITELIST:
            manager.clear()
            self._pending_sampling_observation = None
            return
        active_action_metadata = dict(payload.get("active_action_metadata") or {})
        consumed_sampling = deepcopy(self.sampling_control_summary)
        sampling_updates = (
            int(self.fixed_budget_sampling_controller.update_count)
            if self.fixed_budget_sampling_controller is not None else 0
        )
        consumed_curriculum = deepcopy(self.curriculum_control_summary)
        curriculum_counts = (
            (
                int(self.curriculum_advance_controller.advance_count),
                int(self.curriculum_advance_controller.forced_advance_count),
                int(self.curriculum_advance_controller.rollback_count),
            )
            if self.curriculum_advance_controller is not None
            else (0, 0, 0)
        )
        best_state = (
            self.best_training_loss_checkpoint_manager.state_dict()
            if self.best_training_loss_checkpoint_manager is not None else None
        )
        self._restore_runtime_state(payload, preserve_consumed_budget=True)
        for name in (
            "sampling_diagnostic_evaluations", "sampling_points_diagnosed",
            "candidate_points_evaluated", "sampling_updates_executed",
            "sampling_updates_rejected", "sampling_points_replaced",
        ):
            self.sampling_control_summary[name] = max(
                int(self.sampling_control_summary.get(name, 0)),
                int(consumed_sampling.get(name, 0)),
            )
        self.sampling_control_summary["sampling_compute_time"] = max(
            float(self.sampling_control_summary.get("sampling_compute_time", 0.0)),
            float(consumed_sampling.get("sampling_compute_time", 0.0)),
        )
        if best_state is not None and self.best_training_loss_checkpoint_manager is not None:
            self.best_training_loss_checkpoint_manager.load_state_dict(best_state)
        signature = str(payload.get("action_signature") or checkpoint_id)
        manager.mark_rolled_back(signature)
        if self.controller_coordinator is not None:
            coordinator_signature = payload.get("coordinator_action_signature")
            if coordinator_signature:
                self.controller_coordinator.failed_signatures[
                    str(coordinator_signature)
                ] = int(self._actual_iterations)
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.mark_action_applied(
                self._actual_iterations
                + int(
                    self.gradient_balance_controller.config[
                        "cooldown_iterations"
                    ]
                )
            )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.update_count = max(
                int(self.fixed_budget_sampling_controller.update_count),
                sampling_updates,
            )
            self.fixed_budget_sampling_controller.mark_action_rolled_back(
                self._actual_iterations
            )
            if active_action_type == "replace_sampling_subset":
                self.sampling_control_summary["sampling_updates_rolled_back"] = int(
                    consumed_sampling.get("sampling_updates_rolled_back", 0)
                ) + 1
        if (
            active_action_type == "advance_curriculum_state"
            and self.curriculum_advance_controller is not None
        ):
            controller = self.curriculum_advance_controller
            controller.advance_count = max(
                int(controller.advance_count), curriculum_counts[0]
            )
            controller.forced_advance_count = max(
                int(controller.forced_advance_count), curriculum_counts[1]
            )
            controller.rollback_count = max(
                int(controller.rollback_count), curriculum_counts[2]
            )
            curriculum_id = str(active_action_metadata.get("curriculum_id") or "")
            controller.mark_action_rolled_back(
                curriculum_id, self._actual_iterations
            )
            runtime = self.curriculum_runtimes.get(curriculum_id)
            if runtime is not None:
                runtime.record_rollback(self._actual_iterations)
            integer_fields = (
                "curriculum_diagnostic_checks", "curriculum_advances_proposed",
                "curriculum_advances_executed", "curriculum_advances_forced",
                "curriculum_advances_rejected",
            )
            for name in integer_fields:
                self.curriculum_control_summary[name] = max(
                    int(self.curriculum_control_summary.get(name, 0)),
                    int(consumed_curriculum.get(name, 0)),
                )
            for name in (
                "curriculum_runtime_refresh_cost", "curriculum_validation_cost",
            ):
                self.curriculum_control_summary[name] = max(
                    float(self.curriculum_control_summary.get(name, 0.0)),
                    float(consumed_curriculum.get(name, 0.0)),
                )
            self.curriculum_control_summary["curriculum_advances_rolled_back"] = int(
                consumed_curriculum.get("curriculum_advances_rolled_back", 0)
            ) + 1
            self.curriculum_control_summary["iterations_per_level"] = deepcopy(
                consumed_curriculum.get("iterations_per_level") or {}
            )
        self._pending_sampling_observation = None
        self._pending_curriculum_observation = None
        self.adaptive_control_summary["rollbacks"] = int(
            self.adaptive_control_summary.get("rollbacks", 0)
        ) + 1
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller="checkpoint_rollback",
            event_type="rollback",
            trigger={"reason": reason},
            checkpoint_id=checkpoint_id,
            rollback_status="completed",
        )

    def _parameter_update_norm(
        self, before: list[torch.Tensor]
    ) -> float | None:
        squared = 0.0
        found = False
        for old, parameter in zip(before, self.built.model.parameters()):
            difference = parameter.detach() - old.to(parameter.device)
            value = float(torch.linalg.vector_norm(difference).cpu())
            squared += value * value
            found = True
        return squared**0.5 if found else None

    def _run_recovery_adam(
        self,
        budget: int,
        clipping_norm: float | None,
        clipping_norm_type: float,
        adaptive_interval: int,
    ) -> tuple[float | None, int, str | None]:
        controller = self.lbfgs_stall_controller
        # The label-free L-BFGS physics guard can independently reallocate the
        # remaining budget even when optional stall recovery is disabled.
        # In that path there is no stall controller from which to read a
        # learning-rate scale, so retain the established recovery default.
        scale = float(
            controller.config.get("adam_learning_rate_scale", 0.1)
            if controller is not None
            else 0.1
        )
        source_lr = 1e-3
        for configured in reversed(self.built.optimization_phases):
            if configured.optimizer_name in {"adam", "adamw", "radam", "nadam"}:
                source_lr = float(configured.optimizer.param_groups[0].get("lr", source_lr))
                break
        optimizer = torch.optim.Adam(
            self.built.model.parameters(), lr=max(1e-12, source_lr * scale)
        )
        phase = OptimizationPhase(
            name="adaptive_recovery_adam",
            optimizer_name="adam",
            iterations=int(budget),
            optimizer=optimizer,
            scheduler=None,
            parameters={"learning_rate": max(1e-12, source_lr * scale)},
        )
        self._last_training_phase = phase
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=phase.name,
            controller="lbfgs_stall",
            event_type="recovery_adam_started",
            trigger={"source_learning_rate": source_lr, "scale": scale},
            after={"iterations": int(budget), "learning_rate": source_lr * scale},
        )
        latest: float | None = None
        executed = 0
        failure: str | None = None
        for _ in range(int(budget)):
            stage = self._configure_training_stage(
                phase=phase, non_lbfgs_iteration=self._actual_iterations + 1
            )
            batch = self.built.sampler.sample()
            self._last_training_batch = batch
            optimizer.zero_grad(set_to_none=True)
            loss, components = self.built.loss_function(self.built.model, batch)
            if not torch.isfinite(loss):
                failure = "non_finite_recovery_loss"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            projected = self._actual_iterations + 1
            due = bool(
                self.gradient_balance_controller is not None
                and projected
                % int(self.gradient_balance_controller.config["diagnostic_interval"])
                == 0
            )
            diagnostic = (
                self._collect_component_gradient_diagnostic(
                    total_loss=loss,
                    iteration=projected,
                    phase=phase,
                    checkpoint_kind="adaptive_control",
                )
                if due
                else None
            )
            projection_audit = self._backward_with_runtime_gradient_projection(loss)
            if projection_audit is not None:
                self.optimizer_audit[phase.name] = {
                    **dict(self.optimizer_audit.get(phase.name) or {}),
                    "gradient_projection": projection_audit,
                }
            if clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.built.model.parameters(),
                    clipping_norm,
                    norm_type=clipping_norm_type,
                )
            if not self._gradients_finite():
                failure = "non_finite_recovery_gradient"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            gradient_norm = self._gradient_norm()
            optimizer.step()
            if not self._parameters_finite():
                failure = "non_finite_recovery_parameter"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            latest = float(loss.detach().cpu())
            executed += 1
            self._actual_iterations += 1
            # Recovery steps consume the same global optimizer-step budget as
            # the original phases and must satisfy the same held-out physics probe.
            self._observe_gradient_balance(
                phase=phase,
                loss=latest,
                components=components,
                gradient_norm=gradient_norm,
                diagnostic=diagnostic,
            )
            if self.rollback_manager is not None and self.rollback_manager.observe_loss(latest):
                rollback_baseline = getattr(
                    self.rollback_manager, "baseline_loss", None
                )
                self._rollback_active_action(
                    phase=phase, reason="recovery_loss_degradation"
                )
                if rollback_baseline is not None and math.isfinite(
                    float(rollback_baseline)
                ):
                    latest = float(rollback_baseline)
                # A rollback is the successful safety response to a harmful
                # adaptive action, not a numerical training failure.  Continue
                # the remaining recovery budget from the restored checkpoint
                # with fresh, more conservative Adam state.
                optimizer.state.clear()
                for group in optimizer.param_groups:
                    group["lr"] = max(1e-12, float(group["lr"]) * 0.5)
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=phase.name,
                    controller="checkpoint_rollback",
                    event_type="recovery_rollback_continued",
                    trigger={"reason": "recovery_loss_degradation"},
                    after={
                        "remaining_iterations": max(0, int(budget) - executed),
                        "learning_rates": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                    },
                    rollback_status="completed",
                )
                self._release_iteration_graphs()
                continue
            self._maybe_run_physics_validation(
                phase=phase,
                force=False,
                training_loss=latest,
            )
            adaptive_event = self._actual_iterations % adaptive_interval == 0
            if adaptive_event:
                self.built.sampler.adaptive_update(self.built.model)
            if self._should_record_history(
                self._actual_iterations, phase.name, event=adaptive_event
            ):
                self.history.append(
                    self._history_entry(
                        iteration=self._actual_iterations,
                        phase=phase,
                        total_loss=latest,
                        components=components,
                        batch=batch,
                        gradient_norm=gradient_norm,
                        gradient_statistics=diagnostic,
                        adaptive_refinement_event=adaptive_event,
                        training_stage=stage,
                    )
                )
            self._release_iteration_graphs()
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=phase.name,
            controller="lbfgs_stall",
            event_type="recovery_adam_finished",
            trigger={"failure": failure},
            after={"executed_iterations": executed, "final_loss": latest},
        )
        return latest, executed, failure

    def _stage_descriptor(self, phase: Any) -> Any:
        if self.registry is not None:
            try:
                return self.registry.stage(str(phase.name))
            except KeyError:
                pass
        second_order = isinstance(
            getattr(phase, "optimizer", None), torch.optim.LBFGS
        ) or str(getattr(phase, "optimizer_name", "")).casefold() in {
            "lbfgs", "l_bfgs"
        }
        return SimpleNamespace(
            stage_id=str(phase.name),
            optimizer_family=(
                "second_order_line_search"
                if second_order
                else "first_order_gradient"
            ),
            first_order=not second_order,
            second_order=second_order,
            dynamic_objective_allowed=not second_order,
            recoverable=second_order,
        )

    def _next_second_order_phase(self, current_phase: Any) -> Any | None:
        seen_current = False
        for phase in self.built.optimization_phases:
            if phase is current_phase:
                seen_current = True
                continue
            if not seen_current:
                continue
            if self._stage_descriptor(phase).second_order:
                return phase
        return None

    def _transfer_unused_first_order_budget_to_lbfgs(
        self,
        *,
        current_phase: Any,
        unused_iterations: int,
        reason: str,
    ) -> int:
        target = self._next_second_order_phase(current_phase)
        transfer = max(0, int(unused_iterations))
        if target is None or transfer <= 0:
            return 0
        target.iterations = int(getattr(target, "iterations", 0)) + transfer
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(current_phase.name),
            controller="optimizer_phase_switch",
            event_type="first_order_budget_transferred_to_lbfgs",
            trigger={"reason": reason},
            after={
                "target_phase": str(target.name),
                "transferred_iterations": transfer,
                "target_phase_iterations": int(target.iterations),
            },
        )
        return transfer

    def _optimizer_phase_switch_config(self) -> dict[str, Any]:
        training = self.built.normalized_spec.get("training") or {}
        adaptive = (
            (training.get("extra_parameters") or {}).get("adaptive_control", {})
        )
        config = dict((adaptive or {}).get("optimizer_phase_switch") or {})
        return {
            "enabled": bool(config.get("enabled", True)),
            "latest_lbfgs_entry_step": int(
                config.get("latest_lbfgs_entry_step", 14_000)
            ),
            "require_trial_completion": bool(config.get("require_trial_completion", False)),
        }

    def _initialize_physics_validation(self) -> None:
        manager = self.validation_probe_manager
        if manager is None or self.registry is None:
            return
        if manager.probe_set is None:
            manager.build_probes(self.registry, self.built.sampler)
        if manager.probe_set is None or manager.initial_scales:
            return
        result = manager.evaluate(
            self.built.model,
            self.registry,
            self.built.loss_function,
        )
        if result is not None:
            manager.record(0, "training_start", result)
            if self.gradient_balance_controller is not None:
                self.gradient_balance_controller.record_calibration_probe(
                    result.component_scores
                )
        if manager.audit_probe_set is not None and not manager.audit_initial_scales:
            audit_result = manager.evaluate(
                self.built.model,
                self.registry,
                self.built.loss_function,
                channel="audit",
            )
            if audit_result is not None:
                manager.record(
                    0, "training_start", audit_result, channel="audit"
                )

    def _bind_online_morph_physics_probe(self) -> None:
        """Bind online decisions to the one run-level physics-validation probe."""

        if self.online_morph_controller is None:
            return
        manager = self.validation_probe_manager
        if manager is None or manager.probe_set is None:
            raise RuntimeError(
                "Online morphing requires the initialized physics_validation probe"
            )
        self.physics_validation_probe = manager.probe_set
        self.physics_validation_probe_metadata = {
            "probe_set_id": manager.probe_set.probe_set_id,
            "probe_seed": int(manager.probe_set.seed),
            "component_budgets": dict(manager.probe_set.component_budgets),
            "used_for_gradient_training": False,
            "used_for_adaptive_sampling": False,
            "reference_solution_access": False,
            "audit_probe_set_id": (
                manager.audit_probe_set.probe_set_id
                if manager.audit_probe_set is not None
                else None
            ),
            "audit_probe_seed": (
                int(manager.audit_probe_set.seed)
                if manager.audit_probe_set is not None
                else None
            ),
            "audit_component_budgets": (
                dict(manager.audit_probe_set.component_budgets)
                if manager.audit_probe_set is not None
                else {}
            ),
            "audit_sampling_strategy": manager.audit_sampling_strategy,
        }

    def _evaluate_online_morph_physics_metrics(
        self,
        *,
        channel: str = "decision",
    ) -> PhysicsValidationMetrics:
        timing_started = self._online_morph_timing_start()
        manager = self.validation_probe_manager
        controller = self.online_morph_controller
        if manager is None or controller is None or self.registry is None:
            raise RuntimeError("Online morph held-out physics probe is unavailable")
        if manager.probe_set is None:
            self._initialize_physics_validation()
            self._bind_online_morph_physics_probe()
        utility = dict(controller.config.get("utility") or {})
        probe_set = (
            manager.audit_probe_set
            if str(channel) == "audit"
            else manager.probe_set
        )
        if probe_set is None:
            raise RuntimeError(f"Held-out physics {channel} probe is unavailable")
        result = manager.evaluate_physics_residual(
            self.built.model,
            self.registry,
            self.built.loss_function,
            weights=dict(utility.get("weights") or {}),
            probe_set=probe_set,
        )
        if result is None:
            raise RuntimeError("Held-out physics metrics could not be evaluated")
        self.adaptive_compute_budget_ledger.physics_probe_evaluations += 1
        self.adaptive_compute_budget_ledger.physics_probe_points_evaluated += sum(
            int(value)
            for value in probe_set.component_budgets.values()
        )
        self._record_online_morph_timing(
            f"heldout_{channel}_probe", timing_started
        )
        return result

    def _online_morph_timing_start(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _record_online_morph_timing(self, category: str, started: float) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing = self.adaptive_control_summary.setdefault("online_morph_timing", {})
        timing[str(category)] = float(timing.get(str(category), 0.0)) + max(
            0.0, time.perf_counter() - float(started)
        )

    def _online_morph_module_telemetry(self) -> dict[str, Any]:
        morphable = self.morphable_model
        probe_batch = getattr(self.physics_validation_probe, "batch", None)
        inputs = getattr(probe_batch, "domain_samples", None)
        if morphable is None or not morphable.interventions or not isinstance(inputs, torch.Tensor):
            return {
                "branch_contribution_ratio": None,
                "gate_mean": None,
                "new_module_parameter_norm": None,
                "reason": "module_telemetry_unavailable",
            }
        latest = morphable.interventions[-1]
        gate_logits: list[torch.Tensor] = []
        handles = []

        def _capture_gate_logits(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                gate_logits.append(output.detach())

        for layer in getattr(latest, "gates", ()):
            handles.append(
                layer.register_forward_hook(_capture_gate_logits)
            )
        was_training = self.built.model.training
        self.built.model.eval()
        try:
            with torch.no_grad():
                prediction = morphable.base_model(inputs)
                specs = [s for s in morphable.architecture_state() if not s.get("direct_topology")]
                branch_inputs = morphable.branch_input_transform(inputs)
                for spec, intervention in zip(specs[:-1], list(morphable.interventions)[:-1]):
                    if spec.get("disabled"):
                        continue
                    modified = intervention(branch_inputs, prediction)
                    prediction = prediction + float(spec.get("adaptive_influence", 1.0)) * (modified - prediction)
                before_latest = prediction
                modified = latest(branch_inputs, before_latest)
                after_latest = before_latest + float(specs[-1].get("adaptive_influence", 1.0)) * (modified - before_latest)
                contribution = after_latest - before_latest
                contribution_ratio = float(
                    contribution.norm().div(after_latest.norm().clamp_min(1e-12)).cpu()
                )
                if gate_logits:
                    gate_mean = float(
                        torch.cat([value.reshape(-1) for value in gate_logits])
                        .sigmoid()
                        .mean()
                        .cpu()
                    )
                else:
                    gate = getattr(latest, "gate", None)
                    gate_mean = (
                        float(gate.detach().mean().cpu())
                        if isinstance(gate, torch.Tensor)
                        else None
                    )
                parameter_norm = math.sqrt(
                    sum(
                        float(parameter.detach().square().sum().cpu())
                        for parameter in latest.parameters()
                    )
                )
        finally:
            for handle in handles:
                handle.remove()
            self.built.model.train(was_training)
        return {
            "branch_contribution_ratio": contribution_ratio,
            "gate_mean": gate_mean,
            "new_module_parameter_norm": parameter_norm,
            "used_for_acceptance": False,
        }

    def _restore_final_best_checkpoint(self) -> float | None:
        manager = self.best_training_loss_checkpoint_manager
        if manager.restore_best_model(self.built.model):
            self._selected_final_model_policy = manager.final_model_policy
            self.adaptive_control_summary.pop("checkpoint_selection_fallback", None)
            value = (manager.best_metadata or {}).get("training_loss")
            return float(value) if value is not None else None
        self._selected_final_model_policy = "last"
        self.adaptive_control_summary["checkpoint_selection_fallback"] = {
            "reason": "no_eligible_best_checkpoint",
            "initial_condition_guard": deepcopy(manager.last_initial_condition_guard),
        }
        return None

    def _maybe_run_physics_validation(
        self,
        *,
        phase: Any | None,
        force: bool,
        training_loss: float | None = None,
    ) -> None:
        manager = self.validation_probe_manager
        checkpoint_manager = self.best_training_loss_checkpoint_manager
        stage_id = str(getattr(phase, "name", "training_end"))
        training_loss_updated = bool(
            checkpoint_manager is not None
            and checkpoint_manager.final_model_policy == "best_train_loss"
            and training_loss is not None
            and (
                force
                or self._actual_iterations % checkpoint_manager.evaluation_interval == 0
            )
            and checkpoint_manager.consider(
                model=self.built.model,
                training_loss=training_loss,
                iteration=self._actual_iterations,
                stage_id=stage_id,
                algorithm_spec_id=self._algorithm_spec_id,
                directory=self.checkpoint_dir,
            )
        )
        if manager is None or self.registry is None or manager.probe_set is None:
            return
        if not force:
            if not manager.is_due(self._actual_iterations):
                return
        if (
            manager.history
            and int(manager.history[-1].get("iteration", -1))
            == int(self._actual_iterations)
        ):
            return
        self._guard_adaptive_influence()
        result = manager.evaluate(
            self.built.model,
            self.registry,
            self.built.loss_function,
        )
        if result is None:
            return
        manager.record(self._actual_iterations, stage_id, result)
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.record_calibration_probe(
                result.component_scores
            )
        audit_result = manager.evaluate(
            self.built.model,
            self.registry,
            self.built.loss_function,
            channel="audit",
        )
        if audit_result is not None:
            manager.record(
                self._actual_iterations,
                stage_id,
                audit_result,
                channel="audit",
            )
        physics_checkpoint_updated = bool(
            checkpoint_manager is not None
            and checkpoint_manager.final_model_policy
            == "best_physics_validation"
            and audit_result is not None
            and manager.normalization_ready("audit")
            and not self._adaptive_trial_pending()
            and checkpoint_manager.consider(
                model=self.built.model,
                training_loss=training_loss,
                physics_validation_score=float(audit_result.aggregate_score),
                score_context=manager.audit_score_context(),
                initial_condition_scores=(
                    {
                        component.component_id: audit_result.normalized_component_scores.get(
                            component.component_id
                        )
                        for component in self.registry.loss_components
                        if component.category == "initial_constraint"
                    }
                    if any(component.category == "initial_constraint"
                           for component in self.registry.loss_components)
                    else None
                ),
                iteration=self._actual_iterations,
                stage_id=stage_id,
                algorithm_spec_id=self._algorithm_spec_id,
                directory=self.checkpoint_dir,
            )
        )
        if checkpoint_manager is not None:
            self.adaptive_control_summary["checkpoint_initial_condition_guard"] = deepcopy(
                checkpoint_manager.last_initial_condition_guard
            )
        self._observe_adaptive_time_progress()
        self._observe_directional_frequency()
        self._assess_pending_sampling_action(result, phase)
        curriculum_pending_before_validation = (
            self._pending_curriculum_observation is not None
        )
        curriculum_started = time.perf_counter()
        curriculum_accepted = self._assess_pending_curriculum_action(
            result, phase
        )
        if curriculum_pending_before_validation:
            self.curriculum_control_summary["curriculum_validation_cost"] += (
                time.perf_counter() - curriculum_started
            )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=stage_id,
            controller="physics_validation",
            event_type="physics_validation",
            trigger={
                **result.to_dict(),
                "probe_channel": "decision",
                "audit_probe": (
                    audit_result.to_dict() if audit_result is not None else None
                ),
            },
            after={
                "best_train_checkpoint_updated": training_loss_updated,
                "best_physics_checkpoint_updated": physics_checkpoint_updated,
                "checkpoint_initial_condition_guard": deepcopy(
                    checkpoint_manager.last_initial_condition_guard
                    if checkpoint_manager is not None else {}
                ),
            },
        )

    def _sampling_observable_state(
        self,
        *,
        phase: Any,
        diagnostics: dict[str, Any] | None = None,
        component_states: dict[str, Any] | None = None,
    ) -> TrainerObservableState:
        controller = self.fixed_budget_sampling_controller
        maximum = int(
            getattr(controller, "config", {}).get(
                "maximum_candidate_points_evaluated", 0
            )
        )
        consumed = int(
            self.sampling_control_summary.get("candidate_points_evaluated", 0)
        )
        coordinator_active = bool(
            self.controller_coordinator is not None
            and self.controller_coordinator.observation_window_active(
                self._actual_iterations
            )
        )
        online_controller = getattr(self, "online_morph_controller", None)
        return TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(getattr(phase, "optimizer_name", "")),
            stage_descriptor=self._stage_descriptor(phase),
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
            sampling_component_states=dict(component_states or {}),
            sampling_diagnostics=dict(diagnostics or {}),
            sampling_budget_state={
                "maximum_candidate_points": maximum,
                "consumed_candidate_points": consumed,
                "remaining_candidate_points": max(0, maximum - consumed),
            },
            pending_rollback=bool(
                (self.rollback_manager is not None
                 and self.rollback_manager.active_checkpoint_id is not None)
                or getattr(online_controller, "active_trial", None)
                or getattr(online_controller, "pending_persistent_feedback", ())
            ),
            coordinator_observation_active=coordinator_active,
        )

    def _observe_sampling_control(self, *, phase: Any) -> None:
        controller = self.fixed_budget_sampling_controller
        sampler = getattr(self.built, "sampler", None)
        if controller is None or self.registry is None or sampler is None:
            return
        preliminary = self._sampling_observable_state(phase=phase)
        due, reason = controller.diagnostic_due(preliminary)
        if not due:
            if reason not in {"diagnostic_interval_not_reached", "minimum_iterations_not_reached"}:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="sampling_diagnostic_skipped",
                    trigger={"reason": reason},
                )
            return
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {}
        component_states: dict[str, Any] = {}
        self._sampling_score_cache = {}
        for descriptor in self.registry.sampling_components:
            if not descriptor.replacement_supported:
                continue
            try:
                points = sampler.get_training_points(descriptor.sampler_id)
                context = self._sampling_observable_state(phase=phase)
                scored = sampler.score_points(
                    self.built.model,
                    descriptor.associated_loss_component_ids,
                    points,
                    context,
                )
            except Exception as exc:
                self.sampling_control_summary["sampling_updates_rejected"] += 1
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="sampling_diagnostic_failed",
                    trigger={
                        "sampler_id": descriptor.sampler_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            component_states[descriptor.sampler_id] = {
                "point_count": points.point_count,
                "sampler_state_version": points.state_version,
            }
            diagnostics[descriptor.sampler_id] = {
                "point_count": points.point_count,
                "sampler_state_version": points.state_version,
                "statistics": dict(scored.statistics),
                "scoring_component_ids": tuple(scored.scoring_component_ids),
            }
            self._sampling_score_cache[descriptor.sampler_id] = (points, scored)
            self.sampling_control_summary["sampling_diagnostic_evaluations"] += 1
            self.sampling_control_summary["sampling_points_diagnosed"] += points.point_count
        self.sampling_control_summary["sampling_compute_time"] += (
            time.perf_counter() - started
        )
        state = self._sampling_observable_state(
            phase=phase,
            diagnostics=diagnostics,
            component_states=component_states,
        )
        observation = (
            self.controller_coordinator.observe_controller(controller, state)
            if self.controller_coordinator is not None
            else controller.observe(state)
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="sampling_diagnostic",
            trigger=observation,
        )
        action = controller.propose_action()
        if action is None:
            return
        if self.controller_coordinator is not None:
            selected, rejected = self.controller_coordinator.select_action(
                (action,), iteration=self._actual_iterations
            )
            if selected is None:
                self.sampling_control_summary["sampling_updates_rejected"] += 1
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller="controller_coordinator",
                    event_type="sampling_action_rejected",
                    trigger={"rejections": rejected},
                )
                return
            action = selected
        try:
            mutation = self.apply_validated_action(action, observable_state=state)
        except Exception as exc:
            self.sampling_control_summary["sampling_updates_rejected"] += 1
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action, iteration=self._actual_iterations, outcome="failed"
                )
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="sampling_action_failed",
                trigger={"error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
            )
            return
        controller.mark_action_applied(self._actual_iterations)
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action, iteration=self._actual_iterations, outcome="applied"
            )
        self.sampling_control_summary["sampling_updates_executed"] += 1
        self.sampling_control_summary["sampling_points_replaced"] += int(
            mutation.points_replaced
        )
        self._pending_sampling_observation = None
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="action_applied",
            trigger={
                **dict(action.reason),
                "post_action_validation": "skipped_sampling_whitelist",
                "rollback_allowed": False,
            },
            before=dict(action.before),
            after={
                **dict(action.after),
                "point_count": mutation.point_count_after,
                "points_replaced": mutation.points_replaced,
                "sampler_state_version": mutation.state_version_after,
            },
            limits=dict(action.limits),
        )

    def _execute_sampling_action(
        self, action: ReplaceSamplingSubsetAction
    ) -> Any:
        """Rebuild the complete PDE cloud using the requested local/global allocation."""
        compute_started = time.perf_counter()
        sampler = self.built.sampler
        cached = self._sampling_score_cache.get(action.sampler_id)
        if cached is None:
            raise RuntimeError("Sampling action score cache is unavailable or stale")
        current, _ = cached
        if int(current.state_version) != int(action.sampler_state_version):
            raise RuntimeError("Sampler state changed before action execution")
        local_fraction, _ = sampling_policy_fractions({
            "local_fraction": action.reason.get("local_fraction", action.replacement_fraction),
            "global_fraction": action.reason.get("global_fraction", action.exploration_fraction),
        })
        target_point_count = int(action.reason.get("point_count", current.point_count))
        if target_point_count <= 0:
            raise ValueError("Sampling point_count must be positive")
        local_count, global_count = fixed_pde_cloud_counts(target_point_count, local_fraction)
        local_points = current.points[:0]
        metadata = ()
        if local_count:
            candidates = sampler.generate_candidate_points(
                action.sampler_id, action.candidate_count
            )
            self.sampling_control_summary["candidate_points_evaluated"] += int(
                action.candidate_count
            )
            candidate_scores = sampler.score_points(
                self.built.model,
                action.scoring_component_ids,
                candidates,
                None,
            )
            ranking = candidate_scores.scores.detach().cpu().clone()
            if str(action.reason.get("region_target") or "") == "boundary_regions":
                reference = current.points.detach().cpu().float()
                proposed = candidates.points.detach().cpu().float()
                lower = reference.amin(dim=0)
                extent = (reference.amax(dim=0) - lower).clamp_min(1.0e-8)
                unit = ((proposed - lower) / extent).clamp(0.0, 1.0)
                distance = torch.minimum(unit, 1.0 - unit).amin(dim=1)
                ranking = ranking + ranking.median().clamp_min(1.0e-12) * torch.exp(-distance / 0.05)
            selected = torch.argsort(ranking, descending=True)[:local_count].tolist()
            local_points = candidates.points[selected]
            metadata = tuple(
                {**dict(candidates.point_metadata[index]), "origin": "local"}
                for index in selected
            )
        points = local_points
        if global_count:
            global_points = sampler.generate_candidate_points(action.sampler_id, global_count)
            points = torch.cat((local_points, global_points.points), dim=0)
            metadata += tuple({**dict(item), "origin": "global"} for item in global_points.point_metadata)
        replacement = SamplingBatch(
            sampler_id=action.sampler_id,
            points=points.detach().clone(),
            point_metadata=metadata,
            state_version=current.state_version,
        )
        mutation = sampler.replace_subset(
            action.sampler_id,
            [],
            replacement,
            exploration_count=global_count,
            iteration=self._actual_iterations,
        )
        self.sampling_control_summary["sampling_compute_time"] += (
            time.perf_counter() - compute_started
        )
        return mutation

    def _assess_pending_sampling_action(self, result: Any, phase: Any | None) -> None:
        """Discard legacy pending feedback; sampling changes are whitelisted."""
        self._pending_sampling_observation = None

    def _note_curriculum_training_iteration(self) -> None:
        if self._curriculum_resume_disabled:
            return
        for runtime in self.curriculum_runtimes.values():
            runtime.note_training_iteration(self._actual_iterations)

    def _curriculum_observable_state(self, *, phase: Any) -> TrainerObservableState:
        snapshots: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for curriculum_id, runtime in sorted(self.curriculum_runtimes.items()):
            state = runtime.get_current_state()
            descriptor = self.registry.curriculum(curriculum_id)
            snapshots[curriculum_id] = state.to_dict()
            diagnostics[curriculum_id] = {
                "level_ids": tuple(descriptor.level_ids),
                "axis_role": descriptor.axis_role,
                "associated_component_ids": tuple(
                    descriptor.associated_loss_component_ids
                ),
                "associated_sampler_ids": tuple(descriptor.associated_sampler_ids),
                "critical_component_ids": tuple(descriptor.critical_component_ids),
                "force_advance_supported": bool(
                    descriptor.force_advance_supported
                ),
                "probe_comparability_mode": descriptor.metadata.get(
                    "probe_comparability_mode", "global_fixed_probe"
                ),
            }
        total_budget = sum(
            int(item.iterations) for item in self.built.optimization_phases
        )
        remaining = max(0, total_budget - int(self._actual_iterations))
        coordinator_active = bool(
            self.controller_coordinator is not None
            and self.controller_coordinator.observation_window_active(
                self._actual_iterations
            )
        )
        return TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(getattr(phase, "optimizer_name", "")),
            remaining_iteration_budget=remaining,
            stage_descriptor=self._stage_descriptor(phase),
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
            curriculum_states=snapshots,
            curriculum_diagnostics=diagnostics,
            curriculum_budget_state={
                "total_iteration_budget": total_budget,
                "consumed_iteration_budget": int(self._actual_iterations),
                "remaining_iteration_budget": remaining,
            },
            pending_rollback=bool(
                self.rollback_manager is not None
                and self.rollback_manager.active_checkpoint_id is not None
            ),
            coordinator_observation_active=coordinator_active,
        )

    def _observe_curriculum_control(self, *, phase: Any) -> None:
        controller = self.curriculum_advance_controller
        if (
            controller is None
            or self.registry is None
            or not self.curriculum_runtimes
            or self._curriculum_resume_disabled
        ):
            return
        preliminary = self._curriculum_observable_state(phase=phase)
        due, reason = controller.diagnostic_due(preliminary)
        if not due:
            if reason not in {"validation_interval_not_reached"}:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="curriculum_diagnostic_skipped",
                    trigger={"reason": reason},
                )
            return
        manager = self.validation_probe_manager
        if (
            manager is not None
            and (
                not manager.history
                or int(manager.history[-1].get("iteration", -1))
                != self._actual_iterations
            )
        ):
            started = time.perf_counter()
            self._maybe_run_physics_validation(phase=phase, force=True)
            self.curriculum_control_summary["curriculum_validation_cost"] += (
                time.perf_counter() - started
            )
        state = self._curriculum_observable_state(phase=phase)
        self.curriculum_control_summary["curriculum_diagnostic_checks"] += 1
        observation = (
            self.controller_coordinator.observe_controller(controller, state)
            if self.controller_coordinator is not None
            else controller.observe(state)
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="curriculum_diagnostic",
            trigger=dict(observation),
        )
        action = controller.propose_action()
        if action is None:
            return
        self.curriculum_control_summary["curriculum_advances_proposed"] += 1
        selected = action
        rejected: dict[str, str] = {}
        if self.controller_coordinator is not None:
            selected, rejected = self.controller_coordinator.select_action(
                (action,), iteration=self._actual_iterations
            )
        if selected is None:
            self.curriculum_control_summary["curriculum_advances_rejected"] += 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="curriculum_action_rejected",
                trigger={
                    "reason": rejected.get(action.action_id, "coordinator_rejected")
                },
                before=dict(action.before),
                after=dict(action.after),
            )
            return
        try:
            self.apply_validated_action(action, observable_state=state)
        except Exception as exc:
            self.curriculum_control_summary["curriculum_advances_rejected"] += 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="curriculum_action_failed",
                trigger={"error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
            )
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action, iteration=self._actual_iterations, outcome="failed"
                )

    def _execute_curriculum_advance_action(
        self,
        action: AdvanceCurriculumStateAction,
        observable_state: TrainerObservableState,
    ) -> Any:
        runtime = self.curriculum_runtimes.get(action.curriculum_id)
        if runtime is None:
            raise ValueError("Curriculum Runtime is not registered")
        before_state = runtime.get_current_state()
        if (
            before_state.runtime_state_version != action.runtime_state_version
            or before_state.state_digest != action.runtime_state_digest
        ):
            raise ValueError("Curriculum Runtime changed after boundary validation")
        stage = self._stage_descriptor(self._last_training_phase)
        if not runtime.can_advance(action.to_level_index, stage):
            raise ValueError("Curriculum Runtime rejected the adjacent target level")
        baseline_result = getattr(
            self.validation_probe_manager, "last_result", None
        )
        if baseline_result is None:
            raise ValueError("Curriculum advance requires a fixed physics baseline")
        checkpoint_id = self._save_pre_action_checkpoint(
            action,
            float(self.history[-1]["total_loss"])
            if self.history and self.history[-1].get("total_loss") is not None
            else None,
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(self._last_training_phase.name),
            controller=action.controller,
            event_type="curriculum_action_started",
            trigger=dict(action.trigger_statistics),
            before=dict(action.before),
            after=dict(action.after),
            checkpoint_id=checkpoint_id,
        )
        registry_components = tuple(sorted(self.registry.component_ids))
        registry_samplers = tuple(sorted(
            item.sampler_id for item in self.registry.sampling_components
        ))
        model_structure = tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in self.built.model.state_dict().items()
        )
        spec_id = self._algorithm_spec_id
        unaffected_sampler_ids = set(registry_samplers) - set(
            action.associated_sampler_ids
        )
        sampler_versions_before = {
            sampler_id: int(value)
            for sampler_id, value in dict(
                getattr(self.built.sampler, "_sampling_state_versions", {}) or {}
            ).items()
            if sampler_id in unaffected_sampler_ids
        }
        descriptor = self.registry.curriculum(action.curriculum_id)
        try:
            preview = runtime.preview_advance(action.to_level_index)
            if (
                preview.curriculum_id != action.curriculum_id
                or preview.from_level_index != action.from_level_index
                or preview.to_level_index != action.to_level_index
                or tuple(preview.structural_component_ids) != registry_components
                or tuple(preview.structural_sampler_ids) != registry_samplers
                or tuple(preview.associated_loss_component_ids)
                != tuple(descriptor.associated_loss_component_ids)
                or tuple(preview.associated_sampler_ids)
                != tuple(descriptor.associated_sampler_ids)
                or not preview.fixed_point_budget
            ):
                raise ValueError("Curriculum preview violates structural invariants")
            result = runtime.apply_advance(
                action.to_level_index,
                iteration=self._actual_iterations,
                forced=action.forced,
            )
            after_state = runtime.get_current_state()
            if (
                result.curriculum_id != action.curriculum_id
                or result.from_level_index != action.from_level_index
                or result.to_level_index != action.to_level_index
                or after_state.current_level_index != action.to_level_index
                or after_state.current_level_id != action.to_level_id
                or result.runtime_state_version != after_state.runtime_state_version
                or result.state_digest != after_state.state_digest
                or not result.fixed_point_budget_preserved
                or not runtime.validate_runtime_state()
            ):
                raise ValueError("Curriculum Runtime returned an inconsistent transition")
            if (
                tuple(sorted(self.registry.component_ids)) != registry_components
                or tuple(sorted(
                    item.sampler_id for item in self.registry.sampling_components
                )) != registry_samplers
                or tuple(
                    (name, tuple(value.shape), str(value.dtype))
                    for name, value in self.built.model.state_dict().items()
                ) != model_structure
                or hashlib.sha256(
                    json.dumps(
                        self.built.normalized_spec,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()[:20] != spec_id
                or {
                    sampler_id: int(value)
                    for sampler_id, value in dict(
                        getattr(
                            self.built.sampler, "_sampling_state_versions", {}
                        )
                        or {}
                    ).items()
                    if sampler_id in unaffected_sampler_ids
                }
                != sampler_versions_before
            ):
                raise ValueError("Curriculum advance changed structural training state")
        except Exception:
            if self.rollback_manager is not None and self.rollback_manager.active_checkpoint_id:
                self._rollback_active_action(
                    phase=self._last_training_phase,
                    reason="curriculum_runtime_transition_failed",
                )
            raise

        controller = self.curriculum_advance_controller
        controller.mark_action_applied(
            self._actual_iterations, forced=action.forced
        )
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.on_curriculum_advanced(
                self._actual_iterations,
                float(controller.config["post_advance_controller_ema_decay"]),
            )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.on_curriculum_advanced(
                self._actual_iterations, action.associated_sampler_ids
            )
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action, iteration=self._actual_iterations, outcome="applied"
            )
            self.controller_coordinator.extend_observation_window(
                self._actual_iterations
                + int(action.limits["observation_window_iterations"])
            )
        self.curriculum_control_summary["curriculum_advances_executed"] += 1
        self.curriculum_control_summary["curriculum_advances_forced"] += int(
            action.forced
        )
        self.curriculum_control_summary["curriculum_runtime_refresh_cost"] += float(
            result.refresh_cost
        )
        per_level = self.curriculum_control_summary.setdefault(
            "iterations_per_level", {}
        ).setdefault(action.curriculum_id, {})
        per_level[action.from_level_id] = int(
            before_state.iterations_in_current_level
        )
        baseline = baseline_result.to_dict()
        self._pending_curriculum_observation = {
            "action": action,
            "curriculum_id": action.curriculum_id,
            "start_iteration": int(self._actual_iterations),
            "target_level_index": int(action.to_level_index),
            "target_runtime_state_version": int(result.runtime_state_version),
            "baseline": baseline,
            "validation_checks": 0,
            "degradation_checks": 0,
            "observation_window_iterations": int(
                action.limits["observation_window_iterations"]
            ),
            "checkpoint_id": checkpoint_id,
            "forced": bool(action.forced),
        }
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(self._last_training_phase.name),
            controller=action.controller,
            event_type="action_applied",
            trigger=dict(action.trigger_statistics),
            before={
                "curriculum_id": action.curriculum_id,
                "level": {
                    "index": action.from_level_index,
                    "id": action.from_level_id,
                },
                "runtime_state": {
                    "version": action.runtime_state_version,
                    "digest": action.runtime_state_digest,
                },
            },
            after={
                "level": {
                    "index": action.to_level_index,
                    "id": action.to_level_id,
                },
                "runtime_state": {
                    "version": result.runtime_state_version,
                    "digest": result.state_digest,
                },
                "forced": action.forced,
                "associated_component_ids": list(action.associated_component_ids),
                "associated_sampler_ids": list(action.associated_sampler_ids),
            },
            limits=dict(action.limits),
            checkpoint_id=checkpoint_id,
        )
        return result

    def _assess_pending_curriculum_action(
        self, result: Any, phase: Any | None
    ) -> bool:
        pending = self._pending_curriculum_observation
        controller = self.curriculum_advance_controller
        if pending is None or controller is None:
            return False
        runtime = self.curriculum_runtimes.get(pending["curriculum_id"])
        runtime_state = runtime.get_current_state() if runtime is not None else None
        integrity_bad = bool(
            runtime is None
            or runtime_state.current_level_index != pending["target_level_index"]
            or runtime_state.runtime_state_version
            != pending["target_runtime_state_version"]
            or not runtime.validate_runtime_state()
        )
        baseline = dict(pending["baseline"])
        pending["validation_checks"] = int(pending["validation_checks"]) + 1
        finite = math.isfinite(float(result.aggregate_score)) and all(
            math.isfinite(float(value))
            for value in result.normalized_component_scores.values()
        )
        aggregate_bad = bool(
            finite
            and float(result.aggregate_score)
            > float(baseline["aggregate_score"])
            * float(controller.config["aggregate_degradation_ratio"])
        )
        action = pending["action"]
        baseline_components = dict(
            baseline.get("normalized_component_scores") or {}
        )
        associated_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["associated_component_degradation_ratio"])
            for component_id in action.associated_component_ids
        )
        descriptor = self.registry.curriculum(action.curriculum_id)
        critical_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["critical_component_degradation_ratio"])
            for component_id in descriptor.critical_component_ids
        )
        non_finite_bad = bool(
            not finite and controller.config["rollback_on_non_finite"]
        )
        degraded = aggregate_bad or associated_bad or critical_bad or non_finite_bad
        pending["degradation_checks"] = (
            int(pending["degradation_checks"]) + 1 if degraded else 0
        )
        minimum_checks = int(controller.config["minimum_validation_checks"])
        if integrity_bad or non_finite_bad:
            self._rollback_active_action(
                phase=phase or self._last_training_phase,
                reason=(
                    "curriculum_runtime_state_inconsistent"
                    if integrity_bad
                    else "curriculum_non_finite_emergency"
                ),
            )
            return False
        if int(pending["degradation_checks"]) >= minimum_checks:
            if self.rollback_manager is not None:
                self.rollback_manager.clear()
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(getattr(phase, "name", "training")),
                controller=controller.name,
                event_type="curriculum_delayed_feedback",
                trigger={
                    "outcome": "failure",
                    "reason": "curriculum_physics_validation_degradation",
                    "continued_without_rollback": True,
                },
                checkpoint_id=pending.get("checkpoint_id"),
            )
            self._pending_curriculum_observation = None
            return False
        complete = bool(
            self._actual_iterations - int(pending["start_iteration"])
            >= int(pending["observation_window_iterations"])
            and int(pending["validation_checks"]) >= minimum_checks
        )
        if not complete:
            return False
        if self.rollback_manager is not None:
            self.rollback_manager.clear()
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(getattr(phase, "name", "training")),
            controller="curriculum_advance",
            event_type="curriculum_observation_succeeded",
            trigger={
                "curriculum_id": pending["curriculum_id"],
                "validation_checks": pending["validation_checks"],
            },
            checkpoint_id=pending.get("checkpoint_id"),
        )
        self._pending_curriculum_observation = None
        return True

    def _finalize_curriculum_control_summary(self) -> dict[str, Any]:
        summary = deepcopy(self.curriculum_control_summary)
        per_level = summary.setdefault("iterations_per_level", {})
        unused: dict[str, list[str]] = {}
        for curriculum_id, runtime in sorted(self.curriculum_runtimes.items()):
            state = runtime.get_current_state()
            descriptor = self.registry.curriculum(curriculum_id)
            per_level.setdefault(curriculum_id, {})[
                state.current_level_id
            ] = int(state.iterations_in_current_level)
            unused[curriculum_id] = list(
                descriptor.level_ids[state.current_level_index + 1 :]
            )
        summary["unused_levels"] = unused
        return summary

    def _configure_training_stage(
        self,
        *,
        phase: Any,
        non_lbfgs_iteration: int,
    ) -> dict[str, Any]:
        """Apply component-category pretraining and role-based curriculum."""

        training = self.built.normalized_spec.get("training") or {}
        sampler = getattr(self.built, "sampler", None)
        loss_function = self.built.loss_function
        descriptor = self._stage_descriptor(phase)
        runtime_curriculum_active = bool(
            self.curriculum_runtimes
            and self.curriculum_advance_controller is not None
            and not self._curriculum_resume_disabled
        )
        if not descriptor.dynamic_objective_allowed:
            if (
                not runtime_curriculum_active
                and callable(getattr(sampler, "set_time_window_fraction", None))
            ):
                sampler.set_time_window_fraction(1.0)
            if callable(getattr(loss_function, "set_active_term_names", None)):
                loss_function.set_active_term_names(None)
            stage = {
                "name": (
                    "frozen_curriculum_state"
                    if runtime_curriculum_active
                    else "frozen_objective_full_domain"
                ),
                "time_window_fraction": float(
                    getattr(sampler, "time_window_fraction", 1.0)
                ),
                "initial_state_only": False,
            }
        else:
            pretraining = training.get("initial_state_pretraining") or {}
            pretraining_iterations = (
                int((pretraining.get("parameters") or {}).get("iterations") or 0)
                if pretraining.get("enabled")
                else 0
            )
            initial_only = non_lbfgs_iteration <= pretraining_iterations
            if callable(getattr(loss_function, "set_active_term_names", None)):
                pretraining_component_ids = {
                    component.component_id
                    for component in (
                        self.registry.loss_components
                        if self.registry is not None
                        else ()
                    )
                    if component.category == "initial_constraint"
                }
                loss_function.set_active_term_names(
                    pretraining_component_ids
                    if initial_only and pretraining_component_ids
                    else None
                )
            strategy = training.get("time_strategy") or {
                "name": "none",
                "parameters": {},
            }
            fraction = float(getattr(sampler, "time_window_fraction", 1.0))
            if not runtime_curriculum_active:
                fraction = 1.0
            if (
                not runtime_curriculum_active
                and strategy.get("name") == "time_window_curriculum"
            ):
                parameters = strategy.get("parameters") or {}
                fractions = [
                    float(value)
                    for value in parameters.get("window_fractions") or [1.0]
                ]
                iterations = [
                    int(value)
                    for value in parameters.get("iterations_per_window")
                    or [10**18]
                ]
                cursor = 0
                fraction = fractions[-1]
                for candidate_fraction, count in zip(fractions, iterations):
                    cursor += count
                    if non_lbfgs_iteration <= cursor:
                        fraction = candidate_fraction
                        break
            if (
                not runtime_curriculum_active
                and callable(getattr(sampler, "set_time_window_fraction", None))
            ):
                sampler.set_time_window_fraction(fraction)
            stage = {
                "name": (
                    "initial_state_pretraining"
                    if initial_only
                    else "runtime_curriculum"
                    if runtime_curriculum_active
                    else str(strategy.get("name") or "full_domain")
                ),
                "time_window_fraction": float(fraction),
                "initial_state_only": bool(initial_only),
            }
        self.time_strategy_audit = {
            "configured_time_strategy": dict(
                training.get("time_strategy") or {"name": "none", "parameters": {}}
            ),
            "configured_initial_state_pretraining": dict(
                training.get("initial_state_pretraining")
                or {"enabled": False, "parameters": {}}
            ),
            "window_history": list(
                getattr(sampler, "time_window_history", []) or []
            ),
            "latest_stage": dict(stage),
            "frozen_objective_full_domain": not descriptor.dynamic_objective_allowed,
            # Retained for schema/report compatibility with Trainer Controller
            # v1; the decision itself is now stage-capability driven.
            "lbfgs_full_domain": bool(
                descriptor.second_order and not descriptor.dynamic_objective_allowed
            ),
        }
        return stage

    def _run_lbfgs(
        self,
        phase: Any,
        budget: int,
        clipping_norm: float | None,
        clipping_norm_type: float,
    ) -> tuple[float | None, int, str | None, dict[str, Any]]:
        """Run L-BFGS in persistent chunks until its internal-iteration budget is met.

        ``phase.iterations`` denotes accepted PyTorch L-BFGS internal iterations,
        matching Adam's optimizer-iteration accounting.  Closure evaluations are
        audited separately because Strong-Wolfe can evaluate several trial points
        per internal iteration.  One full-batch sample is frozen for the complete
        phase so every line-search closure evaluates the same objective.
        """

        completed_iterations = 0
        closure_evaluations = 0
        accepted_loss_evaluations = 0
        internal_iterations = 0
        outer_calls = 0
        stalled_calls = 0
        consecutive_stalls = 0
        state_restarts = 0
        latest: float | None = None
        failure_reason: str | None = None
        termination_reason = "budget_fulfilled"
        converged_early = False
        controlled_stall = False
        recovery_adam_iterations = 0
        recovery_action: str | None = None
        stall_iteration: int | None = None
        physics_guard_triggered = False
        physics_guard_baseline_score: float | None = None
        physics_guard_rejected_score: float | None = None
        physics_guard_baseline_groups: dict[str, float] = {}
        physics_guard_rejected_groups: dict[str, dict[str, float]] = {}
        physics_guard_metric = str(
            getattr(
                self.validation_probe_manager,
                "second_order_guard_metric",
                "physics_validation_score",
            )
        )
        physics_guard_ratio = float(
            getattr(
                self.validation_probe_manager,
                "second_order_degradation_ratio",
                1.0,
            )
        )
        physics_guard_baseline = getattr(
            self.validation_probe_manager, "guard_result", None
        )
        guard_score = getattr(
            self.validation_probe_manager,
            "second_order_guard_score",
            None,
        )
        if physics_guard_baseline is not None:
            physics_guard_baseline_score = (
                guard_score(physics_guard_baseline)
                if callable(guard_score)
                else float(physics_guard_baseline.aggregate_score)
            )
            physics_guard_baseline_groups = {
                str(group): float(score)
                for group, score in dict(
                    getattr(physics_guard_baseline, "category_scores", {}) or {}
                ).items()
                if math.isfinite(float(score))
            }
        physics_guard_model_state = (
            [
                parameter.detach().clone()
                for parameter in self.built.model.parameters()
            ]
            if physics_guard_ratio > 1.0
            and physics_guard_baseline_score is not None
            else None
        )
        best_train_checkpoint_manager = (
            self.best_training_loss_checkpoint_manager
        )
        physics_guard_best_train_checkpoint_state = (
            best_train_checkpoint_manager.state_dict()
            if physics_guard_model_state is not None
            and best_train_checkpoint_manager is not None
            else None
        )
        parameters = dict(getattr(phase, "parameters", {}) or {})
        first_group = phase.optimizer.param_groups[0]
        configured_max_iter = max(
            1, int(parameters.get("max_iter") or first_group.get("max_iter") or 20)
        )
        configured_max_eval = max(
            1, int(parameters.get("max_eval") or first_group.get("max_eval") or 25)
        )
        tolerance_grad = float(
            parameters.get("tolerance_grad")
            or first_group.get("tolerance_grad")
            or 1e-7
        )
        original_learning_rates = [float(group.get("lr", 1.0)) for group in phase.optimizer.param_groups]
        line_search_fn = first_group.get("line_search_fn")
        sampler = getattr(self.built, "sampler", None)
        phase_batch = sampler.sample() if sampler is not None else None
        frozen_loss = {}
        freeze = getattr(self.built.loss_function, "freeze_dynamic_weighting", None)
        if callable(freeze):
            frozen_loss = dict(freeze())
        registered_sampling_state = (
            sampler.snapshot_registered_components(self.registry, phase_batch)
            if sampler is not None
            and self.registry is not None
            and phase_batch is not None
            else {}
        )
        frozen_objective = {
            **frozen_loss,
            "frozen_sampling_components": registered_sampling_state,
            "frozen_sampling_ratios": dict(
                getattr(sampler, "last_sampling_snapshot", {}) or {}
            ),
            "frozen_time_window": float(
                getattr(sampler, "time_window_fraction", 1.0)
            ),
            "frozen_constraint_state": deepcopy(
                getattr(sampler, "_constraint_cache", {})
            ),
        }
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage="lbfgs",
            controller="trainer_boundary",
            event_type="objective_frozen",
            trigger={"reason": "lbfgs_phase_entry"},
            after={
                "loss_weights": frozen_objective.get("frozen_loss_weights", {}),
                "loss_terms": frozen_objective.get("frozen_loss_terms", []),
                "time_window": frozen_objective["frozen_time_window"],
                "sampling_snapshot": frozen_objective["frozen_sampling_ratios"],
                "registered_sampling_component_ids": sorted(
                    registered_sampling_state
                ),
            },
        )
        self._last_training_batch = phase_batch
        self._last_training_phase = phase
        last_maximum_gradient: float | None = None
        last_step_size: float | None = None
        max_state_restarts = max(
            8,
            int(
                getattr(self.lbfgs_stall_controller, "config", {}).get(
                    "patience", 0
                )
            )
            + 2,
        )
        restart_after_stalls = 2
        max_outer_calls = max(16, int(budget) * 2 + 16)
        first_order_recovery_available = _has_first_order_recovery_phase(
            self.built.optimization_phases
        )

        def reallocate_non_finite_step(
            reason: str,
            parameter_values: list[torch.Tensor],
            loss_state: dict[str, Any],
            *,
            rejected_iterations: int = 0,
        ) -> bool:
            nonlocal completed_iterations, internal_iterations
            nonlocal failure_reason, termination_reason, controlled_stall
            nonlocal recovery_adam_iterations, recovery_action, stall_iteration
            if not first_order_recovery_available:
                return False
            if rejected_iterations > 0:
                completed_iterations = max(
                    0, completed_iterations - int(rejected_iterations)
                )
                internal_iterations = max(
                    0, internal_iterations - int(rejected_iterations)
                )
            with torch.no_grad():
                for parameter, previous in zip(
                    self.built.model.parameters(), parameter_values
                ):
                    parameter.copy_(previous.to(parameter.device))
            self._restore_lbfgs_loss_state(loss_state)
            self._reset_lbfgs_state(phase.optimizer)
            failure_reason = None
            controlled_stall = True
            recovery_action = "reallocate_to_adam_after_non_finite_lbfgs"
            recovery_adam_iterations = max(
                0, int(budget) - int(completed_iterations)
            )
            stall_iteration = int(self._actual_iterations)
            termination_reason = f"{reason}_reallocated_to_adam"
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage="lbfgs",
                controller="lbfgs_non_finite_guard",
                event_type="budget_reallocated",
                trigger={"reason": reason},
                before={
                    "completed_lbfgs_iterations": int(completed_iterations),
                    "rejected_lbfgs_iterations": int(rejected_iterations),
                },
                after={
                    "recovery_adam_iterations": int(
                        recovery_adam_iterations
                    ),
                    "parameters_restored": True,
                },
            )
            return True

        while completed_iterations < budget and outer_calls < max_outer_calls:
            remaining_iterations = budget - completed_iterations
            call_max_iter = min(configured_max_iter, remaining_iterations)
            validation_manager = self.validation_probe_manager
            if (
                validation_manager is not None
                and validation_manager.probe_set is not None
                and validation_manager.history
            ):
                next_validation = validation_manager.next_due_iteration(
                    self._actual_iterations
                )
                if next_validation > self._actual_iterations:
                    call_max_iter = min(
                        call_max_iter,
                        max(1, next_validation - self._actual_iterations),
                    )
            call_max_eval = max(
                1,
                min(
                    configured_max_eval,
                    math.ceil(
                        configured_max_eval * call_max_iter / configured_max_iter
                    ),
                ),
            )
            for group in phase.optimizer.param_groups:
                group["max_iter"] = call_max_iter
                group["max_eval"] = call_max_eval

            state_before = self._lbfgs_optimizer_state(phase.optimizer)
            n_iter_before = int(state_before.get("n_iter", 0)) if state_before else 0
            closure_calls_this_step = 0
            last_maximum_gradient = None
            loss_state_before = self._snapshot_lbfgs_loss_state()
            parameter_values_before = [
                parameter.detach().clone()
                for parameter in self.built.model.parameters()
            ]

            def closure():
                nonlocal closure_evaluations, closure_calls_this_step
                nonlocal failure_reason, last_maximum_gradient
                phase.optimizer.zero_grad(set_to_none=True)
                self._restore_lbfgs_loss_state(loss_state_before)
                self.built.loss_function.capture_residual_statistics = False
                loss, _ = self.built.loss_function(self.built.model, phase_batch)
                if not torch.isfinite(loss):
                    failure_reason = "non_finite_loss"
                    self._release_iteration_graphs()
                    raise _StopLBFGS(failure_reason)
                loss.backward()
                if clipping_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.built.model.parameters(),
                        clipping_norm,
                        norm_type=clipping_norm_type,
                    )
                if not self._gradients_finite():
                    failure_reason = "non_finite_gradient"
                    self._release_iteration_graphs()
                    raise _StopLBFGS(failure_reason)
                last_maximum_gradient = self._maximum_absolute_gradient()
                closure_evaluations += 1
                closure_calls_this_step += 1
                self._release_iteration_graphs()
                return loss

            outer_calls += 1
            try:
                phase.optimizer.step(closure)
            except _StopLBFGS:
                reason = str(failure_reason or "non_finite_lbfgs_step")
                reallocate_non_finite_step(
                    reason,
                    parameter_values_before,
                    loss_state_before,
                )
                break

            if not self._parameters_finite():
                failure_reason = "non_finite_parameter"
                reallocate_non_finite_step(
                    failure_reason,
                    parameter_values_before,
                    loss_state_before,
                )
                break

            state_after = self._lbfgs_optimizer_state(phase.optimizer)
            if state_after is None:
                # Minimal optimizer doubles used by unit tests have no PyTorch
                # state.  A successful step is one optimizer iteration there.
                iteration_delta = 1 if closure_calls_this_step else 0
                last_step_size = None
            else:
                n_iter_after = int(state_after.get("n_iter", n_iter_before))
                iteration_delta = max(0, n_iter_after - n_iter_before)
                raw_step_size = state_after.get("t")
                try:
                    last_step_size = (
                        float(raw_step_size.detach().cpu())
                        if isinstance(raw_step_size, torch.Tensor)
                        else float(raw_step_size)
                        if raw_step_size is not None
                        else None
                    )
                except (TypeError, ValueError):
                    last_step_size = None

            iteration_delta = min(iteration_delta, remaining_iterations)
            internal_iterations += iteration_delta
            completed_iterations += iteration_delta

            projected_iteration = self._actual_iterations + iteration_delta
            should_record = iteration_delta > 0 and self._should_record_history(
                projected_iteration,
                phase.name,
            )
            self._restore_lbfgs_loss_state(loss_state_before)
            self.built.loss_function.capture_residual_statistics = should_record
            phase.optimizer.zero_grad(set_to_none=True)
            accepted_loss, accepted_components = self.built.loss_function(
                self.built.model, phase_batch
            )
            accepted_loss_evaluations += 1
            if not torch.isfinite(accepted_loss):
                failure_reason = "non_finite_accepted_loss"
                reallocate_non_finite_step(
                    failure_reason,
                    parameter_values_before,
                    loss_state_before,
                    rejected_iterations=iteration_delta,
                )
                self._release_iteration_graphs()
                break
            latest = float(accepted_loss.detach().cpu())
            parameter_update_norm = self._parameter_update_norm(
                parameter_values_before
            )
            if iteration_delta > 0:
                self._actual_iterations += iteration_delta
                if should_record:
                    gradient_statistics = None
                    if self._gradient_diagnostic_due(self._actual_iterations):
                        gradient_statistics = (
                            self._collect_component_gradient_diagnostic(
                                total_loss=accepted_loss,
                                iteration=self._actual_iterations,
                                phase=phase,
                                checkpoint_kind=self._gradient_checkpoint_kind(
                                    self._actual_iterations
                                ),
                            )
                        )
                    self.history.append(
                        self._history_entry(
                            iteration=self._actual_iterations,
                            phase=phase,
                            total_loss=latest,
                            components=accepted_components,
                            batch=phase_batch,
                            gradient_norm=None,
                            gradient_statistics=gradient_statistics,
                            adaptive_refinement_event=False,
                        )
                    )
                    self._history_phase_names_recorded.add(str(phase.name))
            self._release_iteration_graphs()
            if iteration_delta > 0:
                self._maybe_run_physics_validation(
                    phase=phase,
                    # A short L-BFGS phase may end before the next periodic
                    # probe. Always evaluate its terminal state so the
                    # label-free degradation guard cannot be skipped.
                    force=completed_iterations >= budget,
                    training_loss=latest,
                )
                latest_validation = getattr(
                    self.validation_probe_manager, "guard_result", None
                )
                validation_history = list(
                    getattr(
                        self.validation_probe_manager, "guard_history", []
                    )
                    or []
                )
                validation_is_current = bool(
                    validation_history
                    and int(
                        validation_history[-1].get("iteration", -1)
                    )
                    == int(self._actual_iterations)
                )
                current_guard_score = (
                    guard_score(latest_validation)
                    if callable(guard_score)
                    else float(latest_validation.aggregate_score)
                    if latest_validation is not None
                    else None
                )
                degradation_limit = (
                    float(physics_guard_baseline_score)
                    + (physics_guard_ratio - 1.0)
                    * max(abs(float(physics_guard_baseline_score)), 1.0e-12)
                    if physics_guard_baseline_score is not None
                    else None
                )
                degraded_groups = _physics_group_degradations(
                    physics_guard_baseline_groups,
                    getattr(latest_validation, "category_scores", None),
                    maximum_ratio=physics_guard_ratio,
                )
                aggregate_degraded = bool(
                    current_guard_score is not None
                    and degradation_limit is not None
                    and float(current_guard_score) > degradation_limit
                )
                if (
                    physics_guard_model_state is not None
                    and validation_is_current
                    and (aggregate_degraded or bool(degraded_groups))
                ):
                    physics_guard_triggered = True
                    physics_guard_rejected_score = (
                        float(current_guard_score)
                        if current_guard_score is not None
                        else None
                    )
                    physics_guard_rejected_groups = dict(degraded_groups)
                    with torch.no_grad():
                        for parameter, previous in zip(
                            self.built.model.parameters(),
                            physics_guard_model_state,
                        ):
                            parameter.copy_(
                                previous.to(parameter.device)
                            )
                    if (
                        best_train_checkpoint_manager is not None
                        and physics_guard_best_train_checkpoint_state
                        is not None
                    ):
                        # A rejected L-BFGS state may already have replaced
                        # either the best training-loss checkpoint or the best
                        # independent-physics checkpoint. Restore the manager
                        # and its on-disk artifact alongside the guarded model
                        # so final selection cannot reload rejected weights.
                        best_train_checkpoint_manager.load_state_dict(
                            physics_guard_best_train_checkpoint_state
                        )
                        if self.checkpoint_dir is not None:
                            best_train_checkpoint_manager.save_checkpoint(
                                self.checkpoint_dir
                            )
                        if (
                            best_train_checkpoint_manager.final_model_policy
                            == "best_train_loss"
                            and best_train_checkpoint_manager.best_score
                            is not None
                            and math.isfinite(
                                float(
                                    best_train_checkpoint_manager.best_score
                                )
                            )
                        ):
                            # Keep the returned/final training loss aligned
                            # with the restored model. Otherwise the terminal
                            # bookkeeping pass could pair the rejected
                            # L-BFGS loss with the guarded weights and create
                            # a misleading new best checkpoint.
                            latest = float(
                                best_train_checkpoint_manager.best_score
                            )
                    self._reset_lbfgs_state(phase.optimizer)
                    controlled_stall = True
                    if first_order_recovery_available:
                        recovery_action = (
                            "bounded_recovery_adam_then_return_to_lbfgs_after_physics_degradation"
                        )
                        remaining_after_guard = max(
                            0,
                            int(budget) - int(completed_iterations),
                        )
                        recovery_adam_iterations = min(
                            remaining_after_guard,
                            max(
                                0,
                                int(
                                    getattr(
                                        self,
                                        "second_order_guard_recovery_iterations",
                                        500,
                                    )
                                ),
                            ),
                        )
                    else:
                        recovery_action = (
                            "terminate_lbfgs_after_physics_degradation"
                        )
                        recovery_adam_iterations = 0
                    stall_iteration = int(self._actual_iterations)
                    termination_reason = (
                        "physics_validation_degradation_bounded_adam_recovery"
                        if first_order_recovery_available
                        else "physics_validation_degradation_terminated_lbfgs"
                    )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller="lbfgs_physics_guard",
                        event_type=(
                            "budget_reallocated"
                            if first_order_recovery_available
                            else "stage_terminated"
                        ),
                        trigger={
                            "metric": physics_guard_metric,
                            "baseline_score": (
                                physics_guard_baseline_score
                            ),
                            "rejected_score": (
                                physics_guard_rejected_score
                            ),
                            "maximum_ratio": physics_guard_ratio,
                            "aggregate_degraded": aggregate_degraded,
                            "baseline_group_scores": dict(
                                physics_guard_baseline_groups
                            ),
                            "rejected_group_scores": dict(
                                getattr(
                                    latest_validation,
                                    "category_scores",
                                    {},
                                )
                                or {}
                            ),
                            "degraded_groups": dict(
                                physics_guard_rejected_groups
                            ),
                        },
                        after={
                            "model_restored_to": "pre_second_order",
                            "recovery_adam_iterations": int(
                                recovery_adam_iterations
                            ),
                            "return_to_lbfgs": bool(
                                first_order_recovery_available
                                and recovery_adam_iterations > 0
                            ),
                            "remaining_lbfgs_iterations_after_recovery": max(
                                0,
                                int(budget)
                                - int(completed_iterations)
                                - int(recovery_adam_iterations),
                            ),
                        },
                    )
                    break
                if (
                    physics_guard_model_state is not None
                    and current_guard_score is not None
                    and validation_is_current
                    and physics_guard_baseline_score is not None
                    and float(current_guard_score)
                    < float(physics_guard_baseline_score)
                ):
                    physics_guard_baseline_score = float(current_guard_score)
                    physics_guard_baseline_groups = {
                        str(group): float(score)
                        for group, score in dict(
                            getattr(
                                latest_validation,
                                "category_scores",
                                {},
                            )
                            or {}
                        ).items()
                        if math.isfinite(float(score))
                    }
                    physics_guard_model_state = [
                        parameter.detach().clone()
                        for parameter in self.built.model.parameters()
                    ]
                    if best_train_checkpoint_manager is not None:
                        physics_guard_best_train_checkpoint_state = (
                            best_train_checkpoint_manager.state_dict()
                        )

            if self.lbfgs_stall_controller is not None:
                state = TrainerObservableState(
                    iteration=int(self._actual_iterations),
                    stage=str(phase.name),
                    optimizer_name=str(phase.optimizer_name),
                    current_loss=latest,
                    loss_components=dict(accepted_components),
                    loss_weights=dict(
                        getattr(
                            self.built.loss_function,
                            "last_effective_loss_weights",
                            {},
                        )
                    ),
                    learning_rates=tuple(
                        float(group.get("lr", 0.0))
                        for group in phase.optimizer.param_groups
                    ),
                    parameter_update_norm=parameter_update_norm,
                    gradient_norm=last_maximum_gradient,
                    lbfgs_outer_step=outer_calls,
                    closure_evaluations=closure_evaluations,
                    closure_evaluations_this_step=closure_calls_this_step,
                    step_size=last_step_size,
                    line_search_failed=iteration_delta == 0,
                    completed_stage_iterations=completed_iterations,
                    remaining_iteration_budget=max(
                        0, budget - completed_iterations
                    ),
                    sampling_state=dict(
                        getattr(sampler, "last_sampling_snapshot", {}) or {}
                    ),
                    time_window_fraction=float(
                        getattr(sampler, "time_window_fraction", 1.0)
                    ),
                    numerical_finite=True,
                    stage_descriptor=self._stage_descriptor(phase),
                    capability_snapshot=(
                        self.registry.capabilities
                        if self.registry is not None
                        else None
                    ),
                    eligible_component_ids=(),
                    optimizer_diagnostics={
                        "outer_step": outer_calls,
                        "closure_evaluations": closure_evaluations,
                        "step_size": last_step_size,
                        "parameter_update_norm": parameter_update_norm,
                    },
                    budget_state={
                        "completed_stage_iterations": completed_iterations,
                        "remaining_stage_iterations": max(
                            0, budget - completed_iterations
                        ),
                    },
                    validation_state=getattr(
                        self.validation_probe_manager, "last_result", None
                    ),
                )
                if self.controller_coordinator is not None:
                    diagnostic = self.controller_coordinator.observe_controller(
                        self.lbfgs_stall_controller, state
                    )
                else:
                    diagnostic = self.lbfgs_stall_controller.observe(state)
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage="lbfgs",
                    controller="lbfgs_stall",
                    event_type="diagnostic",
                    trigger=diagnostic,
                )
                proposals = (
                    self.controller_coordinator.collect_proposals()
                    if self.controller_coordinator is not None
                    else tuple(
                        item
                        for item in (
                            self.lbfgs_stall_controller.propose_action(),
                        )
                        if item is not None
                    )
                )
                if self.controller_coordinator is not None:
                    action, rejected = self.controller_coordinator.select_action(
                        proposals, iteration=self._actual_iterations
                    )
                    for action_id, reason in rejected.items():
                        self._emit_adaptive_audit(
                            iteration=self._actual_iterations,
                            stage=str(phase.name),
                            controller="controller_coordinator",
                            event_type="action_rejected_by_coordinator",
                            trigger={"action_id": action_id, "reason": reason},
                        )
                else:
                    action = proposals[0] if proposals else None
                if action is not None:
                    if self.action_boundary_validator is not None and self.registry is not None:
                        self.action_boundary_validator.validate(
                            action=action,
                            registry=self.registry,
                            stage=self._stage_descriptor(phase),
                            observable_state=state,
                        )
                    checkpoint_id = self._save_pre_action_checkpoint(
                        action, latest
                    )
                    recovery_action = action.action_type
                    recovery_adam_iterations = int(
                        action.after.get("recovery_adam_iterations", 0)
                    )
                    controlled_stall = True
                    stall_iteration = int(self._actual_iterations)
                    termination_reason = "controlled_lbfgs_stall"
                    self.lbfgs_stall_controller.mark_action_applied(
                        self._actual_iterations
                    )
                    if self.controller_coordinator is not None:
                        self.controller_coordinator.record_outcome(
                            action,
                            iteration=self._actual_iterations,
                            outcome="applied",
                        )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller=action.controller,
                        event_type="lbfgs_stall",
                        trigger=dict(action.reason),
                        before=dict(action.before),
                        after=dict(action.after),
                        limits=dict(action.limits),
                        checkpoint_id=checkpoint_id,
                    )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller=action.controller,
                        event_type="budget_reallocated",
                        trigger={"action": recovery_action},
                        before={"remaining_lbfgs_iterations": budget - completed_iterations},
                        after={"recovery_adam_iterations": recovery_adam_iterations},
                        limits=dict(action.limits),
                        checkpoint_id=checkpoint_id,
                    )
                    break

            genuinely_converged = (
                last_maximum_gradient is not None
                and math.isfinite(last_maximum_gradient)
                and last_maximum_gradient <= tolerance_grad
            )
            zero_step = last_step_size is not None and abs(last_step_size) == 0.0
            stalled = iteration_delta == 0 or (zero_step and not genuinely_converged)
            if stalled:
                stalled_calls += 1
                consecutive_stalls += 1
            else:
                consecutive_stalls = 0

            if genuinely_converged and iteration_delta == 0:
                converged_early = True
                termination_reason = "gradient_tolerance_reached"
                break

            if consecutive_stalls >= restart_after_stalls:
                if state_restarts >= max_state_restarts:
                    if iteration_delta == 0:
                        failure_reason = (
                            "lbfgs_iteration_budget_unreachable: "
                            f"completed={completed_iterations}, requested={budget}"
                        )
                        termination_reason = "stall_recovery_exhausted"
                        break
                else:
                    state_restarts += 1
                    self._reset_lbfgs_state(phase.optimizer)
                    for index, group in enumerate(phase.optimizer.param_groups):
                        group["lr"] = max(
                            original_learning_rates[index] * (0.5**state_restarts),
                            1e-8,
                        )
                    consecutive_stalls = 0

        if completed_iterations < budget and failure_reason is None and not converged_early and not controlled_stall:
            failure_reason = (
                "lbfgs_iteration_budget_unreachable: "
                f"completed={completed_iterations}, requested={budget}"
            )
            termination_reason = "outer_call_limit_reached"
        elif failure_reason is not None and termination_reason == "budget_fulfilled":
            termination_reason = failure_reason

        for group in phase.optimizer.param_groups:
            group["max_iter"] = configured_max_iter
            group["max_eval"] = configured_max_eval

        audit = {
            "policy": "persistent_chunked_internal_iterations_v1",
            "configured_phase_iterations": int(budget),
            "completed_internal_iterations": int(completed_iterations),
            "outer_step_calls": int(outer_calls),
            "closure_evaluations": int(closure_evaluations),
            "accepted_loss_evaluations": int(accepted_loss_evaluations),
            "configured_max_iter_per_call": int(configured_max_iter),
            "configured_max_eval_per_call": int(configured_max_eval),
            "line_search_fn": line_search_fn,
            "fixed_full_batch_within_phase": True,
            "fixed_dynamic_loss_state_within_outer_step": True,
            "fixed_dynamic_loss_state_within_phase": True,
            "frozen_objective": {
                "loss_weights": frozen_objective.get("frozen_loss_weights", {}),
                "loss_terms": frozen_objective.get("frozen_loss_terms", []),
                "time_window": frozen_objective.get("frozen_time_window"),
                "sampling_snapshot": frozen_objective.get("frozen_sampling_ratios", {}),
                "registered_sampling_component_ids": sorted(
                    frozen_objective.get("frozen_sampling_components", {})
                ),
            },
            "stalled_calls": int(stalled_calls),
            "state_restarts": int(state_restarts),
            "last_step_size": last_step_size,
            "last_maximum_absolute_gradient": last_maximum_gradient,
            "converged_early": bool(converged_early),
            "budget_fulfilled": completed_iterations >= budget,
            "termination_reason": termination_reason,
            "controlled_stall_detected": controlled_stall,
            "stall_iteration": stall_iteration,
            "recovery_action": recovery_action,
            "recovery_adam_iterations": recovery_adam_iterations,
            "physics_guard_triggered": physics_guard_triggered,
            "physics_guard_metric": physics_guard_metric,
            "physics_guard_baseline_score": (
                physics_guard_baseline_score
            ),
            "physics_guard_rejected_score": (
                physics_guard_rejected_score
            ),
            "physics_guard_baseline_group_scores": dict(
                physics_guard_baseline_groups
            ),
            "physics_guard_rejected_groups": dict(
                physics_guard_rejected_groups
            ),
            "physics_guard_maximum_ratio": physics_guard_ratio,
            "skipped_stalled_iterations": max(
                0, int(budget) - int(completed_iterations)
            ),
            "final_learning_rates": [
                float(group.get("lr", 0.0)) for group in phase.optimizer.param_groups
            ],
        }
        return latest, completed_iterations, failure_reason, audit

    @staticmethod
    def _lbfgs_optimizer_state(optimizer: Any) -> dict[str, Any] | None:
        state = getattr(optimizer, "state", None)
        groups = getattr(optimizer, "param_groups", None) or []
        if not state or not groups:
            return None
        parameters = groups[0].get("params") or []
        if parameters and parameters[0] in state and isinstance(state[parameters[0]], dict):
            return state[parameters[0]]
        for value in state.values():
            if isinstance(value, dict) and ("n_iter" in value or "func_evals" in value):
                return value
        return None

    @staticmethod
    def _reset_lbfgs_state(optimizer: Any) -> None:
        state = getattr(optimizer, "state", None)
        if hasattr(state, "clear"):
            state.clear()

    def _snapshot_lbfgs_loss_state(self) -> dict[str, torch.Tensor | None]:
        loss_function = self.built.loss_function
        snapshot: dict[str, torch.Tensor | None] = {}
        for name in (
            "_adaptive_weights",
            "_ema_magnitudes",
            "_ntk_weights",
        ):
            value = getattr(loss_function, name, None)
            snapshot[name] = value.detach().clone() if isinstance(value, torch.Tensor) else None
        return snapshot

    def _restore_lbfgs_loss_state(
        self, snapshot: dict[str, torch.Tensor | None]
    ) -> None:
        loss_function = self.built.loss_function
        for name, value in snapshot.items():
            setattr(loss_function, name, value.detach().clone() if value is not None else None)

    def _configure_history_boundary_iterations(
        self,
        *,
        start_iteration: int,
        maximum_iterations: int,
    ) -> None:
        """Mark the first and final optimizer iterations in this train call."""

        cursor = int(start_iteration)
        remaining = max(0, int(maximum_iterations))
        boundaries: set[int] = set()
        for phase in self.built.optimization_phases:
            completed = int(self.completed_phase_iterations.get(phase.name, 0))
            available = max(0, int(phase.iterations) - completed)
            count = min(available, remaining)
            if count <= 0:
                continue
            boundaries.add(cursor + 1)
            boundaries.add(cursor + count)
            cursor += count
            remaining -= count
            if remaining <= 0:
                break
        self._history_boundary_iterations = boundaries

    def _should_record_history(
        self,
        iteration: int,
        phase_name: str | None = None,
        *,
        event: bool = False,
    ) -> bool:
        return (
            bool(event)
            or iteration == 1
            or iteration in self._history_boundary_iterations
            or (
                phase_name is not None
                and str(phase_name) not in self._history_phase_names_recorded
            )
            or iteration % self.history_interval == 0
        )

    def _release_iteration_graphs(self) -> None:
        """Drop graph references that must not survive an optimizer step."""

        loss_function = self.built.loss_function
        if hasattr(loss_function, "last_term_values"):
            loss_function.last_term_values = []
        if hasattr(loss_function, "last_term_names"):
            loss_function.last_term_names = []
        problem = getattr(loss_function, "problem", None)
        clear = getattr(problem, "clear_autodiff_cache", None)
        if callable(clear):
            clear()

    def _history_entry(
        self,
        *,
        iteration: int,
        phase: Any,
        total_loss: float,
        components: dict[str, float],
        batch: Any | None,
        gradient_norm: float | None,
        gradient_statistics: dict[str, Any] | None,
        adaptive_refinement_event: bool,
        training_stage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        residual_statistics = dict(
            getattr(self.built.loss_function, "last_residual_statistics", {}) or {}
        )
        entry = {
            "iteration": int(iteration),
            "phase": phase.name,
            "elapsed_seconds": time.perf_counter() - self._train_started,
            "total_loss": total_loss,
            "component_losses": {
                str(name): float(value)
                for name, value in components.items()
                if name != "total_loss"
            },
            "component_category_losses": self._component_category_losses(
                components
            ),
            "learning_rate": float(phase.optimizer.param_groups[0].get("lr", 0.0)),
            "sampling_point_count": (
                self._sampling_point_count(batch)
                if batch is not None
                else self._configured_sampling_point_count()
            ),
            "maximum_residual": residual_statistics.get("maximum_residual"),
            "mean_residual": residual_statistics.get("mean_residual"),
            "raw_maximum_residual": residual_statistics.get(
                "raw_maximum_residual"
            ),
            "raw_mean_residual": residual_statistics.get(
                "raw_mean_residual"
            ),
            "residual_region_statistics": residual_statistics.get(
                "region_statistics"
            ),
            "mean_region_training_residual_rmse": residual_statistics.get(
                "mean_region_training_residual_rmse"
            ),
            "worst_region_training_residual_rmse": residual_statistics.get(
                "worst_region_training_residual_rmse"
            ),
            "mean_region_raw_residual_rmse": residual_statistics.get(
                "mean_region_raw_residual_rmse"
            ),
            "worst_region_raw_residual_rmse": residual_statistics.get(
                "worst_region_raw_residual_rmse"
            ),
            "gradient_norm": gradient_norm,
            "gradient_statistics": (
                dict(gradient_statistics) if gradient_statistics else None
            ),
            "nan_inf_status": "finite",
            "gpu_memory_allocated": (
                int(torch.cuda.memory_allocated(next(self.built.model.parameters()).device))
                if next(self.built.model.parameters()).device.type == "cuda" and torch.cuda.is_available()
                else 0
            ),
            "adaptive_refinement_event": bool(adaptive_refinement_event),
            "raw_loss_components": dict(components),
            "raw_losses": dict(
                getattr(
                    self.built.loss_function,
                    "last_raw_losses",
                    {},
                )
            ),
            "dynamic_weighting": (
                dict(self.built.loss_function.last_dynamic_weighting)
                if getattr(
                    self.built.loss_function,
                    "last_dynamic_weighting",
                    None,
                )
                else None
            ),
            "effective_loss_weights": dict(
                getattr(
                    self.built.loss_function,
                    "last_effective_loss_weights",
                    {},
                )
            ),
            "weighted_losses": dict(
                getattr(
                    self.built.loss_function,
                    "last_weighted_losses",
                    {},
                )
            ),
            "training_stage": dict(training_stage or {}),
            "time_window_fraction": float(
                (training_stage or {}).get("time_window_fraction", 1.0)
            ),
        }
        return entry

    @staticmethod
    def _sampling_point_count(batch: Any | None) -> int | None:
        if batch is None:
            return None
        total = int(getattr(batch.domain_samples, "shape", [0])[0])
        for value in (getattr(batch, "constraint_samples", {}) or {}).values():
            points = getattr(value, "points", value)
            if hasattr(points, "shape") and len(points.shape) > 0:
                total += int(points.shape[0])
        return total

    def _configured_sampling_point_count(self) -> int:
        if self.registry is not None:
            return sum(
                int(item.point_budget or 0)
                for item in self.registry.sampling_components
            )
        sampling = self.built.normalized_spec.get("sampling") or {}
        return sum(
            int(((section or {}).get("parameters") or {}).get("n_points") or 0)
            for section in sampling.values()
            if isinstance(section, dict)
        )

    def _component_category_losses(
        self, components: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        if self.registry is None:
            return {"unclassified": dict(components)}
        result: dict[str, dict[str, float]] = {}
        for descriptor in self.registry.loss_components:
            if descriptor.component_id in components:
                result.setdefault(descriptor.category, {})[
                    descriptor.component_id
                ] = float(components[descriptor.component_id])
        return result

    def _gradient_norm(self) -> float | None:
        squared = 0.0
        found = False
        for parameter in self.built.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
            squared += norm * norm
        return squared**0.5 if found else None

    def _backward_with_runtime_gradient_projection(
        self, total_loss: torch.Tensor
    ) -> dict[str, Any] | None:
        """Backpropagate with deterministic PCGrad when an online policy enabled it."""

        getter = getattr(
            self.built.loss_function, "get_gradient_projection_strength", None
        )
        strength = float(getter()) if callable(getter) else 0.0
        names = list(getattr(self.built.loss_function, "last_term_names", ()) or ())
        values = list(getattr(self.built.loss_function, "last_term_values", ()) or ())
        weights = dict(
            getattr(self.built.loss_function, "last_effective_loss_weights", {}) or {}
        )
        parameters = [
            parameter
            for parameter in self.built.model.parameters()
            if parameter.requires_grad
        ]
        if strength <= 0.0 or len(values) < 2 or not parameters:
            total_loss.backward()
            return None
        component_losses = [
            value * float(weights.get(str(name), 1.0))
            for name, value in zip(names, values)
        ]
        gradients = [
            torch.autograd.grad(
                component,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            for component in component_losses
        ]
        projected: list[list[torch.Tensor | None]] = [
            [None if value is None else value.clone() for value in row]
            for row in gradients
        ]
        conflicts = 0
        epsilon = 1.0e-20
        for left_index, left in enumerate(projected):
            for right_index, right in enumerate(gradients):
                if left_index == right_index:
                    continue
                dot = sum(
                    (lhs * rhs).sum()
                    for lhs, rhs in zip(left, right)
                    if lhs is not None and rhs is not None
                )
                if not isinstance(dot, torch.Tensor) or float(dot.detach()) >= 0.0:
                    continue
                norm = sum(
                    rhs.square().sum() for rhs in right if rhs is not None
                )
                if not isinstance(norm, torch.Tensor) or float(norm.detach()) <= epsilon:
                    continue
                coefficient = strength * dot / norm.clamp_min(epsilon)
                for parameter_index, rhs in enumerate(right):
                    lhs = left[parameter_index]
                    if lhs is not None and rhs is not None:
                        left[parameter_index] = lhs - coefficient * rhs
                conflicts += 1
        for parameter_index, parameter in enumerate(parameters):
            terms = [
                row[parameter_index]
                for row in projected
                if row[parameter_index] is not None
            ]
            parameter.grad = (
                torch.stack(terms, dim=0).sum(dim=0) if terms else None
            )
        return {
            "method": "pcgrad",
            "projection_strength": strength,
            "component_count": len(component_losses),
            "projected_conflicts": int(conflicts),
        }

    def _configure_gradient_audit_targets(
        self,
        *,
        start_iteration: int,
        planned_iterations: int,
    ) -> None:
        if not self.component_gradient_diagnostics_enabled or planned_iterations <= 0:
            self._gradient_audit_targets = []
            return
        first = int(start_iteration) + 1
        middle = int(start_iteration) + max(1, math.ceil(planned_iterations / 2))
        last = int(start_iteration) + int(planned_iterations)
        self._gradient_audit_targets = sorted({first, middle, last})

    def _gradient_diagnostic_due(self, iteration: int) -> bool:
        if not self.component_gradient_diagnostics_enabled:
            return False
        return any(
            target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
            for target in self._gradient_audit_targets
        )

    def _gradient_checkpoint_kind(self, iteration: int) -> str:
        pending = [
            target
            for target in self._gradient_audit_targets
            if target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
        ]
        if not pending:
            return "unscheduled"
        target = max(pending)
        if target == self._gradient_audit_targets[0]:
            return "first"
        if target == self._gradient_audit_targets[-1]:
            return "last"
        return "middle"

    def _collect_component_gradient_diagnostic(
        self,
        *,
        total_loss: torch.Tensor,
        iteration: int,
        phase: Any,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        names = list(
            getattr(self.built.loss_function, "last_term_names", []) or []
        )
        values = list(
            getattr(self.built.loss_function, "last_term_values", []) or []
        )
        weights = dict(
            getattr(
                self.built.loss_function,
                "last_effective_loss_weights",
                {},
            )
            or {}
        )
        eligible = (
            {
                item.component_id
                for item in self.registry.loss_components
                if item.trainable
                and (not item.active_stages or str(phase.name) in item.active_stages)
            }
            if self.registry is not None
            else set(str(name) for name in names)
        )
        components = {
            str(name): value * float(weights.get(str(name), 1.0))
            for name, value in sorted(
                zip(names, values), key=lambda item: str(item[0])
            )[: self.maximum_gradient_components]
            if str(name) in eligible
        }
        diagnostic = collect_component_gradient_diagnostics(
            total_loss=total_loss,
            component_losses=components,
            parameters=list(self.built.model.parameters()),
            audit_iteration=int(iteration),
            gradient_imbalance_threshold=self.gradient_imbalance_threshold,
            gradient_conflict_threshold=self.gradient_conflict_threshold,
            maximum_components=self.maximum_gradient_components,
            maximum_conflicting_pairs=self.maximum_conflicting_pairs,
        )
        diagnostic.update(
            {
                "checkpoint_kind": checkpoint_kind,
                "phase": str(phase.name),
                "optimizer": str(getattr(phase, "optimizer_name", phase.name)),
                "learning_rates": [
                    float(group.get("lr", 0.0))
                    for group in phase.optimizer.param_groups
                ],
                "scheduler": (
                    type(phase.scheduler).__name__
                    if phase.scheduler is not None
                    else None
                ),
                "scheduler_last_epoch": (
                    int(getattr(phase.scheduler, "last_epoch"))
                    if phase.scheduler is not None
                    and getattr(phase.scheduler, "last_epoch", None) is not None
                    else None
                ),
                "component_weight_basis": "effective_registered_loss_weights",
            }
        )
        crossed = [
            target
            for target in self._gradient_audit_targets
            if target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
        ]
        diagnostic["satisfied_audit_targets"] = crossed
        self._gradient_audit_targets_completed.update(crossed)
        self.component_gradient_audits.append(diagnostic)
        return diagnostic

    def _collect_final_component_gradient_diagnostic(self) -> None:
        if (
            not self.component_gradient_diagnostics_enabled
            or self._actual_iterations <= 0
            or self._last_training_batch is None
            or self._last_training_phase is None
        ):
            return
        if (
            self.component_gradient_audits
            and int(
                self.component_gradient_audits[-1].get("audit_iteration") or -1
            )
            == self._actual_iterations
        ):
            return
        loss_state = self._snapshot_lbfgs_loss_state()
        try:
            loss, _ = self.built.loss_function(
                self.built.model,
                self._last_training_batch,
            )
            if torch.isfinite(loss):
                self._collect_component_gradient_diagnostic(
                    total_loss=loss,
                    iteration=self._actual_iterations,
                    phase=self._last_training_phase,
                    checkpoint_kind="last_before_evaluation",
                )
        finally:
            self._restore_lbfgs_loss_state(loss_state)
            self._release_iteration_graphs()

    def _maximum_absolute_gradient(self) -> float | None:
        maximum: float | None = None
        for parameter in self.built.model.parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().abs().max().cpu())
            maximum = value if maximum is None else max(maximum, value)
        return maximum

    def _gradients_finite(self) -> bool:
        return all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.built.model.parameters()
        )

    def _parameters_finite(self) -> bool:
        return all(torch.isfinite(parameter).all() for parameter in self.built.model.parameters())

    @staticmethod
    def _scheduler_step(scheduler: object | None, loss: float) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # Old checkpoints and pre-v17 budget adaptation can carry a
            # shorter scheduler horizon than the optimizer phase. Never let
            # that metadata mismatch abort an otherwise valid candidate.
            if int(scheduler.last_epoch) >= int(scheduler.total_steps):
                return
            scheduler.step()
        else:
            scheduler.step()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.adaptive_compute_budget_ledger.checkpoint_count += 1
        self.adaptive_compute_budget_ledger.optimizer_steps = max(
            self.adaptive_compute_budget_ledger.optimizer_steps,
            int(self._actual_iterations),
        )
        payload = self._runtime_state_payload(
            current_loss=(
                float(self.history[-1]["total_loss"])
                if self.history and self.history[-1].get("total_loss") is not None
                else None
            )
        )
        payload.update(
            {
                "checkpoint_schema_version": 10,
                "normalized_spec": self.built.normalized_spec,
                "time_strategy_audit": self.time_strategy_audit,
                "adaptive_audit_logger_state": (
                    self.adaptive_audit_logger.state_dict()
                    if self.adaptive_audit_logger is not None
                    else None
                ),
            }
        )
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, history, and phase progress."""

        device = next(self.built.model.parameters()).device
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if int(checkpoint.get("checkpoint_schema_version") or 0) >= 2:
            self._restore_runtime_state(checkpoint)
            self.time_strategy_audit = dict(
                checkpoint.get("time_strategy_audit") or {}
            )
            if self.adaptive_audit_logger is not None:
                self.adaptive_audit_logger.load_state_dict(
                    checkpoint.get("adaptive_audit_logger_state")
                )
            if self.fixed_budget_sampling_controller is not None:
                versions = dict(
                    getattr(self.built.sampler, "_sampling_state_versions", {})
                    or {}
                )
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(
                        checkpoint.get("active_phase_name") or "resume"
                    ),
                    controller="fixed_budget_sampling",
                    event_type="sampling_resume_restored",
                    trigger={
                        "checkpoint_schema_version": int(
                            checkpoint.get("checkpoint_schema_version") or 0
                        ),
                        "sampler_state_versions": versions,
                        "pending_observation": bool(
                            self._pending_sampling_observation
                        ),
                    },
                )
            if self.curriculum_advance_controller is not None:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(checkpoint.get("active_phase_name") or "resume"),
                    controller="curriculum_advance",
                    event_type="curriculum_resume_restored",
                    trigger={
                        "checkpoint_schema_version": int(
                            checkpoint.get("checkpoint_schema_version") or 0
                        ),
                        "resume_disabled": bool(self._curriculum_resume_disabled),
                        "pending_observation": bool(
                            self._pending_curriculum_observation
                        ),
                        "runtime_states": {
                            curriculum_id: runtime.get_current_state().to_dict()
                            for curriculum_id, runtime in self.curriculum_runtimes.items()
                        },
                    },
                )
            return
        # Schema-v1 and terminal model checkpoints can still contain a
        # dynamically grown network.  Rebuild that registered module tree
        # before loading its tensors, just as the current runtime checkpoint
        # path does above.
        if self.morphable_model is not None:
            self.morphable_model.restore_architecture(
                checkpoint.get("online_morph_architecture") or []
            )
            self._synchronize_optimizer_parameters_exact()
        self.built.model.load_state_dict(checkpoint["model_state_dict"])
        self._load_optimizer_state_by_param_names(
            self.built.optimization_phases,
            checkpoint.get("optimizer_states") or [],
            checkpoint.get("optimizer_param_names"),
        )
        for phase, state in zip(self.built.optimization_phases, checkpoint.get("scheduler_states") or []):
            if phase.scheduler is not None and state is not None:
                phase.scheduler.load_state_dict(state)
        self.history = list(checkpoint.get("history") or [])
        self.adaptive_refinement_events = list(checkpoint.get("adaptive_refinement_events") or [])
        self.optimizer_audit = dict(checkpoint.get("optimizer_audit") or {})
        self.component_gradient_audits = list(
            checkpoint.get("component_gradient_audits") or []
        )
        # Safe defaults retain compatibility with schema-v1 checkpoints.
        if checkpoint.get("loss_state"):
            self.built.loss_function.load_state_dict(checkpoint["loss_state"])
        if checkpoint.get("sampler_state"):
            self.built.sampler.load_state_dict(checkpoint["sampler_state"])
        stored = checkpoint.get("completed_phase_iterations") or {}
        if stored:
            self.completed_phase_iterations = {str(name): int(count) for name, count in stored.items()}
        else:
            counts: dict[str, int] = {}
            for item in self.history:
                phase_name = str(item.get("phase") or "")
                if phase_name:
                    counts[phase_name] = counts.get(phase_name, 0) + 1
            self.completed_phase_iterations = counts
        self._actual_iterations = int(
            checkpoint.get("actual_iterations")
            or checkpoint.get("iteration")
            or len(self.history)
        )
