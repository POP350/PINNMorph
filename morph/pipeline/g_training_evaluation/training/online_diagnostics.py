"""Deterministic, execution-grounded diagnostics for online PINN morphing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class OptimizationDiagnostics:
    current_total_loss: float
    relative_loss_improvement: float
    loss_slope: float
    plateau_score: float
    plateau_duration: int
    steps_since_last_improvement: int
    plateau: bool
    insufficient_progress: bool
    generic_capacity_or_representation_limit: bool = False
    training_plateau: bool = False
    physics_plateau: bool = False
    physics_relative_improvement: float | None = None
    insufficient_progress_source: str = "none"


@dataclass(frozen=True)
class GradientDiagnostics:
    available: bool
    component_gradient_norms: dict[str, float] = field(default_factory=dict)
    conflicting_component_pairs: tuple[dict[str, Any], ...] = ()
    gradient_norm_pde: float | None = None
    gradient_norm_bc: float | None = None
    gradient_norm_ic: float | None = None
    gradient_norm_data: float | None = None
    max_min_gradient_norm_ratio: float | None = None
    minimum_gradient_cosine: float | None = None
    mean_gradient_cosine: float | None = None
    gradient_conflict_detected: bool = False
    gradient_imbalance_detected: bool = False
    gradient_pathology: bool = False
    gradient_magnitude_imbalance: bool = False
    persistent_gradient_conflict: bool = False
    transient_gradient_conflict: bool = False
    gradient_status: str = "normal"
    reason: str | None = None


@dataclass(frozen=True)
class SpatialResidualDiagnostics:
    available: bool
    maximum_residual: float | None = None
    mean_residual: float | None = None
    max_mean_residual_ratio: float | None = None
    top_k_fraction: float = 0.05
    top_k_residual_mass_ratio: float | None = None
    residual_concentration_score: float | None = None
    localized_residual_concentration: bool = False
    point_count: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class FrequencyDiagnostics:
    has_multiscale_or_oscillatory_evidence: bool
    periodic: bool = False
    oscillatory: bool = False
    high_frequency: bool = False
    multiscale: bool = False
    heterogeneous_coefficient: bool = False
    known_characteristic_frequency: float | None = None
    evidence_tags: tuple[str, ...] = ()
    measured_axes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetDiagnostics:
    remaining_iterations: int
    minimum_remaining_iterations: int
    remaining_budget_sufficient: bool
    initial_parameter_count: int
    current_parameter_count: int
    morph_count: int
    cooldown_active: bool
    steps_since_last_successful_morph: int | None


@dataclass(frozen=True)
class PhysicsHealthDiagnostics:
    """Compact held-out physics state exposed to online policy evolution."""

    available: bool
    total_score: float | None = None
    pde_loss: float | None = None
    boundary_loss: float | None = None
    initial_loss: float | None = None
    observation_loss: float | None = None
    dominant_category: str = "unavailable"
    dominant_category_ratio: float | None = None
    initial_to_pde_ratio: float | None = None
    initial_condition_dominant: bool = False
    boundary_condition_dominant: bool = False
    pde_residual_dominant: bool = False
    worst_component_id: str | None = None


@dataclass(frozen=True)
class OnlineMorphDiagnostics:
    step: int
    optimization: OptimizationDiagnostics
    gradient: GradientDiagnostics
    spatial: SpatialResidualDiagnostics
    frequency: FrequencyDiagnostics
    budget: BudgetDiagnostics
    physics: PhysicsHealthDiagnostics = field(
        default_factory=lambda: PhysicsHealthDiagnostics(available=False)
    )
    pde: dict[str, Any] = field(default_factory=dict)
    current_architecture: dict[str, Any] = field(default_factory=dict)
    previous_morph_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimization_diagnostics(
    loss_history: Sequence[float],
    *,
    window: int,
    relative_improvement_threshold: float,
    plateau_window_count: int,
    plateau_min_windows: int,
    diagnostic_interval: int,
    step: int,
    last_improvement_step: int | None,
) -> tuple[OptimizationDiagnostics, int, int | None]:
    values = [float(value) for value in loss_history if math.isfinite(float(value))]
    window = max(2, int(window))
    recent = values[-window:]
    current = recent[-1] if recent else float("nan")
    if len(recent) >= 2:
        baseline = recent[0]
        improvement = (baseline - current) / max(abs(baseline), 1e-12)
        x_mean = 0.5 * (len(recent) - 1)
        denominator = sum((index - x_mean) ** 2 for index in range(len(recent)))
        y_mean = sum(recent) / len(recent)
        slope = (
            sum((index - x_mean) * (value - y_mean) for index, value in enumerate(recent))
            / max(denominator, 1e-12)
        )
    else:
        improvement = 0.0
        slope = 0.0
    threshold = max(0.0, float(relative_improvement_threshold))
    low_progress_window = bool(len(recent) >= window and improvement <= threshold)
    next_count = int(plateau_window_count) + 1 if low_progress_window else 0
    if not low_progress_window:
        last_improvement_step = int(step)
    plateau = next_count >= max(1, int(plateau_min_windows))
    plateau_score = (
        max(0.0, 1.0 - improvement / max(threshold, 1e-12))
        if low_progress_window
        else 0.0
    )
    duration = next_count * max(1, int(diagnostic_interval))
    since = (
        max(0, int(step) - int(last_improvement_step))
        if last_improvement_step is not None
        else duration
    )
    return (
        OptimizationDiagnostics(
            current_total_loss=current,
            relative_loss_improvement=improvement,
            loss_slope=slope,
            plateau_score=plateau_score,
            plateau_duration=duration,
            steps_since_last_improvement=since,
            plateau=plateau,
            insufficient_progress=plateau,
            training_plateau=plateau,
            insufficient_progress_source=("training_loss" if plateau else "none"),
        ),
        next_count,
        last_improvement_step,
    )


def build_gradient_diagnostics(
    statistics: Mapping[str, Any] | None,
    *,
    component_categories: Mapping[str, str] | None = None,
) -> GradientDiagnostics:
    stats = dict(statistics or {})
    norms = {
        str(name): float(value)
        for name, value in (stats.get("component_grad_norms") or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    categories = {str(name): str(value) for name, value in (component_categories or {}).items()}

    def aggregate(*accepted: str) -> float | None:
        values = [
            value
            for name, value in norms.items()
            if categories.get(name, "").casefold() in accepted
        ]
        return math.sqrt(sum(value * value for value in values)) if values else None

    pairwise = dict(stats.get("pairwise_cosine_summary") or {})
    conflicting_pairs = tuple(
        {
            "component_a": str(item.get("component_a")),
            "component_b": str(item.get("component_b")),
            "cosine_similarity": float(item.get("cosine_similarity")),
        }
        for item in (pairwise.get("conflicting_pairs") or [])
        if isinstance(item, Mapping)
        and item.get("component_a") is not None
        and item.get("component_b") is not None
        and isinstance(item.get("cosine_similarity"), (int, float))
        and math.isfinite(float(item.get("cosine_similarity")))
    )
    minimum = pairwise.get("minimum")
    mean = pairwise.get("mean", pairwise.get("median"))
    imbalance = bool(stats.get("gradient_imbalance_detected", False))
    conflict = bool(stats.get("gradient_conflict_detected", False))
    available = bool(stats.get("available", bool(norms)))
    return GradientDiagnostics(
        available=available,
        component_gradient_norms=norms,
        conflicting_component_pairs=conflicting_pairs,
        gradient_norm_pde=aggregate("equation_residual"),
        gradient_norm_bc=aggregate("boundary_constraint", "boundary"),
        gradient_norm_ic=aggregate("initial_constraint", "initial"),
        gradient_norm_data=aggregate("data", "observation", "pointset_constraint"),
        max_min_gradient_norm_ratio=_finite_or_none(
            stats.get("max_to_min_nonzero_component_ratio")
        ),
        minimum_gradient_cosine=_finite_or_none(minimum),
        mean_gradient_cosine=_finite_or_none(mean),
        gradient_conflict_detected=conflict,
        gradient_imbalance_detected=imbalance,
        gradient_pathology=bool(available and (imbalance or conflict)),
        gradient_magnitude_imbalance=bool(available and imbalance),
        gradient_status=(
            "magnitude_imbalance"
            if available and imbalance
            else "transient_directional_conflict"
            if available and conflict
            else "normal"
        ),
        reason=str(stats.get("reason")) if stats.get("reason") else None,
    )


def build_spatial_residual_diagnostics(
    pointwise_scores: torch.Tensor | Sequence[float] | None,
    *,
    top_k_fraction: float,
    max_mean_threshold: float,
    top_k_mass_threshold: float,
) -> SpatialResidualDiagnostics:
    if pointwise_scores is None:
        return SpatialResidualDiagnostics(available=False, reason="pointwise_residual_unavailable")
    scores = torch.as_tensor(pointwise_scores, dtype=torch.float64).detach().reshape(-1).abs().cpu()
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        return SpatialResidualDiagnostics(available=False, reason="no_finite_pointwise_residual")
    fraction = min(max(float(top_k_fraction), 1.0 / scores.numel()), 1.0)
    top_count = max(1, math.ceil(scores.numel() * fraction))
    maximum = float(scores.max())
    mean = float(scores.mean())
    ratio = maximum / max(mean, 1e-12)
    mass = float(torch.topk(scores, top_count).values.sum() / scores.sum().clamp_min(1e-12))
    score = min(ratio / max(float(max_mean_threshold), 1e-12), mass / max(float(top_k_mass_threshold), 1e-12))
    localized = bool(
        ratio >= float(max_mean_threshold)
        and mass >= float(top_k_mass_threshold)
    )
    return SpatialResidualDiagnostics(
        available=True,
        maximum_residual=maximum,
        mean_residual=mean,
        max_mean_residual_ratio=ratio,
        top_k_fraction=fraction,
        top_k_residual_mass_ratio=mass,
        residual_concentration_score=score,
        localized_residual_concentration=localized,
        point_count=int(scores.numel()),
    )


def build_frequency_diagnostics(metadata: Mapping[str, Any] | None) -> FrequencyDiagnostics:
    raw = dict(metadata or {})
    tags = _metadata_tags(raw)

    def flag(*names: str) -> bool:
        return any(bool(raw.get(name)) for name in names) or bool(tags.intersection(names))

    periodic = flag("periodic", "has_periodicity")
    oscillatory = flag("oscillatory", "has_oscillatory_solution")
    high_frequency = flag("high_frequency", "has_high_frequency")
    multiscale = flag("multiscale", "has_multiscale", "is_multiscale")
    heterogeneous = flag(
        "heterogeneous_coefficient",
        "heterogeneous",
        "varying_coefficient",
        "piecewise_coefficient",
        "has_variable_coefficients",
    )
    known = _finite_or_none(
        raw.get("known_characteristic_frequency", raw.get("characteristic_frequency"))
    )
    measured = dict(raw.get("residual_frequency_axes") or {})
    measured_high = any(v.get("high_band_energy_fraction", 0) >= 0.2 for v in measured.values())
    evidence = periodic or oscillatory or high_frequency or multiscale or heterogeneous or known is not None or measured_high
    return FrequencyDiagnostics(
        has_multiscale_or_oscillatory_evidence=bool(evidence),
        periodic=periodic,
        oscillatory=oscillatory,
        high_frequency=high_frequency,
        multiscale=multiscale,
        heterogeneous_coefficient=heterogeneous,
        known_characteristic_frequency=known,
        evidence_tags=tuple(sorted(tags)),
        measured_axes=measured,
    )


def build_budget_diagnostics(
    *,
    step: int,
    remaining_iterations: int,
    minimum_remaining_iterations: int,
    initial_parameter_count: int,
    current_parameter_count: int,
    morph_count: int,
    last_successful_morph_step: int | None,
    min_steps_between_morphs: int,
) -> BudgetDiagnostics:
    initial = max(1, int(initial_parameter_count))
    since = (
        int(step) - int(last_successful_morph_step)
        if last_successful_morph_step is not None
        else None
    )
    cooldown = bool(
        since is not None and since < max(0, int(min_steps_between_morphs))
    )
    return BudgetDiagnostics(
        remaining_iterations=int(remaining_iterations),
        minimum_remaining_iterations=int(minimum_remaining_iterations),
        remaining_budget_sufficient=int(remaining_iterations) >= int(minimum_remaining_iterations),
        initial_parameter_count=initial,
        current_parameter_count=int(current_parameter_count),
        morph_count=int(morph_count),
        cooldown_active=cooldown,
        steps_since_last_successful_morph=since,
    )


def build_physics_health_diagnostics(
    metrics: Mapping[str, Any] | None,
) -> PhysicsHealthDiagnostics:
    """Summarize fixed-probe physics residuals without reference-solution data."""

    raw = dict(metrics or {})
    categories = {
        "pde": _finite_or_none(raw.get("pde_loss")),
        "boundary": _finite_or_none(raw.get("boundary_loss")),
        "initial": _finite_or_none(raw.get("initial_loss")),
        "observation": _finite_or_none(raw.get("observation_loss")),
    }
    finite = {name: value for name, value in categories.items() if value is not None}
    components = {
        str(name): float(value)
        for name, value in dict(raw.get("component_metrics") or {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if not finite and not components:
        return PhysicsHealthDiagnostics(available=False)

    dominant = max(finite, key=lambda name: finite[name]) if finite else "unavailable"
    ordered = sorted(finite.values(), reverse=True)
    dominant_ratio = (
        ordered[0] / max(ordered[1], 1e-12)
        if len(ordered) >= 2
        else None
    )
    initial = categories["initial"]
    pde = categories["pde"]
    return PhysicsHealthDiagnostics(
        available=True,
        total_score=_finite_or_none(raw.get("total_score")),
        pde_loss=pde,
        boundary_loss=categories["boundary"],
        initial_loss=initial,
        observation_loss=categories["observation"],
        dominant_category=dominant,
        dominant_category_ratio=dominant_ratio,
        initial_to_pde_ratio=(
            initial / max(pde, 1e-12)
            if initial is not None and pde is not None
            else None
        ),
        initial_condition_dominant=dominant == "initial",
        boundary_condition_dominant=dominant == "boundary",
        pde_residual_dominant=dominant == "pde",
        worst_component_id=(
            max(components, key=lambda name: components[name])
            if components
            else None
        ),
    )
def _metadata_tags(metadata: Mapping[str, Any]) -> set[str]:
    tags: set[str] = set()
    for key in ("challenge_tags", "tags", "physics_tags"):
        value = metadata.get(key)
        if isinstance(value, str):
            tags.add(value.casefold())
        elif isinstance(value, (list, tuple, set)):
            tags.update(str(item).casefold() for item in value)
    for key, value in metadata.items():
        if isinstance(value, bool) and value:
            tags.add(str(key).casefold())
    return tags


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


__all__ = [
    "BudgetDiagnostics",
    "FrequencyDiagnostics",
    "GradientDiagnostics",
    "OnlineMorphDiagnostics",
    "OptimizationDiagnostics",
    "PhysicsHealthDiagnostics",
    "SpatialResidualDiagnostics",
    "build_budget_diagnostics",
    "build_frequency_diagnostics",
    "build_gradient_diagnostics",
    "build_optimization_diagnostics",
    "build_physics_health_diagnostics",
    "build_spatial_residual_diagnostics",
]
