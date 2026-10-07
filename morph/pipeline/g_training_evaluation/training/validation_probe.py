"""Fixed, component-driven physics validation without evaluator data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .registry import TrainingComponentRegistry


@dataclass(frozen=True)
class ValidationProbeBudgetPolicy:
    points_per_component: int = 32
    maximum_total_points: int = 512


@dataclass(frozen=True)
class ValidationProbeSet:
    probe_set_id: str
    seed: int
    component_budgets: Mapping[str, int]
    batch: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicsValidationResult:
    aggregate_score: float
    category_scores: Mapping[str, float]
    component_scores: Mapping[str, float]
    normalized_component_scores: Mapping[str, float]
    worst_component_id: str | None
    worst_component_score: float | None
    median_component_score: float | None
    p90_component_score: float | None
    normalized_region_scores: Mapping[int, float] = field(default_factory=dict)
    mean_region_score: float | None = None
    worst_region_id: int | None = None
    worst_region_score: float | None = None
    auxiliary_scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_score": self.aggregate_score,
            "category_scores": dict(self.category_scores),
            "component_scores": dict(self.component_scores),
            "normalized_component_scores": dict(self.normalized_component_scores),
            "worst_component_id": self.worst_component_id,
            "worst_component_score": self.worst_component_score,
            "median_component_score": self.median_component_score,
            "p90_component_score": self.p90_component_score,
            "normalized_region_scores": {
                str(name): float(value)
                for name, value in self.normalized_region_scores.items()
            },
            "mean_region_score": self.mean_region_score,
            "worst_region_id": self.worst_region_id,
            "worst_region_score": self.worst_region_score,
            "auxiliary_scores": {
                str(name): float(value)
                for name, value in self.auxiliary_scores.items()
            },
        }


@dataclass(frozen=True)
class PhysicsValidationMetrics:
    """Reference-free squared physics residuals on a fixed held-out probe."""

    total_score: float
    pde_loss: float | None
    boundary_loss: float | None
    initial_loss: float | None
    observation_loss: float | None
    component_metrics: Mapping[str, float]
    component_categories: Mapping[str, str] = field(default_factory=dict)
    region_metrics: Mapping[str, float] = field(default_factory=dict)
    region_metadata: Mapping[str, Any] = field(default_factory=dict)
    worst_region_id: str | None = None
    worst_region_loss: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PhysicsValidationMetrics":
        """Read physics-residual telemetry without reference-solution metrics."""

        payload = dict(raw)
        return cls(
            total_score=float(payload["total_score"]),
            pde_loss=payload.get("pde_loss"),
            boundary_loss=payload.get("boundary_loss"),
            initial_loss=payload.get("initial_loss"),
            observation_loss=payload.get("observation_loss"),
            component_metrics=dict(payload.get("component_metrics") or {}),
            component_categories=dict(payload.get("component_categories") or {}),
            region_metrics={
                str(name): float(value)
                for name, value in dict(payload.get("region_metrics") or {}).items()
            },
            region_metadata=dict(payload.get("region_metadata") or {}),
            worst_region_id=(
                str(payload["worst_region_id"])
                if payload.get("worst_region_id") is not None
                else None
            ),
            worst_region_loss=payload.get("worst_region_loss"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": float(self.total_score),
            "pde_loss": self.pde_loss,
            "boundary_loss": self.boundary_loss,
            "initial_loss": self.initial_loss,
            "observation_loss": self.observation_loss,
            "component_metrics": dict(self.component_metrics),
            "component_categories": dict(self.component_categories),
            "region_metrics": dict(self.region_metrics),
            "region_metadata": dict(self.region_metadata),
            "worst_region_id": self.worst_region_id,
            "worst_region_loss": self.worst_region_loss,
        }


@dataclass(frozen=True)
class PhysicsUtilityDecision:
    before: PhysicsValidationMetrics
    after: PhysicsValidationMetrics
    relative_component_improvements: Mapping[str, float]
    aggregate_relative_improvement: float
    component_degradations: Mapping[str, float]
    max_component_degradation: float
    accepted: bool
    reason: str
    group_relative_improvements: Mapping[str, float] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "relative_component_improvements": dict(
                self.relative_component_improvements
            ),
            "aggregate_relative_improvement": float(
                self.aggregate_relative_improvement
            ),
            "component_degradations": dict(self.component_degradations),
            "max_component_degradation": float(
                self.max_component_degradation
            ),
            "group_relative_improvements": dict(
                self.group_relative_improvements
            ),
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
        }


def fit_log_linear_continuation(
    history: Sequence[Mapping[str, Any]],
    *,
    target_steps: Sequence[int],
    history_points: int = 7,
    epsilon: float = 1e-12,
    minimum_confidence: float = 0.35,
    maximum_extrapolation_factor: float = 4.0,
    reject_clipped_forecast: bool = True,
) -> dict[str, Any]:
    """Fit a conservative frozen continuation forecast.

    A least-squares log-linear extrapolation is extremely sensitive to a
    single noisy PINN residual.  Use a Theil-Sen-style median pairwise slope,
    quantify dispersion with MAD, and clamp every prediction to a multiple of
    the actually observed recent envelope.  Clamped or low-confidence
    forecasts remain auditable but are explicitly ineligible for policy
    credit.
    """

    required = max(2, int(history_points))
    rows = list(history)[-required:]
    if len(rows) < required:
        return {
            "valid": False,
            "reason": "insufficient_continuation_history",
            "required_history_points": required,
            "available_history_points": len(rows),
        }
    steps = [int(row["step"]) for row in rows]
    if len(set(steps)) != len(steps):
        return {"valid": False, "reason": "unstable_continuation_fit_duplicate_steps"}
    metric_rows = [dict(row.get("metrics") or {}) for row in rows]
    component_sets = [set(dict(row.get("component_metrics") or {})) for row in metric_rows]
    if not component_sets or not component_sets[0] or any(
        item != component_sets[0] for item in component_sets[1:]
    ):
        return {"valid": False, "reason": "continuation_component_set_changed"}

    max_factor = max(1.0, float(maximum_extrapolation_factor))

    def fit(values: Sequence[float]) -> dict[str, float] | None:
        if len(values) != len(steps) or any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in values
        ):
            return None
        x0 = float(steps[-1])
        xs = [float(step) - x0 for step in steps]
        ys = [math.log(float(value) + float(epsilon)) for value in values]
        pairwise_slopes = [
            (ys[j] - ys[i]) / (xs[j] - xs[i])
            for i in range(len(xs))
            for j in range(i + 1, len(xs))
            if abs(xs[j] - xs[i]) > float(epsilon)
        ]
        if not pairwise_slopes:
            return None
        slope = float(median(pairwise_slopes))
        intercept = float(median([y - slope * x for x, y in zip(xs, ys)]))
        residuals = [abs(y - (intercept + slope * x)) for x, y in zip(xs, ys)]
        mad = float(median(residuals))
        if not math.isfinite(slope) or not math.isfinite(intercept):
            return None
        return {
            "intercept": intercept,
            "slope": slope,
            "origin_step": x0,
            "log_residual_mad": mad,
            "history_span": max(float(max(xs) - min(xs)), 1.0),
            "recent_min": min(float(value) for value in values),
            "recent_max": max(float(value) for value in values),
        }

    total_fit = fit([float(row["total_score"]) for row in metric_rows])
    component_fits = {
        name: fit([float(row["component_metrics"][name]) for row in metric_rows])
        for name in sorted(component_sets[0])
    }
    region_sets = [set(dict(row.get("region_metrics") or {})) for row in metric_rows]
    stable_regions = bool(
        region_sets
        and region_sets[0]
        and all(item == region_sets[0] for item in region_sets[1:])
    )
    region_fits = (
        {
            name: fit([float(row["region_metrics"][name]) for row in metric_rows])
            for name in sorted(region_sets[0])
        }
        if stable_regions
        else {}
    )
    if (
        total_fit is None
        or any(value is None for value in component_fits.values())
        or any(value is None for value in region_fits.values())
    ):
        return {"valid": False, "reason": "unstable_continuation_fit"}

    def predict(model: Mapping[str, float], target: int) -> dict[str, Any] | None:
        exponent = float(model["intercept"]) + float(model["slope"]) * (
            float(target) - float(model["origin_step"])
        )
        try:
            raw_value = math.exp(exponent) - float(epsilon)
        except OverflowError:
            return None
        if not math.isfinite(raw_value):
            return None
        raw_value = max(0.0, raw_value)
        recent_min = max(0.0, float(model["recent_min"]))
        recent_max = max(recent_min, float(model["recent_max"]))
        lower = recent_min / max_factor
        upper = max(recent_max * max_factor, float(epsilon))
        value = min(max(raw_value, lower), upper)
        clipped = not math.isclose(
            value, raw_value, rel_tol=1e-12, abs_tol=float(epsilon)
        )
        horizon_ratio = abs(
            float(target) - float(model["origin_step"])
        ) / float(model["history_span"])
        uncertainty = 1.4826 * float(model["log_residual_mad"]) * (
            1.0 + horizon_ratio
        )
        confidence = math.exp(-max(0.0, uncertainty))
        if clipped:
            confidence *= 0.25
        return {
            "value": float(value),
            "raw_value": float(raw_value),
            "clipped": bool(clipped),
            "confidence": float(min(1.0, max(0.0, confidence))),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }

    categories = dict(metric_rows[-1].get("component_categories") or {})

    def physics_group(component_name: str) -> str | None:
        category = str(categories.get(component_name, ""))
        if category == "equation_residual":
            return "pde"
        if category == "initial_constraint":
            return "initial"
        if category == "observation":
            return "observation"
        if category and category != "auxiliary_constraint":
            return "boundary"
        return None

    def geometric_mean(values: Sequence[float]) -> float:
        finite = [min(1.0, max(float(epsilon), float(value))) for value in values]
        if not finite:
            return 0.0
        return math.exp(sum(math.log(value) for value in finite) / len(finite))

    predictions: dict[str, Any] = {}
    forecast_confidences: list[float] = []
    any_clipped = False
    for target in target_steps:
        total = predict(total_fit, int(target))
        components = {
            name: predict(model, int(target))
            for name, model in component_fits.items()
            if model is not None
        }
        regions = {
            name: predict(model, int(target))
            for name, model in region_fits.items()
            if model is not None
        }
        if (
            total is None
            or any(value is None for value in components.values())
            or any(value is None for value in regions.values())
        ):
            return {"valid": False, "reason": "unstable_continuation_prediction"}
        component_details = {
            name: dict(value) for name, value in components.items()
        }
        region_details = {name: dict(value) for name, value in regions.items()}
        confidence_by_group: dict[str, list[float]] = {}
        clipping_by_group: dict[str, bool] = {}
        for name, detail in component_details.items():
            group = physics_group(name)
            if group is None:
                continue
            confidence_by_group.setdefault(group, []).append(
                float(detail["confidence"])
            )
            clipping_by_group[group] = bool(
                clipping_by_group.get(group, False) or detail["clipped"]
            )
        if region_details:
            confidence_by_group["worst_region"] = [
                float(value["confidence"]) for value in region_details.values()
            ]
            clipping_by_group["worst_region"] = any(
                bool(value["clipped"]) for value in region_details.values()
            )
        group_confidences = {
            name: geometric_mean(values)
            for name, values in confidence_by_group.items()
            if values
        }
        # Confidence is aggregated over active physical groups. A single tiny,
        # noisy BC component can lower confidence, but cannot veto otherwise
        # stable PDE/IC/worst-region evidence by itself.
        target_confidence = geometric_mean(list(group_confidences.values()))
        target_clipped = bool(total["clipped"]) or any(
            clipping_by_group.values()
        )
        forecast_confidences.append(target_confidence)
        any_clipped = any_clipped or target_clipped
        predictions[str(int(target))] = {
            "step": int(target),
            "total_score": float(total["value"]),
            "component_metrics": {
                name: float(value["value"])
                for name, value in component_details.items()
            },
            "component_categories": categories,
            "region_metrics": {
                name: float(value["value"])
                for name, value in region_details.items()
            },
            "region_metadata": dict(metric_rows[-1].get("region_metadata") or {}),
            "forecast_confidence": float(target_confidence),
            "group_forecast_confidences": group_confidences,
            "confidence_aggregation": "geometric_mean_active_physics_groups",
            "forecast_clipped": bool(target_clipped),
            "prediction_details": {
                "total": dict(total),
                "components": component_details,
                "regions": region_details,
            },
        }
    confidence = min(forecast_confidences) if forecast_confidences else 0.0
    credit_eligible = bool(
        confidence >= float(minimum_confidence)
        and (not bool(reject_clipped_forecast) or not any_clipped)
    )
    for prediction in predictions.values():
        prediction["credit_eligible"] = bool(
            float(prediction["forecast_confidence"])
            >= float(minimum_confidence)
            and (
                not bool(reject_clipped_forecast)
                or not bool(prediction["forecast_clipped"])
            )
        )
    return {
        "valid": True,
        "reason": (
            "continuation_fit_available"
            if credit_eligible
            else "continuation_forecast_clipped"
            if any_clipped and bool(reject_clipped_forecast)
            else "continuation_forecast_low_confidence"
        ),
        "credit_eligible": credit_eligible,
        "forecast_confidence": float(confidence),
        "confidence_aggregation": "geometric_mean_active_physics_groups",
        "minimum_confidence": float(minimum_confidence),
        "maximum_extrapolation_factor": float(max_factor),
        "forecast_clipped": bool(any_clipped),
        "history_steps": steps,
        "history_points_used": len(rows),
        "epsilon": float(epsilon),
        "model": {
            "total": total_fit,
            "components": component_fits,
            "regions": region_fits,
        },
        "predictions": predictions,
    }


def continuation_forecast_supports_targets(
    forecast: Mapping[str, Any], target_steps: Sequence[int]
) -> bool:
    """Return whether a frozen forecast can be evaluated at every target.

    Forecast confidence controls causal credit, not whether a safe structural
    experiment may execute. Requiring ``credit_eligible`` here creates a
    self-reinforcing deadlock: no experiment means no new outcome evidence.
    Execution therefore only requires a valid, finite prediction. Low-
    confidence or clipped forecasts remain ineligible for policy credit.
    """

    if not bool(forecast.get("valid", False)):
        return False
    predictions = dict(forecast.get("predictions") or {})
    for step in target_steps:
        prediction = dict(predictions.get(str(int(step))) or {})
        total = prediction.get("total_score")
        components = dict(prediction.get("component_metrics") or {})
        if (
            not isinstance(total, (int, float))
            or not math.isfinite(float(total))
            or not components
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in components.values()
            )
        ):
            return False
    return True


def missing_continuation_feedback(
    prediction: Mapping[str, Any] | None,
    audit_prediction: Mapping[str, Any] | None,
    actual: PhysicsValidationMetrics,
    audit_actual: PhysicsValidationMetrics | None,
    *,
    target_step: int,
) -> dict[str, Any] | None:
    """Keep real observations without inventing an independent audit forecast."""
    missing = []
    if prediction is None:
        missing.append("decision")
    if audit_prediction is None or audit_actual is None:
        missing.append("audit")
    if not missing:
        return None
    return {
        "accepted": False,
        "reason": "continuation_forecast_unavailable",
        "withheld_kind": "continuation_forecast_unavailable",
        "credit_eligible": False,
        "aggregate_continuation_utility": 0.0,
        "forecast_confidence": 0.0,
        "missing_forecast_channels": missing,
        "forecast_target_step": int(target_step),
        "decision_probe_feedback": {
            "credit_eligible": False,
            "reason": "continuation_forecast_unavailable",
            "actual": actual.to_dict(),
        },
        "audit_probe_feedback": {
            "credit_eligible": False,
            "reason": "continuation_forecast_unavailable",
            "actual": audit_actual.to_dict() if audit_actual is not None else None,
        },
        "directional_evidence": "forecast_unavailable",
        "llm_feedback_summary": (
            "Frozen continuation evidence is unavailable for one or both probes; "
            "retain real measurements without assigning policy credit."
        ),
    }


def compare_continuation_aware_metrics(
    predicted: Mapping[str, Any],
    actual: PhysicsValidationMetrics,
    *,
    weights: Mapping[str, float] | None = None,
    minimum_relative_improvement: float,
    maximum_component_degradation: float,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate utility against predicted no-intervention continuation."""

    predicted_components = dict(predicted.get("component_metrics") or {})
    forecast_confidence = float(predicted.get("forecast_confidence", 1.0))
    credit_eligible = bool(predicted.get("credit_eligible", True))
    if bool(predicted.get("forecast_clipped", False)):
        credit_eligible = False
    actual_components = dict(actual.component_metrics)
    if not predicted_components or set(predicted_components) != set(actual_components):
        return {
            "accepted": False,
            "reason": "continuation_component_set_changed",
            "aggregate_continuation_utility": float("-inf"),
            "credit_eligible": False,
        }
    configured = {
        "pde": 1.0,
        "boundary": 1.0,
        "initial": 1.0,
        "observation": 1.0,
        "worst_region": 1.0,
        **{str(name): float(value) for name, value in dict(weights or {}).items()},
    }

    def category(name: str) -> str:
        return {
            "equation_residual": "pde",
            "initial_constraint": "initial",
            "observation": "observation",
        }.get(str(actual.component_categories.get(name, "")), "boundary")

    component_utility: dict[str, float] = {}
    component_degradation: dict[str, float] = {}
    by_group: dict[str, list[float]] = {}
    for name in sorted(actual_components):
        forecast = float(predicted_components[name])
        observed = float(actual_components[name])
        scale = max(abs(forecast), float(epsilon))
        utility = (forecast - observed) / scale
        degradation = (observed - forecast) / scale
        component_utility[name] = utility
        component_degradation[name] = degradation
        by_group.setdefault(category(name), []).append(utility)
    group_utility = {
        name: sum(values) / len(values)
        for name, values in by_group.items()
        if values
    }
    predicted_regions = {
        str(name): float(value)
        for name, value in dict(predicted.get("region_metrics") or {}).items()
    }
    actual_regions = {
        str(name): float(value)
        for name, value in dict(actual.region_metrics or {}).items()
    }
    if predicted_regions or actual_regions:
        if not predicted_regions or set(predicted_regions) != set(actual_regions):
            return {
                "accepted": False,
                "reason": "continuation_region_set_changed",
                "aggregate_continuation_utility": float("-inf"),
                "credit_eligible": False,
            }
    region_utility: dict[str, float] = {}
    region_degradation: dict[str, float] = {}
    for name in sorted(actual_regions):
        forecast = predicted_regions[name]
        observed = actual_regions[name]
        scale = max(abs(forecast), float(epsilon))
        region_utility[name] = (forecast - observed) / scale
        region_degradation[name] = (observed - forecast) / scale
    worst_region_id = (
        min(region_utility, key=region_utility.get) if region_utility else None
    )
    worst_region_utility = (
        region_utility[worst_region_id]
        if worst_region_id is not None
        else None
    )
    region_labels = {
        str(name): str(label)
        for name, label in dict(
            actual.region_metadata.get("region_labels") or {}
        ).items()
    }
    weighted = [
        (value, configured[name]) for name, value in group_utility.items()
    ]
    if worst_region_utility is not None:
        # Localized shock failure must not be averaged away by the much larger
        # smooth part of the domain.  It enters as one explicit physics group.
        weighted.append((worst_region_utility, configured["worst_region"]))
    weight_sum = sum(weight for _, weight in weighted)
    aggregate = sum(value * weight for value, weight in weighted) / max(
        weight_sum, float(epsilon)
    )
    all_degradations = list(component_degradation.values()) + list(
        region_degradation.values()
    )
    largest = max(0.0, max(all_degradations)) if all_degradations else 0.0
    accepted = bool(
        credit_eligible
        and
        math.isfinite(aggregate)
        and aggregate >= float(minimum_relative_improvement)
        and largest <= float(maximum_component_degradation)
    )
    return {
        "predicted_continuation": dict(predicted),
        "actual": actual.to_dict(),
        "component_continuation_utility": component_utility,
        "group_continuation_utility": group_utility,
        "aggregate_continuation_utility": float(aggregate),
        "component_degradations": component_degradation,
        "region_continuation_utility": region_utility,
        "region_degradations": region_degradation,
        "worst_region_id": worst_region_id,
        "worst_region_label": region_labels.get(str(worst_region_id)),
        "region_labels": region_labels,
        "worst_region_continuation_utility": worst_region_utility,
        "worst_region_degradation": (
            max(0.0, max(region_degradation.values()))
            if region_degradation
            else None
        ),
        "max_component_degradation": float(largest),
        "forecast_confidence": forecast_confidence,
        "credit_eligible": credit_eligible,
        "accepted": accepted,
        "reason": (
            "continuation_utility_accepted"
            if accepted
            else "continuation_credit_withheld"
            if not credit_eligible
            else "component_degradation_guard_failed"
            if largest > float(maximum_component_degradation)
            else "insufficient_continuation_utility"
        ),
    }


def compare_physics_validation_metrics(
    before: PhysicsValidationMetrics,
    after: PhysicsValidationMetrics,
    *,
    weights: Mapping[str, float] | None = None,
    minimum_relative_improvement: float,
    maximum_component_degradation: float,
    epsilon: float = 1e-12,
) -> PhysicsUtilityDecision:
    """Make the keep/rollback decision from fixed-probe physics only."""

    before_components = dict(before.component_metrics)
    after_components = dict(after.component_metrics)
    shared = sorted(set(before_components).intersection(after_components))
    if not shared or set(before_components) != set(after_components):
        return PhysicsUtilityDecision(
            before=before,
            after=after,
            relative_component_improvements={},
            aggregate_relative_improvement=float("-inf"),
            component_degradations={},
            max_component_degradation=float("inf"),
            accepted=False,
            reason="physics_component_set_changed",
        )
    configured = {
        "pde": 1.0,
        "boundary": 1.0,
        "initial": 1.0,
        "observation": 1.0,
        **{str(name): float(value) for name, value in dict(weights or {}).items()},
    }

    def utility_category(category: str) -> str:
        return {
            "equation_residual": "pde",
            "initial_constraint": "initial",
            "observation": "observation",
        }.get(str(category), "boundary")

    improvements: dict[str, float] = {}
    degradations: dict[str, float] = {}
    by_group: dict[str, list[float]] = {}
    for name in shared:
        baseline = float(before_components[name])
        current = float(after_components[name])
        scale = max(abs(baseline), float(epsilon))
        improvement = (baseline - current) / scale
        degradation = (current - baseline) / scale
        improvements[name] = improvement
        degradations[name] = degradation
        category = utility_category(before.component_categories.get(name, ""))
        by_group.setdefault(category, []).append(improvement)
    group_improvements = {
        name: sum(values) / len(values)
        for name, values in by_group.items()
        if values
    }
    weighted = [
        (value, configured[name])
        for name, value in group_improvements.items()
    ]
    weight_sum = sum(weight for _, weight in weighted)
    aggregate = sum(value * weight for value, weight in weighted) / max(
        weight_sum, float(epsilon)
    )
    largest = max(0.0, max(degradations.values())) if degradations else 0.0
    accepted = bool(
        math.isfinite(aggregate)
        and aggregate >= float(minimum_relative_improvement)
        and largest <= float(maximum_component_degradation)
    )
    reason = (
        "accepted"
        if accepted
        else "component_degradation_exceeded"
        if largest > float(maximum_component_degradation)
        else "insufficient_physics_improvement"
    )
    return PhysicsUtilityDecision(
        before=before,
        after=after,
        relative_component_improvements=improvements,
        aggregate_relative_improvement=float(aggregate),
        component_degradations=degradations,
        max_component_degradation=float(largest),
        accepted=accepted,
        reason=reason,
        group_relative_improvements=group_improvements,
    )


def combine_cross_probe_feedback(
    decision_feedback: Mapping[str, Any],
    audit_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Require agreement across disjoint decision and audit probes.

    The conservative minimum utility prevents a small decision probe from
    awarding credit that is not reproduced on the larger independent audit
    probe.  Both payloads are retained for causal auditability.
    """

    decision = deepcopy(dict(decision_feedback))
    audit = deepcopy(dict(audit_feedback))
    decision_utility = float(
        decision.get("aggregate_continuation_utility", float("-inf"))
    )
    audit_utility = float(
        audit.get("aggregate_continuation_utility", float("-inf"))
    )
    finite = math.isfinite(decision_utility) and math.isfinite(audit_utility)
    base_credit_eligible = bool(
        finite
        and decision.get("credit_eligible", True)
        and audit.get("credit_eligible", True)
    )
    direction_epsilon = 1.0e-12

    def direction(value: float) -> str:
        if not math.isfinite(value) or abs(value) <= direction_epsilon:
            return "neutral"
        return "improving" if value > 0.0 else "degrading"

    decision_direction = direction(decision_utility)
    audit_direction = direction(audit_utility)
    opposite = {decision_direction, audit_direction} == {"improving", "degrading"}
    same_direction = decision_direction == audit_direction
    decision_regions = {
        str(name): float(value)
        for name, value in dict(
            decision.get("region_continuation_utility") or {}
        ).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    audit_regions = {
        str(name): float(value)
        for name, value in dict(
            audit.get("region_continuation_utility") or {}
        ).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    focus_region_ids = {
        str(value)
        for value in (
            decision.get("worst_region_id"), audit.get("worst_region_id")
        )
        if value is not None
    }
    region_probe_directions = {
        region_id: {
            "decision": direction(decision_regions[region_id]),
            "audit": direction(audit_regions[region_id]),
        }
        for region_id in sorted(focus_region_ids)
        if region_id in decision_regions and region_id in audit_regions
    }
    conflicting_worst_regions = [
        region_id
        for region_id, directions in region_probe_directions.items()
        if {directions["decision"], directions["audit"]}
        == {"improving", "degrading"}
    ]
    worst_region_opposite = bool(conflicting_worst_regions)
    low_confidence_same_direction = bool(
        finite
        and same_direction
        and not opposite
        and not worst_region_opposite
        and not base_credit_eligible
    )
    # Opposing held-out probes are causal ambiguity, not proof of failure.
    # Keep genuine, eligible same-direction degradation available as failure
    # evidence, while withholding credit in the two ambiguous cases.
    credit_eligible = bool(
        base_credit_eligible and not opposite and not worst_region_opposite
    )
    accepted = bool(
        credit_eligible
        and decision.get("accepted", False)
        and audit.get("accepted", False)
    )
    utility = min(decision_utility, audit_utility) if finite else float("-inf")
    if opposite:
        reason = "cross_probe_opposite_direction"
        withheld_kind = "opposite_probe_directions"
        directional_evidence = "probe_conflict"
    elif worst_region_opposite:
        reason = "cross_probe_worst_region_opposite_direction"
        withheld_kind = "opposite_worst_region_directions"
        directional_evidence = "worst_region_probe_conflict"
    elif low_confidence_same_direction:
        reason = "cross_probe_same_direction_low_confidence"
        withheld_kind = "same_direction_low_confidence"
        directional_evidence = f"consistent_{decision_direction}_trend"
    elif accepted:
        reason = "cross_probe_utility_accepted"
        withheld_kind = None
        directional_evidence = "consistent_improving_trend"
    elif not credit_eligible:
        reason = "cross_probe_inconclusive"
        withheld_kind = "inconclusive_or_nonfinite"
        directional_evidence = "inconclusive"
    elif same_direction and decision_direction == "degrading":
        reason = "cross_probe_agreed_degradation"
        withheld_kind = None
        directional_evidence = "consistent_degrading_trend"
    else:
        reason = "cross_probe_component_risk_or_insufficient_utility"
        withheld_kind = None
        directional_evidence = f"consistent_{decision_direction}_trend"
    qualitative_credit_hint = 0.0
    if low_confidence_same_direction:
        minimum_confidence = min(
            float(decision.get("forecast_confidence", 0.0)),
            float(audit.get("forecast_confidence", 0.0)),
        )
        qualitative_credit_hint = max(
            -0.25,
            min(
                0.25,
                ((decision_utility + audit_utility) / 2.0)
                * max(0.0, min(1.0, minimum_confidence)),
            ),
        )
    return {
        **decision,
        "decision_probe_feedback": decision,
        "audit_probe_feedback": audit,
        "aggregate_continuation_utility": float(utility),
        "max_component_degradation": max(
            float(decision.get("max_component_degradation", 0.0)),
            float(audit.get("max_component_degradation", 0.0)),
        ),
        "forecast_confidence": min(
            float(decision.get("forecast_confidence", 0.0)),
            float(audit.get("forecast_confidence", 0.0)),
        ),
        "decision_worst_region_id": decision.get("worst_region_id"),
        "decision_worst_region_utility": decision.get(
            "worst_region_continuation_utility"
        ),
        "audit_worst_region_id": audit.get("worst_region_id"),
        "audit_worst_region_utility": audit.get(
            "worst_region_continuation_utility"
        ),
        "decision_probe_direction": decision_direction,
        "audit_probe_direction": audit_direction,
        "probe_direction_agreement": bool(same_direction),
        "worst_region_probe_directions": region_probe_directions,
        "conflicting_worst_region_ids": conflicting_worst_regions,
        "worst_region_direction_agreement": not worst_region_opposite,
        "directional_evidence": directional_evidence,
        "withheld_kind": withheld_kind,
        "qualitative_credit_hint": float(qualitative_credit_hint),
        "qualitative_credit_bound": 0.25,
        "llm_feedback_summary": (
            "Both physics probes move in the same direction, but continuation "
            "confidence is insufficient for numeric policy credit; use the "
            "direction only as qualitative evidence."
            if low_confidence_same_direction
            else "Global physics may agree, but the worst physical region moves "
            "in opposite directions across probes; withhold credit and treat "
            "the action as locally non-generalizing."
            if worst_region_opposite
            else "Decision and audit physics probes move in opposite directions; "
            "treat the intervention as non-generalizing/uncertain rather than "
            "as a success or failure."
            if opposite
            else "Both physics probes support the credited improvement."
            if accepted
            else "Both physics probes support degradation."
            if reason == "cross_probe_agreed_degradation"
            else "Physics evidence is inconclusive or fails a component safety guard."
        ),
        "credit_eligible": credit_eligible,
        "accepted": accepted,
        "reason": reason,
    }


class ValidationProbeManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.evaluation_interval = max(1, int(config.get("evaluation_interval", 100)))
        self.normalization = str(config.get("normalization", "warmup_envelope"))
        self.normalization_calibration_observations = max(
            1,
            int(config.get("normalization_calibration_observations", 3)),
        )
        self.category_aggregation = str(
            config.get(
                "category_aggregation", "mean_active_physics_groups"
            )
        )
        self.epsilon = float(config.get("epsilon", 1e-12))
        self.minimum_component_scale = max(
            self.epsilon,
            float(config.get("minimum_component_scale", 1.0e-3)),
        )
        # Regional diagnostics are RMSE values while component scores are
        # squared residual means, so their compatible floor is the square root.
        self.minimum_region_scale = math.sqrt(self.minimum_component_scale)
        self.region_normalization = str(
            config.get("region_normalization", "per_region_initial")
        )
        self.second_order_degradation_ratio = max(
            1.0,
            float(config.get("second_order_degradation_ratio", 1.0)),
        )
        self.second_order_guard_metric = str(
            config.get(
                "second_order_guard_metric",
                "physics_validation_score",
            )
        )
        self.seed = int(config.get("seed", 1729))
        self.category_weights = {
            str(name): float(value)
            for name, value in (config.get("category_weights") or {}).items()
        }
        self.policy = ValidationProbeBudgetPolicy(
            points_per_component=max(1, int(config.get("points_per_component", 32))),
            maximum_total_points=max(1, int(config.get("maximum_total_points", 512))),
        )
        self.audit_enabled = bool(config.get("audit_enabled", True))
        self.audit_seed = int(
            config.get("audit_seed", self.seed + int(config.get("audit_seed_offset", 7919)))
        )
        self.audit_sampling_strategy = str(
            config.get("audit_sampling_strategy", "uniform")
        )
        self.audit_policy = ValidationProbeBudgetPolicy(
            points_per_component=max(
                1,
                int(
                    config.get(
                        "audit_points_per_component",
                        max(2048, self.policy.points_per_component * 8),
                    )
                ),
            ),
            maximum_total_points=max(
                1,
                int(
                    config.get(
                        "audit_maximum_total_points",
                        max(32768, self.policy.maximum_total_points * 8),
                    )
                ),
            ),
        )
        self.respect_training_component_budget = bool(
            config.get("respect_training_component_budget", True)
        )
        self.probe_set: ValidationProbeSet | None = None
        self.audit_probe_set: ValidationProbeSet | None = None
        self.initial_scales: dict[str, float] = {}
        self.initial_region_scales: dict[int, float] = {}
        self.last_result: PhysicsValidationResult | None = None
        self.history: list[dict[str, Any]] = []
        self.audit_initial_scales: dict[str, float] = {}
        self.audit_initial_region_scales: dict[int, float] = {}
        self.audit_last_result: PhysicsValidationResult | None = None
        self.audit_history: list[dict[str, Any]] = []
        self.normalization_observations = 0
        self.audit_normalization_observations = 0
        self._expected_probe_set_id: str | None = None
        self._expected_probe_tensor_snapshot: dict[str, torch.Tensor] | None = None
        self._expected_audit_probe_set_id: str | None = None
        self._expected_audit_probe_tensor_snapshot: dict[str, torch.Tensor] | None = None

    def build_probes(
        self,
        registry: TrainingComponentRegistry,
        sampler: Any,
        budget_policy: ValidationProbeBudgetPolicy | None = None,
        seed: int | None = None,
    ) -> ValidationProbeSet | None:
        """Build disjoint decision and audit probes under one manager."""

        if not self.enabled or not registry.capabilities.supports_validation_probes:
            return None
        self.probe_set = self._build_probe_set(
            registry,
            sampler,
            budget_policy or self.policy,
            self.seed if seed is None else int(seed),
            respect_training_budget=self.respect_training_component_budget,
        )
        if self.audit_enabled:
            audit_seed = self.audit_seed
            if audit_seed == self.probe_set.seed:
                audit_seed += 1
            self.audit_probe_set = self._build_probe_set(
                registry,
                sampler,
                self.audit_policy,
                audit_seed,
                respect_training_budget=False,
                interior_strategy=self.audit_sampling_strategy,
            )
        self._validate_restored_probe(
            self.probe_set,
            self._expected_probe_set_id,
            self._expected_probe_tensor_snapshot,
            "decision",
        )
        if self.audit_probe_set is not None:
            self._validate_restored_probe(
                self.audit_probe_set,
                self._expected_audit_probe_set_id,
                self._expected_audit_probe_tensor_snapshot,
                "audit",
            )
        return self.probe_set

    def _validate_restored_probe(
        self,
        probe_set: ValidationProbeSet,
        expected_id: str | None,
        expected_snapshot: Mapping[str, torch.Tensor] | None,
        channel: str,
    ) -> None:
        if expected_id is not None and expected_id != probe_set.probe_set_id:
            raise ValueError(f"Validation {channel} probe set does not match the checkpoint")
        if expected_snapshot is None:
            return
        actual = self._batch_tensor_snapshot(probe_set.batch)
        if set(expected_snapshot) != set(actual) or any(
            not torch.equal(expected_snapshot[name], actual[name])
            for name in expected_snapshot
        ):
            raise ValueError(
                f"Validation {channel} probe coordinates do not match the checkpoint"
            )

    def _build_probe_set(
        self,
        registry: TrainingComponentRegistry,
        sampler: Any,
        policy: ValidationProbeBudgetPolicy,
        probe_seed: int,
        *,
        respect_training_budget: bool,
        interior_strategy: str | None = None,
    ) -> ValidationProbeSet:
        supported = [
            item
            for item in registry.sampling_components
            if item.validation_probe_supported
        ]
        total_point_multiplier = sum(
            max(1, int(item.metadata.get("point_multiplier", 1)))
            for item in supported
        )
        per_component = min(
            policy.points_per_component,
            max(1, policy.maximum_total_points // max(1, total_point_multiplier)),
        )
        component_budgets = {
            item.sampler_id: (
                min(
                    per_component,
                    int(item.point_budget)
                    if item.point_budget is not None and item.point_budget > 0
                    else per_component,
                )
                if respect_training_budget
                else per_component
            )
            for item in supported
        }
        overrides: dict[str, int] = {}
        for item in supported:
            overrides[str(item.sampler_id)] = component_budgets[
                item.sampler_id
            ]
            if item.domain_role == "interior":
                overrides["interior"] = component_budgets[item.sampler_id]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(probe_seed)
            np.random.seed(probe_seed % (2**32 - 1))
            torch.manual_seed(probe_seed)
            probe_sampler = sampler.probe_clone()
            if not respect_training_budget:
                probe_sampler.maximum_sampling_points = None
            if interior_strategy:
                # The audit channel owns its coverage policy. ``uniform`` is
                # deliberately domain-aware because it delegates to the PDE's
                # sample_domain implementation, including irregular geometry.
                probe_sampler.spec["interior"]["strategy"] = str(
                    interior_strategy
                )
            # A separate RNG seed is insufficient for unscrambled Halton,
            # Hammersley, configured Sobol, and grid samplers.  Move every
            # low-discrepancy stream to a deterministic validation-only range.
            sequence_offset = 1 + abs(probe_seed * 104729) % 1_000_003
            if hasattr(probe_sampler, "_sequence_offsets"):
                probe_sampler._sequence_offsets = {
                    name: sequence_offset
                    for name in probe_sampler._sequence_offsets
                }
            if hasattr(probe_sampler, "_restored_sobol_num_generated"):
                probe_sampler._restored_sobol_num_generated = sequence_offset
            batch = probe_sampler.sample(overrides)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        tensor_snapshot = self._batch_tensor_snapshot(batch)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "schema": registry.schema_version,
                    "seed": probe_seed,
                    "budgets": component_budgets,
                    "components": sorted(registry.component_ids),
                    "interior_strategy": interior_strategy,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for name, tensor in sorted(tensor_snapshot.items()):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            digest.update(contiguous.numpy().tobytes())
        fingerprint = digest.hexdigest()[:20]
        probe_set = ValidationProbeSet(
            probe_set_id=fingerprint,
            seed=probe_seed,
            component_budgets=component_budgets,
            batch=batch,
        )
        return probe_set

    @staticmethod
    def _batch_tensor_snapshot(batch: Any) -> dict[str, torch.Tensor]:
        """Capture probe coordinates/targets without serializing PDE callables."""

        result: dict[str, torch.Tensor] = {}

        def add(name: str, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                result[name] = value.detach().clone().cpu()
            elif isinstance(value, np.ndarray):
                result[name] = torch.as_tensor(value).detach().clone().cpu()
            elif isinstance(value, Mapping):
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                    add(f"{name}.{key}", item)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    add(f"{name}.{index}", item)

        add("domain_samples", getattr(batch, "domain_samples", None))
        for constraint_name, constraint in sorted(
            dict(getattr(batch, "constraint_samples", {}) or {}).items()
        ):
            for field_name in (
                "points", "numpy_points", "pointset_values", "paired_points",
                "normal_axes", "coefficient_left", "coefficient_right",
            ):
                add(
                    f"constraint_samples.{constraint_name}.{field_name}",
                    getattr(constraint, field_name, None),
                )
        add("observation_samples", getattr(batch, "observation_samples", None))
        return result

    def evaluate_physics_residual(
        self,
        model: torch.nn.Module,
        registry: TrainingComponentRegistry,
        loss_runtime: Any,
        *,
        weights: Mapping[str, float] | None = None,
        probe_set: ValidationProbeSet | None = None,
    ) -> PhysicsValidationMetrics | None:
        """Evaluate grouped squared physics residuals without reference data."""

        selected = probe_set or self.probe_set
        if not self.enabled or selected is None:
            return None
        component_ids = tuple(
            item.component_id
            for item in registry.loss_components
            if item.validation_enabled
        )
        categories = {
            item.component_id: item.category
            for item in registry.loss_components
            if item.validation_enabled
        }
        was_training = bool(model.training)
        loss_state = loss_runtime.state_dict()
        probe_active_before = getattr(loss_runtime, "_physics_probe_active", False)
        loss_runtime._physics_probe_active = True
        capture_residual_statistics = bool(
            getattr(loss_runtime, "capture_residual_statistics", False)
        )
        try:
            model.eval()
            loss_runtime.set_active_term_names(component_ids)
            loss_runtime.freeze_dynamic_weighting()
            loss_runtime.capture_residual_statistics = True
            with torch.enable_grad():
                _, raw = loss_runtime(model, selected.batch)
            residual_statistics = dict(
                getattr(loss_runtime, "last_residual_statistics", {}) or {}
            )
        finally:
            model.train(was_training)
            loss_runtime.capture_residual_statistics = capture_residual_statistics
            loss_runtime.load_state_dict(loss_state)
            loss_runtime._physics_probe_active = probe_active_before
            loss_runtime.last_term_values = []
            loss_runtime.last_term_names = []
            clear = getattr(loss_runtime.problem, "clear_autodiff_cache", None)
            if callable(clear):
                clear()
        component_metrics = {
            str(name): float(value)
            for name, value in dict(raw or {}).items()
            if str(name) in component_ids
            and str(categories.get(str(name), "")) != "auxiliary_constraint"
            and math.isfinite(float(value))
        }
        if not component_metrics:
            return None
        # Region statistics are emitted as RMSE by the loss runtime.  Square
        # them here so they use the same unit as component residual losses.
        region_metrics = {
            str(int(item["region_id"])): float(item["training_residual_rmse"]) ** 2
            for item in residual_statistics.get("region_statistics") or []
            if isinstance(item, Mapping)
            and item.get("region_id") is not None
            and item.get("training_residual_rmse") is not None
            and math.isfinite(float(item["training_residual_rmse"]))
        }
        region_metadata_provider = getattr(
            loss_runtime.problem, "training_residual_region_metadata", None
        )
        region_metadata = (
            dict(region_metadata_provider() or {})
            if callable(region_metadata_provider)
            else {}
        )
        worst_region_id = (
            max(region_metrics, key=region_metrics.get)
            if region_metrics
            else None
        )
        worst_region_loss = (
            region_metrics[worst_region_id]
            if worst_region_id is not None
            else None
        )
        utility_weights = {
            "pde": 1.0,
            "boundary": 1.0,
            "initial": 1.0,
            "observation": 1.0,
            **{str(name): float(value) for name, value in dict(weights or {}).items()},
        }

        def utility_category(category: str) -> str:
            return {
                "equation_residual": "pde",
                "initial_constraint": "initial",
                "observation": "observation",
            }.get(str(category), "boundary")

        def category_mean(category: str) -> float | None:
            values = [
                value
                for name, value in component_metrics.items()
                if utility_category(categories.get(name, "")) == category
            ]
            return sum(values) / len(values) if values else None

        pde_global = category_mean("pde")
        # The worst held-out physical region is a first-class PDE probe.  A
        # localized shock residual therefore cannot disappear in a global
        # domain average.
        pde_with_worst = (
            (pde_global + worst_region_loss) / 2.0
            if pde_global is not None and worst_region_loss is not None
            else pde_global
        )
        group_losses = {
            name: value
            for name in ("pde", "boundary", "initial", "observation")
            if (
                value := (
                    pde_with_worst if name == "pde" else category_mean(name)
                )
            ) is not None
        }
        weighted = [
            (value, utility_weights[name])
            for name, value in group_losses.items()
        ]
        weight_sum = sum(weight for _, weight in weighted)
        total = (
            sum(value * weight for value, weight in weighted)
            / max(weight_sum, self.epsilon)
        )
        return PhysicsValidationMetrics(
            total_score=float(total),
            pde_loss=pde_with_worst,
            boundary_loss=category_mean("boundary"),
            initial_loss=category_mean("initial"),
            observation_loss=category_mean("observation"),
            component_metrics=component_metrics,
            component_categories={
                name: categories.get(name, "auxiliary_constraint")
                for name in component_metrics
            },
            region_metrics=region_metrics,
            region_metadata=region_metadata,
            worst_region_id=worst_region_id,
            worst_region_loss=worst_region_loss,
        )

    def evaluate(
        self,
        model: torch.nn.Module,
        registry: TrainingComponentRegistry,
        loss_runtime: Any,
        probe_set: ValidationProbeSet | None = None,
        *,
        channel: str = "decision",
    ) -> PhysicsValidationResult | None:
        is_audit = str(channel) == "audit"
        selected = probe_set or (
            self.audit_probe_set if is_audit else self.probe_set
        )
        if not self.enabled or selected is None:
            return None
        initial_scales = (
            self.audit_initial_scales if is_audit else self.initial_scales
        )
        initial_region_scales = (
            self.audit_initial_region_scales
            if is_audit
            else self.initial_region_scales
        )
        loss_state = loss_runtime.state_dict()
        probe_active_before = getattr(loss_runtime, "_physics_probe_active", False)
        loss_runtime._physics_probe_active = True
        was_training = bool(model.training)
        capture_residual_statistics = bool(
            getattr(loss_runtime, "capture_residual_statistics", False)
        )
        component_ids = {
            item.component_id
            for item in registry.loss_components
            if item.validation_enabled
        }
        try:
            model.eval()
            loss_runtime.set_active_term_names(component_ids)
            loss_runtime.freeze_dynamic_weighting()
            loss_runtime.capture_residual_statistics = True
            with torch.enable_grad():
                _, components = loss_runtime(model, selected.batch)
            residual_statistics = dict(
                getattr(loss_runtime, "last_residual_statistics", {}) or {}
            )
            raw = {
                component_id: float(components[component_id])
                for component_id in sorted(component_ids)
                if component_id in components
                and math.isfinite(float(components[component_id]))
            }
            auxiliary_scores: dict[str, float] = {}
            score_provider = getattr(
                loss_runtime.problem,
                "physics_validation_scores",
                None,
            )
            if callable(score_provider):
                provided = score_provider(model, selected.batch) or {}
                auxiliary_scores = {
                    str(name): float(value)
                    for name, value in provided.items()
                    if math.isfinite(float(value))
                }
        finally:
            model.train(was_training)
            loss_runtime.capture_residual_statistics = (
                capture_residual_statistics
            )
            loss_runtime.load_state_dict(loss_state)
            loss_runtime._physics_probe_active = probe_active_before
            loss_runtime.last_term_values = []
            loss_runtime.last_term_names = []
            clear = getattr(loss_runtime.problem, "clear_autodiff_cache", None)
            if callable(clear):
                clear()
        observations = (
            self.audit_normalization_observations
            if is_audit
            else self.normalization_observations
        )
        calibrating = bool(
            self.normalization == "warmup_envelope"
            and observations < self.normalization_calibration_observations
        )
        if not initial_scales or calibrating:
            for name, value in raw.items():
                initial_scales[name] = max(
                    float(initial_scales.get(name, 0.0)),
                    abs(value),
                    self.minimum_component_scale,
                )
        normalized = {
            name: value / max(initial_scales.get(name, abs(value)), self.epsilon)
            for name, value in raw.items()
        }
        raw_region_scores = {
            int(item["region_id"]): float(
                item["training_residual_rmse"]
            )
            for item in residual_statistics.get("region_statistics") or []
            if isinstance(item, Mapping)
            and item.get("region_id") is not None
            and item.get("training_residual_rmse") is not None
            and math.isfinite(float(item["training_residual_rmse"]))
        }
        if (
            raw_region_scores
            and self.region_normalization != "shared_pde_rmse"
            and (not initial_region_scales or calibrating)
        ):
            for region_id, value in raw_region_scores.items():
                initial_region_scales[region_id] = max(
                    float(initial_region_scales.get(region_id, 0.0)),
                    abs(value),
                    self.minimum_region_scale,
                )
        if self.region_normalization == "shared_pde_rmse":
            governing_scales = [
                initial_scales[descriptor.component_id]
                for descriptor in registry.loss_components
                if descriptor.category == "equation_residual"
                and descriptor.component_id in initial_scales
            ]
            shared_region_scale = math.sqrt(
                max(
                    sum(governing_scales) / len(governing_scales)
                    if governing_scales
                    else self.minimum_component_scale,
                    self.epsilon,
                )
            )
            normalized_region_scores = {
                region_id: value / shared_region_scale
                for region_id, value in raw_region_scores.items()
            }
        else:
            normalized_region_scores = {
                region_id: value
                / max(
                    initial_region_scales.get(
                        region_id, abs(value)
                    ),
                    self.minimum_region_scale,
                )
                for region_id, value in raw_region_scores.items()
            }
        result = self._aggregate(
            registry,
            raw,
            normalized,
            normalized_region_scores=normalized_region_scores,
            auxiliary_scores=auxiliary_scores,
        )
        if is_audit:
            self.audit_last_result = result
        else:
            self.last_result = result
        return result

    def next_due_iteration(self, current_iteration: int) -> int:
        last_iteration = (
            int(self.history[-1].get("iteration", 0))
            if self.history
            else 0
        )
        return max(
            int(current_iteration),
            last_iteration + int(self.evaluation_interval),
        )

    def is_due(self, current_iteration: int) -> bool:
        last_iteration = (
            int(self.history[-1].get("iteration", 0))
            if self.history
            else 0
        )
        return (
            int(current_iteration) - last_iteration
            >= int(self.evaluation_interval)
        )

    def _aggregate(
        self,
        registry: TrainingComponentRegistry,
        raw: dict[str, float],
        normalized: dict[str, float],
        *,
        normalized_region_scores: dict[int, float] | None = None,
        auxiliary_scores: dict[str, float] | None = None,
    ) -> PhysicsValidationResult:
        by_category: dict[str, list[float]] = {}
        for descriptor in registry.loss_components:
            if descriptor.component_id in normalized:
                by_category.setdefault(descriptor.category, []).append(
                    normalized[descriptor.component_id]
                )
        regional = dict(normalized_region_scores or {})
        if regional:
            pde_category = next(
                (
                    descriptor.category
                    for descriptor in registry.loss_components
                    if descriptor.category == "equation_residual"
                ),
                "equation_residual",
            )
            by_category.setdefault(pde_category, []).extend(
                regional.values()
            )
        category_scores: dict[str, float] = {}
        for category, values in sorted(by_category.items()):
            category_scores[category] = sum(values) / len(values)
        weights = {
            category: self.category_weights.get(category, 1.0)
            for category in category_scores
        }
        weight_total = sum(weights.values())
        aggregate = (
            sum(category_scores[name] * weights[name] for name in category_scores)
            / max(weight_total, self.epsilon)
            if category_scores
            else float("inf")
        )
        ordered = sorted(normalized.values())
        p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1) if ordered else 0
        worst_id = max(normalized, key=normalized.get) if normalized else None
        worst_region_id = (
            max(regional, key=regional.get) if regional else None
        )
        return PhysicsValidationResult(
            aggregate_score=float(aggregate),
            category_scores=category_scores,
            component_scores=raw,
            normalized_component_scores=normalized,
            worst_component_id=worst_id,
            worst_component_score=normalized.get(worst_id) if worst_id else None,
            median_component_score=median(ordered) if ordered else None,
            p90_component_score=ordered[p90_index] if ordered else None,
            normalized_region_scores=regional,
            mean_region_score=(
                sum(regional.values()) / len(regional)
                if regional
                else None
            ),
            worst_region_id=worst_region_id,
            worst_region_score=(
                regional.get(worst_region_id)
                if worst_region_id is not None
                else None
            ),
            auxiliary_scores=dict(auxiliary_scores or {}),
        )

    def score_for_metric(
        self,
        result: PhysicsValidationResult | None,
        metric_name: str,
    ) -> float | None:
        """Return a finite, label-free score exposed by a fixed probe."""

        if result is None:
            return None
        if metric_name == "physics_validation_score":
            value = float(result.aggregate_score)
        else:
            raw = result.auxiliary_scores.get(str(metric_name))
            if raw is None:
                return None
            value = float(raw)
        return value if math.isfinite(value) else None

    def second_order_guard_score(
        self,
        result: PhysicsValidationResult | None,
    ) -> float | None:
        return self.score_for_metric(result, self.second_order_guard_metric)

    def record(
        self,
        iteration: int,
        stage_id: str,
        result: PhysicsValidationResult,
        *,
        channel: str = "decision",
    ) -> None:
        history = self.audit_history if str(channel) == "audit" else self.history
        history.append(
            {"iteration": int(iteration), "stage_id": str(stage_id), **result.to_dict()}
        )
        if str(channel) == "audit":
            self.audit_normalization_observations += 1
        else:
            self.normalization_observations += 1
        history[-1]["normalization_calibrated"] = self.normalization_ready(channel)

    def normalization_ready(self, channel: str = "decision") -> bool:
        observations = (self.audit_normalization_observations
                        if channel == "audit" else self.normalization_observations)
        return (self.normalization != "warmup_envelope"
                or observations >= self.normalization_calibration_observations)

    def audit_score_context(self) -> str | None:
        """Identify a frozen score scale; warmup scores cannot select a model."""
        if not self.normalization_ready("audit"):
            return None
        payload = {"probe_set_id": self.audit_probe_set.probe_set_id if self.audit_probe_set else None,
                   "components": self.audit_initial_scales,
                   "regions": self.audit_initial_region_scales,
                   "category_aggregation": self.category_aggregation,
                   "region_normalization": self.region_normalization,
                   "weights": self.category_weights}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def guard_result(self) -> PhysicsValidationResult | None:
        return self.audit_last_result or self.last_result

    @property
    def guard_history(self) -> list[dict[str, Any]]:
        return self.audit_history or self.history

    def state_dict(self) -> dict[str, Any]:
        return {
            "probe_set_id": self.probe_set.probe_set_id if self.probe_set else None,
            "probe_seed": self.probe_set.seed if self.probe_set else self.seed,
            "component_budgets": dict(self.probe_set.component_budgets) if self.probe_set else {},
            "probe_tensor_snapshot": (
                self._batch_tensor_snapshot(self.probe_set.batch)
                if self.probe_set is not None
                else None
            ),
            "audit_probe_set_id": (
                self.audit_probe_set.probe_set_id
                if self.audit_probe_set is not None
                else None
            ),
            "audit_probe_seed": (
                self.audit_probe_set.seed
                if self.audit_probe_set is not None
                else self.audit_seed
            ),
            "audit_component_budgets": (
                dict(self.audit_probe_set.component_budgets)
                if self.audit_probe_set is not None
                else {}
            ),
            "audit_probe_tensor_snapshot": (
                self._batch_tensor_snapshot(self.audit_probe_set.batch)
                if self.audit_probe_set is not None
                else None
            ),
            "initial_scales": dict(self.initial_scales),
            "normalization_observations": int(self.normalization_observations),
            "initial_region_scales": {
                str(name): float(value)
                for name, value in self.initial_region_scales.items()
            },
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "history": list(self.history),
            "audit_initial_scales": dict(self.audit_initial_scales),
            "audit_normalization_observations": int(
                self.audit_normalization_observations
            ),
            "audit_initial_region_scales": {
                str(name): float(value)
                for name, value in self.audit_initial_region_scales.items()
            },
            "audit_last_result": (
                self.audit_last_result.to_dict()
                if self.audit_last_result is not None
                else None
            ),
            "audit_history": list(self.audit_history),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        expected = state.get("probe_set_id")
        self._expected_probe_set_id = str(expected) if expected else None
        stored_seed = state.get("probe_seed")
        if stored_seed is not None:
            self.seed = int(stored_seed)
        snapshot = state.get("probe_tensor_snapshot")
        self._expected_probe_tensor_snapshot = (
            {
                str(name): value.detach().clone().cpu()
                for name, value in dict(snapshot).items()
                if isinstance(value, torch.Tensor)
            }
            if isinstance(snapshot, Mapping)
            else None
        )
        audit_expected = state.get("audit_probe_set_id")
        self._expected_audit_probe_set_id = (
            str(audit_expected) if audit_expected else None
        )
        stored_audit_seed = state.get("audit_probe_seed")
        if stored_audit_seed is not None:
            self.audit_seed = int(stored_audit_seed)
        audit_snapshot = state.get("audit_probe_tensor_snapshot")
        self._expected_audit_probe_tensor_snapshot = (
            {
                str(name): value.detach().clone().cpu()
                for name, value in dict(audit_snapshot).items()
                if isinstance(value, torch.Tensor)
            }
            if isinstance(audit_snapshot, Mapping)
            else None
        )
        if self.probe_set is not None:
            self._validate_restored_probe(
                self.probe_set,
                self._expected_probe_set_id,
                self._expected_probe_tensor_snapshot,
                "decision",
            )
        if self.audit_probe_set is not None:
            self._validate_restored_probe(
                self.audit_probe_set,
                self._expected_audit_probe_set_id,
                self._expected_audit_probe_tensor_snapshot,
                "audit",
            )
        self.initial_scales = {
            str(name): max(float(value), self.minimum_component_scale)
            for name, value in (state.get("initial_scales") or {}).items()
        }
        self.initial_region_scales = {
            int(name): max(float(value), self.minimum_region_scale)
            for name, value in (
                state.get("initial_region_scales") or {}
            ).items()
        }
        self.audit_initial_scales = {
            str(name): max(float(value), self.minimum_component_scale)
            for name, value in (state.get("audit_initial_scales") or {}).items()
        }
        self.audit_initial_region_scales = {
            int(name): max(float(value), self.minimum_region_scale)
            for name, value in (
                state.get("audit_initial_region_scales") or {}
            ).items()
        }
        # Checkpoints created before warm-up envelope calibration already own
        # a fixed denominator.  Treat those scales as locked instead of
        # silently changing their score semantics after resume.
        legacy_locked = bool(
            "normalization_observations" not in state
            and (self.initial_scales or state.get("history"))
        )
        legacy_audit_locked = bool(
            "audit_normalization_observations" not in state
            and (self.audit_initial_scales or state.get("audit_history"))
        )
        self.normalization_observations = int(
            state.get(
                "normalization_observations",
                self.normalization_calibration_observations if legacy_locked else 0,
            )
        )
        self.audit_normalization_observations = int(
            state.get(
                "audit_normalization_observations",
                self.normalization_calibration_observations
                if legacy_audit_locked
                else 0,
            )
        )
        result = state.get("last_result")
        if result:
            payload = dict(result)
            payload["normalized_region_scores"] = {
                int(name): float(value)
                for name, value in (
                    payload.get("normalized_region_scores") or {}
                ).items()
            }
            payload["auxiliary_scores"] = {
                str(name): float(value)
                for name, value in (
                    payload.get("auxiliary_scores") or {}
                ).items()
            }
            self.last_result = PhysicsValidationResult(**payload)
        else:
            self.last_result = None
        self.history = list(state.get("history") or [])
        audit_result = state.get("audit_last_result")
        if audit_result:
            audit_payload = dict(audit_result)
            audit_payload["normalized_region_scores"] = {
                int(name): float(value)
                for name, value in (
                    audit_payload.get("normalized_region_scores") or {}
                ).items()
            }
            audit_payload["auxiliary_scores"] = {
                str(name): float(value)
                for name, value in (
                    audit_payload.get("auxiliary_scores") or {}
                ).items()
            }
            self.audit_last_result = PhysicsValidationResult(**audit_payload)
        else:
            self.audit_last_result = None
        self.audit_history = list(state.get("audit_history") or [])
