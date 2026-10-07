"""Reference-free bottleneck contracts and bounded online trial control."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Mapping


GRADIENT_ACTIONS = {"GRADIENT_SURGERY", "GRADIENT_CONFLICT_MITIGATION", "GRADIENT_TOPOLOGY_ADAPTATION"}


def validate_strategy(config):
    if not isinstance(config, dict):
        raise ValueError("adaptive_strategy must be an object")
    unknown = set(config) - {"bottleneck_enabled", "progressive_enabled", "maximum_same_family", "freeze_step", "temporal", "trial_control", "influence_guard", "llm_efficiency"}
    if unknown:
        raise ValueError(f"Unknown adaptive_strategy fields: {sorted(unknown)}")
    for name in ("bottleneck_enabled", "progressive_enabled"):
        if name in config and not isinstance(config[name], bool):
            raise ValueError(f"{name} must be boolean")
    for name, low, high in (("maximum_same_family", 1, 5), ("freeze_step", 1000, 20000)):
        if name in config and (type(config[name]) is not int or not low <= config[name] <= high):
            raise ValueError(f"{name} outside [{low}, {high}]")
    temporal = config.get("temporal", {})
    if not isinstance(temporal, dict) or set(temporal) - {"enabled", "problem_ids", "bins", "future_weight", "minimum_dwell", "maximum_dwell", "advance_ratio", "full_domain_step", "health_gate", "initial_tolerance", "stable_checks", "retreat_ratio", "initial_weight_multiplier", "boundary_ratio"}:
        raise ValueError("Invalid temporal settings")
    if "enabled" in temporal and not isinstance(temporal["enabled"], bool):
        raise ValueError("temporal.enabled must be boolean")
    for section in ("trial_control", "influence_guard", "llm_efficiency"):
        settings = config.get(section, {})
        if not isinstance(settings, dict):
            raise ValueError(f"{section} must be an object")
        allowed = {
            "trial_control": {"enabled", "extension_steps", "retirement_steps", "safety_degradation"},
            "influence_guard": {"enabled", "points", "scale_floor", "maximum_output_ratio", "maximum_gradient_ratio", "maximum_residual_ratio"},
            "llm_efficiency": {"enabled", "maximum_calls", "maximum_wall_seconds", "refresh_steps", "relative_change"},
        }[section]
        if set(settings) - allowed:
            raise ValueError(f"Unknown {section} fields: {sorted(set(settings) - allowed)}")
        if "enabled" in settings and type(settings["enabled"]) is not bool:
            raise ValueError(f"{section}.enabled must be boolean")
        for name, value in settings.items():
            if name != "enabled" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid {section}.{name}")
        for name in ("extension_steps", "retirement_steps", "points", "maximum_calls", "refresh_steps"):
            if name in settings and type(settings[name]) is not int:
                raise ValueError(f"{section}.{name} must be integer")
    if "health_gate" in temporal and type(temporal["health_gate"]) is not bool:
        raise ValueError("temporal.health_gate must be boolean")
    for name in ("initial_tolerance", "initial_weight_multiplier", "retreat_ratio", "boundary_ratio"):
        if name in temporal and (isinstance(temporal[name], bool) or not isinstance(temporal[name], (int, float)) or not math.isfinite(temporal[name]) or temporal[name] <= 0):
            raise ValueError(f"Invalid temporal.{name}")
    if "stable_checks" in temporal and (type(temporal["stable_checks"]) is not int or temporal["stable_checks"] < 1):
        raise ValueError("temporal.stable_checks must be a positive integer")
    for name in ("advance_ratio", "future_weight"):
        value = temporal.get(name, 0.5)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"temporal.{name} must be in (0, 1]")
    for name in ("bins", "minimum_dwell", "maximum_dwell", "full_domain_step"):
        if name in temporal and (type(temporal[name]) is not int or temporal[name] < 1):
            raise ValueError(f"temporal.{name} must be a positive integer")
    if temporal.get("minimum_dwell", 1000) > temporal.get("maximum_dwell", 3000):
        raise ValueError("temporal minimum_dwell exceeds maximum_dwell")
    if temporal.get("full_domain_step", 12000) > config.get("freeze_step", 14000):
        raise ValueError("full-domain learning must begin by adaptation freeze")


def mechanism_family(primitive: str) -> str:
    if primitive == "ENTER_LBFGS_PHASE":
        return "optimization"
    if primitive in {"PROJECT_CONFLICTING_GRADIENTS", "MITIGATE_GRADIENT_CONFLICT", "SPLIT_CONFLICTING_TOPOLOGY"}:
        return "gradient"
    if "FOURIER" in primitive:
        return "representation"
    if "LOCALIZED" in primitive or primitive == "REWEIGHT_RESIDUAL_REGIONS":
        return "local"
    if "SAMPLING" in primitive:
        return "sampling"
    if any(word in primitive for word in ("WEIGHT", "BALANCE", "EMPHAS")):
        return "objective"
    return "capacity"


def bottleneck_contract(state: Mapping[str, Any], primitives, config: Mapping[str, Any]) -> dict:
    opt, grad, spatial, frequency, physics = (dict(state.get(k) or {}) for k in
        ("optimization", "gradient", "spatial", "frequency", "physics"))
    hypotheses = []
    if spatial.get("localized_residual_concentration"):
        hypotheses += ["local", "sampling"]
    if frequency.get("has_multiscale_or_oscillatory_evidence"):
        hypotheses += ["representation"]
    if physics.get("initial_condition_dominant") or physics.get("boundary_condition_dominant") or grad.get("gradient_magnitude_imbalance"):
        hypotheses += ["objective"]
    if opt.get("generic_capacity_or_representation_limit"):
        hypotheses += ["capacity"]
    gradient_supported = bool(grad.get("persistent_gradient_conflict") and opt.get("physics_plateau"))
    if gradient_supported:
        hypotheses += ["gradient"]
    if opt.get("physics_plateau") or opt.get("training_plateau"):
        hypotheses += ["sampling", "capacity", "optimization"]
    hypotheses = list(dict.fromkeys(hypotheses))
    return {"enabled": True, "allowed_primitives": list(primitives),
        "supported_hypotheses": hypotheses, "gradient_supported": gradient_supported,
        "maximum_same_family": int(config.get("maximum_same_family", 2)),
        "directive": "Propose distinct evidence-backed bottleneck hypotheses. Gradient conflict alone is insufficient: gradient interventions require persistent conflict AND held-out physics plateau. The allowed_primitives list overrides the global capability catalogue. KEEP is valid when evidence is insufficient. Do not invent reference-solution errors.",
        "time_progress": deepcopy(state.get("pde", {}).get("online_time_progress"))}


def program_rejection(program: Mapping[str, Any], contract: Mapping[str, Any], peers=()) -> str | None:
    if not contract or program.get("action") == "KEEP":
        return None
    operations = list(program.get("operations") or [])
    if not operations:
        mechanism = program.get("mechanism")
        if mechanism in {None, "AUTO", "NONE"}:
            if not contract.get("enabled", True):
                from .adaptation_program import PRIMITIVES_BY_ACTION
                possible = PRIMITIVES_BY_ACTION.get(program.get("action"), set())
                if possible and possible.issubset(set(contract["allowed_primitives"])):
                    return None
            return "online trial requires explicit operations from the allowed capability list"
        operations = [{"primitive": mechanism}]
    names = [op.get("primitive") for op in operations]
    if any(name not in contract["allowed_primitives"] for name in names):
        return "candidate uses a capability disabled in this experiment"
    if not contract.get("enabled", True):
        return None
    families = set(map(mechanism_family, names))
    if (program.get("action") in GRADIENT_ACTIONS or "gradient" in families) and not contract.get("gradient_supported"):
        return "gradient intervention requires persistent conflict and held-out physics plateau"
    supported = set(contract.get("supported_hypotheses", []))
    if not families.issubset(supported):
        return "candidate does not address a currently supported bottleneck hypothesis"
    used = Counter()
    for peer in peers:
        p = peer.program.to_dict() if hasattr(peer, "program") else peer.policy_spec["program"]
        if p.get("action") != "KEEP":
            operations = p.get("operations") or [{"primitive": p.get("mechanism")}]
            used.update(set(mechanism_family(op["primitive"]) for op in operations if op.get("primitive")))
    if len(supported) > 1 and families and all(used[f] >= contract.get("maximum_same_family", 2) for f in families):
        return "this bottleneck family is already represented; propose a distinct supported hypothesis or KEEP"
    return None


def trial_transition(current: float, feedback: Mapping[str, Any], horizon: str) -> tuple[float, str]:
    """Only credible improvement earns more influence; uncertainty never does."""
    if feedback.get("confounded") or not feedback.get("credit_eligible", False):
        return current, "probation_uncertain"
    if feedback.get("accepted", False):
        return max(current, {"short": 0.5, "medium": 0.75, "persistent": 1.0}[horizon]), ("confirmed" if horizon == "persistent" else "provisional_success")
    return (max(0.0625, current * 0.5), "attenuated") if horizon == "short" else (0.0, "rejected")


def control_snapshot(loss_state: Mapping[str, Any]) -> dict:
    return {name: deepcopy(loss_state.get(name)) for name in (
        "runtime_base_weights", "gradient_projection_strength", "online_point_weighting", "online_gradnorm")}


def blend_control(before: Mapping[str, Any], target: Mapping[str, Any], influence: float) -> dict:
    alpha = min(1.0, max(0.0, float(influence)))
    if alpha == 0:
        return deepcopy(dict(before))
    result = deepcopy(dict(target))
    result["runtime_base_weights"] = {key: float(value) * (float(target["runtime_base_weights"].get(key, value)) / float(value)) ** alpha
        for key, value in before["runtime_base_weights"].items()}
    old, new = float(before.get("gradient_projection_strength") or 0), float(target.get("gradient_projection_strength") or 0)
    result["gradient_projection_strength"] = old + alpha * (new - old)
    for name in ("online_point_weighting", "online_gradnorm"):
        if result.get(name):
            start = float((before.get(name) or {}).get("strength", 0))
            result[name]["strength"] = start + alpha * (float(result[name].get("strength", 0)) - start)
    return result


def advance_time_frontier(state: dict, bin_scores: list[float | None], step: int, *, blocked: bool, config: Mapping[str, Any]) -> dict | None:
    """Observe fixed physical probes, retain early-time coverage, freeze before L-BFGS."""
    if config.get("health_gate"):
        return advance_healthy_time_frontier(state, bin_scores, step, blocked=blocked, config=config)
    bins = int(config.get("bins", 4))
    state.setdefault("level", 1)
    state.setdefault("last_advance", 0)
    if "reference_scores" not in state and all(v is not None and math.isfinite(v) for v in bin_scores):
        state["reference_scores"] = [max(float(v), 1e-12) for v in bin_scores]
    state["bin_scores"] = bin_scores
    old = int(state["level"])
    if old >= bins:
        return None
    freeze = int(config.get("full_domain_step", 12000))
    if step >= freeze and not blocked:
        level, reason = bins, "full_domain_budget_reserve"
    elif blocked or "reference_scores" not in state:
        return None
    else:
        prefix = bin_scores[:old]
        if any(v is None or not math.isfinite(v) for v in prefix):
            return None
        ratios = [float(v) / ref for v, ref in zip(prefix, state["reference_scores"])]
        dwell = step - int(state["last_advance"])
        ready = max(ratios) <= float(config.get("advance_ratio", 0.7))
        if dwell >= int(config.get("minimum_dwell", 1000)) and ready:
            level, reason = old + 1, "prefix_residual_improved"
        elif dwell >= int(config.get("maximum_dwell", 3000)):
            level, reason = old + 1, "bounded_dwell_budget_reserve"
        else:
            return None
    state.update(level=level, last_advance=int(step))
    return {"event": "ONLINE_TIME_FRONTIER_ADVANCED", "step": step, "old_level": old, "level": level,
        "bins": bins, "reason": reason, "bin_scores": bin_scores, "reference_solution_used": False}


def advance_healthy_time_frontier(state, scores, step, *, blocked, config):
    """Evidence controls the frontier; budget expiry releases weighting, not mastery."""
    bins = int(config.get("bins", 4))
    state.setdefault("level", 1)
    state.setdefault("last_advance", 0)
    state["bin_scores"] = scores
    old = state["level"]
    if blocked:
        state["stable_count"] = 0
        return None
    if state.get("coverage_released"):
        return None
    if step >= config.get("full_domain_step", 12000):
        state.update(coverage_released=True, initial_weight_multiplier=1.0)
        return {"event": "ONLINE_TIME_COVERAGE_RELEASED", "step": step,
                "level": old, "reason": "budget_reserve_not_mastery", "reference_solution_used": False}
    if len(scores) != bins or any(v is None or not math.isfinite(v) for v in scores):
        state["stable_count"] = 0
        return None
    state.setdefault("reference_scores", [max(float(v), 1e-12) for v in scores])
    initial = state.get("initial_errors", {})
    initial_ok = bool(initial) and all(math.isfinite(v) and v <= config.get("initial_tolerance", 0.1)
                                       for v in initial.values())
    boundary = state.get("boundary_scores", [])
    if boundary:
        state.setdefault("boundary_reference", [max(float(v), 1e-12) for v in boundary])
    boundary_ok = bool(boundary) and all(math.isfinite(v) and v <= r * config.get("boundary_ratio", 1.1)
                                      for v, r in zip(boundary, state.get("boundary_reference", [])))
    state["initial_ready"] = initial_ok
    state["initial_weight_multiplier"] = 1.0 if initial_ok else config.get("initial_weight_multiplier", 4.0)
    best = state.setdefault("best_prefix_scores", list(scores))
    degraded = any(v > max(b, 1e-12) * config.get("retreat_ratio", 1.5) for v, b in zip(scores[:old], best[:old]))
    for i, value in enumerate(scores[:old]):
        best[i] = min(best[i], value)
    if old > 1 and (not initial_ok or degraded):
        state.update(level=old - 1, last_advance=step, stable_count=0)
        state["best_prefix_scores"] = list(scores)
        return {"event": "ONLINE_TIME_FRONTIER_RETREATED", "step": step, "old_level": old,
                "level": old - 1, "reason": "initial_or_prefix_degraded", "reference_solution_used": False}
    ready = initial_ok and boundary_ok and max(v / r for v, r in zip(scores[:old], state["reference_scores"][:old])) <= config.get("advance_ratio", 0.7)
    # Track a narrow band at the next frontier using the same fixed coordinates.
    junction = state.get("junction_score")
    if junction is not None and math.isfinite(junction):
        key = str(old)
        baseline = state.setdefault("junction_reference", {}).setdefault(key, max(junction, 1e-12))
        ready = ready and junction <= baseline * config.get("boundary_ratio", 1.1)
    else:
        ready = False
    state["stable_count"] = state.get("stable_count", 0) + 1 if ready else 0
    if old < bins and state["stable_count"] >= config.get("stable_checks", 3) and step - state["last_advance"] >= config.get("minimum_dwell", 1000):
        state.update(level=old + 1, last_advance=step, stable_count=0)
        state["best_prefix_scores"] = list(scores)
        return {"event": "ONLINE_TIME_FRONTIER_ADVANCED", "step": step, "old_level": old,
                "level": old + 1, "reason": "initial_and_prefix_stable", "reference_solution_used": False}
    return None
