"""Configuration loading for the single PINN-Online experiment architecture."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import floor
from pathlib import Path
from typing import Any, Mapping

import yaml

from morph.pipeline.g_training_evaluation.training.adaptation_program import (
    PRIMITIVE_LIBRARY,
)


ONLINE_METHOD = "single_trajectory_eoh_policy_population"
LEGACY_ONLINE_METHODS = frozenset({"single_trajectory_eohs_policy_population"})
PINNMORPH_DECISION_MODE = "llm_evolution"
RANDOM_MORPH_DECISION_MODE = "random_morph"
NO_MORPH_ABLATION = "no_morph"
RANDOM_MORPH_ABLATION = "random_morph"
RANDOM_SELECTION_ABLATION = "random_selection"
RANDOM_EVOLUTION_ABLATION = "random_evolution"
LEGACY_STRUCTURAL_PRIMITIVES = (
    "WIDEN", "ADD_LAYER", "PRUNE", "REGROW", "REDISTRIBUTE", "LOCAL_REINIT",
    "ADD_RESIDUAL_BRANCH", "ADD_CORRECTION_BRANCH", "ADD_FOURIER_BRANCH",
    "ADD_GATED_BRANCH", "ADD_LOCALIZED_EXPERT",
)
SUPPORTED_ABLATION_MODES = frozenset(
    {"none", "", NO_MORPH_ABLATION, RANDOM_MORPH_ABLATION,
     RANDOM_SELECTION_ABLATION, RANDOM_EVOLUTION_ABLATION}
)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    source = Path(path).resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise ValueError(f"Cyclic experiment config inheritance: {chain}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {source}")
    parent = payload.get("extends")
    if parent:
        parent_path = (source.parent / str(parent)).resolve()
        resolved = deep_merge(
            load_experiment_config(parent_path, (*_stack, source)), payload
        )
    else:
        resolved = deepcopy(payload)
    resolved["_config_path"] = str(source)
    if not _stack:
        if str(resolved.get("method") or "") in LEGACY_ONLINE_METHODS:
            resolved["_legacy_method"] = str(resolved["method"])
            resolved["method"] = ONLINE_METHOD
        validate_experiment_config(resolved)
    return resolved


def apply_ablation_mode(config: Mapping[str, Any], mode: str | None) -> dict[str, Any]:
    """Return a validated config with a PINNMorph ablation mode applied."""

    selected = str(mode or "none").strip().lower()
    if selected in {"", "none"}:
        result = deepcopy(dict(config))
        result["ablation_mode"] = "none"
        validate_experiment_config(result)
        return result
    if selected not in SUPPORTED_ABLATION_MODES:
        raise ValueError(
            f"Unsupported ablation_mode: {selected}"
        )
    result = deepcopy(dict(config))
    result["ablation_mode"] = selected
    online = deepcopy(dict(result.get("online_morphing") or {}))
    policy_set = deepcopy(dict(online.get("policy_set") or {}))
    evolution = deepcopy(dict(policy_set.get("evolution") or {}))
    if selected == NO_MORPH_ABLATION:
        result["phase"] = "pinn_online_no_morph"
        result["result_group"] = "pinn_online_no_morph"
        result["research_question"] = "Fixed base PINN without any online adaptation"
        online["enabled"] = False
        online["decision_mode"] = NO_MORPH_ABLATION
        policy_set["enabled"] = False
        evolution["enabled"] = False
        # Keep explicit matched schedules; legacy Adam-only ablations remain supported.
        switch_enabled = result.get("no_morph_optimizer_schedule", "adam_only") == "matched"
        result.setdefault("algorithm_overrides", {}).setdefault("training", {}).setdefault(
            "extra_parameters", {}
        ).setdefault("adaptive_control", {}).setdefault("optimizer_phase_switch", {})["enabled"] = switch_enabled
    elif selected == RANDOM_MORPH_ABLATION:
        result["phase"] = "pinn_online_random_morph"
        result["result_group"] = "pinn_online_random_morph"
        result["research_question"] = (
            "Random valid online interventions under the PINNMorph executor"
        )
        online["enabled"] = True
        online["decision_mode"] = RANDOM_MORPH_DECISION_MODE
        policy_set["enabled"] = False
        evolution["enabled"] = False
    else:
        result["phase"] = f"pinn_online_{selected}"
        result["result_group"] = f"pinn_online_{selected}"
        result["research_question"] = (
            "Uniform final selection from executable population proposals"
            if selected == RANDOM_SELECTION_ABLATION
            else "Population evolution with randomized direction and no execution feedback"
        )
        online["enabled"] = True
        online["decision_mode"] = (
            RANDOM_SELECTION_ABLATION if selected == RANDOM_SELECTION_ABLATION
            else PINNMORPH_DECISION_MODE
        )
        policy_set["enabled"] = True
        evolution["enabled"] = True
        evolution["direction"] = (
            "random" if selected == RANDOM_EVOLUTION_ABLATION else "feedback"
        )
        if selected == RANDOM_EVOLUTION_ABLATION:
            policy_set["prior_evidence_auto_refresh"] = False
            policy_set["prior_evidence_path"] = None
    policy_set["evolution"] = evolution
    online["policy_set"] = policy_set
    result["online_morphing"] = online
    validate_experiment_config(result)
    return result


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    required = ("schema_version", "phase", "pde", "method", "seeds", "budget", "runtime")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing experiment config fields: {', '.join(missing)}")
    if int(config["schema_version"]) != 1:
        raise ValueError("Unsupported experiment config schema_version")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(item, bool) for item in seeds):
        raise ValueError("seeds must be a non-empty arbitrary integer list")
    if len({int(item) for item in seeds}) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    budget = dict(config["budget"])
    if int(budget.get("total_optimizer_steps") or 0) != 20_000:
        raise ValueError("PINN-Online requires exactly 20000 optimizer steps")
    runtime = dict(config["runtime"])
    if str(runtime.get("dtype") or "") != "float32":
        raise ValueError("Formal experiment configs currently require dtype=float32")
    method = str(config["method"])
    ablation_mode = str(config.get("ablation_mode") or "none").strip().lower()
    if ablation_mode not in SUPPORTED_ABLATION_MODES:
        raise ValueError(
            f"Unsupported ablation_mode: {ablation_mode}"
        )
    enabled = bool(dict(config.get("online_morphing") or {}).get("enabled", False))
    if method != ONLINE_METHOD and method not in LEGACY_ONLINE_METHODS:
        raise ValueError(
            f"Only the current PINN-Online method is supported: {ONLINE_METHOD}"
        )
    if ablation_mode != NO_MORPH_ABLATION and not enabled:
        raise ValueError("The PINN-Online method requires online_morphing.enabled=true")
    online = dict(config.get("online_morphing") or {})
    if ablation_mode == NO_MORPH_ABLATION:
        if enabled:
            raise ValueError("No Morph ablation requires online_morphing.enabled=false")
    else:
        expected_decision_mode = {
            RANDOM_MORPH_ABLATION: RANDOM_MORPH_DECISION_MODE,
            RANDOM_SELECTION_ABLATION: RANDOM_SELECTION_ABLATION,
        }.get(ablation_mode, PINNMORPH_DECISION_MODE)
        if str(online.get("decision_mode")) != expected_decision_mode:
            raise ValueError(
                f"ablation_mode={ablation_mode or 'none'} requires "
                f"decision_mode={expected_decision_mode}"
            )
        capability_ablation = config.get("capability_ablation", "none")
        if capability_ablation not in {"none", "structural_only", "fixed_optimizer_schedule"}:
            raise ValueError("Unknown capability_ablation")
        expected_primitives = (
            LEGACY_STRUCTURAL_PRIMITIVES
            if capability_ablation == "structural_only" else PRIMITIVE_LIBRARY
        )
        if capability_ablation == "fixed_optimizer_schedule":
            expected_primitives = tuple(p for p in PRIMITIVE_LIBRARY if p != "ENTER_LBFGS_PHASE")
        if tuple(online.get("allowed_primitives") or ()) != expected_primitives:
            raise ValueError(
                "Allowed primitives must match the declared capability registry/ablation"
            )
        if int(online.get("adaptation_freeze_step") or 0) != 14_000:
            raise ValueError("PINN-Online requires adaptation_freeze_step=14000")
        policy_set = dict(online.get("policy_set") or {})
        if ablation_mode == RANDOM_MORPH_ABLATION:
            if bool(policy_set.get("enabled")):
                raise ValueError("Random Morph ablation requires policy_set.enabled=false")
            evolution = dict(policy_set.get("evolution") or {})
            if bool(evolution.get("enabled", False)):
                raise ValueError(
                    "Random Morph ablation requires policy_set.evolution.enabled=false"
                )
        else:
            if not bool(policy_set.get("enabled")):
                raise ValueError("PINN-Online requires policy_set.enabled=true")
            if int(policy_set.get("population_size") or 0) != 5:
                raise ValueError("PINN-Online requires exactly 5 peer policy candidates")
            scoring = dict(policy_set.get("scoring") or {})
            if not scoring or not 0.0 <= float(scoring.get("temperature", -1.0)) <= 2.0:
                raise ValueError(
                    "PINN-Online requires independent LLM candidate scoring configuration"
                )
            if "exploration_weight" in policy_set:
                raise ValueError(
                    "PINN-Online candidate ranking cannot use the retired heuristic exploration weight"
                )
            evolution = dict(policy_set.get("evolution") or {})
            if not bool(evolution.get("enabled")):
                raise ValueError("PINN-Online requires policy_set.evolution.enabled=true")
            if int(evolution.get("offspring_per_generation") or 0) != 5:
                raise ValueError(
                    "PINN-Online requires the five EOH LLM generation modes"
                )
            if int(evolution.get("minimum_valid_offspring") or 0) != 5:
                raise ValueError("PINN-Online requires all 5 evolution modes to be valid")
            if ablation_mode == RANDOM_EVOLUTION_ABLATION and evolution.get("direction") != "random":
                raise ValueError("Random Evolution requires evolution.direction=random")
    if bool(config.get("outer_evolution", False)):
        raise ValueError("PINN-Online does not support outer candidate evolution")
    if config.get("no_morph_optimizer_schedule", "adam_only") not in {"adam_only", "matched"}:
        raise ValueError("no_morph_optimizer_schedule must be adam_only or matched")
    training = dict(dict(config.get("algorithm_overrides") or {}).get("training") or {})
    extra = dict(training.get("extra_parameters") or {})
    if not bool(dict(extra.get("physics_validation") or {}).get("enabled")):
        raise ValueError("PINN-Online requires physics_validation.enabled=true")
    best = dict(extra.get("best_checkpoint") or {})
    if (
        not bool(best.get("enabled"))
        or str(best.get("final_model_policy")) != "best_physics_validation"
        or str(best.get("selection_metric")) != "physics_audit_score"
    ):
        raise ValueError(
            "PINN-Online requires independent physics-audit checkpoint selection"
        )
    if config.get("paper_protocol"):
        from .paper_contract import validate_paper_protocol
        validate_paper_protocol(config)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in config.items() if not str(key).startswith("_")})


def scale_optimization_phases_to_budget(
    phases: list[dict[str, Any]],
    target_iterations: int,
    *,
    min_lbfgs_iterations: int = 1,
    preserve_enabled_training_stages: bool = True,
) -> list[dict[str, Any]]:
    """Scale one continuous optimization plan to the configured step budget."""

    if not isinstance(phases, list) or not phases:
        raise ValueError("At least one optimization phase is required")
    target = int(target_iterations)
    if target < len(phases):
        raise ValueError(
            "Target iterations cannot provide one step to every optimization phase"
        )
    original: list[int] = []
    for index, phase in enumerate(phases):
        value = phase.get("iterations") if isinstance(phase, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"optimization phase {index} must have at least one iteration"
            )
        original.append(value)
    total = sum(original)
    quotas = [target * value / total for value in original]
    floors = [
        max(1, int(min_lbfgs_iterations))
        if str(phase.get("optimizer") or "").strip().lower() == "lbfgs"
        else 1
        for phase in phases
    ]
    if not preserve_enabled_training_stages:
        floors = [0 for _ in phases]
    if sum(floors) > target:
        raise ValueError("Step budget cannot preserve the configured phase floors")
    allocated = [
        max(minimum, floor(quota))
        for quota, minimum in zip(quotas, floors)
    ]
    while sum(allocated) > target:
        removable = [
            index for index, value in enumerate(allocated)
            if value > floors[index]
        ]
        if not removable:
            raise ValueError("Unable to honor the optimization phase floors")
        index = min(
            removable,
            key=lambda item: (quotas[item] - allocated[item], item),
        )
        allocated[index] -= 1
    while sum(allocated) < target:
        index = max(
            range(len(allocated)),
            key=lambda item: (quotas[item] - allocated[item], -item),
        )
        allocated[index] += 1
    scaled = deepcopy(phases)
    for phase, source_iterations, iterations in zip(
        scaled, original, allocated
    ):
        phase["iterations"] = iterations
        _align_scheduler_horizon(
            phase,
            source_iterations=source_iterations,
            target_iterations=iterations,
        )
    return scaled


def _align_scheduler_horizon(
    phase: dict[str, Any],
    *,
    source_iterations: int,
    target_iterations: int,
) -> None:
    scheduler = phase.get("scheduler")
    if not isinstance(scheduler, dict):
        return
    name = str(scheduler.get("name") or "none").strip().lower()
    parameters = scheduler.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
        scheduler["parameters"] = parameters
    if name in {"one_cycle", "onecycle", "onecyclelr"}:
        parameters.pop("epochs", None)
        parameters.pop("steps_per_epoch", None)
        parameters["total_steps"] = max(1, int(target_iterations))
        return
    horizon_field = {
        "cosine_annealing": "T_max",
        "cosine": "T_max",
        "cosineannealinglr": "T_max",
        "linear_lr": "total_iters",
        "linearlr": "total_iters",
        "polynomial_lr": "total_iters",
        "polynomiallr": "total_iters",
    }.get(name)
    if horizon_field is not None and (
        parameters.get(horizon_field) is None
        or int(parameters[horizon_field]) == int(source_iterations)
    ):
        parameters[horizon_field] = max(1, int(target_iterations))


__all__ = [
    "ONLINE_METHOD",
    "LEGACY_ONLINE_METHODS",
    "config_hash",
    "deep_merge",
    "load_experiment_config",
    "scale_optimization_phases_to_budget",
    "stable_hash",
    "validate_experiment_config",
]
