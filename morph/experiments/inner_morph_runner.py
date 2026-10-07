"""Formal, resumable runner for fixed-AlgorithmSpec inner-morph experiments."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from morph.experiments.experiment_config import (
    ONLINE_METHOD,
    apply_ablation_mode,
    config_hash,
    deep_merge,
    load_experiment_config,
    scale_optimization_phases_to_budget,
    stable_hash,
)
from morph.pipeline.a_problem_definition.problems import get_problem
from morph.pipeline.c_knowledge_retrieval import (
    build_online_execution_evidence,
    retrieve_online_prior_context,
)
from morph.pipeline.d_algorithm_generation.llm.provider import (
    build_provider,
    redact_secret,
)
from morph.pipeline.d_algorithm_generation.specs import algorithm_spec_template
from morph.pipeline.f_pinn_construction.builders.pinn_builder import (
    AlgorithmSpecValidationError,
    OpenAlgorithmSpecRuntime,
)
from morph.pipeline.g_training_evaluation.evaluation.evaluator import OpenSpecEvaluator
from morph.pipeline.g_training_evaluation.training.trainer import OpenSpecTrainer
from morph.utils.experiment_provenance import assert_run_compatible, source_identity
from morph.utils.reproducibility import get_environment_info, set_global_seed


RUN_SUMMARY = "run_summary.json"
FORMAL_HISTORY_INTERVAL = 10


class TrackingLLMProvider:
    """Record one explicit LLM role without exposing credentials."""

    def __init__(
        self,
        provider: Any,
        evidence_path: Path | None = None,
        *,
        role: str = "unspecified",
    ) -> None:
        self.provider = provider
        self.evidence_path = evidence_path
        self.role = str(role)
        self.calls = 0
        self.failures = 0
        self.total_latency_seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.records: list[dict[str, Any]] = []

    def complete_text(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        started = time.perf_counter()
        try:
            response = self.provider.complete_text(*args, **kwargs)
        except Exception as exc:
            self.failures += 1
            self.total_latency_seconds += time.perf_counter() - started
            self._record(
                {
                    "call": self.calls,
                    "task": kwargs.get("task"),
                    "success": False,
                    "error": redact_secret(str(exc), getattr(self.provider, "api_key", None)),
                    "messages": deepcopy(args[0] if args else kwargs.get("messages")),
                }
            )
            raise
        latency = getattr(response, "latency_seconds", None)
        self.total_latency_seconds += (
            float(latency) if isinstance(latency, (int, float)) else time.perf_counter() - started
        )
        if not bool(getattr(response, "success", False)):
            self.failures += 1
        usage = dict(getattr(response, "token_usage", {}) or {})
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self._record(
            {
                "call": self.calls,
                "task": kwargs.get("task"),
                "success": bool(getattr(response, "success", False)),
                "provider": getattr(response, "provider", None),
                "model": getattr(response, "model", None),
                "model_version": getattr(response, "model_version", None),
                "finish_reason": getattr(response, "finish_reason", None),
                "latency_seconds": float(getattr(response, "latency_seconds", 0.0) or 0.0),
                "token_usage": usage,
                "messages": deepcopy(args[0] if args else kwargs.get("messages")),
                "response_text": str(getattr(response, "text", "")),
                "error": redact_secret(
                    str(getattr(response, "error", None) or ""),
                    getattr(self.provider, "api_key", None),
                ) or None,
            }
        )
        return response

    def _record(self, payload: Mapping[str, Any]) -> None:
        record = deepcopy(dict(payload))
        record["role"] = self.role
        record["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        self.records.append(record)
        if self.evidence_path is not None:
            self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            with self.evidence_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str))
                stream.write("\n")


def build_algorithm_spec(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    problem = get_problem(str(config["pde"]))
    spec = deep_merge(algorithm_spec_template(), dict(config.get("algorithm_overrides") or {}))
    morph = dict(config.get("online_morphing") or {})
    adaptive = (
        spec["training"]
        .setdefault("extra_parameters", {})
        .setdefault("adaptive_control", {})
    )
    adaptive["online_morphing"] = morph
    total_steps = int(config["budget"]["total_optimizer_steps"])
    if (str(config.get("ablation_mode") or "") == "no_morph"
            and config.get("no_morph_optimizer_schedule", "adam_only") == "adam_only"):
        adam = next(
            phase for phase in spec["optimization"]["phases"]
            if str(phase.get("optimizer", "")).lower() == "adam"
        )
        spec["optimization"]["phases"] = scale_optimization_phases_to_budget(
            [adam], total_steps
        )
    else:
        spec["optimization"]["phases"] = scale_optimization_phases_to_budget(
            spec["optimization"]["phases"], total_steps
        )
    metadata = spec.setdefault("generation_metadata", {})
    metadata.update(
        {
            "proposal_strategy": "single_online_trajectory",
            "design_hypothesis": str(config.get("research_question") or ""),
            "online_method": str(config["method"]),
            "iteration_budget_rationale": f"one continuous {total_steps}-step trajectory",
        }
    )
    augment = getattr(problem, "augment_training_algorithm_spec", None)
    if callable(augment):
        spec = augment(spec)
    return problem, spec


def normalized_spec(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], Any]:
    problem, spec = build_algorithm_spec(config)
    runtime = OpenAlgorithmSpecRuntime()
    report = runtime.validate(spec, problem, _validation_budget(config))
    if not report.valid or report.normalized_spec is None:
        raise AlgorithmSpecValidationError(report)
    return problem, report.normalized_spec, report


def _validation_budget(config: Mapping[str, Any]) -> dict[str, Any]:
    budget = dict(config["budget"])
    steps = int(budget["total_optimizer_steps"])
    return {
        "recommended_total_iterations": steps,
        "required_total_iterations": steps,
        "maximum_total_iterations": steps,
        "iteration_recommendation_is_advisory": False,
        "maximum_sampling_points": budget.get("maximum_sampling_points"),
        "constraint_resample_interval": int(budget.get("constraint_resample_interval", 50)),
    }


def run_directory(results_root: str | Path, config: Mapping[str, Any], seed: int) -> Path:
    return Path(results_root) / str(config.get("result_group") or config["phase"]) / str(config["pde"]) / str(config["method"]) / f"seed_{int(seed)}"


def completed_run(path: str | Path) -> bool:
    summary = Path(path) / RUN_SUMMARY
    if not summary.exists():
        return False
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("status") == "success"
    except (OSError, json.JSONDecodeError):
        return False


def failed_run(path: str | Path) -> bool:
    summary = Path(path) / RUN_SUMMARY
    if not summary.exists():
        return False
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("status") == "failed"
    except (OSError, json.JSONDecodeError):
        return False


def run_one(
    config: Mapping[str, Any],
    *,
    seed: int,
    results_root: str | Path,
    force: bool = False,
    retry_failed: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    output = run_directory(results_root, config, seed)
    source = source_identity(Path(__file__).resolve().parents[2])
    previous_on_disk = _read_existing_summary(output / RUN_SUMMARY)
    if not force:
        assert_run_compatible(previous_on_disk, config_hash(config), source["sha256"])
        if not previous_on_disk and output.exists() and any(output.rglob("*.pt")):
            raise ValueError("Checkpoint exists without run provenance; use a fresh results root")
    if completed_run(output) and not force:
        return {"status": "skipped_completed", "output_dir": str(output)}
    if failed_run(output) and not (force or retry_failed or resume):
        return {"status": "skipped_failed", "output_dir": str(output)}
    output.mkdir(parents=True, exist_ok=True)
    previous_summary = (
        _read_existing_summary(output / RUN_SUMMARY)
        if resume and not force else {}
    )
    previous_cumulative_wall = float(
        (previous_summary.get("resources") or {}).get("cumulative_wall_time")
        or (previous_summary.get("resources") or {}).get("wall_clock_time")
        or 0.0
    )
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    partial = _base_summary(config, seed, started_at)
    partial["hashes"]["source_tree"] = source["sha256"]
    policy_prior_evidence: dict[str, Any] = {}
    _write_json(output / RUN_SUMMARY, partial)
    try:
        set_global_seed(int(seed))
        if str(config["runtime"]["dtype"]) != "float32":
            raise ValueError("Only the project-native float32 experiment dtype is supported")
        torch.set_default_dtype(torch.float32)
        device = str(config["runtime"].get("device") or "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA hardware is unavailable")
        problem, spec, validation = normalized_spec(config)
        _write_json(output / "algorithm_spec_normalized.json", spec)
        _write_json(output / "experiment_config_resolved.json", {
            key: value for key, value in config.items() if not key.startswith("_")
        })
        runtime = OpenAlgorithmSpecRuntime()
        built = runtime.build(
            spec,
            problem,
            {
                "device": device,
                "maximum_sampling_points": config["budget"].get("maximum_sampling_points"),
                "constraint_resample_interval": int(config["budget"].get("constraint_resample_interval", 50)),
            },
        )
        initial_parameters = _parameter_count(built.model)
        if built.online_morph_controller is not None:
            static_prior = _seed_static_policy_knowledge(
                config, problem, built.online_morph_controller
            )
            execution_prior = _seed_policy_evolution_evidence(
                config, built.online_morph_controller
            )
            policy_prior_evidence = {
                **execution_prior,
                "static_knowledge": static_prior,
            }
        llm_trackers = _build_dual_llm_policy(
            config, output / "llm_calls.jsonl"
        )
        trainer = OpenSpecTrainer(
            built,
            checkpoint_dir=output,
            history_interval=int(config.get("history_interval", FORMAL_HISTORY_INTERVAL)),
            component_gradient_diagnostics_enabled=bool(config["budget"].get("component_gradient_diagnostics_enabled", True)),
            online_policy_generation_provider=llm_trackers.get("candidate_generation"),
            online_candidate_scoring_provider=llm_trackers.get("candidate_scoring"),
        )
        resume_candidates = (
            output / "adaptive_control" / "recovery_resume_checkpoint.pt",
            output / "last_optimizer_checkpoint.pt",
        )
        available_checkpoints = [path for path in resume_candidates if path.exists()]
        resume_checkpoint = max(available_checkpoints, key=lambda path: path.stat().st_mtime_ns, default=None)
        resumed_from_checkpoint = bool(
            resume and not force and resume_checkpoint is not None
        )
        if resumed_from_checkpoint:
            trainer.load_checkpoint(resume_checkpoint)
            # Old checkpoints predate the restored static-prior field.  Loading
            # one must not silently erase the context installed above.  Runtime
            # execution evidence, on the other hand, remains checkpoint-owned
            # so that a resumed causal history is not duplicated.
            if built.online_morph_controller is not None:
                restored_static = _seed_static_policy_knowledge(
                    config, problem, built.online_morph_controller
                )
                restored_static["reapplied_after_checkpoint"] = True
                policy_prior_evidence["static_knowledge"] = restored_static
        report = trainer.train(maximum_iterations=int(config["budget"]["total_optimizer_steps"]))
        if not report.success:
            raise RuntimeError(str(report.failure_reason or "training_failed"))
        metrics = OpenSpecEvaluator(problem, device).evaluate_final(
            built.model, report
        )
        final_parameters = _parameter_count(built.model)
        morph = _morph_summary(report.adaptive_control_summary)
        llm = _llm_summary(llm_trackers, report.adaptive_control_summary)
        segment_wall_time = time.perf_counter() - started
        cumulative_wall_time = previous_cumulative_wall + segment_wall_time
        summary = {
            **partial,
            "status": "success",
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "relative_l2": metrics.get("relative_l2_error"),
                "mse": metrics.get("mse"),
                "pde_residual": metrics.get("pde_residual_error"),
                "bc_residual": metrics.get("boundary_error"),
                "ic_residual": metrics.get("initial_error"),
            },
            "resources": {
                "wall_clock_time": cumulative_wall_time,
                "segment_wall_time": segment_wall_time,
                "cumulative_wall_time": cumulative_wall_time,
                "training_time": report.training_time,
                "morph_overhead_time": float(report.adaptive_control_summary.get("online_morph_wall_time") or 0.0),
                "peak_gpu_memory": int(metrics.get("peak_gpu_memory") or report.peak_gpu_memory or 0),
                "initial_parameter_count": initial_parameters,
                "final_parameter_count": final_parameters,
                "parameter_growth_ratio": (final_parameters - initial_parameters) / max(1, initial_parameters),
                "actual_optimizer_steps": int(
                    getattr(trainer, "_actual_iterations", report.actual_iterations)
                ),
                "actual_peak_sampling_points": int(report.actual_peak_sampling_points),
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_checkpoint": (
                    str(resume_checkpoint) if resumed_from_checkpoint else None
                ),
            },
            "checkpoint_selection": {
                "source": report.selected_checkpoint_source,
                "step": report.selected_checkpoint_step,
                "metric": report.selected_checkpoint_metric,
            },
            "evaluation_diagnostics": {
                key: metrics.get(key) for key in (
                    "prediction_mean", "prediction_std", "reference_std",
                    "temporal_l2re", "temporal_mse", "temporal_amplitude_ratio",
                    "initial_displacement_relative_l2", "initial_velocity_relative_l2",
                )
            },
            "morphing": morph,
            "llm": llm,
            "policy_prior_evidence": policy_prior_evidence,
            "hashes": {
                **partial["hashes"],
                "normalized_algorithm_spec": stable_hash(spec),
            },
            "validation_warnings": [vars(item) for item in validation.warnings],
            "failure_reason": None,
        }
        _write_json(output / "algorithm_spec_normalized.json", spec)
        _write_json(output / "training_report.json", report.to_dict())
        _write_training_curve(output / "training_curve.csv", report, initial_parameters, final_parameters)
        _write_json(output / RUN_SUMMARY, summary)
        return summary
    except Exception as exc:
        segment_wall_time = time.perf_counter() - started
        cumulative_wall_time = previous_cumulative_wall + segment_wall_time
        failed = {
            **partial,
            "status": "failed",
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "resources": {
                "wall_clock_time": cumulative_wall_time,
                "segment_wall_time": segment_wall_time,
                "cumulative_wall_time": cumulative_wall_time,
            },
        }
        _write_json(output / RUN_SUMMARY, failed)
        return failed


def _base_summary(config: Mapping[str, Any], seed: int, started_at: str) -> dict[str, Any]:
    device = str(config["runtime"].get("device") or "cuda")
    online = dict(config.get("online_morphing") or {})
    return {
        "schema_version": 1,
        "pde": str(config["pde"]),
        "method": str(config["method"]),
        "phase": str(config["phase"]),
        "seed": int(seed),
        "status": "running",
        "started_at": started_at,
        "budget": deepcopy(dict(config["budget"])),
        "runtime": deepcopy(dict(config["runtime"])),
        "environment": get_environment_info(device),
        "ablation_mode": str(config.get("ablation_mode") or "none"),
        "online_morphing_enabled": bool(online.get("enabled", False)),
        "online_decision_mode": online.get("decision_mode"),
        "hashes": {"experiment_config": config_hash(config)},
    }


def _build_dual_llm_policy(
    config: Mapping[str, Any], evidence_path: Path | None = None
) -> dict[str, TrackingLLMProvider]:
    method = str(config["method"])
    if method != ONLINE_METHOD:
        raise ValueError(f"Unsupported experiment method: {method}")
    online = dict(config.get("online_morphing") or {})
    if str(online.get("decision_mode") or "") not in {"llm_evolution", "random_selection"}:
        return {}
    llm = dict(config.get("llm") or {})
    channels: dict[str, TrackingLLMProvider] = {}
    for role, index in (("candidate_generation", 1), ("candidate_scoring", 2)):
        if role == "candidate_scoring" and str(online.get("decision_mode")) == "random_selection":
            continue
        channel = dict(llm.get(role) or {})
        prefix = f"DUAL_LLM_{index}"
        provider_name = channel.get("provider") or os.environ.get(
            f"{prefix}_PROVIDER"
        )
        model = channel.get("model") or os.environ.get(f"{prefix}_MODEL")
        api_key = channel.get("api_key") or os.environ.get(f"{prefix}_API_KEY")
        base_url = channel.get("api_base") or os.environ.get(
            f"{prefix}_API_BASE_URL"
        )
        missing = [
            name
            for name, value in (
                (f"{prefix}_PROVIDER", provider_name),
                (f"{prefix}_MODEL", model),
                (f"{prefix}_API_KEY", api_key),
                (f"{prefix}_API_BASE_URL", base_url),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{method} requires a complete {role} channel; missing "
                + ", ".join(missing)
            )
        provider = build_provider(
            provider=str(provider_name),
            model=str(model),
            api_key=str(api_key),
            base_url=str(base_url),
            timeout_seconds=int(
                channel.get("timeout_seconds", llm.get("timeout_seconds", 60))
            ),
            enable_thinking=channel.get("enable_thinking"),
        )
        channels[role] = TrackingLLMProvider(
            provider, evidence_path, role=role
        )
    return channels


def _seed_policy_evolution_evidence(
    config: Mapping[str, Any], controller: Any
) -> dict[str, Any]:
    policy_set = getattr(controller, "policy_population", None)
    policy_config = dict(
        dict(config.get("online_morphing") or {}).get("policy_set") or {}
    )
    configured = policy_config.get("prior_evidence_path")
    if policy_set is None or not configured:
        return {"configured": False, "loaded": False}
    path = Path(str(configured))
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    refresh_audit: dict[str, Any] = {"attempted": False}
    if bool(policy_config.get("prior_evidence_auto_refresh", False)):
        refresh_audit["attempted"] = True
        try:
            generated = build_online_execution_evidence(
                Path(__file__).resolve().parents[2] / "results",
                path,
                maximum_runs=int(
                    policy_config.get("prior_evidence_maximum_runs", 64)
                ),
            )
            refresh_audit.update(
                {"success": True, "run_count": int(generated["run_count"])}
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            refresh_audit.update(
                {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    if not path.is_file():
        return {
            "configured": True,
            "loaded": False,
            "cold_start": True,
            "path": str(path),
            "reason": "configured_prior_evidence_missing",
            "contexts": 0,
            "states_loaded": 0,
            "experiences_loaded": 0,
            "refresh": refresh_audit,
        }
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    from morph.pipeline.g_training_evaluation.training.adaptation_program import (
        PolicyCandidate,
    )

    contexts = 0
    experiences_before = len(policy_set.shared_history)
    prior_candidates: list[PolicyCandidate] = []
    target_pde = str(config.get("pde") or "")
    for run in payload.get("runs") or []:
        run_pde = str(run.get("pde") or "")
        if target_pde and run_pde and run_pde != target_pde:
            continue
        context_id = str(run.get("context_id") or f"context-{contexts:04d}")
        online = dict(run.get("online_evidence") or {})
        for item in online.get("recent_experiences") or []:
            evidence_item = deepcopy(dict(item))
            original_id = str(
                evidence_item.get("intervention_id")
                or f"experience-{len(policy_set.shared_history):06d}"
            )
            if original_id.startswith("prior-"):
                continue
            evidence_item["intervention_id"] = (
                f"prior-{context_id}-{original_id}"
            )
            policy_set.append_shared_evidence(evidence_item)
        for item in online.get("policies") or []:
            try:
                prior_candidates.append(PolicyCandidate.from_dict(item))
            except (TypeError, ValueError):
                continue
        contexts += 1
    unique: dict[str, PolicyCandidate] = {}
    for candidate in sorted(
        prior_candidates,
        key=lambda item: (item.global_credit, item.execution_count),
        reverse=True,
    ):
        signature = json.dumps(
            candidate.policy_spec, sort_keys=True, ensure_ascii=False, default=str
        )
        unique.setdefault(signature, candidate)
    selected = list(unique.values())[:5]
    if len(selected) == 5:
        policy_set.policies = [
            PolicyCandidate(
                policy_id=f"prior-policy-{index + 1}",
                version=candidate.version,
                # Preserve one source edge, not an indefinitely nested chain
                # of imported prior-policy ancestors across repeated runs.
                parent_ids=(candidate.policy_id,),
                creation_step=candidate.creation_step,
                creation_mode="prior_evidence",
                policy_spec=deepcopy(candidate.policy_spec),
                execution_count=candidate.execution_count,
                success_count=candidate.success_count,
                failure_count=candidate.failure_count,
                global_credit=candidate.global_credit,
                state_conditioned_credit=deepcopy(
                    candidate.state_conditioned_credit
                ),
                last_selected_step=candidate.last_selected_step,
                recent_results=deepcopy(candidate.recent_results),
            )
            for index, candidate in enumerate(selected)
        ]
    experiences_loaded = len(policy_set.shared_history) - experiences_before
    loaded = bool(contexts and (len(selected) == 5 or experiences_loaded > 0))
    return {
        "configured": True,
        "loaded": loaded,
        "cold_start": not loaded,
        "path": str(path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "contexts": contexts,
        "policies_loaded": len(selected) if len(selected) == 5 else 0,
        "experiences_loaded": experiences_loaded,
        "target_pde": target_pde,
        "reference_solution_metrics_loaded": False,
        "refresh": refresh_audit,
    }


def _seed_static_policy_knowledge(
    config: Mapping[str, Any], problem: Any, controller: Any
) -> dict[str, Any]:
    policy_set = getattr(controller, "policy_population", None)
    policy_config = dict(
        dict(config.get("online_morphing") or {}).get("policy_set") or {}
    )
    static_config = dict(policy_config.get("static_prior") or {})
    if policy_set is None or not bool(static_config.get("enabled", True)):
        return {"configured": False, "loaded": False, "items_loaded": 0}
    problem_spec = problem.get_spec()
    metadata = deepcopy(dict(getattr(problem_spec, "metadata", {}) or {}))
    metadata.setdefault("problem_id", str(getattr(problem_spec, "problem_id", "")))
    metadata.setdefault("benchmark_id", str(config.get("pde") or ""))
    principles, audit = retrieve_online_prior_context(
        metadata,
        configured_path=static_config.get("path"),
        maximum_items=int(static_config.get("maximum_items", 8)),
    )
    policy_set.set_prior_context(principles)
    return audit
def _llm_summary(
    trackers: Mapping[str, TrackingLLMProvider],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    values = list(trackers.values())
    failures = sum(int(tracker.failures) for tracker in values)
    calls = sum(int(tracker.calls) for tracker in values)
    rejected = int(control.get("online_morph_rejected_decisions") or 0)
    failures = max(failures, rejected)
    channels = {
        role: {
            "role": role,
            "provider": getattr(tracker.provider, "provider", None),
            "model": getattr(tracker.provider, "model", None),
            "calls": int(tracker.calls),
            "failures": int(tracker.failures),
            "latency_seconds": float(tracker.total_latency_seconds),
            "prompt_tokens": int(tracker.prompt_tokens),
            "completion_tokens": int(tracker.completion_tokens),
            "total_tokens": int(tracker.total_tokens),
        }
        for role, tracker in trackers.items()
    }
    return {
        "calls": calls,
        "failures": failures,
        "failure_rate": failures / calls if calls else 0.0,
        "latency_seconds": sum(float(item.total_latency_seconds) for item in values),
        "prompt_tokens": sum(int(item.prompt_tokens) for item in values),
        "completion_tokens": sum(int(item.completion_tokens) for item in values),
        "total_tokens": sum(int(item.total_tokens) for item in values),
        "evidence_file": (
            str(values[0].evidence_path)
            if values and values[0].evidence_path else None
        ),
        "channels": channels,
        "records": [
            deepcopy(record)
            for tracker in values
            for record in tracker.records
        ],
        "fallback_used": bool(failures),
        "fallback_decision": "KEEP" if failures else None,
    }


def _morph_summary(control: Mapping[str, Any]) -> dict[str, Any]:
    events = list(control.get("online_morph_events") or [])
    successes = sum(
        str(item.get("feedback_status") or "").lower() == "success"
        for item in events
    )
    failures = sum(
        str(item.get("feedback_status") or "").lower() == "failure"
        for item in events
    )
    withheld = sum(
        str(item.get("feedback_status") or "").lower() == "credit_withheld"
        for item in events
    )
    emergency_recoveries = sum(
        item.get("event") == "HEURISTIC_EMERGENCY_RECOVERY" for item in events
    )
    applied = int(control.get("online_morphs_applied") or 0)
    rolled_back = int(control.get("online_morphs_rolled_back") or 0)
    return {
        "diagnostic_events": int(control.get("online_morph_diagnostics") or 0),
        "attempts": int(control.get("online_morph_triggers") or control.get("online_morphs_applied") or 0),
        "applied": applied,
        "retained": int(
            control.get("online_morphs_retained")
            if control.get("online_morphs_retained") is not None
            else max(0, applied - rolled_back)
        ),
        "rolled_back": rolled_back,
        "trial_rollbacks": int(control.get("online_morph_trial_rollbacks") or 0),
        "time_progress": deepcopy(control.get("online_time_progress") or {}),
        "residual_frequency_axes": deepcopy(control.get("residual_frequency_axes") or {}),
        "llm_budget": deepcopy(control.get("online_llm_budget") or {}),
        "soft_retirements": int(control.get("online_morph_soft_retirements") or 0),
        "successful_feedback": successes,
        "failed_feedback": failures,
        "withheld_feedback": withheld,
        "emergency_recoveries": emergency_recoveries,
        "rejected": int(control.get("online_morphs_rejected") or 0),
        "operators": [str(item.get("operator")) for item in events if item.get("operator")],
        "events": events,
        "diagnostics": list(control.get("online_morph_diagnostic_events") or []),
        "policy_population": deepcopy(control.get("online_policy_population")),
        "policy_evolution": {
            "calls": int(control.get("online_policy_evolution_calls") or 0),
            "cycles": int(control.get("online_policy_evolution_cycles") or 0),
            "failures": int(
                control.get("online_policy_evolution_failures") or 0
            ),
            "events": deepcopy(
                control.get("online_policy_evolution_events") or []
            ),
        },
        "candidate_scoring": {
            "calls": int(control.get("online_policy_scoring_calls") or 0),
            "failures": int(
                control.get("online_policy_scoring_failures") or 0
            ),
            "events": deepcopy(
                control.get("online_policy_scoring_events") or []
            ),
        },
    }


def _write_training_curve(path: Path, report: Any, initial: int, final: int) -> None:
    events = _read_jsonl(path.parent / "morph_events.jsonl")
    event_by_step: dict[int, list[str]] = {}
    changes: list[tuple[int, int]] = []
    for item in events:
        # Feedback and rollback records retain the intervention's start
        # iteration for attribution, but their observable event time is step.
        step = int(item.get("step", item.get("iteration", -1)))
        name = str(item.get("event") or "")
        if step >= 0 and name:
            event_by_step.setdefault(step, []).append(name)
        if item.get("parameter_count_after") is not None:
            changes.append((step, int(item["parameter_count_after"])))
    physics = {
        int(item.get("iteration", item.get("step", -1))): item.get("aggregate_score")
        for item in report.physics_validation_history
        if isinstance(item, dict)
    }
    physics_audit = {
        int(item.get("iteration", item.get("step", -1))): item.get(
            "aggregate_score"
        )
        for item in report.physics_audit_history
        if isinstance(item, dict)
    }
    rows = []
    for item in report.history:
        step = int(item.get("iteration") or 0)
        count = initial
        for change_step, changed_count in sorted(changes):
            if change_step <= step:
                count = changed_count
        if step >= int(report.actual_iterations):
            count = final
        stage = dict(item.get("training_stage") or {})
        rows.append(
            {
                "step": step,
                "training_total_loss": item.get("total_loss"),
                "heldout_physics_score": physics.get(step),
                "audit_physics_score": physics_audit.get(step),
                "morph_phase": stage.get("online_morph_phase", "idle"),
                "parameter_count": count,
                "event": "|".join(event_by_step.get(step, ())) or "NONE",
                "phase": item.get("phase"),
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["step", "training_total_loss", "heldout_physics_score", "morph_phase", "parameter_count", "event", "phase", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--ablation-mode",
        choices=("none", "no_morph", "random_morph", "random_selection", "random_evolution"),
        default=None,
        help="Override the loaded config with a PINNMorph ablation mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outcomes = []
    for path in args.config:
        config = load_experiment_config(path)
        if args.ablation_mode is not None:
            config = apply_ablation_mode(config, args.ablation_mode)
        seeds = args.seeds if args.seeds else [int(item) for item in config["seeds"]]
        for seed in seeds:
            outcome = run_one(
                config,
                seed=seed,
                results_root=args.results_root,
                force=args.force,
                retry_failed=args.retry_failed,
                resume=not args.no_resume,
            )
            outcomes.append({"config": str(path), "seed": seed, **outcome})
            # Full telemetry remains in run_summary.json, not a multi-megabyte log line.
            console_outcome = {
                key: value for key, value in outcomes[-1].items()
                if key not in {"morphing", "llm", "policy_prior_evidence"}
            }
            print(json.dumps(console_outcome, ensure_ascii=False, default=str))
    if any(item.get("status") == "failed" for item in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
