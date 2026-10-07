"""Build reference-free cross-run policy evidence from completed online runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ONLINE_METHOD = "single_trajectory_eoh_policy_population"
LEGACY_ONLINE_METHODS = frozenset({"single_trajectory_eohs_policy_population"})
_REFERENCE_SOLUTION_KEYS = {
    "mse",
    "mean_squared_error",
    "relative_l2",
    "relative_l2_error",
    "reference_error",
    "reference_metrics",
    "reference_solution",
    "final_metrics",
}


def build_online_execution_evidence(
    search_root: str | Path,
    output_path: str | Path,
    *,
    maximum_runs: int = 64,
) -> dict[str, Any]:
    """Write an atomic, label-free snapshot of reusable policy experience."""

    root = Path(search_root).resolve()
    output = Path(output_path).resolve()
    rows: list[dict[str, Any]] = []
    for summary_path in root.rglob("run_summary.json") if root.is_dir() else ():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(summary, dict)
            or summary.get("status") != "success"
            or str(summary.get("method") or "")
            not in {ONLINE_METHOD, *LEGACY_ONLINE_METHODS}
        ):
            continue
        population = dict(
            dict(summary.get("morphing") or {}).get("policy_population") or {}
        )
        policies = [
            item
            for item in list(population.get("policies") or [])
            if not str(dict(item).get("policy_id") or "").startswith("prior-policy-")
            and str(dict(item).get("creation_mode") or "") != "prior_evidence"
        ]
        # Imported prior rows are context, not newly observed execution. Never
        # export them again or IDs grow as prior-X-prior-Y-... across runs.
        history = [
            item
            for item in list(population.get("shared_history") or [])
            if not str(dict(item).get("intervention_id") or "").startswith("prior-")
        ]
        if not policies and not history:
            continue
        run_key = hashlib.sha256(
            str(summary_path.parent).encode("utf-8")
        ).hexdigest()[:12]
        rows.append(
            {
                "context_id": run_key,
                "pde": str(summary.get("pde") or ""),
                "seed": int(summary.get("seed") or 0),
                "ended_at": summary.get("ended_at"),
                "config_hash": dict(summary.get("hashes") or {}).get(
                    "experiment_config"
                ),
                "online_evidence": {
                    "recent_experiences": _reference_free_copy(history[-50:]),
                    "policies": _reference_free_copy(policies),
                },
                "source_run": str(summary_path.parent),
            }
        )
    rows.sort(key=lambda row: str(row.get("ended_at") or ""), reverse=True)
    rows = rows[: max(1, int(maximum_runs))]
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_kind": "reference_free_online_policy_execution",
        "online_decision_uses_reference_solution": False,
        "reference_solution_metrics_included": False,
        "search_root": str(root),
        "run_count": len(rows),
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def _reference_free_copy(value: Any) -> Any:
    """Recursively drop reference-solution fields from reusable evidence."""

    if isinstance(value, dict):
        return {
            str(key): _reference_free_copy(item)
            for key, item in value.items()
            if str(key).casefold() not in _REFERENCE_SOLUTION_KEYS
        }
    if isinstance(value, list):
        return [_reference_free_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_reference_free_copy(item) for item in value]
    return deepcopy(value)


__all__ = ["build_online_execution_evidence"]
