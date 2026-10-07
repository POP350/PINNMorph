"""Retrieve the legacy PINNsForge knowledge base for online policy guidance.

The online project deliberately keeps the legacy knowledge read-only.  It is
soft guidance for the registered high-level architecture actions, never a source of
reference-solution metrics or an alternative execution path.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


_CATALOGS = (
    ("macro_physics_principles.json", "macro_physics_principles"),
    ("architecture_design_principles.json", "architecture_design_principles"),
)
_COMPACT_FIELDS = (
    "principle_id",
    "category",
    "title",
    "required_conditions",
    "relevance_tags",
    "implication",
    "design_requirements",
    "recommendations",
    "risks",
    "tradeoffs",
    "evidence_sources",
)
_FORBIDDEN_REFERENCE_KEYS = {
    "mse",
    "relative_l2",
    "relative_l2_error",
    "reference_metrics",
    "reference_solution",
    "final_metrics",
}


def resolve_legacy_knowledge_base(
    configured_path: str | Path | None = None,
) -> Path | None:
    """Resolve a bundled, configured, or sibling PINNsForge knowledge base."""

    project_root = Path(__file__).resolve().parents[3]
    configured = str(configured_path or "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(os.environ["PINNSFORGE_KNOWLEDGE_BASE_PATH"]).expanduser()
        if os.environ.get("PINNSFORGE_KNOWLEDGE_BASE_PATH")
        else None,
        Path(__file__).resolve().parent / "knowledge_base",
        project_root.parent
        / "PINNsForge"
        / "forge"
        / "pipeline"
        / "c_knowledge_retrieval"
        / "knowledge_base",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if all((path / filename).is_file() for filename, _ in _CATALOGS):
            return path.resolve()
    return None


def retrieve_online_prior_context(
    problem_metadata: Mapping[str, Any],
    *,
    configured_path: str | Path | None = None,
    maximum_items: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return bounded legacy principles relevant to one online PDE context."""

    root = resolve_legacy_knowledge_base(configured_path)
    if root is None:
        return [], {
            "configured": bool(configured_path),
            "loaded": False,
            "reason": "legacy_knowledge_base_not_found",
            "items_loaded": 0,
        }
    context = _problem_context(problem_metadata)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    source_hash = hashlib.sha256()
    candidate_count = 0
    for filename, collection_key in _CATALOGS:
        path = root / filename
        raw = path.read_bytes()
        source_hash.update(filename.encode("utf-8"))
        source_hash.update(raw)
        payload = json.loads(raw.decode("utf-8-sig"))
        items = (
            payload.get(collection_key, [])
            if isinstance(payload, Mapping)
            else []
        )
        for item in items:
            if not isinstance(item, Mapping):
                continue
            candidate_count += 1
            score = _relevance_score(item, context)
            if score <= 0.0:
                continue
            compact = _compact_principle(item, source_file=filename)
            ranked.append(
                (
                    score,
                    str(compact.get("principle_id") or compact.get("title") or ""),
                    compact,
                )
            )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, _, item in ranked:
        identity = str(item.get("principle_id") or item.get("title") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        item["relevance_score"] = float(score)
        selected.append(item)
        if len(selected) >= max(1, int(maximum_items)):
            break
    return selected, {
        "configured": True,
        "loaded": bool(selected),
        "path": str(root),
        "sha256": source_hash.hexdigest(),
        "candidate_principles": candidate_count,
        "items_loaded": len(selected),
        "principle_ids": [item["principle_id"] for item in selected],
        "reference_solution_metrics_exposed": False,
        "binding": False,
    }


def _problem_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(metadata or {}))
    profile = dict(source.get("pinnacle_profile") or {})
    context = {**profile, **source}
    tags: set[str] = set()
    for key in ("challenge_tags", "evidence_tags", "pde_tags"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tags.update(str(item).casefold() for item in value)
    for key, value in context.items():
        if value is True:
            tags.add(str(key).casefold())
        elif isinstance(value, str):
            tags.add(value.casefold())
    context["_tags"] = tags
    return context


def _relevance_score(item: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    required = dict(item.get("required_conditions") or {})
    for key, expected in required.items():
        if not _condition_matches(context.get(key), expected):
            return 0.0
    score = 10.0 * len(required)
    tags = {
        str(value).casefold()
        for key in ("relevance_tags", "pde_families", "problem_features")
        for value in _as_list(item.get(key))
    }
    context_tags = set(context.get("_tags") or ())
    overlap = tags.intersection(context_tags)
    score += 2.0 * len(overlap)
    problem_ids = {
        str(context.get("problem_id") or "").casefold(),
        str(context.get("benchmark_id") or "").casefold(),
    }
    if any(value and value in tags for value in problem_ids):
        score += 12.0
    family = str(
        context.get("pde_family") or context.get("family") or ""
    ).casefold()
    if family and family in tags:
        score += 6.0
    # Generic derivative/conditioning principles remain available, but exact
    # PDE and feature matches always rank ahead of them.
    if not required and not overlap and not (family and family in tags):
        generic = {
            "derivative",
            "gradient",
            "conditioning",
            "capacity",
            "architecture",
        }
        if not tags.intersection(generic):
            return 0.0
        score = 0.5
    return score


def _condition_matches(actual: Any, expected: Any) -> bool:
    options = _as_list(expected)
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        actual_values = {str(value).casefold() for value in actual}
        return any(str(value).casefold() in actual_values for value in options)
    return any(
        actual == value or str(actual).casefold() == str(value).casefold()
        for value in options
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _compact_principle(
    item: Mapping[str, Any], *, source_file: str
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _COMPACT_FIELDS:
        if key in _FORBIDDEN_REFERENCE_KEYS or key not in item:
            continue
        value = deepcopy(item[key])
        if isinstance(value, list):
            value = value[:8]
        compact[key] = value
    compact.setdefault("principle_id", str(item.get("id") or item.get("title") or ""))
    compact["source"] = f"legacy_pinnsforge:{source_file}"
    compact["binding"] = False
    compact["applicable_online_actions"] = _applicable_actions(compact)
    return compact


def _applicable_actions(principle: Mapping[str, Any]) -> list[str]:
    text = json.dumps(principle, ensure_ascii=False, default=str).casefold()
    result = ["KEEP"]
    if any(token in text for token in ("capacity", "width", "depth", "underfit")):
        result.append("CONSERVATIVE_GROWTH")
    if any(token in text for token in ("topology", "depth", "connection")):
        result.append("EXPLORATORY_MORPH")
    if any(
        token in text
        for token in ("representation", "frequency", "fourier", "sine", "wave")
    ):
        result.append("REPRESENTATION_AUGMENTATION")
    if any(
        token in text
        for token in ("localized", "local expert", "subdomain", "domain decomposition")
    ):
        result.append("LOCAL_SPECIALIZATION")
    if any(
        token in text
        for token in ("gradient", "utility", "allocation", "balance", "conditioning")
    ):
        result.append("REALLOCATION")
    if len(result) == 1:
        result.extend(
            [
                "CONSERVATIVE_GROWTH",
                "EXPLORATORY_MORPH",
                "REALLOCATION",
                "REPRESENTATION_AUGMENTATION",
                "LOCAL_SPECIALIZATION",
            ]
        )
    return list(dict.fromkeys(result))


__all__ = [
    "resolve_legacy_knowledge_base",
    "retrieve_online_prior_context",
]
