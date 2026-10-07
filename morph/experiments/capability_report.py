#!/usr/bin/env python3
"""Aggregate audit-only LLM capability gaps across PDE runs and seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: {path}:{line_number}: {exc}")
                continue
            if isinstance(value, Mapping):
                yield dict(value)


def _problem_id(record: Mapping[str, Any]) -> str:
    pde = dict(dict(record.get("context") or {}).get("pde") or {})
    return str(
        pde.get("problem_id")
        or pde.get("benchmark_id")
        or pde.get("pde_type")
        or "unknown"
    )


def summarize_capability_gaps(results_root: Path) -> dict[str, Any]:
    root = results_root.resolve()
    files = sorted(root.rglob("capability_gaps.jsonl"))
    raw_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in files:
        run = str(path.parent.relative_to(root))
        for record in _read_jsonl(path):
            gap_id = str(record.get("gap_id") or "")
            # Checkpoint resumes can replay an evolution event. Count a stable
            # observation once per run while preserving occurrences in other runs.
            identity = f"{run}:{gap_id}" if gap_id else json.dumps(
                [run, record], ensure_ascii=False, sort_keys=True, default=str
            )
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            record["run"] = run
            record["problem_id"] = _problem_id(record)
            raw_records.append(record)

    # Schema-repair retries are not independent scientific support. Preserve
    # the raw count for audit, but rank one semantic mention per run/state/role.
    records: list[dict[str, Any]] = []
    mention_keys: set[tuple[str, str, str, str, str]] = set()
    for record in raw_records:
        mention_key = (
            str(record.get("run")),
            str(record.get("state_id")),
            str(record.get("mode")),
            str(record.get("kind")),
            str(record.get("proposed")),
        )
        if mention_key in mention_keys:
            continue
        mention_keys.add(mention_key)
        records.append(record)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("kind")), str(record.get("proposed")))].append(record)

    ranked: list[dict[str, Any]] = []
    for (kind, proposed), items in groups.items():
        pdes = sorted({str(item["problem_id"]) for item in items})
        runs = sorted({str(item["run"]) for item in items})
        modes = sorted({str(item.get("mode") or "unknown") for item in items})
        examples = []
        for item in items[:3]:
            examples.append(
                {
                    key: item.get(key)
                    for key in (
                        "gap_id", "run", "problem_id", "step", "mode",
                        "action", "mechanism", "proposal", "context",
                    )
                    if item.get(key) is not None
                }
            )
        ranked.append(
            {
                "rank_score": len(items) * max(1, len(pdes)),
                "occurrences": len(items),
                "pde_coverage": len(pdes),
                "run_coverage": len(runs),
                "kind": kind,
                "proposed": proposed,
                "pdes": pdes,
                "runs": runs,
                "modes": modes,
                "examples": examples,
            }
        )
    ranked.sort(
        key=lambda item: (
            -int(item["pde_coverage"]),
            -int(item["run_coverage"]),
            -int(item["occurrences"]),
            str(item["kind"]),
            str(item["proposed"]),
        )
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_root": str(root),
        "source_files": [str(path) for path in files],
        "raw_observation_count": len(raw_records),
        "observation_count": len(records),
        "distinct_gap_count": len(ranked),
        "ranking_policy": "PDE coverage, then run coverage, then occurrence count",
        "ranked_gaps": ranked,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# PINNMorph capability-gap report",
        "",
        f"- Observations: {report['observation_count']}",
        f"- Raw observations before retry de-duplication: {report['raw_observation_count']}",
        f"- Distinct gaps: {report['distinct_gap_count']}",
        f"- Source files: {len(report['source_files'])}",
        "- Safety: audit records only; no missing capability was executed.",
        "",
        "| Priority | Missing capability | Kind | Occurrences | PDEs | Runs |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, item in enumerate(report.get("ranked_gaps") or (), 1):
        proposed = str(item["proposed"]).replace("|", "\\|")
        lines.append(
            f"| {index} | `{proposed}` | {item['kind']} | "
            f"{item['occurrences']} | {item['pde_coverage']} | {item['run_coverage']} |"
        )
    if report.get("ranked_gaps"):
        lines.extend(["", "## Evidence details", ""])
    for item in report.get("ranked_gaps") or ():
        proposed = str(item["proposed"])
        pdes = ", ".join(item.get("pdes") or ())
        lines.extend([f"### `{proposed}`", "", f"PDE coverage: {pdes}.", ""])
        proposal = next(
            (
                example.get("proposal")
                for example in item.get("examples") or ()
                if example.get("proposal")
            ),
            None,
        )
        if proposal:
            lines.append(
                "Representative rationale: "
                + str(proposal.get("rationale") or "not supplied")
                + "\n"
            )
    if not report.get("ranked_gaps"):
        lines.extend(["", "No capability-gap observations were recorded."])
    return "\n".join(lines).rstrip() + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results-root", type=Path, required=True)
    result.add_argument("--json", type=Path, default=None)
    result.add_argument("--markdown", type=Path, default=None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.results_root.resolve()
    report = summarize_capability_gaps(root)
    json_path = (args.json or root / "capability_gap_report.json").resolve()
    markdown_path = (
        args.markdown or root / "capability_gap_report.md"
    ).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Capability gaps: {report['observation_count']} observations, "
        f"{report['distinct_gap_count']} distinct gaps"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
