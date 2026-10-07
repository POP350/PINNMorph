#!/usr/bin/env python3
"""Run the thirteen PINNMorph benchmarks and their matched ablations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morph.experiments.benchmark_config import SUITE_PDES, VARIANTS, profile
from morph.experiments.experiment_config import config_hash, load_experiment_config
from morph.utils.experiment_provenance import source_identity

MANIFEST = "comparison_manifest.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pdes", nargs="+", choices=SUITE_PDES, default=SUITE_PDES)
    value.add_argument("--variants", nargs="+", choices=VARIANTS, default=("full",))
    value.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    value.add_argument("--gpus", nargs="+", metavar="GPU")
    value.add_argument("--per-gpu-concurrency", type=int, default=1)
    value.add_argument("--python", default=sys.executable)
    value.add_argument("--results-root", type=Path, default=Path("results/benchmarks"))
    value.add_argument("--retries", type=int, default=1, help="Clean retries after a failed suite.")
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    value.add_argument("--no-resume", action="store_true")
    value.add_argument("--stop-on-failure", action="store_true")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    configs = {
        variant: {pde: profile(pde, variant, args.seeds) for pde in args.pdes}
        for variant in args.variants
    }
    identity = {
        "pdes": list(args.pdes), "variants": list(args.variants), "seeds": list(args.seeds),
        "source_sha256": source_identity(PROJECT_ROOT)["sha256"],
        "resolved_config_hashes": {
            variant: {pde: config_hash(config) for pde, config in entries.items()}
            for variant, entries in configs.items()
        },
    }
    manifest_path = args.results_root / MANIFEST
    previous = read_json(manifest_path)
    if previous:
        if previous.get("experiment_identity") != identity:
            raise ValueError("Results contain a different experiment. Use a fresh --results-root.")
        for variant, entries in configs.items():
            for pde, config in entries.items():
                saved = args.results_root / "_configs" / variant / f"{pde}.yaml"
                if not saved.is_file() or config_hash(load_experiment_config(saved)) != config_hash(config):
                    raise ValueError(f"Saved configuration changed or is missing: {saved}")
        return previous
    if args.results_root.exists() and any(args.results_root.iterdir()):
        raise ValueError("Results root is nonempty without a benchmark manifest. Use a fresh --results-root.")
    args.results_root.mkdir(parents=True, exist_ok=True)
    for variant, entries in configs.items():
        folder = args.results_root / "_configs" / variant
        folder.mkdir(parents=True, exist_ok=True)
        for pde, config in entries.items():
            (folder / f"{pde}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
    manifest = {"schema_version": 1, "experiment_identity": identity}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def scheduler_command(args: argparse.Namespace, variant: str, *, retry: bool = False) -> list[str]:
    command = [
        args.python, "-u", "-m", "morph.experiments.suite_scheduler",
        "--gpus", *args.gpus, "--pdes", *args.pdes,
        "--seeds", *map(str, args.seeds),
        "--per-gpu-concurrency", str(args.per_gpu_concurrency),
        "--python", args.python,
        "--config-root", str(args.results_root / "_configs" / variant),
        "--results-root", str(args.results_root / variant),
    ]
    for flag in ("dry_run", "preflight_only", "no_resume", "stop_on_failure"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    if retry:
        command.append("--retry-failed-clean")
    return command


def summarize(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    design = manifest["experiment_identity"]
    rows = []
    for variant in design["variants"]:
        for pde in design["pdes"]:
            config = load_experiment_config(root / "_configs" / variant / f"{pde}.yaml")
            for seed in design["seeds"]:
                directory = (root / variant / str(config.get("result_group") or config["phase"])
                             / pde / config["method"] / f"seed_{seed}")
                saved = read_json(directory / "run_summary.json")
                metrics, resources = saved.get("metrics") or {}, saved.get("resources") or {}
                rows.append({
                    "variant": variant, "pde": pde, "seed": seed,
                    "status": saved.get("status") or "missing",
                    "relative_l2": metrics.get("relative_l2"), "mse": metrics.get("mse"),
                    "wall_clock_time": resources.get("wall_clock_time"),
                    "peak_gpu_memory": resources.get("peak_gpu_memory"),
                    "actual_optimizer_steps": resources.get("actual_optimizer_steps"),
                    "morphs_applied": (saved.get("morphing") or {}).get("applied"),
                    "llm_calls": (saved.get("llm") or {}).get("calls"),
                    "run_directory": directory.relative_to(root).as_posix(),
                })
    with (root / "comparison_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# PINNMorph benchmark comparison", "",
             "Only successful runs with finite MSE enter the mean and sample standard deviation.",
             "Incomplete seed sets are marked explicitly.", "",
             "| PDE | Variant | Successful seeds | MSE mean | Sample std | Complete |",
             "| --- | --- | --- | --- | --- | --- |"]
    for pde in design["pdes"]:
        for variant in design["variants"]:
            values = [row["mse"] for row in rows
                      if row["pde"] == pde and row["variant"] == variant
                      and row["status"] == "success" and isinstance(row["mse"], (int, float))
                      and math.isfinite(row["mse"])]
            mean = f"{statistics.mean(values):.6g}" if values else "—"
            std = f"{statistics.stdev(values):.6g}" if len(values) > 1 else "—"
            expected = len(design["seeds"])
            complete = "yes" if len(values) == expected else "no"
            lines.append(f"| {pde} | {variant} | {len(values)}/{expected} | {mean} | {std} | {complete} |")
    (root / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison: {root / 'comparison_summary.md'}", flush=True)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    args.results_root = (PROJECT_ROOT / args.results_root).resolve()
    if args.summary_only:
        manifest = read_json(args.results_root / MANIFEST)
        if not manifest:
            cli.error("--summary-only requires an existing benchmark manifest")
        summarize(args.results_root, manifest)
        return 0
    if not args.gpus:
        cli.error("--gpus is required unless --summary-only is selected")
    for name in ("gpus", "pdes", "variants", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            cli.error(f"--{name} values must be unique")
    if args.per_gpu_concurrency < 1 or args.retries < 0 or min(args.seeds) < 0:
        cli.error("concurrency must be positive; retries and seeds must be nonnegative")
    manifest = prepare(args)
    failure = 0
    for variant in args.variants:
        attempts = 1 if args.dry_run or args.preflight_only else args.retries + 1
        for attempt in range(attempts):
            print(f"Variant {variant}, attempt {attempt + 1}/{attempts}", flush=True)
            code = subprocess.run(
                scheduler_command(args, variant, retry=attempt > 0), cwd=PROJECT_ROOT
            ).returncode
            if code == 0 or code in (130, -2):
                break
        if code in (130, -2):
            return 130
        failure = failure or code
        if code and args.stop_on_failure:
            break
    if not args.dry_run and not args.preflight_only:
        summarize(args.results_root, manifest)
    return failure


if __name__ == "__main__":
    raise SystemExit(main())
