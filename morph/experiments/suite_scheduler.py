#!/usr/bin/env python3
"""Schedule resumable PINN-Online runs on explicitly selected GPUs.

Each PDE/seed pair runs in an independent inner_morph_runner process. The
number of workers per selected GPU is configurable and defaults to one.
Completed runs are skipped; interrupted runs resume from their checkpoints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import statistics
import shutil
import signal
import subprocess
import sys
from threading import Event, Lock
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morph.experiments.experiment_config import (  # noqa: E402
    apply_ablation_mode,
    config_hash,
    load_experiment_config,
)
from morph.utils.experiment_provenance import assert_manifest_compatible, assert_run_compatible, source_identity
from morph.experiments.capability_report import (  # noqa: E402
    render_markdown as render_capability_gap_markdown,
    summarize_capability_gaps,
)


CONFIG_ROOT = PROJECT_ROOT / "configs"
EXPECTED_OPTIMIZER_STEPS = 20000
DEFAULT_RESULTS_ROOT = Path("results/pinn_online_evolution_full_suite")
from morph.experiments.benchmark_config import SUITE_PDES
PRINT_LOCK = Lock()
_LIVE_PROCESSES: set[subprocess.Popen] = set()
_LIVE_LOCK = Lock()
_SHUTDOWN = Event()


@dataclass(frozen=True)
class Job:
    order: int
    pde: str
    seed: int
    config: str
    result_group: str
    method: str


@dataclass
class JobResult:
    order: int
    lane: int
    gpu: str
    pde: str
    seed: int
    config: str
    command: list[str]
    log_file: str
    run_directory: str
    status: str
    exit_code: int
    elapsed_seconds: float
    started_at_utc: str | None = None
    ended_at_utc: str | None = None


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gpus", nargs="+", required=True, metavar="GPU")
    value.add_argument("--per-gpu-concurrency", type=int, default=1, metavar="N")
    value.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4), metavar="SEED")
    value.add_argument("--pdes", nargs="+", choices=SUITE_PDES, default=SUITE_PDES, metavar="PDE")
    value.add_argument("--python", default=sys.executable)
    value.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    value.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--no-resume", action="store_true")
    value.add_argument(
        "--ablation-mode",
        choices=("none", "no_morph", "random_morph", "random_selection", "random_evolution"),
        default="none",
        help="Run an ablation variant without editing every PDE config.",
    )
    value.add_argument(
        "--retry-failed-clean",
        action="store_true",
        help=(
            "Archive failed run directories under _failed_attempts and rerun "
            "them cleanly; completed runs remain untouched."
        ),
    )
    value.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Halt the queue when a job fails (default: keep going).",
    )
    return value


def load_config_for_args(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    config = load_experiment_config(path)
    if str(args.ablation_mode) != "none":
        config = apply_ablation_mode(config, str(args.ablation_mode))
    return config


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    order = 0
    for seed in args.seeds:
        for pde in args.pdes:
            config = args.config_root / f"{pde}.yaml"
            if not config.is_file():
                raise SystemExit(f"Missing experiment config: {config}")
            resolved = load_config_for_args(args, config)
            jobs.append(
                Job(
                    order,
                    pde,
                    int(seed),
                    str(config),
                    str(resolved.get("result_group") or resolved["phase"]),
                    str(resolved["method"]),
                )
            )
            order += 1
    return jobs


def validate_configs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    from morph.experiments.inner_morph_runner import normalized_spec

    seen = {job.config for job in jobs}
    for config_path in sorted(seen):
        config = load_config_for_args(args, Path(config_path))
        _, spec, report = normalized_spec(config)
        if not report.valid:
            raise SystemExit(f"Normalized experiment config is invalid: {config_path}")
        steps = sum(int(item["iterations"]) for item in spec["optimization"]["phases"])
        if steps != EXPECTED_OPTIMIZER_STEPS:
            raise SystemExit(
                f"{config_path} resolved to {steps} optimizer steps instead of "
                f"{EXPECTED_OPTIMIZER_STEPS}"
            )
    _print(f"Config validation passed for {len(seen)} PDE configs ({EXPECTED_OPTIMIZER_STEPS} steps each).")


def check_prior_evidence(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    config = load_config_for_args(args, Path(jobs[0].config))
    path = (
        (config.get("online_morphing") or {}).get("policy_set") or {}
    ).get("prior_evidence_path")
    if not path:
        _print("Execution evidence is not configured; using the built-in 5-policy cold start plus the PINNacle PDE prior.")
        return
    evidence = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    if not evidence.is_file():
        _print(
            "External execution evidence is absent; using the built-in 5-policy "
            f"cold start plus the PINNacle PDE prior: {evidence}"
        )
        return
    _print(f"Prior evidence: {evidence}")


def child_environment(job: Job, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(job.seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PINNSFORGE_PERFORMANCE_MODE"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for gpu in dict.fromkeys(args.gpus):
        code = (
            "import json,torch,numpy,scipy,pydantic,yaml; "
            "assert torch.cuda.is_available(),'CUDA unavailable'; "
            "p=torch.cuda.get_device_properties(0); "
            "print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
            "'numpy':numpy.__version__,'scipy':scipy.__version__,"
            "'pydantic':pydantic.__version__,'pyyaml':yaml.__version__,"
            "'device_name':p.name,'total_memory_bytes':p.total_memory,"
            "'capability':[p.major,p.minor]}))"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        completed = subprocess.run(
            [args.python, "-c", code],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise SystemExit(
                f"GPU preflight failed for physical GPU {gpu}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append({"gpu": str(gpu), "status": "ok", **payload})
        _print(
            f"GPU preflight ok: physical GPU {gpu} = {payload['device_name']} "
            f"({int(payload['total_memory_bytes']) // (1024 ** 2)} MiB)"
        )
    return results


def llm_preflight(args: argparse.Namespace, jobs: Sequence[Job]) -> dict[str, Any]:
    if args.dry_run:
        _print("LLM preflight: skipped for dry-run invocation.")
        return {"required": True, "status": "skipped_dry_run"}
    from morph.experiments.inner_morph_runner import (
        _build_dual_llm_policy,
    )
    from morph.pipeline.d_algorithm_generation.llm.provider import redact_secret

    config = load_config_for_args(args, Path(jobs[0].config))
    trackers = _build_dual_llm_policy(config)
    if not trackers:
        _print("LLM preflight: not required for this ablation mode.")
        return {"required": False, "status": "not_required"}
    channels: dict[str, Any] = {}
    for role, tracker in trackers.items():
        response = tracker.complete_text(
            [{"role": "user", "content": 'Return only JSON: {"status":"ok"}'}],
            temperature=0.0,
            max_tokens=64,
            task=f"pinn_online_{role}_preflight",
        )
        if not response.success:
            raise SystemExit(
                f"LLM preflight failed for {role}: "
                + redact_secret(
                    str(response.error or response.text), tracker.provider.api_key
                )
            )
        _print(
            f"LLM preflight ok [{role}]: provider={response.provider} "
            f"model={response.model} latency={response.latency_seconds:.1f}s"
        )
        channels[role] = {
            "provider": response.provider,
            "requested_model": response.model,
            "response_model": response.model_version,
            "finish_reason": response.finish_reason,
            "latency_seconds": response.latency_seconds,
            "token_usage": response.token_usage,
        }
    return {
        "required": True,
        "status": "ok",
        "channels": channels,
    }


def run_directory(results_root: Path, job: Job) -> Path:
    return (
        results_root
        / job.result_group
        / job.pde
        / job.method
        / f"seed_{job.seed}"
    )


def log_path(results_root: Path, job: Job) -> Path:
    return results_root / "_launcher_logs" / f"{job.pde}_seed_{job.seed}.log"


def command_for(args: argparse.Namespace, job: Job) -> list[str]:
    command = [
        args.python,
        "-u",
        "-m",
        "morph.experiments.inner_morph_runner",
        "--config",
        job.config,
        "--results-root",
        str(args.results_root),
        "--seeds",
        str(job.seed),
    ]
    if str(args.ablation_mode) != "none":
        command.extend(["--ablation-mode", str(args.ablation_mode)])
    if args.no_resume:
        command.append("--no-resume")
    return command


def _queue_depth(job_q: "queue.Queue[Job]") -> int:
    return job_q.qsize()


def run_job(
    args: argparse.Namespace, job: Job, lane: int, gpu: str, job_q: "queue.Queue[Job]"
) -> JobResult:
    log = log_path(args.results_root, job)
    target = run_directory(args.results_root, job)
    summary = target / "run_summary.json"
    existing = _read_json(summary)
    if existing:
        assert_run_compatible(existing, config_hash(load_config_for_args(args, Path(job.config))), args._source_evidence["sha256"])
    if not args.no_resume and _completed(summary):
        return JobResult(
            job.order, lane, gpu, job.pde, job.seed, job.config,
            command_for(args, job), str(log), str(target), "skipped_completed", 0, 0.0,
        )
    if args.retry_failed_clean and _failed(summary) and not args.dry_run:
        archived = _archive_failed_run(args.results_root, job, target)
        _print(
            f"[lane {lane}] archived failed {job.pde} seed={job.seed} to {archived}"
        )
    command = command_for(args, job)
    if args.dry_run:
        return JobResult(
            job.order, lane, gpu, job.pde, job.seed, job.config,
            command, str(log), str(target), "dry_run", 0, 0.0,
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    _print(
        f"[lane {lane} | GPU {gpu}] start {job.pde} seed={job.seed} "
        f"({_queue_depth(job_q)} jobs left after this one)"
    )
    returncode: int | None = None
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n=== attempt {started_at}; lane={lane}; physical_gpu={gpu}; "
            f"pde={job.pde}; seed={job.seed} ===\n"
        )
        stream.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment(job, gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with _LIVE_LOCK:
            _LIVE_PROCESSES.add(process)
        try:
            while process.poll() is None:
                if _SHUTDOWN.is_set():
                    process.terminate()
                    break
                time.sleep(2.0)
            returncode = process.wait()
        finally:
            with _LIVE_LOCK:
                _LIVE_PROCESSES.discard(process)
    elapsed = time.monotonic() - started
    if returncode is None:
        returncode = 1
    status = "completed" if returncode == 0 and _completed(summary) else "failed"
    if status == "failed" and _SHUTDOWN.is_set():
        status = "interrupted"
    _print(
        f"[lane {lane}] {status} {job.pde} seed={job.seed} "
        f"({elapsed / 60.0:.1f} min; log: {log.name})"
    )
    if status == "failed":
        tail = _tail(log, 30)
        if tail:
            _print(f"--- log tail: {job.pde} seed={job.seed} ---\n{tail}")
    return JobResult(
        job.order, lane, gpu, job.pde, job.seed, job.config,
        command, str(log), str(target), status, returncode, round(elapsed, 3),
        started_at, _now(),
    )


def run_lane(
    args: argparse.Namespace,
    job_q: "queue.Queue[Job]",
    lane: int,
    gpu: str,
) -> list[JobResult]:
    results: list[JobResult] = []

    def drain_not_started() -> None:
        while True:
            try:
                job = job_q.get_nowait()
            except queue.Empty:
                return
            results.append(
                JobResult(
                    job.order, lane, gpu, job.pde, job.seed, job.config,
                    command_for(args, job), str(log_path(args.results_root, job)),
                    str(run_directory(args.results_root, job)), "not_started", 1, 0.0,
                )
            )

    while True:
        if _SHUTDOWN.is_set():
            drain_not_started()
            return results
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return results
        result = run_job(args, job, lane, gpu, job_q)
        results.append(result)
        if result.status == "failed" and args.stop_on_failure:
            _SHUTDOWN.set()
            drain_not_started()
            return results


def summarize(args: argparse.Namespace, jobs: Sequence[Job]) -> dict[str, Any]:
    table: dict[str, dict[int, dict[str, Any]]] = {}
    for job in jobs:
        summary = _read_json(run_directory(args.results_root, job) / "run_summary.json")
        metrics = dict(summary.get("metrics") or {})
        resources = dict(summary.get("resources") or {})
        table.setdefault(job.pde, {})[job.seed] = {
            "status": summary.get("status") or "missing",
            "relative_l2": metrics.get("relative_l2"),
            "mse": metrics.get("mse"),
            "wall_clock_time": resources.get("wall_clock_time"),
        }
    _print("\n=== Suite summary (MSE mean ± sample standard deviation) ===")
    mse_summary = {}
    for pde in args.pdes:
        values = [entry["mse"] for entry in table.get(pde, {}).values()
                  if entry["status"] == "success" and isinstance(entry.get("mse"), (int, float))]
        mse_summary[pde] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                            "std": statistics.stdev(values) if len(values) > 1 else None}
        item = mse_summary[pde]
        label = (f"{item['mean']:.6g} ± {item['std']:.6g}"
                 if item["n"] == len(args.seeds) and item["std"] is not None else "incomplete")
        _print(f"{pde:<32} {label} (n={item['n']})")
    counts: dict[str, int] = {}
    for seeds in table.values():
        for entry in seeds.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    _print(f"Totals: {json.dumps(counts, ensure_ascii=False)}")
    return {"per_pde": table, "mse_mean_std": mse_summary, "status_counts": counts}


def write_manifest(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    gpu_evidence: Sequence[dict[str, Any]],
    llm_evidence: Mapping[str, Any] | None = None,
    results: Sequence[JobResult] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> Path:
    args.results_root.mkdir(parents=True, exist_ok=True)
    path = args.results_root / "experiment_manifest.json"
    previous = _read_json(path)
    identity = experiment_identity(args, jobs)
    assert_manifest_compatible(previous, identity)
    payload = {
        "schema_version": 1,
        "experiment": "pinn_online_llm_evolution_full_suite",
        "experiment_identity": identity,
        "source_identity": args._source_evidence,
        "created_at_utc": previous.get("created_at_utc") or _now(),
        "updated_at_utc": _now(),
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_evidence(),
        "gpu_preflight": list(gpu_evidence),
        "llm_preflight": dict(llm_evidence or {}),
        "design": {
            "pdes": list(args.pdes),
            "seeds": list(map(int, args.seeds)),
            "gpus": list(args.gpus),
            "per_gpu_concurrency": int(args.per_gpu_concurrency),
            "total_jobs": len(jobs),
            "job_ordering": "seed-major: all PDEs at seed 0 first, then seed 1, ...",
            "lane_assignment": "jobs pulled from one FIFO queue; lane -> physical GPU via CUDA_VISIBLE_DEVICES",
            "reference_solution_visible_to_online_controller": False,
        },
        "configs": {
            pde: {
                "path": str(args.config_root / f"{pde}.yaml"),
                "sha256": _sha256(args.config_root / f"{pde}.yaml"),
                "resolved_sha256": identity["resolved_configs"][pde],
            }
            for pde in args.pdes
        },
        "jobs": [asdict(item) for item in jobs],
        "results": ([asdict(item) for item in sorted(results, key=lambda item: item.order)]
                    if results is not None else previous.get("results", [])),
        "final_metrics": dict(summary if summary is not None else previous.get("final_metrics", {})),
    }
    _write_json(path, payload)
    return path


def experiment_identity(args: argparse.Namespace, jobs: Sequence[Job]) -> dict[str, Any]:
    if not hasattr(args, "_source_evidence"):
        args._source_evidence = source_identity(PROJECT_ROOT)
    return {
        "source_sha256": args._source_evidence["sha256"],
        "pdes": list(args.pdes), "seeds": list(map(int, args.seeds)),
        "ablation_mode": args.ablation_mode,
        "resolved_configs": {
            job.pde: config_hash(load_config_for_args(args, Path(job.config)))
            for job in jobs
        },
    }


def _git_evidence() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=PROJECT_ROOT, text=True, capture_output=True
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
        "available": status is not None,
    }


def _archive_failed_run(results_root: Path, job: Job, target: Path) -> Path:
    """Move one exact failed run aside so retry evidence starts cleanly."""

    archive_root = results_root / "_failed_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = archive_root / f"{job.pde}_seed_{job.seed}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"{job.pde}_seed_{job.seed}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(target), str(candidate))
    return candidate


def _completed(path: Path) -> bool:
    return _read_json(path).get("status") == "success"


def _failed(path: Path) -> bool:
    return _read_json(path).get("status") == "failed"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail(path: Path, count: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])
    except OSError:
        return ""


def _print(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def _request_shutdown(signum: int, frame: Any) -> None:
    _print(
        f"\nReceived signal {signum}: stopping in-flight runs; completed runs are "
        "preserved. Relaunch the same command to resume the queue."
    )
    _SHUTDOWN.set()
    with _LIVE_LOCK:
        for process in list(_LIVE_PROCESSES):
            if process.poll() is None:
                process.terminate()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _SHUTDOWN.clear()
    args.config_root = args.config_root.resolve()
    for label, values in (("GPUs", args.gpus), ("seeds", args.seeds), ("PDEs", args.pdes)):
        if not values or len(set(values)) != len(values):
            raise SystemExit(f"{label} must be nonempty and unique")
    if args.per_gpu_concurrency < 1:
        raise SystemExit("--per-gpu-concurrency must be >= 1")
    args.results_root = (
        (PROJECT_ROOT / args.results_root).resolve()
        if not args.results_root.is_absolute()
        else args.results_root.resolve()
    )
    jobs = build_jobs(args)
    identity = experiment_identity(args, jobs)
    assert_manifest_compatible(_read_json(args.results_root / "experiment_manifest.json"), identity)
    for job in jobs:
        saved = _read_json(run_directory(args.results_root, job) / "run_summary.json")
        assert_run_compatible(saved, identity["resolved_configs"][job.pde], identity["source_sha256"])
    lanes = len(args.gpus) * args.per_gpu_concurrency
    _print(
        f"Suite: {len(args.pdes)} PDEs x {len(args.seeds)} seeds = {len(jobs)} jobs | "
        f"{len(args.gpus)} GPUs x {args.per_gpu_concurrency} lanes = {lanes} concurrent runs"
    )
    validate_configs(args, jobs)
    check_prior_evidence(args, jobs)
    _print("\nQueue (seed-major, first 10 shown):")
    for job in jobs[:10]:
        _print(f"  {job.order + 1:02d}. {job.pde} seed={job.seed}")
    if len(jobs) > 10:
        _print(f"  ... and {len(jobs) - 10} more")
    gpu_evidence = [] if args.dry_run else preflight(args)
    llm_evidence = llm_preflight(args, jobs)
    if not args.dry_run:
        write_manifest(args, jobs, gpu_evidence, llm_evidence)
    if args.preflight_only:
        _print("Preflight passed; no training started.")
        return 0

    if args.dry_run:
        _print(
            f"Dry run: all {len(jobs)} commands validated; no training was started."
        )
        for job in jobs[:5]:
            _print(f"  {' '.join(command_for(args, job))}")
        return 0

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_shutdown)

    job_q: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        job_q.put(job)
    lane_specs = [
        (lane, str(args.gpus[lane // args.per_gpu_concurrency]))
        for lane in range(lanes)
    ]
    _print(
        "\nLane assignment: "
        + ", ".join(f"lane {lane} -> GPU {gpu}" for lane, gpu in lane_specs)
        + "\n"
    )
    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=lanes) as executor:
        futures = [
            executor.submit(run_lane, args, job_q, lane, gpu)
            for lane, gpu in lane_specs
        ]
        try:
            for future in as_completed(futures):
                results.extend(future.result())
        except KeyboardInterrupt:
            _SHUTDOWN.set()
            for future in futures:
                future.cancel()
    summary = summarize(args, jobs)
    gap_report = summarize_capability_gaps(args.results_root)
    gap_json = args.results_root / "capability_gap_report.json"
    gap_markdown = args.results_root / "capability_gap_report.md"
    gap_json.write_text(
        json.dumps(gap_report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    gap_markdown.write_text(
        render_capability_gap_markdown(gap_report), encoding="utf-8"
    )
    summary["capability_gap_audit"] = {
        "raw_observation_count": gap_report["raw_observation_count"],
        "observation_count": gap_report["observation_count"],
        "distinct_gap_count": gap_report["distinct_gap_count"],
        "json_report": str(gap_json),
        "markdown_report": str(gap_markdown),
    }
    _print(
        "Capability-gap audit: "
        f"{gap_report['observation_count']} observations, "
        f"{gap_report['distinct_gap_count']} distinct; report: {gap_markdown}"
    )
    manifest = write_manifest(
        args, jobs, gpu_evidence, llm_evidence, results, summary
    )
    _print(f"\nManifest: {manifest}")
    failures = [
        item for item in results if item.status in ("failed", "not_started")
    ]
    if _SHUTDOWN.is_set() and not args.stop_on_failure:
        _print(
            "Queue was interrupted. Completed runs are preserved; relaunch the "
            "same command to resume."
        )
        return 130
    if failures:
        _print(f"{len(failures)} of {len(jobs)} jobs failed or were not started:")
        for item in failures:
            _print(f"  {item.pde} seed={item.seed} [{item.status}] log: {item.log_file}")
        _print(
            "Retry failed runs with --retry-failed-clean, then relaunch the same command."
        )
        return 1
    _print(f"All {len(jobs)} jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
