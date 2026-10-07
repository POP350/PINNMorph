"""Regression checks for package renaming, public entrypoints and run provenance."""
import json
import shutil
import subprocess

import pytest

from morph.experiments.benchmark_config import CONFIG_ROOT, SUITE_PDES, VARIANTS, profile
from morph.experiments.experiment_config import load_experiment_config
from morph.experiments.inner_morph_runner import normalized_spec
from morph.experiments import suite_scheduler
from morph.pipeline.a_problem_definition.problems import list_problems
from morph.utils.experiment_provenance import source_identity
from scripts import run_benchmarks as runner


def arguments(root, *extra):
    return runner.parser().parse_args([
        "--pdes", "burgers_1d", "--seeds", "0", "--gpus", "0",
        "--results-root", str(root), *extra,
    ])


def test_public_scope_and_both_config_defaults():
    assert set(list_problems()) == set(SUITE_PDES)
    assert len(SUITE_PDES) == 13
    assert CONFIG_ROOT == suite_scheduler.CONFIG_ROOT == runner.PROJECT_ROOT / "configs"
    assert {p.stem for p in CONFIG_ROOT.glob("*.yaml")} == set(SUITE_PDES) | {"base", "training_base"}


@pytest.mark.parametrize("pde", SUITE_PDES)
def test_all_variants_preserve_matched_physics_and_step_budget(pde):
    configs = [profile(pde, variant, [0, 1, 2, 3, 4]) for variant in VARIANTS]
    specs = [normalized_spec(config)[1] for config in configs]
    for spec in specs:
        assert sum(phase["iterations"] for phase in spec["optimization"]["phases"]) == 20000
    for section in ("network", "loss", "optimization"):
        assert all(spec[section] == specs[0][section] for spec in specs)
    assert not configs[0]["online_morphing"]["enabled"]
    assert all(config["online_morphing"]["enabled"] for config in configs[1:])


@pytest.mark.parametrize("extra", [
    ["--pdes", "removed_pde"], ["--seeds", "0", "0"],
    ["--retries", "-1"], ["--per-gpu-concurrency", "0"],
])
def test_invalid_requests_do_not_create_output(tmp_path, extra):
    root = tmp_path / "results"
    with pytest.raises(SystemExit):
        runner.main(["--gpus", "0", "--results-root", str(root), *extra])
    assert not root.exists()


def test_gpu_selection_is_required_before_writing(tmp_path):
    root = tmp_path / "results"
    with pytest.raises(SystemExit):
        runner.main(["--dry-run", "--results-root", str(root)])
    assert not root.exists()


def test_changed_design_or_saved_config_is_rejected_without_overwriting(tmp_path):
    args = arguments(tmp_path)
    runner.prepare(args)
    config = tmp_path / "_configs/full/burgers_1d.yaml"
    original = config.read_bytes()
    args.seeds = [1]
    with pytest.raises(ValueError, match="different experiment"):
        runner.prepare(args)
    assert config.read_bytes() == original
    args.seeds = [0]
    config.write_bytes(original.replace(b"total_optimizer_steps: 20000", b"total_optimizer_steps: 19999"))
    changed = config.read_bytes()
    with pytest.raises(ValueError):
        runner.prepare(args)
    assert config.read_bytes() == changed


def test_existing_unidentified_results_are_preserved(tmp_path):
    sentinel = tmp_path / "existing.pt"
    sentinel.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="nonempty"):
        runner.prepare(arguments(tmp_path))
    assert sentinel.read_bytes() == b"checkpoint"


def test_failed_suite_gets_clean_retry_but_dry_run_does_not(tmp_path, monkeypatch):
    commands = []
    results = iter([1, 0, 1])

    def execute(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, next(results))

    monkeypatch.setattr(runner.subprocess, "run", execute)
    base = ["--pdes", "burgers_1d", "--seeds", "0", "--gpus", "0"]
    assert runner.main([*base, "--results-root", str(tmp_path / "real")]) == 0
    assert "--retry-failed-clean" not in commands[0]
    assert "--retry-failed-clean" in commands[1]
    assert commands[0][3] == "morph.experiments.suite_scheduler"
    assert runner.main([*base, "--dry-run", "--results-root", str(tmp_path / "dry")]) == 1
    assert len(commands) == 3
    assert not (tmp_path / "dry/comparison_runs.csv").exists()


def test_summary_uses_saved_design_and_excludes_failed_values(tmp_path):
    args = arguments(tmp_path, "--seeds", "0", "1", "2")
    runner.prepare(args)
    config = load_experiment_config(tmp_path / "_configs/full/burgers_1d.yaml")
    for seed, status, mse in [(0, "success", 1.0), (1, "success", 3.0), (2, "failed", 999.0)]:
        directory = (tmp_path / "full" / config["result_group"] / "burgers_1d"
                     / config["method"] / f"seed_{seed}")
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text(json.dumps({
            "status": status, "metrics": {"mse": mse},
        }))
    assert runner.main(["--summary-only", "--results-root", str(tmp_path)]) == 0
    report = (tmp_path / "comparison_summary.md").read_text(encoding="utf-8")
    assert "| burgers_1d | full | 2/3 | 2 | 1.41421 | no |" in report
    assert "999" not in report
    assert "failed" in (tmp_path / "comparison_runs.csv").read_text()


def test_scheduler_dry_run_skips_gpu_and_llm_requests(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted a GPU or LLM preflight")

    monkeypatch.setattr(suite_scheduler, "preflight", forbidden)
    # llm_preflight imports providers only after its dry-run branch.
    monkeypatch.setattr(suite_scheduler, "write_manifest", forbidden)
    assert suite_scheduler.main([
        "--gpus", "0", "--pdes", "burgers_1d", "--seeds", "0",
        "--dry-run", "--results-root", str(tmp_path / "dry"),
    ]) == 0
    args = suite_scheduler.parser().parse_args(["--gpus", "0", "--pdes", "burgers_1d"])
    job = suite_scheduler.build_jobs(args)[0]
    assert "morph.experiments.inner_morph_runner" in suite_scheduler.command_for(args, job)


def test_source_identity_covers_renamed_package_and_shell_launcher(tmp_path):
    (tmp_path / "morph").mkdir()
    (tmp_path / "scripts").mkdir()
    module = tmp_path / "morph/core.py"
    launcher = tmp_path / "scripts/run.sh"
    module.write_bytes(b"x = 1\r\n")
    launcher.write_bytes(b"python main.py\r\n")
    before = source_identity(tmp_path)
    assert set(before["files"]) == {"morph/core.py", "scripts/run.sh"}
    module.write_bytes(b"x = 1\n")
    assert source_identity(tmp_path) == before
    launcher.write_bytes(b"python other.py\n")
    assert source_identity(tmp_path)["sha256"] != before["sha256"]


def test_git_submission_excludes_local_secrets_and_caches(tmp_path):
    git = shutil.which("git")
    if not git:
        pytest.skip("Git is not installed")
    repository = tmp_path / "metadata.git"
    subprocess.run([git, "init", "--quiet", "--bare", str(repository)], check=True)
    result = subprocess.run([
        git, "-c", "core.bare=false", f"--git-dir={repository}",
        f"--work-tree={runner.PROJECT_ROOT}", "ls-files", "--others", "--exclude-standard", "-z",
    ], cwd=runner.PROJECT_ROOT, capture_output=True, check=True)
    candidates = result.stdout.decode().split("\0")
    assert ".env.local" not in candidates
    assert ".env.example" in candidates
    assert "third_party/pinnacle/LICENSE" in candidates
    assert not any("__pycache__/" in path or path.startswith(("results/", ".pytest_cache/"))
                   for path in candidates)
