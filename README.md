# Running PINNMorph

Run these commands from the project root in your Python environment.

The Python package is `morph`. Benchmark configurations live in `configs/`.
The bundled PINNacle runtime and reference data in `third_party/` are required.

PDE collocation uses a fixed point cloud. The initial cloud defaults to 70% local high-residual points and 30% independent full-domain points. At step 200, only Full makes one mandatory generation-LLM request for a sampling plan and applies it directly, without candidate scoring, a plateau gate or utility approval. The LLM chooses the local/global allocation and total point count freely; neither the 70/30 default nor configured point-count limit constrains an explicit LLM plan. Coordinates stay fixed between explicit sampling or curriculum actions. Sampling creates no post-action validation, delayed utility trial, observation window or rollback. A provider/format/runtime failure is reported instead of silently converting this mandatory action to KEEP.

Ordinary online sampling returns KEEP for an already active policy: identical region, local/global allocation and candidate multiplier do not regenerate points under a different ID, strength, selector, wording or cooldown. Changing these effective parameters replaces the current policy; an earlier policy can be selected again after switching away. The mandatory Full-only step-200 action bypasses this ordinary gate once. Both the active policy and mandatory-action status survive checkpoints. Whitelisted means applied without utility validation, never proven benefit.

Committed Loss Reweighting, Adaptive Sampling, Gradient Adjustment, Residual/Fourier Branch, and Localized Expert actions are exempt from feedback-driven rollback, weight restoration, and branch retirement. Loss, gradient, and architecture actions still record delayed feedback for policy learning; sampling retains its feedback exemption. Other action families keep their existing rollback behavior.

Online component-gradient balancing uses the current weighted gradient contributions and enforces the LLM's `minimum_weight` and `maximum_weight` on every dynamic update. These bounds survive checkpoints, and the resulting weights stay frozen throughout L-BFGS.

Sampling prompts prioritize full-domain coverage and require spatial evidence for increasing the local fraction. IC dominance or gradient conflict alone does not demonstrate localized PDE error, and PDE-cloud changes leave the IC/BC point pools unchanged. This advice is shared by generation, scoring, tie-breaking and the mandatory step-200 request; the LLM still chooses the allocation freely.

Physics-checkpoint selection first requires every registered initial-condition component on the independent audit set to have normalized residual MSE at most `0.1` of its calibrated warmup baseline. The limit is configurable under `best_checkpoint.initial_condition_guard.maximum_normalized_error` and is matched across variants. Eligible candidates are ranked by the existing physics audit score. PDEs without initial conditions keep the original ranking; if no candidate qualifies, the final model is the last optimizer state and the report records the fallback. Selection uses no reference-solution MSE or L2 metrics and does not roll back Morph actions.

Poisson Classic uses soft boundary penalties in every variant: the outer square target remains 1, the circular-hole targets remain 0, and the configured boundary loss weight is 10. The default profile applies no hard output lifting. KS keeps its exact initial-condition transform; its output wrapper preserves dynamic architecture snapshots when saving and restoring checkpoints.

Online LLM requests have no per-run call-count cap by default (`online_morphing.policy_set.evolution.maximum_llm_calls_per_run: 0`). Policy generation, scoring, retries, and tie-breaks continue beyond 38 requests when the existing online adaptation rules permit them. Request counts still survive checkpoint resume. A positive value explicitly enables an optional shared cap; only then does generation reserve calls for independent scoring and an exhausted cap produce KEEP decisions. Connectivity preflight requests are counted separately. Prompts use compact JSON, one capability contract, at most three recent causal records, complete assigned parents, and summaries of other policies; scoring includes only the capabilities actually proposed.

## 1. Install dependencies

Use Python 3.11. Install a PyTorch build compatible with your CUDA environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## 2. Configure LLM access

Copy `.env.example` to `.env.local` on first setup. Keep an existing `.env.local`. Fill in both `DUAL_LLM_1_*` and `DUAL_LLM_2_*` providers, model IDs, API keys, and endpoint URLs. Set `DUAL_LLM_1_MODEL` for policy generation and `DUAL_LLM_2_MODEL` for candidate scoring to model IDs supported by your endpoints, including provider namespaces or deployment aliases. Model names are configurable without a whitelist. Existing shell variables take precedence over `.env.local`. The `full` variant requires both channels; `no_morph_matched` runs without LLM access. Preflight logs and run summaries record the models used.

## 3. Choose an available GPU

Find the GPU index available on your machine:

```bash
nvidia-smi
```

Set `GPU_ID` to that index. **The value `0` below is an example; change it to the GPU you can use.**

Linux / Bash:

```bash
GPU_ID=0
```

PowerShell:

```powershell
$GPU_ID = 0
```

The commands below use `"$GPU_ID"` in either shell. The runner requires an explicit `--gpus` argument and defaults to one worker per selected GPU.

## 4. Validate before training

Validate one PDE's configuration and generated command without training or LLM requests:

```bash
python scripts/run_benchmarks.py --pdes burgers_1d --seeds 0 --gpus "$GPU_ID" --dry-run --results-root results/check_burgers
```

Check the chosen GPU and LLM connectivity. This preflight makes real provider requests:

```bash
python scripts/run_benchmarks.py --pdes burgers_1d --seeds 0 --gpus "$GPU_ID" --preflight-only --results-root results/check_burgers
```

## 5. Run PDEs individually

Each command runs **one PDE with five seeds (0, 1, 2, 3, 4)** using the `full` variant. These are five independent runs with 20,000 optimizer steps each. Run the commands you need, one at a time. Each PDE has its own results root.

### 1. Burgers 1D

```bash
python scripts/run_benchmarks.py --pdes burgers_1d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/burgers_1d
```

### 2. Burgers 2D

```bash
python scripts/run_benchmarks.py --pdes burgers_2d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/burgers_2d
```

### 3. Gray–Scott 2D

```bash
python scripts/run_benchmarks.py --pdes gray_scott_2d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/gray_scott_2d
```

### 4. Heat 2D Complex Geometry

```bash
python scripts/run_benchmarks.py --pdes heat_2d_complex_geometry --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/heat_2d_complex_geometry
```

### 5. Heat 2D Multiscale

```bash
python scripts/run_benchmarks.py --pdes heat_2d_multiscale --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/heat_2d_multiscale
```

### 6. Heat 2D Varying-Coefficient

```bash
python scripts/run_benchmarks.py --pdes heat_2d_varying_coefficient --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/heat_2d_varying_coefficient
```

### 7. Kuramoto–Sivashinsky 1D

```bash
python scripts/run_benchmarks.py --pdes kuramoto_sivashinsky_1d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/kuramoto_sivashinsky_1d
```

### 8. Navier–Stokes 2D Backstep

```bash
python scripts/run_benchmarks.py --pdes navier_stokes_2d_backstep --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/navier_stokes_2d_backstep
```

### 9. Poisson 2D Classic

```bash
python scripts/run_benchmarks.py --pdes poisson_2d_classic --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/poisson_2d_classic
```

### 10. Poisson–Boltzmann 2D

```bash
python scripts/run_benchmarks.py --pdes poisson_boltzmann_2d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/poisson_boltzmann_2d
```

### 11. Poisson 2D Many-Area

```bash
python scripts/run_benchmarks.py --pdes poisson_2d_many_area --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/poisson_2d_many_area
```

### 12. Poisson 3D Complex Geometry

```bash
python scripts/run_benchmarks.py --pdes poisson_3d_complex_geometry --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/poisson_3d_complex_geometry
```

### 13. Poisson 5D

```bash
python scripts/run_benchmarks.py --pdes poisson_5d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --results-root results/poisson_5d
```

## Run one seed or the paper ablations

For a single-seed trial, use a separate results root. A one-seed result is not a five-run reproduction:

```bash
python scripts/run_benchmarks.py --pdes burgers_1d --seeds 0 --gpus "$GPU_ID" --results-root results/burgers_1d_single_seed
```

Run the paper's four variants for one PDE, with five seeds each:

```bash
python scripts/run_benchmarks.py --pdes burgers_1d --seeds 0 1 2 3 4 --gpus "$GPU_ID" --variants no_morph_matched random_selection random_evolution full --results-root results/burgers_1d_paper_ablations
```

The identifiers correspond to No Morph, Random Selection, Random Evolution, and PINNMorph. To run one variant, pass only that identifier after `--variants`. For another PDE, replace `burgers_1d` with its identifier from the individual commands above and choose a corresponding results root. The default seed list is `0 1 2 3 4`.

## Resume training

Rerun the same command to skip completed runs and resume interrupted tasks. After changing code, PDE selection, variants, or seeds, choose a new `--results-root`. Failed tasks receive one clean retry by default; add `--retries 0` to disable retries.

## View results

Open `comparison_summary.md` or `comparison_runs.csv` under the results root used by the command. Individual run folders contain `run_summary.json` and `training_report.json`.

Regenerate a summary without training or selecting a GPU:

```bash
python scripts/run_benchmarks.py --summary-only --results-root results/burgers_1d
```

Use `python scripts/run_benchmarks.py --help` to view available arguments.

## Run checks

```bash
python -m pytest tests -q
```

## License

PINNMorph's first-party code, scripts, configurations and documentation are
licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
Bundled third-party components retain their upstream licenses: PINNacle uses
MIT, and the bundled DeepXDE 1.8.1 runtime uses LGPL-2.1. See [NOTICE](NOTICE)
and the license files in the corresponding `third_party/` directories.
