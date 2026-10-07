# Running PINNMorph

Run these commands from the project root in your Python environment.

The Python package is `morph`. Benchmark configurations live in `configs/`.
The bundled PINNacle runtime and reference data in `third_party/` are required.

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

## Server integration check: GPUs 1, 2 and 3

On a server where GPU 0 must remain unused, this optional script runs all 13 PDEs with seed 0 and the `full` variant. It runs at most three tasks on each of GPUs 1, 2 and 3, for nine tasks in total. It keeps 20,000 steps and the paper method settings so the run can exercise adaptation, delayed feedback and the optimizer phases. This is a single-seed integration check, not a five-run reproduction.

First validate without training or provider requests:

```bash
bash scripts/run_server_check.sh --dry-run --results-root results/server_check_seed0
```

Then run in your configured Python environment:

```bash
bash scripts/run_server_check.sh --results-root results/server_check_seed0
```

For a background run that survives disconnecting from SSH:

```bash
nohup bash scripts/run_server_check.sh --results-root results/server_check_seed0 > server_check.out 2>&1 &
```

Choose either the foreground or background command. Follow progress with `tail -f server_check.out` for the background run. Individual logs are in `full/_launcher_logs/`; the final summary is `comparison_summary.md`. Each `run_summary.json` records failures, solution errors, actual optimizer steps, run time, peak memory, interventions and LLM calls. Do not interpret simultaneous task times as isolated performance measurements.

Rerun the same command and results root to resume. Automatic clean retries are disabled so failed attempts remain available for diagnosis. After changing code or configuration, select a new results root. Inside a child process, `cuda:0` refers to its assigned physical GPU 1, 2 or 3.

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
