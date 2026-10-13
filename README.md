# Running PINNMorph
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


