#!/usr/bin/env bash
# One seed across all PDEs; the documented integration server reserves GPU 0.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
PYTHON_BIN="${PINNMORPH_PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_benchmarks.py \
  --gpus 1 2 3 --per-gpu-concurrency 3 --seeds 0 --variants full \
  --retries 0 --results-root results/server_check_seed0 "$@"
