#!/usr/bin/env bash
# Full sweep: 6 models x 4 methods, then the summary table.
# Usage: scripts/run_all.sh [extra args for `quantarena matrix`]
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/run_all_$(date +%Y%m%d_%H%M%S).log"

uv run --extra eval --extra awq --extra gptq --extra fp8 --extra vllm \
    quantarena matrix --keep-going "$@" 2>&1 | tee "$log"

uv run quantarena report
