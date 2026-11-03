#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: run_matrix.sh hunyuan|cogvideo output-directory [benchmark arguments...]}
output=${2:?Output directory is required}
shift 2
python examples/chunk_dit/benchmark.py --model "$model" --output "$output/dense" "$@"
for chunk in 1024 2048 4096 8192; do
    python examples/chunk_dit/benchmark.py --model "$model" --chunk-size "$chunk" \
        --reference "$output/dense" --output "$output/$chunk" "$@"
done
