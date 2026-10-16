#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IMG="${1:?usage: predict.sh <image>}"
exec "$PYTHON" -u -m ctu.predict --config "${2:-configs/default.yaml}" --checkpoint "${3:-outputs/checkpoints/best.pt}" --image "$IMG" --device cuda
