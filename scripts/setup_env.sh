#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash /home/public/tbiom/.cursor/skills/paper-2-model/templates/setup_env.sh
