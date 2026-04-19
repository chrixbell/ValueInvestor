#!/usr/bin/env bash
# ───────────────────────────────────────────────────
#  ValueInvestor — Autonomous Scorer Improvement Loop
#  Inspired by karpathy/autoresearch
# ───────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Load .env
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Default flags (override via env or CLI args)
FLAGS="${IMPROVE_SCORER_FLAGS:-}"

echo "╔══════════════════════════════════════════════╗"
echo "║  ValueInvestor — Scorer Improvement Agent    ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Press Ctrl-C to stop gracefully             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Pass through any script arguments, plus default flags
exec valueinvestor improve-scorer $FLAGS "$@"
