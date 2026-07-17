#!/usr/bin/env bash
# Compatibility wrapper. The deployment launcher is implemented in Python so
# category planning, manifests, resume state, and subprocess handling are testable.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_batch_persistent.py" "$@"
