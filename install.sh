#!/usr/bin/env bash
# One-shot setup: creates .venv, installs dependencies, verifies the install.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Python 3.9+ is required (found: $("$PYTHON" --version 2>&1))" >&2
  exit 1
fi

echo "==> Creating virtual environment in .venv"
"$PYTHON" -m venv .venv

echo "==> Installing dependencies"
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -e ".[dev]"

echo "==> Verifying"
./.venv/bin/transcribe --check

cat <<'EOF'

Done. Use it with:

  ./.venv/bin/transcribe your-recording.m4a

or activate the environment first:

  source .venv/bin/activate
  transcribe your-recording.m4a

The first run downloads the model (~1.5 GB for the default large-v3-turbo);
every run after that is fully offline.
EOF
