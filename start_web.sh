#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] Creating Python environment..."
  python3 -m venv .venv
fi

echo "[2/3] Installing or checking dependencies..."
.venv/bin/python tools/bootstrap_runtime.py

echo "[3/3] Starting Bee Vision at http://127.0.0.1:8000"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
