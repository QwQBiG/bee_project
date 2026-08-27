#!/bin/sh

set -u

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT" || exit 1
SYSTEM=$(uname -s)

# Finder starts scripts with a minimal PATH. Include the standard Homebrew paths.
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH
PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_ENABLE_MPS_FALLBACK

HOST="127.0.0.1"
PORT="8000"
APP_URL="http://$HOST:$PORT"
HEALTH_URL="$APP_URL/api/health"
LOG_DIR="$PROJECT_ROOT/logs"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_ERROR_LOG="$LOG_DIR/server-error.log"
SERVER_PID=""

say() {
  printf '%s\n' "$1"
}

open_browser() {
  if [ "$SYSTEM" = "Darwin" ]; then
    open "$APP_URL" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$APP_URL" >/dev/null 2>&1 || true
  fi
}

bee_is_healthy() {
  command -v curl >/dev/null 2>&1 || return 1
  response=$(curl --silent --fail --max-time 2 "$HEALTH_URL" 2>/dev/null) || return 1
  printf '%s' "$response" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' || return 1
  printf '%s' "$response" | grep -Eq '"preview"[[:space:]]*:[[:space:]]*"vp8-webm"'
}

port_is_busy() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

python_is_compatible() {
  candidate=$1
  [ -x "$candidate" ] || return 1
  BEE_HOST_ARCH=$(uname -m) "$candidate" -c '
import os
import platform
import struct
import sys

version_ok = (3, 10) <= sys.version_info[:2] < (3, 14)
bits_ok = struct.calcsize("P") == 8
machine = platform.machine().lower()
host = os.environ.get("BEE_HOST_ARCH", "").lower()
arch_ok = host not in {"arm64", "aarch64"} or machine in {"arm64", "aarch64"}
raise SystemExit(0 if version_ok and bits_ok and arch_ok else 1)
' >/dev/null 2>&1
}

find_python() {
  for name in python3.12 python3.13 python3.11 python3.10 python3; do
    candidate=$(command -v "$name" 2>/dev/null || true)
    if [ -n "$candidate" ] && python_is_compatible "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  for candidate in \
    /opt/homebrew/opt/python@3.12/bin/python3.12 \
    /usr/local/opt/python@3.12/bin/python3.12; do
    if python_is_compatible "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

prepare_python() {
  if [ "$SYSTEM" = "Darwin" ]; then
    primary_venv="$PROJECT_ROOT/.venv-macos"
    fallback_venv="$PROJECT_ROOT/.venv-macos-py312"
  else
    primary_venv="$PROJECT_ROOT/.venv"
    fallback_venv="$PROJECT_ROOT/.venv-py312"
  fi

  for venv_dir in "$primary_venv" "$fallback_venv"; do
    if python_is_compatible "$venv_dir/bin/python"; then
      printf '%s\n' "$venv_dir/bin/python"
      return 0
    fi
  done

  base_python=$(find_python || true)
  if [ -z "$base_python" ] && [ "$SYSTEM" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    say "[Bee Vision] 未找到兼容的 Python，正在通过 Homebrew 安装 Python 3.12..." >&2
    if brew install python@3.12 >&2; then
      base_python=$(find_python || true)
    fi
  fi

  if [ -z "$base_python" ]; then
    say "[Bee Vision] 未找到兼容的 64 位 Python（需要 3.10-3.13，推荐 3.12）。" >&2
    if [ "$SYSTEM" = "Darwin" ]; then
      say "[Bee Vision] 请安装 Python 3.12 后重新双击 start_web.command。" >&2
      open "https://www.python.org/downloads/macos/" >/dev/null 2>&1 || true
    fi
    return 1
  fi

  if [ ! -e "$primary_venv" ]; then
    target_venv="$primary_venv"
  elif [ ! -e "$fallback_venv" ]; then
    target_venv="$fallback_venv"
  else
    say "[Bee Vision] 已有的虚拟环境无效，请移走 $primary_venv 和 $fallback_venv 后重试。" >&2
    return 1
  fi

  say "[Bee Vision] 正在使用 $base_python 创建运行环境..." >&2
  "$base_python" -m venv "$target_venv" || return 1
  python_is_compatible "$target_venv/bin/python" || return 1
  printf '%s\n' "$target_venv/bin/python"
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}

say "[1/5] 检查 Bee Vision 服务和端口..."
if bee_is_healthy; then
  say "[Bee Vision] 服务已经运行，正在打开浏览器。"
  open_browser
  exit 0
fi
if port_is_busy; then
  say "[Bee Vision] 端口 $PORT 已被其他程序占用，无法启动。"
  say "[Bee Vision] 请关闭占用该端口的程序后重试。"
  exit 1
fi

say "[2/5] 查找或创建 Mac Python 环境..."
BEE_PYTHON=$(prepare_python) || exit 1
say "[Bee Vision] 使用 Python：$BEE_PYTHON"

say "[3/5] 安装或校验依赖，并检测 MPS/CPU..."
"$BEE_PYTHON" tools/bootstrap_runtime.py || exit 1

say "[4/5] 启动 Bee Vision..."
mkdir -p "$LOG_DIR"
: >"$SERVER_LOG"
: >"$SERVER_ERROR_LOG"
"$BEE_PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" \
  >"$SERVER_LOG" 2>"$SERVER_ERROR_LOG" &
SERVER_PID=$!
trap cleanup HUP INT TERM EXIT

attempt=0
while [ "$attempt" -lt 120 ]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    say "[Bee Vision] 服务启动失败，请检查：$SERVER_ERROR_LOG"
    tail -n 30 "$SERVER_ERROR_LOG" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    exit 1
  fi
  if bee_is_healthy; then
    say "[5/5] Bee Vision 已就绪：$APP_URL"
    open_browser
    break
  fi
  attempt=$((attempt + 1))
  sleep 1
done

if ! bee_is_healthy; then
  say "[Bee Vision] 等待服务就绪超时，请检查：$SERVER_ERROR_LOG"
  exit 1
fi

say "[Bee Vision] 可以在网页中点击“关闭程序”，或在此窗口按 Ctrl+C。"
wait "$SERVER_PID"
status=$?
SERVER_PID=""
trap - HUP INT TERM EXIT

if [ "$status" -eq 0 ] || [ "$status" -eq 130 ]; then
  say "[Bee Vision] 服务已停止。"
  exit 0
fi
say "[Bee Vision] 服务异常退出，请检查：$SERVER_ERROR_LOG"
exit "$status"
