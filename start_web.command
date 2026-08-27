#!/bin/sh

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT" || exit 1

/bin/sh ./start_web.sh
status=$?

if [ "$status" -ne 0 ]; then
  printf '\n启动失败。按回车键关闭此窗口...'
  read -r _unused
fi

exit "$status"
