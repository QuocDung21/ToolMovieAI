#!/bin/bash
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "====================================================="
echo "🚀 Đang khởi động Bilibili AI Sub & Dubbing Studio..."
echo "🌐 Truy cập giao diện Web tại: http://127.0.0.1:8080"
echo "☕ Đã bật chế độ chống ngủ máy (caffeinate) khi treo tác vụ."
echo "====================================================="

CAFFEINATE=""
if command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATE="caffeinate -i"
fi

if [ -d "$PROJECT_DIR/pyvideotrans" ] && command -v uv >/dev/null 2>&1; then
  cd "$PROJECT_DIR/pyvideotrans"
  exec $CAFFEINATE uv run --offline python "$PROJECT_DIR/web_app.py"
fi
exec $CAFFEINATE python3 "$PROJECT_DIR/web_app.py"
