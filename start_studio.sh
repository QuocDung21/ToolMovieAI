#!/bin/bash
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "====================================================="
echo "🚀 Đang khởi động Bilibili AI Sub & Dubbing Studio..."
echo "🌐 Truy cập giao diện Web tại: http://127.0.0.1:8080"
echo "====================================================="
if [ -d "$PROJECT_DIR/pyvideotrans" ] && command -v uv >/dev/null 2>&1; then
  cd "$PROJECT_DIR/pyvideotrans"
  exec uv run --offline python "$PROJECT_DIR/web_app.py"
fi
exec python3 "$PROJECT_DIR/web_app.py"
