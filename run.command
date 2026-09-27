#!/bin/bash
# macOS 双击启动。终端里执行 ./run.command
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "需要 Python 3.10 或更高版本。"
  echo "如果没有：https://www.python.org/downloads/macos/"
  read -r -p "按回车关闭…" _; exit 1
fi

PY=python3
[ -d ".venv" ] && PY=".venv/bin/python"

"$PY" -c "import sounddevice, webrtcvad, PySide6" >/dev/null 2>&1 || {
  echo "[1/2] 安装依赖…"
  [ -d ".venv" ] || python3 -m venv .venv
  .venv/bin/pip install --disable-pip-version-check -r requirements.txt || exit 1
}

if [ ! -f "models/ggml-small.bin" ]; then
  echo "[2/2] 首次运行，下载运行时和模型（约 500MB）…"
  "$PY" tools/fetch_assets.py || exit 1
fi

echo "启动中…"
exec "$PY" app.py
