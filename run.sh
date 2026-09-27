#!/bin/bash
# Linux 启动脚本
cd "$(dirname "$0")"

echo "== SoundTrace 环境自检 =="
python3 tools/selfcheck.py
echo

read -r -p "按回车启动（上面有 [FAIL] 的话建议先修好）…" _

PY=python3
[ -d ".venv" ] && PY=".venv/bin/python"
exec "$PY" app.py
