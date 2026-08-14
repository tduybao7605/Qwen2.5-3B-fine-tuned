#!/bin/bash
# Chạy pipeline bằng interpreter và thư viện của máy local.
#
#   PIPELINE_PYTHON=/path/to/venv/bin/python ./pipeline/run.sh main.py
#
# OLLAMA_LIB_DIR chỉ cần khi Ollama được cài vào $HOME thay vì đường dẫn hệ
# thống — trỏ tới thư mục chứa các lib CUDA của nó.
set -euo pipefail

PYTHON="${PIPELINE_PYTHON:-python3}"

if [ -n "${OLLAMA_LIB_DIR:-}" ]; then
  export LD_LIBRARY_PATH="$OLLAMA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

exec "$PYTHON" "$@"
