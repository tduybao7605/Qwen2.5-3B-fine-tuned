#!/bin/bash
export LD_LIBRARY_PATH="/home/team3/.local/lib:/home/team3/.local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="/home/team3/.local/bin:$PATH"
PYTHON=/home/team3/Isaac-GR00T/.venv/bin/python
exec "$PYTHON" "$@"
