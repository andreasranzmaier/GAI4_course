#!/usr/bin/env bash
set -euo pipefail
# Ensure that the vscode user owns cache and virtualenv directories.
sudo mkdir -p /home/vscode/.cache/pip /home/vscode/.cache/uv /home/vscode/.cache/Microsoft

sudo chown -R vscode:vscode /home/vscode/.cache
[ -d /home/vscode/.venv ] && sudo chown -R vscode:vscode /home/vscode/.venv || true
