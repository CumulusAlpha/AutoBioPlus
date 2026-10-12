#!/usr/bin/env bash
set -euo pipefail

uv sync

# There is a dependency conflict between full LeRobot and this OpenPI stack.
# The exporter and loader only need the LeRobot dataset interface, so install
# the API-compatible LeRobot release without pulling its dependency set into
# the OpenPI environment.
uv pip install "lerobot==0.4.4" --no-deps
