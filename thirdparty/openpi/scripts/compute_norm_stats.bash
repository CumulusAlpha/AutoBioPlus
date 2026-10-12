#!/usr/bin/env bash
set -euo pipefail

TASKS=(
  Aloha-Screw-Tighten
  Shadow-Centrifuge-Insert
  Shadow-Pipette-Insert
  Shadow-Pipette-Manipulate
  Shadow-Tube-Insert
  Shadow-Tube-Rotate
  Wuji-Centrifuge-Close
  Wuji-Tweezers-Manipulate
  XHand-Bunsen-Manipulate
  XHand-MixerPanel-Manipulate
  XHand-ThermalCycler-Close
  XHand-ThermalCycler-Open
  XHand-Tube-Insert
)

for task in "${TASKS[@]}"; do
  uv run python scripts/compute_norm_stats.py "$task" --batch-size=64 --num-workers=16
done
