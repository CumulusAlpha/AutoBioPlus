#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_DIR="$ROOT/assets/scenes"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1cd3pnt8sibf_Ob5gIT24vwH8w2P9XlHv"
SCENE_FILES=(lab1.ply lab2.ply lab3.ply lab4.ply lab5.ply lab6.ply table.ply)

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to download the Gaussian scenes." >&2
  exit 2
fi

mkdir -p "$SCENE_DIR"

echo "Downloading Gaussian scenes to: $SCENE_DIR"
uvx gdown --folder --continue -O "$SCENE_DIR" "$DRIVE_FOLDER_URL"

for scene_file in "${SCENE_FILES[@]}"; do
  scene_path="$SCENE_DIR/$scene_file"
  if [[ ! -s "$scene_path" ]]; then
    echo "Missing or empty Gaussian scene: $scene_path" >&2
    exit 1
  fi
done

echo "Gaussian scenes are ready in: $SCENE_DIR"
