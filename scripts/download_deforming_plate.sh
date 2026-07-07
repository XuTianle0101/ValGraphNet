#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-raw_dataset/deforming_plate/deforming_plate}"
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/deforming_plate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"

download_file() {
  local name="$1"
  local url="$BASE_URL/$name"
  local out="$OUT_DIR/$name"

  if [[ -s "$out" ]]; then
    echo "Found $out"
    return
  fi

  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Neither curl nor wget was found. Install one of them and retry." >&2
    exit 1
  fi
}

download_file meta.json
download_file train.tfrecord
download_file valid.tfrecord
download_file test.tfrecord

echo
echo "Deforming plate data is ready under:"
echo "  $OUT_DIR"
