#!/usr/bin/env bash
set -euo pipefail

TORCH_BACKEND="auto"
PROFILE="dev"
VENV=".venv"
PYTHON_CMD=""
SKIP_TORCH=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [options]

Options:
  --torch-backend auto|cpu|cu118|cu126|cu128
      PyTorch wheel backend. Default: auto.
      auto uses nvidia-smi to choose the highest supported CUDA wheel and
      falls back to CPU when CUDA cannot be detected.

  --profile base|deforming_plate|dev|none
      Dependency profile to install from requirements/. Default: dev.

  --venv PATH
      Virtual environment path. Default: .venv.

  --python COMMAND
      Python executable used to create the venv. If omitted, the script tries
      python3.11, python3.10, python3, then python.

  --skip-torch
      Do not install PyTorch. Useful when torch is already installed manually.

  -h, --help
      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch-backend)
      TORCH_BACKEND="${2:?missing value for --torch-backend}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift 2
      ;;
    --venv)
      VENV="${2:?missing value for --venv}"
      shift 2
      ;;
    --python)
      PYTHON_CMD="${2:?missing value for --python}"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$TORCH_BACKEND" in
  auto|cpu|cu118|cu126|cu128) ;;
  *)
    echo "Invalid --torch-backend: $TORCH_BACKEND" >&2
    exit 2
    ;;
esac

case "$PROFILE" in
  base|deforming_plate|dev|none) ;;
  *)
    echo "Invalid --profile: $PROFILE" >&2
    exit 2
    ;;
esac

find_python() {
  if [[ -n "$PYTHON_CMD" ]]; then
    command -v "$PYTHON_CMD"
    return
  fi

  for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "Python 3.10+ was not found on PATH. Install Python first or pass --python." >&2
  exit 1
}

venv_python() {
  if [[ -x "$VENV/bin/python" ]]; then
    printf '%s\n' "$VENV/bin/python"
  elif [[ -x "$VENV/Scripts/python.exe" ]]; then
    printf '%s\n' "$VENV/Scripts/python.exe"
  else
    echo "Could not find Python inside virtual environment: $VENV" >&2
    exit 1
  fi
}

activation_hint() {
  if [[ -f "$VENV/bin/activate" ]]; then
    printf 'source %s/bin/activate\n' "$VENV"
  elif [[ -f "$VENV/Scripts/activate" ]]; then
    printf 'source %s/Scripts/activate\n' "$VENV"
  else
    printf 'activate the virtual environment under %s\n' "$VENV"
  fi
}

version_ge() {
  local major="$1"
  local minor="$2"
  local req_major="$3"
  local req_minor="$4"
  local value=$((major * 100 + minor))
  local required=$((req_major * 100 + req_minor))
  [[ "$value" -ge "$required" ]]
}

detect_torch_backend() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  local smi_output
  smi_output="$(nvidia-smi 2>/dev/null || true)"
  local cuda_version
  cuda_version="$(
    printf '%s\n' "$smi_output" |
      sed -nE 's/.*CUDA Version: ([0-9]+)\.([0-9]+).*/\1 \2/p' |
      head -n 1
  )"

  if [[ -z "$cuda_version" ]]; then
    echo "Could not parse CUDA Version from nvidia-smi; using CPU wheels." >&2
    echo "Override with --torch-backend cu118|cu126|cu128 if needed." >&2
    echo "cpu"
    return
  fi

  local major minor
  read -r major minor <<<"$cuda_version"

  if version_ge "$major" "$minor" 12 8; then
    echo "cu128"
  elif version_ge "$major" "$minor" 12 6; then
    echo "cu126"
  elif version_ge "$major" "$minor" 11 8; then
    echo "cu118"
  else
    echo "Detected CUDA $major.$minor, but no compatible wheel is configured; using CPU." >&2
    echo "cpu"
  fi
}

install_torch() {
  local python_bin="$1"
  local backend="$2"

  if [[ "$backend" == "auto" ]]; then
    backend="$(detect_torch_backend)"
    echo "Auto-selected PyTorch backend: $backend"
  fi

  if [[ "$backend" == "cpu" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      "$python_bin" -m pip install torch torchvision torchaudio
    else
      "$python_bin" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    "$python_bin" -m pip install torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/$backend"
  fi
}

PYTHON_LAUNCHER="$(find_python)"
echo "Using Python launcher: $PYTHON_LAUNCHER"

"$PYTHON_LAUNCHER" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print("Python", sys.version.split()[0])
PY

if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment at $VENV"
  "$PYTHON_LAUNCHER" -m venv "$VENV"
fi

PYTHON_BIN="$(venv_python)"
echo "Using virtual environment Python: $PYTHON_BIN"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" -eq 0 ]]; then
  install_torch "$PYTHON_BIN" "$TORCH_BACKEND"
else
  echo "Skipping PyTorch installation"
fi

if [[ "$PROFILE" != "none" ]]; then
  REQUIREMENTS="requirements/$PROFILE.txt"
  echo "Installing dependencies from $REQUIREMENTS"
  "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"
fi

echo "Installing ValGraphNet in editable mode"
"$PYTHON_BIN" -m pip install -e . --no-deps

echo "Environment check"
"$PYTHON_BIN" - <<'PY'
import importlib

torch = importlib.import_module("torch")
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available(), "cuda=", torch.version.cuda)

try:
    pyg = importlib.import_module("torch_geometric")
    print("torch_geometric", getattr(pyg, "__version__", "unknown"))
except Exception as exc:
    print("torch_geometric import failed:", exc)

try:
    physicsnemo = importlib.import_module("physicsnemo")
    print("physicsnemo", getattr(physicsnemo, "__version__", "unknown"))
except Exception as exc:
    print("physicsnemo import failed:", exc)
PY

echo
echo "Done. Activate with:"
echo "  $(activation_hint)"
