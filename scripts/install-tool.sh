#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install-tool.sh [options]

Install or refresh the user-facing meeting-asr uv tool from this checkout.

Options:
  --python VALUE          Python interpreter or version for uv tool install. Default: 3.14
  --editable             Install this checkout in editable mode.
  --no-local-voiceprint  Do not install the local SpeechBrain voiceprint extra.
  --print-only           Print the install plan without executing it.
  --check                Inspect the current meeting-asr executable and exit.
  -h, --help             Show this help.
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

quote_command() {
  if [[ $# -eq 0 ]]; then
    printf '\n'
    return
  fi
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

inspect_environment() {
  echo "Meeting-ASR install environment"
  echo "uv: $(command -v uv || echo '<missing>')"
  if command -v uv >/dev/null 2>&1; then
    echo "uv version: $(uv --version)"
    echo "uv python 3.14: $(uv python find 3.14 2>/dev/null || echo '<not found>')"
  fi
  echo "pyenv: $(command -v pyenv || echo '<missing>')"
  if command -v pyenv >/dev/null 2>&1; then
    echo "pyenv version: $(pyenv --version)"
    echo "pyenv active: $(pyenv version)"
    echo "pyenv python3.14: $(pyenv which python3.14 2>/dev/null || echo '<not found>')"
  fi
}

inspect_install() {
  local executable
  executable="$(command -v meeting-asr || true)"
  if [[ -z "$executable" ]]; then
    echo "meeting-asr is not on PATH." >&2
    return 1
  fi

  local python_path
  python_path="$(head -n 1 "$executable" | sed 's/^#!//')"
  if [[ -z "$python_path" || ! -x "$python_path" ]]; then
    echo "Cannot read executable Python from $executable" >&2
    return 1
  fi

  echo "Meeting-ASR install status"
  echo "Executable: $executable"
  echo "Python: $python_path"
  "$python_path" - <<'PY'
from importlib.metadata import distribution
import json
import sys

dist = distribution("meeting-asr")
direct_url = dist.read_text("direct_url.json")
source_url = json.loads(direct_url).get("url") if direct_url else "<unknown>"
print("Python version:", ".".join(str(part) for part in sys.version_info[:3]))
print("Package:", dist.locate_file(""))
print("Source:", source_url)
PY
}

warn_path_pollution() {
  case ":$PATH:" in
    *":$HOME/.local/share/uv/tools/meeting-asr/bin:"*)
      cat >&2 <<'EOF'
Warning: PATH contains ~/.local/share/uv/tools/meeting-asr/bin.
This leaks the tool's private python/python3 into your shell. Keep ~/.local/bin on PATH instead.
Regenerate completion after updating meeting-asr:
  meeting-asr completion install zsh
EOF
      ;;
  esac
}

python_value="3.14"
editable=0
local_voiceprint=1
print_only=0
check_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_value="${2:-}"
      if [[ -z "$python_value" ]]; then
        echo "--python requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --editable)
      editable=1
      shift
      ;;
    --no-local-voiceprint)
      local_voiceprint=0
      shift
      ;;
    --print-only)
      print_only=1
      shift
      ;;
    --check)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$check_only" -eq 1 ]]; then
  inspect_environment
  echo
  inspect_install
  warn_path_pollution
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

source_dir="$(repo_root)"
package="."
if [[ "$local_voiceprint" -eq 1 ]]; then
  package=".[local-voiceprint]"
fi

command=(uv tool install --python "$python_value" --force --reinstall --refresh)
if [[ "$editable" -eq 1 ]]; then
  command+=(--editable)
fi
command+=("$package")

echo "Meeting-ASR install plan"
echo "Source: $source_dir"
echo "Mode: $([[ "$editable" -eq 1 ]] && echo editable || echo wheel)"
echo "Local voiceprint: $([[ "$local_voiceprint" -eq 1 ]] && echo yes || echo no)"
echo "Command:"
echo "  cd $(printf '%q' "$source_dir")"
printf '  '
quote_command "${command[@]}"

if [[ "$print_only" -eq 1 ]]; then
  exit 0
fi

(
  cd "$source_dir"
  "${command[@]}"
)
inspect_install
warn_path_pollution
