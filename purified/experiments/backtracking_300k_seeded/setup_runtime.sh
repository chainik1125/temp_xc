#!/usr/bin/env bash
set -euo pipefail

historical_commit="284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
runtime_parent="${1:?usage: setup_runtime.sh RUNTIME_PARENT CACHE_FILE}"
cache_file="${2:?usage: setup_runtime.sh RUNTIME_PARENT CACHE_FILE}"
repo_root="$(git rev-parse --show-toplevel)"

if [[ "$(git -C "$repo_root" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to continue: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -f "$cache_file" ]]; then
  echo "activation cache missing: $cache_file" >&2
  exit 2
fi

historical_root="$runtime_parent/purified"
if [[ -e "$historical_root" ]]; then
  if [[ ! -f "$historical_root/HISTORICAL_COMMIT" ]] ||
     [[ "$(<"$historical_root/HISTORICAL_COMMIT")" != "$historical_commit" ]]; then
    echo "refusing to reuse an unverified runtime: $historical_root" >&2
    exit 2
  fi
else
  mkdir -p "$runtime_parent"
  git -C "$repo_root" archive "$historical_commit" purified |
    tar -x -C "$runtime_parent"
  printf '%s\n' "$historical_commit" > "$historical_root/HISTORICAL_COMMIT"
fi

cache_dir="$historical_root/results/act_cache/fb2a74be884e512a"
mkdir -p "$cache_dir"
if [[ ! -e "$cache_dir/resid_post_L10.npy" ]]; then
  ln -s "$cache_file" "$cache_dir/resid_post_L10.npy"
fi
if [[ "$(readlink -f "$cache_dir/resid_post_L10.npy")" != "$(readlink -f "$cache_file")" ]]; then
  echo "runtime cache symlink points at the wrong file" >&2
  exit 2
fi
if [[ ! -f "$cache_dir/layer_specs.json" ]]; then
  printf '{\n  "d_model": 4096,\n  "layer": 10\n}\n' > "$cache_dir/layer_specs.json"
fi

echo "$historical_root"
