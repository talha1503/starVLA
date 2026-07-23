#!/usr/bin/env bash

detect_torch_profile() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  local compute_caps
  if ! compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || [[ -z "${compute_caps//[[:space:]]/}" ]]; then
    echo "[torch-profile] unable to query GPU compute capability" >&2
    return 1
  fi

  if echo "${compute_caps}" | awk -F. '$1 >= 10 {found=1} END {exit found ? 0 : 1}'; then
    local driver_version driver_major
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    driver_major="${driver_version%%.*}"
    if ((driver_major >= 580)); then
      echo "cu130"
    else
      echo "cu128"
    fi
  else
    echo "cu126"
  fi
}

resolve_torch_profile() {
  case "$1" in
    auto) detect_torch_profile ;;
    cpu|cu126|cu128|cu130) echo "$1" ;;
    *)
      echo "[torch-profile] unknown profile '$1'; expected auto|cpu|cu126|cu128|cu130" >&2
      return 1
      ;;
  esac
}
