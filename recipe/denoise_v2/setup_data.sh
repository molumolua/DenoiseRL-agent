#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-"${SCRIPT_DIR}/../alfworld_denoise/local_data"}

exec bash "${SCRIPT_DIR}/../alfworld_denoise/setup_data.sh" "$@"
