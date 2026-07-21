#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Final policy evaluation starts every task from its clean initial state. The
# online denoiser used during training is intentionally not loaded here.
export DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-denoise_grpo_complete_eval}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-denoise_grpo_complete_eval}
export EVAL_PROJECT_NAME=${EVAL_PROJECT_NAME:-verl_agent_alfworld_denoise_eval}

exec bash "${SCRIPT_DIR}/run_grpo_eval.sh" "$@"
