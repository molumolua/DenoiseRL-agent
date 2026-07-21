#!/usr/bin/env bash
set -euo pipefail

# Agent analogue of the mathematical DenoiseRL v2 recipe: an ordered gamefile
# pool, one active batch, and slope-driven retirement/replacement.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MAIN_ROLLOUT_N=0
export SUB_ROLLOUT_K=16
# The mathematical v2 keeps only the first wrong solution and cycles it over
# all 16 slots. The agent analogue generates one denoiser trajectory per task.
export DENOISE_PREFIX_CANDIDATES_PER_GROUP=1
export DENOISE_PREFIX_STRATEGY=full_then_ratio
export DENOISE_ADVANTAGE_GROUPING=mixed

V2_INITIAL_RHO=${V2_INITIAL_RHO:-0.0}
V2_MIN_RHO=${V2_MIN_RHO:-0.0}
V2_MAX_RHO=${V2_MAX_RHO:-0.5}
V2_TARGET_ACCURACY=${V2_TARGET_ACCURACY:-0.75}
V2_ALPHA=${V2_ALPHA:-0.2}
V2_HISTORY_WINDOW=${V2_HISTORY_WINDOW:-5}
V2_MIN_HISTORY=${V2_MIN_HISTORY:-2}
V2_SLOPE_THRESHOLD=${V2_SLOPE_THRESHOLD:-0.005}

# The legacy scalar is ignored while v2 is enabled, but keeping it aligned with
# the initial rho makes the fully resolved Hydra config easier to inspect.
export DENOISE_PREFIX_RATIO=${V2_INITIAL_RHO}
export PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld_denoise_v2}
RUN_TAG="rho${V2_INITIAL_RHO}-${V2_MAX_RHO}_target${V2_TARGET_ACCURACY}_alpha${V2_ALPHA}_window${V2_HISTORY_WINDOW}_slope${V2_SLOPE_THRESHOLD}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-denoise_grpo_v2_qwen2.5_7b_1.5b_bsz${TRAIN_BATCH_SIZE:-16}_k16_${RUN_TAG}}
export DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}

exec bash "${SCRIPT_DIR}/run_denoise_grpo_train_base.sh" \
  "data.shuffle=False" \
  "+data.dataloader_num_workers=0" \
  "algorithm.filter_groups.enable=False" \
  "algorithm.use_kl_in_reward=False" \
  "actor_rollout_ref.actor.use_kl_loss=False" \
  "actor_rollout_ref.actor.kl_loss_coef=0.0" \
  "env.denoise.v2.enabled=True" \
  "env.denoise.v2.initial_rho=${V2_INITIAL_RHO}" \
  "env.denoise.v2.min_rho=${V2_MIN_RHO}" \
  "env.denoise.v2.max_rho=${V2_MAX_RHO}" \
  "env.denoise.v2.target_accuracy=${V2_TARGET_ACCURACY}" \
  "env.denoise.v2.alpha=${V2_ALPHA}" \
  "env.denoise.v2.history_window=${V2_HISTORY_WINDOW}" \
  "env.denoise.v2.min_history=${V2_MIN_HISTORY}" \
  "env.denoise.v2.slope_threshold=${V2_SLOPE_THRESHOLD}" \
  "$@"
