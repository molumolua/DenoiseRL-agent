#!/bin/bash

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/params.sh"

PREFIX_POOL_PATH=${PREFIX_POOL_PATH:-""}
MAIN_ROLLOUT_N=${MAIN_ROLLOUT_N:-4}
SUB_ROLLOUT_K=${SUB_ROLLOUT_K:-4}
GROUP_SIZE=$((MAIN_ROLLOUT_N + SUB_ROLLOUT_K))
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}

if [ -z "$PREFIX_POOL_PATH" ]; then
  echo "Set PREFIX_POOL_PATH to a JSONL pool of small-model ALFWorld trajectory prefixes."
  exit 1
fi

prepare_alfworld_data

python3 -m verl.trainer.main_ppo \
  --config-path recipe/alfworld_denoise/config \
  --config-name alfworld_denoise_trainer \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=True" \
  "algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}" \
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}" \
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}" \
  "env.rollout.n=${GROUP_SIZE}" \
  "env.denoise.main_rollout_n=${MAIN_ROLLOUT_N}" \
  "env.denoise.sub_rollout_k=${SUB_ROLLOUT_K}" \
  "env.denoise.prefix_pool_path=${PREFIX_POOL_PATH}" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-denoise_dapo_qwen2.5_1.5b_unified}" \
  "$@"
