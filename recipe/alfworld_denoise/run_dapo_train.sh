#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=offline
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-dapo_qwen2.5_1.5b_unified}
source "${SCRIPT_DIR}/params.sh"

MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}

prepare_alfworld_data

python3 -m verl.trainer.main_ppo \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=True" \
  "algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}" \
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}" \
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-dapo_qwen2.5_1.5b_unified}" \
  "$@"
