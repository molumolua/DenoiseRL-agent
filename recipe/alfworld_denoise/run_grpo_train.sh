#!/bin/bash

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/params.sh"

prepare_alfworld_data

python3 -m verl.trainer.main_ppo \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=False" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-grpo_qwen2.5_1.5b_unified}" \
  "$@"
