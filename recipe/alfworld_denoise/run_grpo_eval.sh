#!/bin/bash

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/params.sh"

if [ -z "${CKPT_DIR:-}" ]; then
  echo "Set CKPT_DIR to the checkpoint directory to evaluate."
  exit 1
fi

prepare_alfworld_data

python3 -m verl.trainer.main_ppo \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=False" \
  "trainer.logger=['console']" \
  "trainer.project_name=${EVAL_PROJECT_NAME:-verl_agent_alfworld_eval}" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-grpo_eval}" \
  "trainer.default_local_dir=${CKPT_DIR}" \
  "trainer.val_only=True" \
  "trainer.val_before_train=True" \
  "$@"
