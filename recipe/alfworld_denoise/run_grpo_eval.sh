#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=offline
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Final evaluation defaults: report both ALFWorld splits (seen + unseen).
# Override via env vars, e.g. EVAL_SPLIT=unseen bash run_grpo_eval.sh.
EVAL_SPLIT=${EVAL_SPLIT:-both}
DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-grpo_qwen2.5_7b_unified_eval}

source "${SCRIPT_DIR}/params.sh"

if [ -z "${CKPT_DIR:-}" ]; then
  echo "Set CKPT_DIR to the checkpoint directory to evaluate."
  exit 1
fi

prepare_alfworld_data

BASE_EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_eval}

echo "============================================================"
echo " Evaluating split(s)=${EVAL_SPLITS}"
echo " val_batch_size=${VAL_BATCH_SIZE}  val_n=${VAL_N}  temperature=${VAL_TEMPERATURE}"
echo "============================================================"

python3 -m verl.trainer.main_ppo \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=False" \
  "trainer.logger=['console']" \
  "trainer.project_name=${EVAL_PROJECT_NAME:-verl_agent_alfworld_eval}" \
  "trainer.experiment_name=${BASE_EXPERIMENT_NAME}" \
  "trainer.default_local_dir=${CKPT_DIR}" \
  "trainer.val_only=True" \
  "trainer.val_before_train=True" \
  "$@"
