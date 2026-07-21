#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Final evaluation defaults: report both ALFWorld splits (seen + unseen).
# Override via env vars, e.g. EVAL_SPLIT=unseen bash run_grpo_eval.sh.
EVAL_SPLIT=${EVAL_SPLIT:-both}
DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-grpo_qwen2.5_7b_unified_eval}
# The shared entry point still constructs training envs in val-only mode. One
# rollout slot avoids allocating unused GRPO-group env workers.
GROUP_SIZE=${GROUP_SIZE:-1}

source "${SCRIPT_DIR}/params.sh"

if [ -z "${CKPT_DIR:-}" ]; then
  echo "Set CKPT_DIR to an experiment root or global_step_* directory." >&2
  exit 1
fi

if [[ -d "${CKPT_DIR}/actor" && "$(basename "${CKPT_DIR}")" =~ ^global_step_[0-9]+$ ]]; then
  RESUME_PATH=${CKPT_DIR}
elif [[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  IFS= read -r CHECKPOINT_STEP < "${CKPT_DIR}/latest_checkpointed_iteration.txt" || true
  if [[ ! "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint step in ${CKPT_DIR}/latest_checkpointed_iteration.txt: ${CHECKPOINT_STEP}" >&2
    exit 1
  fi
  RESUME_PATH="${CKPT_DIR%/}/global_step_${CHECKPOINT_STEP}"
  if [[ ! -d "${RESUME_PATH}/actor" ]]; then
    echo "Checkpoint tracker points to a missing actor directory: ${RESUME_PATH}/actor" >&2
    exit 1
  fi
else
  echo "Invalid CKPT_DIR: expected global_step_*/actor or latest_checkpointed_iteration.txt." >&2
  exit 1
fi
RESUME_ARGS=(
  "trainer.resume_mode=resume_path"
  "trainer.resume_from_path=${RESUME_PATH}"
)

prepare_alfworld_data

BASE_EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_eval}

echo "============================================================"
echo " Exhaustive ALFWorld policy evaluation"
echo " split(s)=${EVAL_SPLITS}  val_n=${VAL_N}  temperature=${VAL_TEMPERATURE}"
echo " checkpoint=${RESUME_PATH}"
echo " Every gamefile is pinned explicitly and coverage is asserted."
echo "============================================================"

python3 -m verl.trainer.main_ppo \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=False" \
  "algorithm.use_kl_in_reward=False" \
  "actor_rollout_ref.actor.use_kl_loss=False" \
  "trainer.logger=['console']" \
  "trainer.project_name=${EVAL_PROJECT_NAME:-verl_agent_alfworld_eval}" \
  "trainer.experiment_name=${BASE_EXPERIMENT_NAME}" \
  "trainer.val_only=True" \
  "trainer.val_before_train=True" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=-1" \
  "$@"
