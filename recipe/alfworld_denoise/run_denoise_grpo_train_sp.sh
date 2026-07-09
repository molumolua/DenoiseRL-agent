#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=offline
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DUMP_EXPERIMENT_NAME=${DUMP_EXPERIMENT_NAME:-denoise_grpo_qwen2.5_7b_1.5b_unified}
MAIN_ROLLOUT_N=${MAIN_ROLLOUT_N:-8}
SUB_ROLLOUT_K=${SUB_ROLLOUT_K:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
GROUP_SIZE=$((MAIN_ROLLOUT_N + SUB_ROLLOUT_K))
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# GPU memory budget for this recipe only. We intentionally do NOT edit
# params.sh (where GPU_MEMORY_UTILIZATION defaults to 0.6) so other recipes
# that source it keep their behavior. Pre-setting here before `source` makes
# params.sh's `${GPU_MEMORY_UTILIZATION:-0.6}` leave our value untouched.
#   solver  (main rollout model):  0.5
#   denoiser (small online model): 0.2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
source "${SCRIPT_DIR}/params.sh"

DENOISE_MODEL_PATH=${DENOISE_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DENOISE_PREFIX_STRATEGY=${DENOISE_PREFIX_STRATEGY:-full_then_ratio}
DENOISE_PREFIX_RATIO=${DENOISE_PREFIX_RATIO:-0.3}
DENOISE_PREFIX_CANDIDATES_PER_GROUP=${DENOISE_PREFIX_CANDIDATES_PER_GROUP:-${SUB_ROLLOUT_K}}
DENOISE_PREFIX_SAMPLE_SEED=${DENOISE_PREFIX_SAMPLE_SEED:-${SEED:-0}}
DENOISE_FULL_ROLLOUT_MAX_STEPS=${DENOISE_FULL_ROLLOUT_MAX_STEPS:-${MAX_STEPS}}
DENOISE_AVOID_TERMINAL_PREFIX=${DENOISE_AVOID_TERMINAL_PREFIX:-True}
DENOISE_PREFIX_MAX_STEPS=${DENOISE_PREFIX_MAX_STEPS:-${DENOISE_MAX_PREFIX_STEPS:-null}}
DENOISE_PROMPT_LENGTH=${DENOISE_PROMPT_LENGTH:-${PROMPT_LENGTH}}
DENOISE_RESPONSE_LENGTH=${DENOISE_RESPONSE_LENGTH:-512}
DENOISE_TEMPERATURE=${DENOISE_TEMPERATURE:-1.0}
DENOISE_TOP_P=${DENOISE_TOP_P:-1.0}
DENOISE_TOP_K=${DENOISE_TOP_K:--1}
DENOISE_DO_SAMPLE=${DENOISE_DO_SAMPLE:-True}
DENOISE_TP_SIZE=${DENOISE_TP_SIZE:-${TP_SIZE}}
DENOISE_MAX_MODEL_LEN=${DENOISE_MAX_MODEL_LEN:-${MAX_MODEL_LEN}}
DENOISE_MAX_NUM_BATCHED_TOKENS=${DENOISE_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}
if [ -n "${DENOISE_SHARED_GPU_MEMORY_UTILIZATION+x}" ] && \
   [ -z "${DENOISE_DENOISER_GPU_MEMORY_UTILIZATION+x}" ] && \
   [ -z "${DENOISE_SOLVER_GPU_MEMORY_UTILIZATION+x}" ]; then
  DENOISE_DENOISER_GPU_MEMORY_UTILIZATION=null
  DENOISE_SOLVER_GPU_MEMORY_UTILIZATION=null
else
  DENOISE_DENOISER_GPU_MEMORY_UTILIZATION=${DENOISE_DENOISER_GPU_MEMORY_UTILIZATION:-0.2}
  DENOISE_SOLVER_GPU_MEMORY_UTILIZATION=${DENOISE_SOLVER_GPU_MEMORY_UTILIZATION:-0.5}
fi
DENOISE_SHARED_GPU_MEMORY_UTILIZATION=${DENOISE_SHARED_GPU_MEMORY_UTILIZATION:-null}
DENOISE_COLOCATE_GPU_UTIL_CAP=${DENOISE_COLOCATE_GPU_UTIL_CAP:-0.9}
DENOISE_SEPARATE_PROCESS=${DENOISE_SEPARATE_PROCESS:-True}
DENOISE_ADVANTAGE_GROUPING=${DENOISE_ADVANTAGE_GROUPING:-mixed}

if [ -z "$DENOISE_MODEL_PATH" ]; then
  echo "Set DENOISE_MODEL_PATH to the small online denoiser model."
  exit 1
fi
case "${DENOISE_MODEL_PATH}" in
  /*) ;;
  *) DENOISE_MODEL_PATH="${MODEL_ROOT%/}/${DENOISE_MODEL_PATH}" ;;
esac

prepare_alfworld_data

python3 -m recipe.alfworld_denoise.main_online_denoise \
  --config-path config \
  --config-name alfworld_denoise_trainer \
  "${ALFWORLD_COMMON_ARGS[@]}" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.filter_groups.enable=False" \
  "env.rollout.n=${GROUP_SIZE}" \
  "env.denoise.enable=True" \
  "env.denoise.mode=online" \
  "env.denoise.main_rollout_n=${MAIN_ROLLOUT_N}" \
  "env.denoise.sub_rollout_k=${SUB_ROLLOUT_K}" \
  "env.denoise.advantage_grouping=${DENOISE_ADVANTAGE_GROUPING}" \
  "env.denoise.online.model_path=${DENOISE_MODEL_PATH}" \
  "env.denoise.online.prefix_strategy=${DENOISE_PREFIX_STRATEGY}" \
  "env.denoise.online.prefix_ratio=${DENOISE_PREFIX_RATIO}" \
  "env.denoise.online.prefix_candidates_per_group=${DENOISE_PREFIX_CANDIDATES_PER_GROUP}" \
  "env.denoise.online.prefix_sample_seed=${DENOISE_PREFIX_SAMPLE_SEED}" \
  "env.denoise.online.full_rollout_max_steps=${DENOISE_FULL_ROLLOUT_MAX_STEPS}" \
  "env.denoise.online.avoid_terminal_prefix=${DENOISE_AVOID_TERMINAL_PREFIX}" \
  "env.denoise.online.max_prefix_steps=${DENOISE_PREFIX_MAX_STEPS}" \
  "env.denoise.online.prompt_length=${DENOISE_PROMPT_LENGTH}" \
  "env.denoise.online.response_length=${DENOISE_RESPONSE_LENGTH}" \
  "env.denoise.online.temperature=${DENOISE_TEMPERATURE}" \
  "env.denoise.online.top_p=${DENOISE_TOP_P}" \
  "env.denoise.online.top_k=${DENOISE_TOP_K}" \
  "env.denoise.online.do_sample=${DENOISE_DO_SAMPLE}" \
  "env.denoise.online.tensor_model_parallel_size=${DENOISE_TP_SIZE}" \
  "env.denoise.online.max_model_len=${DENOISE_MAX_MODEL_LEN}" \
  "env.denoise.online.max_num_batched_tokens=${DENOISE_MAX_NUM_BATCHED_TOKENS}" \
  "env.denoise.online.denoiser_gpu_memory_utilization=${DENOISE_DENOISER_GPU_MEMORY_UTILIZATION}" \
  "env.denoise.online.solver_gpu_memory_utilization=${DENOISE_SOLVER_GPU_MEMORY_UTILIZATION}" \
  "env.denoise.online.shared_gpu_memory_utilization=${DENOISE_SHARED_GPU_MEMORY_UTILIZATION}" \
  "env.denoise.online.colocate_gpu_util_cap=${DENOISE_COLOCATE_GPU_UTIL_CAP}" \
  "env.denoise.online.separate_denoise_process=${DENOISE_SEPARATE_PROCESS}" \
  "trainer.val_before_train=False" \
  "trainer.experiment_name=${EXPERIMENT_NAME:-denoise_grpo_qwen2.5_7b_1.5b_unified}" \
  "$@"
