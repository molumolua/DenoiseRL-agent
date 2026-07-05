#!/bin/bash

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

RECIPE_DIR=${RECIPE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-"${RECIPE_DIR}/local_data"}
VERL_AGENT_DATA_DIR=${VERL_AGENT_DATA_DIR:-"${LOCAL_DATA_DIR}/verl_agent"}
export ALFWORLD_DATA=${ALFWORLD_DATA:-"${LOCAL_DATA_DIR}/alfworld"}

ENGINE=${ENGINE:-vllm}
MODEL_ROOT=${MODEL_ROOT:-/inspire/hdd/global_user/xucaijun-253108120121/Model}
MODEL_PATH=${MODEL_PATH:-${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}}
case "${MODEL_PATH}" in
  /*) ;;
  *) MODEL_PATH="${MODEL_ROOT%/}/${MODEL_PATH}" ;;
esac

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-50}
HISTORY_LENGTH=${HISTORY_LENGTH:-2}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}

PROMPT_LENGTH=${PROMPT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_N=${VAL_N:-1}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
KL_COEF=${KL_COEF:-0.01}
INVALID_ACTION_PENALTY=${INVALID_ACTION_PENALTY:-0.1}

TP_SIZE=${TP_SIZE:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
NNODES=${NNODES:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4608}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:--1}
PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld_unified}

prepare_alfworld_data() {
  local train_file="${VERL_AGENT_DATA_DIR}/text/train.parquet"
  local val_file="${VERL_AGENT_DATA_DIR}/text/test.parquet"
  if [[ -f "${train_file}" && -f "${val_file}" ]]; then
    echo "Using local trainer parquet files: ${VERL_AGENT_DATA_DIR}/text"
    return 0
  fi

  if [[ "${OFFLINE_DATA_ONLY:-0}" == "1" ]]; then
    echo "Missing local trainer parquet files under ${VERL_AGENT_DATA_DIR}/text and OFFLINE_DATA_ONLY=1."
    echo "Expected: ${train_file} and ${val_file}"
    exit 1
  fi

  python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${VERL_AGENT_DATA_DIR}" \
    --train_data_size "${TRAIN_BATCH_SIZE}" \
    --val_data_size "${VAL_BATCH_SIZE}"
}

ALFWORLD_COMMON_ARGS=(
  "data.train_files=${VERL_AGENT_DATA_DIR}/text/train.parquet"
  "data.val_files=${VERL_AGENT_DATA_DIR}/text/test.parquet"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.val_batch_size=${VAL_BATCH_SIZE}"
  "data.max_prompt_length=${PROMPT_LENGTH}"
  "data.max_response_length=${RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=left"
  "data.return_raw_chat=True"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}"
  "actor_rollout_ref.rollout.do_sample=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.name=${ENGINE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=True"
  "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY}"
  "algorithm.use_kl_in_reward=False"
  "env.env_name=alfworld/AlfredTWEnv"
  "env.seed=${SEED:-0}"
  "env.history_length=${HISTORY_LENGTH}"
  "env.max_steps=${MAX_STEPS}"
  "env.rollout.n=${GROUP_SIZE}"
  "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER}"
  "trainer.critic_warmup=0"
  "trainer.logger=['console','wandb']"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.val_before_train=True"
)
