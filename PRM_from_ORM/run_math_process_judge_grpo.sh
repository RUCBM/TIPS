#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 || true

: "${MODEL_PATH:?Set MODEL_PATH to a local model path or HuggingFace model id.}"

MODE=${MODE:-ORM}
DATA_DIR=${DATA_DIR:-PRM_from_ORM/processed_math_process_judge}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3-4B-PRM_from_${MODE}_MATH_TokenMean_NoNormAdv}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/scan_pro_train.parquet}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/processbench_eval.parquet']"}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${EXPERIMENT_NAME}}
ROLLOUT_DIR=${ROLLOUT_DIR:-outputs/rollout_data}
LOGGER=${LOGGER:-'["console"]'}
PROJECT_NAME=${PROJECT_NAME:-PRM_from_ORM}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-20000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32000}
ROLLOUT_N=${ROLLOUT_N:-8}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}

if [[ ! -f "${TRAIN_FILES}" ]]; then
  echo "Missing TRAIN_FILES: ${TRAIN_FILES}" >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.nccl_timeout=64000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE:-token-mean}" \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-1024}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-65536}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.validation_data_dir="${ROLLOUT_DIR}/${EXPERIMENT_NAME}-validation" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}/${EXPERIMENT_NAME}-train" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" "$@"
