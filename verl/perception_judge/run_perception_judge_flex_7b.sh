set -x
ENGINE=${1:-vllm}

MODEL_PATH=jongwooko/Flex-VL-7B

TRAIN_DATA_PATH=ppjd_3k/train.parquet
TEST_DATA_PATH=ppjd_3k/validation.parquet

PROJECT_NAME=perception_judge
EXPERIMENT_NAME=perception_judge_flex_7b
uid="$(date +%Y%m%d_%H%M%S)"
VAL_DATA_DIR="output_log/${PROJECT_NAME}_${EXPERIMENT_NAME}_${uid}"/validation_data
ROLLOUT_DATA_DIR="output_log/${PROJECT_NAME}_${EXPERIMENT_NAME}_${uid}"/rollout_data_dir

REWARD_FUNCTION_PATH=perception_judge/reward_function.py
REWARD_FUNCTION_NAME=batch_reward_function

# PARAMETERS
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20 # 0 for HF rollout, -1 for vLLM rollout
MAX_RESPONSE_LENGTH=24576
LR=2.5e-7


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=2 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$VAL_DATA_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR\
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    custom_reward_function.name=$REWARD_FUNCTION_NAME \
    trainer.total_epochs=1 $@

