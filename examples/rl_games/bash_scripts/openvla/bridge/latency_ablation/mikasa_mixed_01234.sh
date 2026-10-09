#!/usr/bin/env bash
set -euo pipefail

STARVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${STARVLA_ROOT}")}"
export LATENCY_BENCH_ROOT="${LATENCY_BENCH_ROOT:-${WORKSPACE_DIR}/latency-aware-agents}"

cd "${STARVLA_ROOT}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

bash examples/rl_games/install/install_stack.sh openvla

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starvla_rl_games_openvla

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Same H1 QwenOFT recipe as the paper's zero-latency MIKASA baseline.
# Extra key=value arguments override the values below.
python examples/MIKASA/scripts/launch_train.py \
    model=openvla \
    env=mikasa \
    init=scratch \
    mode=mixed \
    run_id="openvla_mikasa_mixed_01234_h1_latency_prompt_exp1" \
    workspace_dir="${WORKSPACE_DIR}" \
    mikasa.benchmark_root="${LATENCY_BENCH_ROOT}" \
    wandb_entity="talha1503" \
    wandb_project=starvla_tasks \
    seed=0 \
    launch.num_processes=1 \
    trainer.distributed_backend=none \
    checkpoint.load=none \
    checkpoint.hf_repo_id="talha15032/openvla_mikasa_mixed_01234_h1_latency_prompt_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_mikasa_mixed_01234_h1_latency_prompt_exp1" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.converted_name="mikasa_mixed_01234_h1_latency_prompt_train" \
    dataset.latency_filter=[0,1,2,3,4] \
    dataset.episodes_per_latency=250 \
    dataset.validation_episodes_per_latency=25 \
    datasets.vla_data.dataset_py=latency_bench_tasks \
    datasets.vla_data.lerobot_version=v3.0 \
    datasets.vla_data.include_state=true \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    datasets.vla_data.prompt_mode=raw \
    datasets.vla_data.num_obs_frames=1 \
    datasets.vla_data.obs_image_size=[224,224] \
    datasets.vla_data.image_mode=single \
    datasets.vla_data.video_backend=pyav \
    datasets.vla_data.num_workers=0 \
    datasets.vla_data.persistent_workers=false \
    datasets.vla_data.per_device_batch_size=16 \
    framework.name=QwenOFT \
    framework.qwenvl.attn_implementation=flash_attention_2 \
    framework.qwenvl.enable_gradient_checkpointing=false \
    framework.action_model.action_model_type=MLP \
    framework.action_model.state_encoding=continuous_projector \
    framework.action_model.state_dim=7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    framework.action_model.action_horizon=1 \
    framework.action_model.future_action_window_size=0 \
    framework.action_model.past_action_window_size=0 \
    framework.action_model.loss_type=l1 \
    trainer.max_train_steps=5000 \
    trainer.stop_after_steps=5000 \
    trainer.num_warmup_steps=100 \
    trainer.eval_interval=5000 \
    trainer.save_interval=5000 \
    trainer.eval_num_batches=1 \
    trainer.logging_frequency=5 \
    trainer.gradient_accumulation_steps=8 \
    trainer.learning_rate.base=3.0e-5 \
    trainer.learning_rate.action_model=1.0e-4 \
    trainer.lr_scheduler_type=cosine_with_min_lr \
    trainer.scheduler_specific_kwargs.min_lr=1.0e-6 \
    trainer.optimizer.name=AdamW \
    trainer.optimizer.betas=[0.9,0.95] \
    trainer.optimizer.weight_decay=1.0e-8 \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    checkpoint.save_pt_file=true \
    checkpoint.save_training_state=false \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=200 \
    rl_games.env_eval.post_train.max_steps_per_episode=60 \
    "$@"
