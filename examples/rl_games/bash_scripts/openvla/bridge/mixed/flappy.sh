cd starVLA

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

pip install flash-attn --no-build-isolation

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=flappy \
    init=bridge \
    mode=mixed_latency \
    run_id="openvla_bridge_flappy_mixed_latency_exp5" \
    trainer.distributed_backend=deepspeed \
    launch.use_accelerate=true \
    launch.num_processes=2 \
    rl_games.env_eval.distributed_mode=rank_sharded \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_flappy_mixed_latency_exp5" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_flappy_mixed_latency_exp5" \
    dataset.source_hf="talha1503/flappy_bird_mixed_latency_parquet" \
    checkpoint.save_best_model=true \
    trainer.max_train_steps=3000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=300 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=32 \
    datasets.vla_data.per_device_batch_size=2 \
    dataset.latency_filter=[0,1,2,3,4,5] \
    dataset.episodes_per_latency=10 \
    rl_games.env_eval.mid_train.interval_steps=300 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.latencies=[0,1,2,3,4,5] \
    rl_games.env_eval.mid_train.num_episodes=3 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.vectorized.enabled=true \
    rl_games.env_eval.vectorized.batch_size=3