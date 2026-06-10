cd starVLA

bash examples/rl_games/install/install_stack.sh gr00t flappy

conda activate starvla_rl_games_gr00t

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="gr00t_bridge_flappy_single_latency_exp1_lr_2.5x" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_flappy_single_latency_exp1_lr_2.5x" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_flappy_single_latency_exp1_lr_2.5x" \
    dataset.source_hf="talha1503/flappy_bird_zero_latency_parquet" \
    checkpoint.save_best_model=true \
    datasets.max_episodes=3000 \
    trainer.max_train_steps=3000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=100 \
    trainer.logging_frequency=1 \
    trainer.learning_rate.base=5e-6 \
    trainer.learning_rate.qwen_vl_interface=5e-6 \
    trainer.learning_rate.action_model=5e-6 \
    trainer.gradient_accumulation_steps=16 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.interval_steps=100 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
