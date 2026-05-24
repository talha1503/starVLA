cd starVLA

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

bash examples/rl_games/scripts/run_experiment.sh \
    examples/rl_games/experiments/openvla/bridge/single/flappy.yaml \
    run_id="openvla_bridge_flappy_single_latency_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb.entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_flappy_single_latency_exp1" \
    checkpoint.sync_enabled=true \
    checkpoint.sync_repo_id="talha1503/openvla_bridge_flappy_single_latency_exp1" \
    checkpoint.local_keep_last_n=2 \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.save_interval=10 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    trainer.batch_size=16 \
    rl_games.mid_train_eval.interval_steps=500 \
    rl_games.mid_train_eval.max_steps_per_episode=3600 \
    rl_games.mid_train_eval.num_episodes=20 \
    rl_games.post_train_eval.enabled=true \
    rl_games.post_train_eval.latencies=[0,1,2,3,4,5,6,7] \
    rl_games.post_train_eval.num_episodes=20 \
    rl_games.post_train_eval.max_steps_per_episode=3600
