cd starVLA

bash examples/rl_games/install/install_stack.sh openvla cross_task

conda activate starvla_rl_games_openvla

bash examples/rl_games/scripts/run_experiment.sh \
    examples/rl_games/experiments/openvla/bridge/cross_task/flappy_mixed_demon_zero.yaml \
    run_id="openvla_bridge_cross_flappy_mixed_demon_zero_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb.entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_cross_flappy_mixed_demon_zero_exp1" \
    checkpoint.sync_enabled=true \
    checkpoint.sync_repo_id="talha1503/openvla_bridge_cross_flappy_mixed_demon_zero_exp1" \
    checkpoint.local_keep_last_n=2 \
    trainer.max_train_steps=20000 \
    trainer.num_warmup_steps=0 \
    trainer.save_interval=1000 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    trainer.batch_size=16
