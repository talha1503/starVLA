cd starVLA

bash examples/rl_games/install/install_stack.sh openvla deadly_corridor

conda activate starvla_rl_games_openvla

pip install flash-attn --no-build-isolation

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=deadly_corridor \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_deadly_corridor_fix_latency_2_clean_data_bce_exp2" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_deadly_corridor_fix_latency_2_clean_data_bce_exp2" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_deadly_corridor_fix_latency_2_clean_data_bce_exp2" \
    dataset.source_hf=latency-sensitive-bench/deadly_1000ep \
    dataset.source_subdir=deadly_corridor_fix_latency_2_1000ep \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=50 \
    trainer.save_interval=500 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    framework.action_model.loss_type=multibinary_bce \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600