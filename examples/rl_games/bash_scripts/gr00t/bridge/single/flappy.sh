cd starVLA

bash examples/rl_games/install/install_stack.sh gr00t flappy

conda activate starvla_rl_games_gr00t

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    dataset.source_hf="latency-sensitive-bench/flappy_200ep" \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=100 \
    trainer.logging_frequency=1 \
    trainer.learning_rate.base=5e-6 \
    trainer.learning_rate.qwen_vl_interface=5e-6 \
    trainer.learning_rate.action_model=5e-6 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600










cd starVLA

WORK_ROOT=/work/hdd/bfea/tchafekar

export CONDA_ENVS_PATH="$WORK_ROOT/.conda/envs"
export CONDA_PKGS_DIRS="$WORK_ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$WORK_ROOT/.cache/pip"
export HF_HOME="$WORK_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$WORK_ROOT/.cache/huggingface/hub"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache/xdg"
export TMPDIR="$WORK_ROOT/tmp"
export TEMP="$WORK_ROOT/tmp"
export TMP="$WORK_ROOT/tmp"

module load pytorch-conda/2.8
module load aws-ofi-nccl/1.14.2

# Store Conda environments and downloaded packages on /work.
export CONDA_ENVS_PATH=/work/hdd/bfea/tchafekar/.conda/envs
export CONDA_PKGS_DIRS=/work/hdd/bfea/tchafekar/.conda/pkgs


conda activate starvla_rl_games_gr00t

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/work/hdd/bfea/tchafekar/" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_flappy_single_latency_clean_data_exp1" \
    dataset.source_hf="latency-sensitive-bench/flappy_200ep" \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=100 \
    trainer.logging_frequency=1 \
    trainer.learning_rate.base=5e-6 \
    trainer.learning_rate.qwen_vl_interface=5e-6 \
    trainer.learning_rate.action_model=5e-6 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
