export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla/scratch/single/flappy.yaml \
  conda.env_name=starvla_openvla \
  workspace_dir=/inspire/hdd/project/spatialintelligence/public/lzj/starVLA \
  run_id=openvla_flappy_fix_latency_0 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_parquet \
  paths.base_model_dir=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  dataset.source_hf= \
  dataset.converted_name=flappy_train \
  dataset.setup_force=false \
  dataset.force_download=false \
  base_model.repo_id= \
  trainer.gradient_accumulation_steps=64 \
  trainer.batch_size=16 \
  trainer.max_train_steps=20000 \
  trainer.save_interval=1000 \
  trainer.eval_interval=1000 \
  checkpoint.load=none \
  checkpoint.sync_enabled=false \
  rl_games.latencies=[1] \
  rl_games.mid_train_eval.interval_steps=1000 \
  rl_games.mid_train_eval.max_steps_per_episode=3600 \
  rl_games.post_train_eval.max_steps_per_episode=3600 \
  rl_games.mid_train_eval.latencies=[0] \
  rl_games.post_train_eval.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]