export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

: "${WANDB_ENTITY:?Set WANDB_ENTITY to your W&B entity before running this training command}"

bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/pi0/bridge/single/demon_attack.yaml \
  conda.env_name=starvla_pi0 \
  workspace_dir=/inspire/hdd/project/spatialintelligence/public/lzj/starVLA \
  run_id=pi0_demon_attack_fix_latency_1 \
  wandb.entity="$WANDB_ENTITY" \
  paths.dataset_local_dir=data/demon_attack_fix_latency_1 \
  paths.base_model_dir=playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  dataset.source_hf= \
  dataset.converted_name=demon_attack_train \
  dataset.setup_force=false \
  dataset.force_download=false \
  base_model.repo_id= \
  initialization.checkpoint_local_dir=playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1 \
  initialization.checkpoint_hf_repo_id= \
  initialization.checkpoint_filename=checkpoints/steps_50000_pytorch_model.pt \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  trainer.max_train_steps=2000 \
  trainer.save_interval=100 \
  trainer.eval_interval=100 \
  checkpoint.load=none \
  checkpoint.sync_enabled=false \
  checkpoint.local_keep_last_n=2 \
  rl_games.latencies=[1] \
  rl_games.mid_train_eval.interval_steps=100 \
  rl_games.mid_train_eval.latencies=[1] \
  rl_games.post_train_eval.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
