from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from examples.rl_games.scripts import launch_train, run_experiment
from examples.rl_games.scripts.setup_training_assets import _resolve_explicit_resume_checkpoint, _task_converter_and_verifier


REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ("openvla", "pi0", "pi05", "gr00t")
ENVS = ("flappy", "demon_attack", "deadly_corridor")
OPENVLA_DEADLY_CROSS_TASK_SETUPS = (
    "flappy_zero_deadly_mixed",
    "deadly_zero_flappy_mixed",
    "demon_zero_deadly_mixed",
    "deadly_zero_demon_mixed",
    "flappy_demon_deadly_024",
)


def _command_path(model: str, env: str) -> Path:
    return REPO_ROOT / "commands" / f"train_{env}_{model}.sh"


def test_training_command_matrix_targets_hydra_launcher() -> None:
    for model in MODELS:
        for env in ENVS:
            command_path = _command_path(model, env)

            assert command_path.exists(), f"Missing command wrapper: {command_path}"

            command_text = command_path.read_text(encoding="utf-8")
            assert "python examples/rl_games/scripts/launch_train.py" in command_text
            assert "examples/rl_games/experiments/" not in command_text
            assert f"model={model}" in command_text
            assert f"env={env}" in command_text
            assert "init=bridge" in command_text
            assert "mode=single" not in command_text
            assert "WANDB_ENTITY" not in command_text
            assert "wandb_entity=" not in command_text
            assert "rl_games.env_eval.post_train.latencies=" not in command_text
            assert "trainer.batch_size=" not in command_text
            assert "datasets.vla_data.per_device_batch_size=" in command_text
            assert "dataset.source_hf=data/" not in command_text
            assert "checkpoint.save_pt_file=true" not in command_text


def test_training_commands_are_valid_bash() -> None:
    command_paths = [str(_command_path(model, env)) for model in MODELS for env in ENVS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)


def test_wan_oft_commands_are_valid_bash() -> None:
    command_paths = [
        REPO_ROOT / "commands" / "train_flappy_wan_oft.sh",
        REPO_ROOT / "commands" / "train_flappy_wan_oft_horizon1.sh",
        REPO_ROOT / "commands" / "train_flappy_wan_oft_multigpu.sh",
        REPO_ROOT / "commands" / "train_demon_attack_wan_oft.sh",
    ]

    subprocess.run(["bash", "-n", *[str(path) for path in command_paths]], check=True, cwd=REPO_ROOT)


def test_wan_oft_multigpu_command_enables_distributed_eval() -> None:
    command_path = REPO_ROOT / "commands" / "train_flappy_wan_oft_multigpu.sh"
    command_text = command_path.read_text(encoding="utf-8")

    assert "model=wan_oft" in command_text
    assert "env=flappy" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "trainer.distributed_backend=deepspeed" in command_text
    assert "launch.use_accelerate=true" in command_text
    assert "export NCCL_DEBUG=${NCCL_DEBUG:-INFO}" in command_text
    assert "export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}" in command_text
    assert "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}" in command_text
    assert "launch.gpus=\"'${WAN_OFT_GPUS:-0,1}'\"" in command_text
    assert "launch.num_processes=${WAN_OFT_NUM_PROCESSES:-2}" in command_text
    assert "datasets.vla_data.data_mix=flappy_train__bridge" in command_text
    assert "datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in command_text
    assert "rl_games.env_eval.enabled=true" in command_text
    assert "rl_games.env_eval.distributed_mode=rank_sharded" in command_text
    assert "rl_games.env_eval.mid_train.enabled=true" in command_text
    assert "rl_games.env_eval.mid_train.latencies=[0]" in command_text
    assert "rl_games.env_eval.post_train.enabled=true" in command_text
    assert "rl_games.env_eval.post_train.latencies=[0,1,2,3,4]" in command_text


def test_wan_oft_single_gpu_command_enables_held_out_eval_mix() -> None:
    command_path = REPO_ROOT / "commands" / "train_flappy_wan_oft.sh"
    command_text = command_path.read_text(encoding="utf-8")

    assert "model=wan_oft" in command_text
    assert "env=flappy" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "trainer.distributed_backend=none" in command_text
    assert "launch.use_accelerate=false" in command_text
    assert "launch.num_processes=1" in command_text
    assert "run_id=wan_oft_flappy_fix_latency_0_context5_standard_sft_2000_effbs128_224_currentce" in command_text
    assert "paths.dataset_local_dir=data/flappy_fix_latency_0_200ep_context5" in command_text
    assert "dataset.converted_name=flappy_train__bridge" in command_text
    assert "trainer.max_train_steps=2000" in command_text
    assert "trainer.gradient_accumulation_steps=32" in command_text
    assert "datasets.vla_data.per_device_batch_size=4" in command_text
    assert "datasets.vla_data.obs_image_size=[224,224]" in command_text
    assert "datasets.vla_data.image_sequence_length=5" in command_text
    assert "datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]" in command_text
    assert "framework.world_model.num_frames=5" in command_text
    assert "framework.action_model.loss_type=current_discrete_ce" in command_text
    assert "framework.action_model.class_weights" not in command_text
    assert "+trainer.learning_rate.action_query_proj=1.0e-4" in command_text
    assert "trainer.lr_scheduler_type=cosine_with_min_lr" in command_text
    assert "trainer.scheduler_specific_kwargs.min_lr=1.0e-6" in command_text
    assert "trainer.eval_action_classification=false" in command_text
    assert "trainer.save_interval=0" in command_text
    assert "datasets.vla_data.action_balance.enabled=false" in command_text
    assert "datasets.vla_data.data_mix=flappy_train__bridge" in command_text
    assert "datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in command_text
    assert "rl_games.env_eval.enabled=true" in command_text
    assert "rl_games.env_eval.image_size=224" in command_text
    assert "rl_games.env_eval.latency.prompt_map_path=data/flappy_fix_latency_0_200ep_context5/flappy_train__bridge/latency_prompt_map.json" in command_text
    assert "rl_games.env_eval.mid_train.enabled=true" in command_text
    assert "rl_games.env_eval.mid_train.interval_steps=500" in command_text
    assert "rl_games.env_eval.mid_train.latencies=[0]" in command_text
    assert "rl_games.env_eval.mid_train.num_episodes=5" in command_text
    assert "rl_games.env_eval.post_train.enabled=true" in command_text
    assert "rl_games.env_eval.post_train.latencies=[0]" in command_text
    assert "rl_games.env_eval.post_train.num_episodes=20" in command_text
    assert "checkpoint.save_pt_file=true" in command_text
    assert "checkpoint.save_training_state=false" in command_text
    assert "checkpoint.save_best_model=false" in command_text
    assert "checkpoint.save_final_model=true" in command_text


def test_wan_oft_demon_attack_single_gpu_command_derives_baseline_from_latency_arg() -> None:
    command_path = REPO_ROOT / "commands" / "train_demon_attack_wan_oft.sh"
    command_text = command_path.read_text(encoding="utf-8")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}}"' in command_text
    assert 'RUN_ID="${RUN_ID:-wan_oft_demon_attack_fix_latency_${LATENCY}_context${CONTEXT_WINDOW}_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentce}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"' in command_text
    assert 'EVAL_INTERVAL="${EVAL_INTERVAL:-500}"' in command_text
    assert 'MID_TRAIN_LATENCIES="${MID_TRAIN_LATENCIES:-[${LATENCY}]}"' in command_text
    assert 'POST_TRAIN_LATENCIES="${POST_TRAIN_LATENCIES:-[${LATENCY}]}"' in command_text
    assert "model=wan_oft" in command_text
    assert "env=demon_attack" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "trainer.distributed_backend=none" in command_text
    assert "launch.use_accelerate=false" in command_text
    assert 'run_id="${RUN_ID}"' in command_text
    assert 'paths.dataset_local_dir="${DATASET_LOCAL_DIR}"' in command_text
    assert "dataset.converted_name=demon_attack_train__bridge" in command_text
    assert "datasets.vla_data.data_mix=demon_attack_train__bridge" in command_text
    assert "datasets.vla_data.eval_data_mix=demon_attack_train__bridge__val" in command_text
    assert "datasets.vla_data.obs_image_size=[224,224]" in command_text
    assert "datasets.vla_data.image_sequence_length=\"${CONTEXT_WINDOW}\"" in command_text
    assert "datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]" in command_text
    assert "framework.world_model.num_frames=\"${CONTEXT_WINDOW}\"" in command_text
    assert "framework.action_model.loss_type=current_discrete_ce" in command_text
    assert "trainer.max_train_steps=\"${MAX_TRAIN_STEPS}\"" in command_text
    assert "trainer.eval_interval=\"${EVAL_INTERVAL}\"" in command_text
    assert "trainer.eval_action_classification=false" in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/demon_attack_latency_prompt_map.json}"' in command_text
    assert "rl_games.env_eval.latency.prompt_map_path=\"${PROMPT_MAP_PATH}\"" in command_text
    assert "rl_games.env_eval.mid_train.interval_steps=\"${MID_TRAIN_INTERVAL}\"" in command_text
    assert '"rl_games.env_eval.mid_train.latencies=${MID_TRAIN_LATENCIES}"' in command_text
    assert '"rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}"' in command_text


def test_demon_attack_wan_oft_pipeline_script_parameterizes_latency() -> None:
    script_path = REPO_ROOT / "scripts" / "run_demon_attack_wan_oft_pipeline.sh"
    script_text = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True, cwd=REPO_ROOT)

    assert "--latency <N>" in script_text
    assert 'LATENCY=""' in script_text
    assert 'RAW_DATASET_REPO="${RAW_DATASET_REPO:-latency-sensitive-bench/demon_attack_200ep_context${CONTEXT_WINDOW}}"' in script_text
    assert 'RAW_SUBDIR="demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"' in script_text
    assert 'CONVERTED_DATA_ROOT="data/demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"' in script_text
    assert 'RUN_ID="wan_oft_demon_attack_fix_latency_${LATENCY}_context${CONTEXT_WINDOW}_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentce"' in script_text
    assert 'bash examples/rl_games/install/bootstrap.sh' in script_text
    assert 'hf download "${BASE_MODEL_REPO}"' in script_text
    assert '--include "${RAW_SUBDIR}/**"' in script_text
    assert 'convert_demon_attack_to_starvla_lerobot.py' in script_text
    assert 'bash commands/train_demon_attack_wan_oft.sh "${LATENCY}"' in script_text
    assert 'hf upload "${UPLOAD_REPO}" "${RUN_DIR}" "${UPLOAD_PATH_IN_REPO}"' in script_text


def test_wan_oft_chunk8_command_matches_released_checkpoint() -> None:
    command_path = REPO_ROOT / "commands" / "train_flappy_wan_oft_horizon1.sh"
    command_text = command_path.read_text(encoding="utf-8")

    assert "model=wan_oft" in command_text
    assert "env=flappy" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "trainer.distributed_backend=deepspeed" in command_text
    assert "launch.use_accelerate=true" in command_text
    assert "launch.num_processes=1" in command_text
    assert "run_id=wan_oft_flappy_fix_latency_0_context4_chunk8" in command_text
    assert "horizon1" not in command_text
    assert "framework.action_model.action_horizon=8" in command_text
    assert "framework.action_model.future_action_window_size=7" in command_text
    assert "framework.action_model.past_action_window_size=0" in command_text
    assert "datasets.vla_data.action_indices=[0,1,2,3,4,5,6,7]" in command_text
    assert "trainer.reload_modules=" not in command_text
    assert "datasets.vla_data.data_mix=flappy_train__bridge" in command_text
    assert "datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in command_text


def test_openvla_deadly_cross_task_scripts_are_valid_bash() -> None:
    script_dir = REPO_ROOT / "examples" / "rl_games" / "bash_scripts" / "openvla" / "bridge" / "cross_task"
    command_paths = [str(script_dir / f"{setup}.sh") for setup in OPENVLA_DEADLY_CROSS_TASK_SETUPS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)


def test_cross_task_setup_supports_deadly_corridor_converter() -> None:
    _, _, robot_type = _task_converter_and_verifier("deadly_corridor")

    assert robot_type == "rl_games_deadly_corridor"


def test_openvla_deadly_cross_task_setups_compose() -> None:
    for setup_name in OPENVLA_DEADLY_CROSS_TASK_SETUPS:
        cfg = launch_train.compose_training_config(
            config_name="train",
            model="openvla",
            env="cross_task",
            init="bridge",
            mode="cross_task",
            overrides=[f"cross_task_setup={setup_name}"],
        )
        train_tasks = OmegaConf.to_container(cfg.rl_games.cross_task.train_tasks, resolve=True)
        task_names = {task["name"] for task in train_tasks}

        assert "deadly_corridor" in task_names
        assert cfg.framework.name == "QwenOFT"
        assert cfg.framework.action_model.action_dim == 7
        assert cfg.framework.action_model.action_env_dim == 7
        assert cfg.rl_games.cross_task.loss_by_task.deadly_corridor == "multibinary_bce"
        assert cfg.rl_games.env_eval.deadly.action_layout == "multibinary_7"


def test_openvla_three_env_cross_task_setup_uses_024_latencies() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="cross_task",
        init="bridge",
        mode="cross_task",
        overrides=["cross_task_setup=flappy_demon_deadly_024"],
    )
    train_tasks = OmegaConf.to_container(cfg.rl_games.cross_task.train_tasks, resolve=True)

    assert [task["name"] for task in train_tasks] == ["flappy", "demon_attack", "deadly_corridor"]
    for task in train_tasks:
        assert task["train_latency_filter"] == [0, 2, 4]
        assert task["eval_latency_filter"] == [0, 2, 4]
    assert cfg.rl_games.cross_task.loss_by_task.flappy == "discrete_ce"
    assert cfg.rl_games.cross_task.loss_by_task.demon_attack == "discrete_ce"
    assert cfg.rl_games.cross_task.loss_by_task.deadly_corridor == "multibinary_bce"
    assert cfg.rl_games.env_eval.deadly.multibinary_threshold == 0.0
    for task_name in ("flappy", "demon_attack", "deadly_corridor"):
        post_train = getattr(cfg.rl_games.cross_task.eval_tasks, task_name).post_train
        assert list(post_train.latencies) == [0, 2, 4]


def test_openvla_three_env_cross_task_forwards_deadly_threshold(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="cross_task",
        init="bridge",
        mode="cross_task",
        overrides=["cross_task_setup=flappy_demon_deadly_024"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++rl_games.env_eval.deadly.action_layout=multibinary_7" in cmd
    assert "++rl_games.env_eval.deadly.multibinary_threshold=0.0" in cmd


def test_launcher_does_not_translate_trainer_batch_size_alias() -> None:
    launcher_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "launch_train.py").read_text(
        encoding="utf-8"
    )

    assert '"trainer.batch_size"' not in launcher_text


def test_launcher_forwards_canonical_per_device_batch_size_override(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["datasets.vla_data.per_device_batch_size=16"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++datasets.vla_data.per_device_batch_size=16" in cmd


def test_launcher_quotes_comma_separated_reload_modules_for_hydra() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=["trainer.reload_modules='backbone,action_model'"],
    )
    setup = {
        "pretrained_checkpoint": "/tmp/checkpoint.pt",
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, Path("/tmp/workspace"), "results/Checkpoints")

    assert "++trainer.reload_modules='backbone,action_model'" in cmd


def test_launcher_auto_forwards_new_nested_config_fields(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[
            "+trainer.optimizer.extra_flag=true",
            "+datasets.vla_data.synthetic_cache.enabled=true",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++trainer.optimizer.extra_flag=true" in cmd
    assert "++datasets.vla_data.synthetic_cache.enabled=true" in cmd


def test_launcher_runtime_overrides_are_last(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[
            "datasets.vla_data.data_root_dir=config_root",
            "framework.qwenvl.base_vlm=config_base",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "resolved_datasets"),
        "base_model_dir": str(tmp_path / "resolved_base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    data_root_overrides = [item for item in cmd if item.endswith("datasets.vla_data.data_root_dir=config_root") or "datasets.vla_data.data_root_dir=" in item]
    base_vlm_overrides = [item for item in cmd if item.endswith("framework.qwenvl.base_vlm=config_base") or "framework.qwenvl.base_vlm=" in item]

    assert data_root_overrides[-1] == f"++datasets.vla_data.data_root_dir={tmp_path / 'resolved_datasets'}"
    assert base_vlm_overrides[-1] == f"++framework.qwenvl.base_vlm={tmp_path / 'resolved_base_model'}"


def test_run_experiment_auto_forwards_canonical_nested_fields(tmp_path: Path) -> None:
    cfg = {
        "config_name": "train",
        "model": "openvla",
        "env": "flappy",
        "init": "bridge",
        "mode": "single",
        "run_id": "run_experiment_test",
        "seed": 42,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "paths": {"dataset_local_dir": "datasets", "base_model_dir": "base_model"},
        "checkpoint": {
            "load": "none",
            "hf_repo_id": None,
            "sync": {"enabled": False, "keep_last_n": 0, "repo_id": None},
            "local": {"keep_last_n": 1},
            "save_best_model": False,
            "save_pt_file": False,
        },
        "trainer": {"optimizer": {"extra_flag": True}},
        "datasets": {"vla_data": {"per_device_batch_size": 8, "synthetic_cache": {"enabled": True}}},
        "framework": {"qwenvl": {"base_vlm": "config_base"}},
        "rl_games": {"env_eval": {"enabled": False, "latency": {"values": [0]}}},
    }
    setup = {
        "dataset_local_dir": str(tmp_path / "resolved_datasets"),
        "base_model_dir": str(tmp_path / "resolved_base_model"),
        "resume_found": False,
    }

    cmd = run_experiment._trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++trainer.optimizer.extra_flag=true" in cmd
    assert "++datasets.vla_data.synthetic_cache.enabled=true" in cmd
    assert "trainer.batch_size=" not in " ".join(cmd)
    assert [item for item in cmd if "datasets.vla_data.data_root_dir=" in item][-1] == (
        f"++datasets.vla_data.data_root_dir={tmp_path / 'resolved_datasets'}"
    )
    assert [item for item in cmd if "framework.qwenvl.base_vlm=" in item][-1] == (
        f"++framework.qwenvl.base_vlm={tmp_path / 'resolved_base_model'}"
    )


def test_launcher_forwards_vit_and_llm_freeze_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "test")
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["trainer.freeze_vit=true", "trainer.freeze_llm_layers=[0,27]"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++trainer.freeze_vit=true" in cmd
    assert "++trainer.freeze_llm_layers=[0,27]" in cmd
    assert "trainer.freeze_llm_bottom_ratio=" not in cmd


def test_deadly_corridor_loss_selector_forwards_l1(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="deadly_corridor",
        init="bridge",
        mode="single",
        overrides=["rl_games.deadly_corridor_loss_type=l1"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.action_model.loss_type == "l1"
    assert "++framework.action_model.loss_type=l1" in cmd


def test_launcher_setup_namespace_forwards_episodes_per_latency(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="mixed_latency",
        overrides=["dataset.episodes_per_latency=100"],
    )

    setup_args = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.episodes_per_latency == 100


def test_setup_training_assets_exposes_episodes_per_latency_option() -> None:
    setup_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "setup_training_assets.py").read_text(
        encoding="utf-8"
    )

    assert "--episodes-per-latency" in setup_text


def test_deadly_corridor_loss_selector_accepts_multibinary_ce_alias(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="deadly_corridor",
        init="bridge",
        mode="single",
        overrides=["rl_games.deadly_corridor_loss_type=multibinary_ce"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.action_model.loss_type == "multibinary_bce"
    assert "++framework.action_model.loss_type=multibinary_bce" in cmd


def test_launcher_defaults_to_one_last_checkpoint_and_no_pt_file(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++checkpoint.local.keep_last_n=1" in cmd
    assert "++checkpoint.save_best_model=true" in cmd
    assert "++checkpoint.save_pt_file=false" in cmd


def test_run_train_exposes_deadly_loss_type_option() -> None:
    run_train_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(
        encoding="utf-8"
    )

    assert "--deadly-loss-type <l1|multibinary_bce|multibinary_ce>" in run_train_text
    assert "rl_games.deadly_corridor_loss_type=$DEADLY_LOSS_TYPE" in run_train_text


def test_run_train_exposes_distributed_eval_option() -> None:
    run_train_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(
        encoding="utf-8"
    )

    assert "--eval-distributed-mode <none|rank_sharded>" in run_train_text
    assert "rl_games.env_eval.distributed_mode=$EVAL_DISTRIBUTED_MODE" in run_train_text


def test_launch_train_exposes_distributed_eval_option() -> None:
    launcher_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "launch_train.py").read_text(
        encoding="utf-8"
    )

    assert "--eval-distributed-mode" in launcher_text
    assert "rl_games.env_eval.distributed_mode={cli_args.eval_distributed_mode}" in launcher_text


def test_deadly_corridor_openvla_wrapper_passes_extra_args() -> None:
    command_text = _command_path("openvla", "deadly_corridor").read_text(encoding="utf-8")

    assert '"$@"' in command_text


def test_launcher_forwards_cross_task_setup_outputs(tmp_path: Path) -> None:
    cfg = OmegaConf.create({
        "config_name": "train",
        "model": "openvla",
        "env": "cross_task",
        "init": "bridge",
        "mode": "cross_task",
        "run_id": "cross_task_test",
        "seed": 42,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "paths": {"dataset_local_dir": "datasets"},
        "checkpoint": {
            "sync": {"enabled": False, "keep_last_n": 0, "repo_id": ""},
            "local": {"keep_last_n": 3},
            "save_best_model": True,
        },
        "trainer": {"is_resume": False},
        "rl_games": {
            "task": "cross_task",
            "model_alias": "openvla",
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
            "env_eval": {
                "enabled": True,
                "distributed_mode": "rank_sharded",
                "latency": {"mode": "mixed", "values": [0, 1]},
                "mid_train": {"interval_steps": 500},
                "post_train": {"enabled": True},
            },
            "cross_task": {
                "eval_tasks": {
                    "flappy": {
                        "frameskip": 1,
                        "image_size": 224,
                        "mid_train": {
                            "enabled": True,
                            "latencies": [0, 1],
                            "num_episodes": 2,
                            "max_steps_per_episode": 200,
                        },
                    },
                    "demon_attack": {
                        "frameskip": 4,
                        "image_size": 224,
                        "mid_train": {
                            "enabled": True,
                            "latencies": [0],
                            "num_episodes": 3,
                            "max_steps_per_episode": 300,
                        },
                    },
                }
            },
        },
        "datasets": {"vla_data": {"include_state": False}},
    })
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "data_mix": "cross__flappy__demon",
        "eval_data_mix": "cross__flappy__demon__val",
        "custom_mixtures_path": str(tmp_path / "datasets" / "_generated_mixtures" / "cross.json"),
        "base_model_dir": str(tmp_path / "base"),
        "resume_found": False,
        "pretrained_checkpoint": str(tmp_path / "init.pt"),
        "cross_task_prompt_maps": {
            "flappy": str(tmp_path / "flappy_prompt_map.json"),
            "demon_attack": str(tmp_path / "demon_prompt_map.json"),
        },
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++datasets.vla_data.data_mix=cross__flappy__demon" in cmd
    assert "++datasets.vla_data.eval_data_mix=cross__flappy__demon__val" in cmd
    assert f"++datasets.vla_data.custom_mixtures_path={setup['custom_mixtures_path']}" in cmd
    assert "++rl_games.env_eval.distributed_mode=rank_sharded" in cmd
    assert f"++rl_games.cross_task.eval_tasks.flappy.prompt_map_path={setup['cross_task_prompt_maps']['flappy']}" in cmd
    assert (
        f"++rl_games.cross_task.eval_tasks.demon_attack.prompt_map_path="
        f"{setup['cross_task_prompt_maps']['demon_attack']}"
    ) in cmd
    assert "++rl_games.cross_task.eval_tasks.flappy.mid_train.latencies=[0,1]" in cmd
    assert "++rl_games.cross_task.eval_tasks.flappy.mid_train.num_episodes=2" in cmd
    assert "++rl_games.cross_task.eval_tasks.demon_attack.mid_train.max_steps_per_episode=300" in cmd


def test_cross_task_mode_does_not_override_initialization_group() -> None:
    mode_text = (REPO_ROOT / "examples" / "rl_games" / "config" / "mode" / "cross_task.yaml").read_text(
        encoding="utf-8"
    )

    assert "initialization_mode:" not in mode_text
    assert "action_carrier:" not in mode_text


def test_launcher_defaults_workspace_to_repo_root_when_env_is_unset(monkeypatch) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/workspace":
            return True
        return original_exists(path)

    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    monkeypatch.setattr(Path, "exists", fake_exists)

    cfg = OmegaConf.create({"workspace_dir": "WORKSPACE_DIR"})

    assert launch_train._workspace_dir(cfg) == launch_train.REPO_ROOT


def test_launcher_treats_checkpoint_load_path_as_explicit_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "runs" / "run_1" / "checkpoints" / "steps_400_state"
    cfg = OmegaConf.create(
        {
            "run_id": "run_1",
            "model": "openvla",
            "env": "flappy",
            "mode": "single",
            "rl_games": {"initialization_mode": "bridge", "action_carrier": "bridge"},
            "dataset": {"converted_name": "flappy_train"},
            "paths": {
                "dataset_local_dir": "datasets",
                "base_model_dir": "base_model",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "checkpoint": {
                "load": str(checkpoint_path),
                "hf_repo_id": "",
                "save_best_model": True,
                "sync": {"enabled": False, "repo_id": ""},
            },
            "initialization": {
                "checkpoint_local_dir": "",
                "checkpoint_hf_repo_id": "",
                "checkpoint_filename": "",
            },
        }
    )

    namespace = launch_train.setup_namespace_from_cfg(cfg, tmp_path, str(tmp_path / "runs"))

    assert namespace.checkpoint == str(checkpoint_path)
    assert namespace.checkpoint_load == "none"


def test_vla_trainer_saves_last_checkpoints_independently_from_best_model() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert ") and not self._save_best_model_enabled:" not in trainer_text


def test_vla_trainer_pt_checkpoint_file_is_optional() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "self._save_pt_file_enabled" in trainer_text
    assert "safe_serialization=True" in trainer_text
    assert "self.accelerator.get_state_dict(self.model)" in trainer_text
    assert 'model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"' in trainer_text
    assert "torch.save(state_dict, model_checkpoint_path)" in trainer_text
    assert "model_path=model_checkpoint_path" in trainer_text


def test_run_train_accepts_explicit_resume_checkpoint() -> None:
    script_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(encoding="utf-8")

    assert "--checkpoint <path>" in script_text
    assert '--checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;' in script_text
    assert '--checkpoint "${RESUME_CHECKPOINT:-}"' in script_text


def test_explicit_best_state_resume_reads_best_metadata(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    best_state = checkpoint_dir / "best_state"
    best_state.mkdir(parents=True)
    (checkpoint_dir / "best_model_metadata.json").write_text('{"best_step": 2500}', encoding="utf-8")

    checkpoint, step, kind = _resolve_explicit_resume_checkpoint(str(best_state), checkpoint_dir)

    assert checkpoint == best_state.resolve()
    assert step == 2500
    assert kind == "state"


def test_explicit_resume_checkpoint_matches_eval_relative_checkpoint_resolution(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    state_dir = checkpoint_dir / "steps_300_state"
    state_dir.mkdir(parents=True)
    (state_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    checkpoint, step, kind = _resolve_explicit_resume_checkpoint("steps_300_state", checkpoint_dir)

    assert checkpoint == state_dir.resolve()
    assert step == 300
    assert kind == "state"
