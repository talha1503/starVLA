from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


class ActionLatencyQueue:
    def __init__(self, latency: int, default_action):
        self.latency = max(0, int(latency))
        self.default_action = default_action
        self.current_action = default_action
        self.pending = []
        self.step_count = 0

    def reset(self):
        self.current_action = self.default_action
        self.pending = []
        self.step_count = 0

    def schedule_and_get(self, action):
        if self.latency == 0:
            self.step_count += 1
            return action
        apply_at = self.step_count + self.latency
        self.pending.append((apply_at, action))
        while self.pending and self.pending[0][0] <= self.step_count:
            _, matured = self.pending.pop(0)
            self.current_action = matured
        self.step_count += 1
        return self.current_action


@dataclass
class EvalResult:
    per_latency: Dict[str, Dict]
    aggregate: Dict


def decode_discrete_argmax(action_values: Iterable[float], n_actions: int) -> int:
    arr = np.asarray(action_values, dtype=np.float32)
    return int(np.argmax(arr[:n_actions]))


def decode_deadly_multibinary_7(action_values: Iterable[float], threshold: float = 0.5) -> List[int]:
    arr = np.asarray(action_values, dtype=np.float32)
    return [int(x >= threshold) for x in arr[:7]]


def decode_deadly_factorized_11(action_values: Iterable[float]) -> List[int]:
    arr = np.asarray(action_values, dtype=np.float32)
    if arr.shape[0] < 11:
        raise ValueError(f"Expected at least 11 action dims for factorized decode, got {arr.shape[0]}")
    turn = int(np.argmax(arr[0:3]))
    move = int(np.argmax(arr[3:6]))
    strafe = int(np.argmax(arr[6:9]))
    attack = int(np.argmax(arr[9:11]))
    return [turn, move, strafe, attack]


def _factorized_to_semantic_buttons(action_tuple: List[int]) -> List[int]:
    turn, move, strafe, attack = action_tuple
    active = set()
    if turn == 1:
        active.add("TURN_LEFT")
    elif turn == 2:
        active.add("TURN_RIGHT")
    if move == 1:
        active.add("MOVE_FORWARD")
    elif move == 2:
        active.add("MOVE_BACKWARD")
    if strafe == 1:
        active.add("MOVE_LEFT")
    elif strafe == 2:
        active.add("MOVE_RIGHT")
    if attack == 1:
        active.add("ATTACK")
    semantic_order = [
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
        "ATTACK",
    ]
    return [1 if name in active else 0 for name in semantic_order]


def _extract_image_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        for key in (
            "screen",
            "rgb",
            "image",
            "obs",
            "observation",
            "pixels",
        ):
            value = obs.get(key)
            if value is not None and np.asarray(value).ndim >= 2:
                return value
        for value in obs.values():
            if hasattr(value, "shape") and np.asarray(value).ndim >= 2:
                return value
        raise TypeError(f"dict observation has no image-like value; keys={sorted(obs.keys())}")
    return obs


def _resize_rgb(raw_obs: Any, image_size: int) -> np.ndarray:
    raw_obs = _extract_image_observation(raw_obs)
    if raw_obs is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    raw_obs = np.asarray(raw_obs)
    if raw_obs.dtype != np.uint8:
        raw_obs = raw_obs.astype(np.uint8)
    if raw_obs.ndim == 2:
        raw_obs = np.stack([raw_obs] * 3, axis=-1)
    elif raw_obs.ndim == 3 and raw_obs.shape[2] == 1:
        raw_obs = np.repeat(raw_obs, 3, axis=2)
    if raw_obs.shape[:2] == (image_size, image_size):
        return raw_obs
    return np.array(Image.fromarray(raw_obs).resize((image_size, image_size), Image.BILINEAR))


def _load_latency_prompt_map(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    out: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        latency = int(value.get("latency", key))
        out[latency] = {
            "latency": latency,
            "prompt": str(value["prompt"]),
            "latency_ms": value.get("latency_ms", None),
        }
    return out


def _get_available_button_names(env) -> List[str]:
    def _button_name(button):
        name = getattr(button, "name", None)
        if name:
            return str(name)
        text = str(button)
        if "." in text:
            text = text.split(".")[-1]
        return text

    for candidate in (env, getattr(env, "unwrapped", None)):
        if candidate is None:
            continue
        for attr_name in ("game", "_game"):
            game = getattr(candidate, attr_name, None)
            if game is None:
                continue
            getter = getattr(game, "get_available_buttons", None)
            if getter is None:
                continue
            buttons = getter()
            names = [_button_name(button) for button in buttons]
            if names:
                return names
    return ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK"]


def _semantic_to_runtime_multibinary(semantic_values: List[int], runtime_order: List[str]) -> List[int]:
    semantic_order = ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    semantic_map = {name: int(semantic_values[idx]) for idx, name in enumerate(semantic_order)}
    return [semantic_map.get(name, 0) for name in runtime_order]


class _TaskEvaluator:
    def __init__(self, task: str, cfg):
        self.task = task
        self.cfg = cfg
        self.env_eval_cfg = cfg.rl_games.env_eval
        self.default_prompt = str(getattr(self.env_eval_cfg, "task_description", "") or "")
        self.image_size = int(getattr(self.env_eval_cfg, "image_size", 84))
        self.frameskip = max(1, int(getattr(self.env_eval_cfg, "frameskip", 1)))
        self.state_dim = int(getattr(cfg.framework.action_model, "state_dim", 1) or 1)

    def _make_env(self):
        if self.task == "flappy":
            import flappy_bird_gymnasium  # noqa: F401
            import gymnasium as gym

            return gym.make("FlappyBird-v0")

        if self.task == "demon_attack":
            import ale_py  # noqa: F401
            import gymnasium as gym

            attempts = [
                "ALE/DemonAttack-v5",
                "ALE/DemonAttack-v4",
            ]
            last_exc = None
            for env_id in attempts:
                try:
                    return gym.make(
                        env_id,
                        frameskip=self.frameskip,
                        repeat_action_probability=0.0,
                    )
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Failed to create DemonAttack env: {last_exc}")

        if self.task == "deadly_corridor":
            import gymnasium as gym
            import vizdoom.gymnasium_wrapper  # noqa: F401

            attempts = [
                ("VizdoomDeadlyCorridor-MultiBinary-v1", {}),
                ("VizdoomDeadlyCorridor-MultiBinary-v0", {}),
                ("VizdoomDeadlyCorridor-v1", {"max_buttons_pressed": 0}),
                ("VizdoomDeadlyCorridor-v0", {"max_buttons_pressed": 0}),
            ]
            last_exc = None
            for env_id, kwargs in attempts:
                try:
                    return gym.make(
                        env_id,
                        render_mode="rgb_array",
                        frame_skip=self.frameskip,
                        **kwargs,
                    )
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Failed to create Deadly Corridor env: {last_exc}")

        raise ValueError(f"Unsupported task: {self.task}")

    def _uses_native_frameskip(self) -> bool:
        return self.task in {"demon_attack", "deadly_corridor"}

    def _decode_action(self, raw_action: np.ndarray, runtime_button_order: Optional[List[str]] = None):
        if self.task == "flappy":
            return decode_discrete_argmax(raw_action, 2)
        if self.task == "demon_attack":
            return decode_discrete_argmax(raw_action, 6)
        if self.task == "deadly_corridor":
            layout = str(getattr(self.env_eval_cfg.deadly, "action_layout", "multibinary_7"))
            if layout == "multibinary_7":
                semantic = decode_deadly_multibinary_7(raw_action)
            elif layout == "factorized_11":
                semantic = _factorized_to_semantic_buttons(decode_deadly_factorized_11(raw_action))
            else:
                raise ValueError(f"Unknown deadly action layout: {layout}")
            if runtime_button_order is None:
                return semantic
            return _semantic_to_runtime_multibinary(semantic, runtime_button_order)
        return decode_discrete_argmax(raw_action, raw_action.shape[-1])

    def run_latency(
        self,
        model,
        latency: int,
        prompt: str,
        max_steps: int,
        num_episodes: int,
        progress_desc: str | None = None,
        progress_position: int = 0,
    ) -> Dict[str, Any]:
        env = self._make_env()
        runtime_button_order = _get_available_button_names(env) if self.task == "deadly_corridor" else None
        queue_default = [0] * 7 if self.task == "deadly_corridor" else 0
        queue = ActionLatencyQueue(latency=latency, default_action=queue_default)

        rewards: List[float] = []
        lengths: List[int] = []
        action_hist = Counter()

        episode_iter = tqdm(
            range(num_episodes),
            desc=progress_desc or f"{self.task} latency={latency}",
            total=num_episodes,
            leave=False,
            position=progress_position,
            dynamic_ncols=True,
        )
        for episode in episode_iter:
            obs, _ = env.reset()
            queue.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                obs_rgb = _resize_rgb(obs, image_size=self.image_size)
                example = {
                    "image": [Image.fromarray(obs_rgb)],
                    "lang": prompt,
                    "state": np.zeros((1, self.state_dim), dtype=np.float32),
                }
                output = model.predict_action(examples=[example])
                actions = output["normalized_actions"]
                if actions.ndim != 3:
                    raise RuntimeError(f"Expected normalized_actions shape [B,T,D], got {actions.shape}")
                raw_action = actions[0, 0, :]
                decoded = self._decode_action(raw_action, runtime_button_order=runtime_button_order)
                effective_action = queue.schedule_and_get(decoded)
                action_hist[str(effective_action)] += 1

                step_reward = 0.0
                terminated = False
                truncated = False
                repeat_count = 1 if self._uses_native_frameskip() else self.frameskip
                for _ in range(repeat_count):
                    obs, reward, terminated, truncated, _ = env.step(effective_action)
                    step_reward += float(reward)
                    if terminated or truncated:
                        break

                total_reward += step_reward
                steps += 1
                done = terminated or truncated

            rewards.append(total_reward)
            lengths.append(steps)
            episode_iter.set_postfix({
                "reward": f"{total_reward:.2f}",
                "steps": steps,
            })

        env.close()
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        return {
            "latency": int(latency),
            "num_episodes": int(num_episodes),
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "decoded_action_hist": dict(action_hist),
        }


class RlGamesEvalRunner:
    def __init__(self, cfg, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir

    def _stage_cfg(self, stage: str):
        env_eval = self.cfg.rl_games.env_eval
        if stage in {"mid_train", "post_train"}:
            return getattr(env_eval, stage, None)
        return None

    def is_enabled(self, stage: str) -> bool:
        env_eval = self.cfg.rl_games.env_eval
        if not bool(getattr(env_eval, "enabled", False)):
            return False
        stage_cfg = self._stage_cfg(stage)
        if stage_cfg is None:
            return True
        return bool(getattr(stage_cfg, "enabled", True))

    def interval_steps(self, default: int) -> int:
        env_eval = self.cfg.rl_games.env_eval
        stage_cfg = self._stage_cfg("mid_train")
        if stage_cfg is not None and getattr(stage_cfg, "interval_steps", None) is not None:
            return int(stage_cfg.interval_steps)
        return int(getattr(env_eval, "interval_steps", default))

    def _get_latency_values(self, stage: str) -> List[int]:
        env_eval = self.cfg.rl_games.env_eval
        stage_cfg = self._stage_cfg(stage)
        values = None
        if stage_cfg is not None:
            values = getattr(stage_cfg, "latencies", None)
        if not values:
            values = getattr(getattr(env_eval, "latency", {}), "values", []) or [0]
        return [int(v) for v in values]

    def _num_episodes(self, stage: str) -> int:
        env_eval = self.cfg.rl_games.env_eval
        stage_cfg = self._stage_cfg(stage)
        if stage_cfg is not None and getattr(stage_cfg, "num_episodes", None) is not None:
            return int(stage_cfg.num_episodes)
        return int(getattr(env_eval, "num_episodes", 5))

    def _max_steps_per_episode(self, stage: str) -> int:
        env_eval = self.cfg.rl_games.env_eval
        stage_cfg = self._stage_cfg(stage)
        if stage_cfg is not None and getattr(stage_cfg, "max_steps_per_episode", None) is not None:
            return int(stage_cfg.max_steps_per_episode)
        return int(getattr(env_eval, "max_episode_steps", 2000))

    def _resolve_prompt(self, latency: int, mapping: Dict[int, Dict[str, Any]]) -> str:
        if latency in mapping:
            return str(mapping[latency]["prompt"])
        prompt = str(getattr(self.cfg.rl_games.env_eval, "task_description", "") or "")
        if prompt:
            return prompt
        task = str(getattr(self.cfg.rl_games, "task", "flappy"))
        if task == "flappy":
            return "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP."
        if task == "demon_attack":
            return "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE."
        if task == "deadly_corridor":
            return (
                "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, "
                "fight enemies, and reach the green armor vest at the far end. The available actions are: "
                "MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK."
            )
        return "Act optimally in the current environment."

    def _get_tasks(self) -> List[str]:
        task = str(getattr(self.cfg.rl_games, "task", "flappy"))
        if task != "cross_task":
            return [task]
        cross_cfg = getattr(self.cfg.rl_games, "cross_task", None)
        if cross_cfg is not None and getattr(cross_cfg, "eval_tasks", None):
            return [str(x) for x in cross_cfg.eval_tasks]
        return ["flappy", "demon_attack"]

    def run(self, model, step: int, stage: str = "mid_train") -> EvalResult:
        latency_values = self._get_latency_values(stage=stage)
        max_steps = self._max_steps_per_episode(stage=stage)
        num_episodes = self._num_episodes(stage=stage)

        latency_map_path = getattr(getattr(self.cfg.rl_games.env_eval, "latency", {}), "prompt_map_path", None)
        latency_prompt_map = _load_latency_prompt_map(latency_map_path)

        tasks = self._get_tasks()
        per_latency: Dict[str, Dict] = {}
        aggregate = {
            "stage": stage,
            "step": int(step),
            "task": str(getattr(self.cfg.rl_games, "task", "flappy")),
            "model_alias": str(getattr(self.cfg.rl_games, "model_alias", "openvla")),
            "total_episodes": 0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "task_count": len(tasks),
        }

        all_rewards = []
        all_lengths = []

        total_rollouts = len(tasks) * len(latency_values)
        rollout_iter = tqdm(
            [(task_name, latency) for task_name in tasks for latency in latency_values],
            desc=f"rl-games {stage} eval step={step}",
            total=total_rollouts,
            leave=True,
            position=0,
            dynamic_ncols=True,
        )

        for task_name, latency in rollout_iter:
            rollout_iter.set_postfix({"task": task_name, "latency": latency})
            task_eval = _TaskEvaluator(task=task_name, cfg=self.cfg)
            prompt = self._resolve_prompt(latency=latency, mapping=latency_prompt_map)
            metrics = task_eval.run_latency(
                model=model,
                latency=latency,
                prompt=prompt,
                max_steps=max_steps,
                num_episodes=num_episodes,
                progress_desc=f"{stage} {task_name} latency={latency}",
                progress_position=1,
            )
            key = f"{task_name}/latency_{latency}"
            per_latency[key] = metrics
            aggregate["total_episodes"] += metrics["num_episodes"]
            all_rewards.append(metrics["mean_reward"])
            all_lengths.append(metrics["mean_length"])

        if all_rewards:
            aggregate["mean_reward"] = float(np.mean(all_rewards))
        if all_lengths:
            aggregate["mean_length"] = float(np.mean(all_lengths))

        result = EvalResult(per_latency=per_latency, aggregate=aggregate)
        self._save(result=result, step=step, stage=stage)
        return result

    def _save(self, result: EvalResult, step: int, stage: str) -> None:
        eval_dir = os.path.join(self.output_dir, "eval", stage)
        os.makedirs(eval_dir, exist_ok=True)
        payload = {
            "per_latency": result.per_latency,
            "aggregate": result.aggregate,
        }
        with open(os.path.join(eval_dir, f"step_{step}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
