"""RL-games data registry for StarVLA LeRobot datasets."""

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class FlappyDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.image"]
    state_keys = ["state.game_state"]
    action_keys = ["action.button"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = [0]

    def modality_config(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
        ])


class DemonAttackDataConfig(FlappyDataConfig):
    pass


class DeadlyCorridorDataConfig(FlappyDataConfig):
    pass


ROBOT_TYPE_CONFIG_MAP = {
    "rl_games_flappy": FlappyDataConfig(),
    "rl_games_demon_attack": DemonAttackDataConfig(),
    "rl_games_deadly_corridor": DeadlyCorridorDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "rl_games_flappy": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_demon_attack": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_deadly_corridor": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "flappy_train": [("flappy_train", 1.0, "rl_games_flappy")],
    "flappy_mixed_latency_train": [("flappy_mixed_latency_train", 1.0, "rl_games_flappy")],
    "demon_attack_train": [("demon_attack_train", 1.0, "rl_games_demon_attack")],
    "demon_attack_mixed_latency_train": [("demon_attack_mixed_latency_train", 1.0, "rl_games_demon_attack")],
    "deadly_corridor_train": [("deadly_corridor_train", 1.0, "rl_games_deadly_corridor")],
    "deadly_corridor_mixed_latency_train": [
        ("deadly_corridor_mixed_latency_train", 1.0, "rl_games_deadly_corridor")
    ],
}
