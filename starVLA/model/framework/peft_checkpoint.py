from __future__ import annotations

from pathlib import Path

import torch


LORA_ADAPTER_CHECKPOINT_SUFFIX = "_lora_adapter"
ACTION_MODEL_PT = "action_model.pt"
ACTION_MODEL_SAFE = "action_model.safetensors"


def _cfg_get(node, key: str, default=None):
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def is_lora_enabled(cfg) -> bool:
    framework = _cfg_get(cfg, "framework", {})
    qwenvl = _cfg_get(framework, "qwenvl", {})
    lora = _cfg_get(qwenvl, "lora", {})
    return bool(_cfg_get(lora, "enabled", False))


def lora_adapter_checkpoint_path(checkpoint_base_path: str | Path) -> Path:
    return Path(f"{checkpoint_base_path}{LORA_ADAPTER_CHECKPOINT_SUFFIX}")


def is_lora_adapter_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _strip_state_dict_prefix(state_dict, prefix: str):
    prefix_len = len(prefix)
    return {name[prefix_len:]: tensor.detach().cpu() for name, tensor in state_dict.items() if name.startswith(prefix)}


def save_lora_adapter_checkpoint(model, output_dir: str | Path, save_format: str, model_state_dict=None) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_model = model.qwen_vl_interface.model
    peft_state = None
    if model_state_dict is not None:
        peft_state = _strip_state_dict_prefix(model_state_dict, "qwen_vl_interface.model.")
    peft_model.save_pretrained(str(output_dir), safe_serialization=True, state_dict=peft_state)

    if model_state_dict is not None:
        action_state = _strip_state_dict_prefix(model_state_dict, "action_model.")
    else:
        action_state = {name: tensor.detach().cpu() for name, tensor in model.action_model.state_dict().items()}
    if save_format == "safetensors":
        from safetensors.torch import save_file

        save_file(action_state, str(output_dir / ACTION_MODEL_SAFE))
    elif save_format == "pt":
        torch.save(action_state, output_dir / ACTION_MODEL_PT)
    else:
        raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
    return str(output_dir)


def _load_adapter_state(checkpoint_dir: Path):
    adapter_safe = checkpoint_dir / "adapter_model.safetensors"
    if adapter_safe.exists():
        from safetensors.torch import load_file

        return load_file(str(adapter_safe))
    return torch.load(checkpoint_dir / "adapter_model.bin", map_location="cpu")


def _load_action_state(checkpoint_dir: Path):
    action_safe = checkpoint_dir / ACTION_MODEL_SAFE
    if action_safe.exists():
        from safetensors.torch import load_file

        return load_file(str(action_safe))
    return torch.load(checkpoint_dir / ACTION_MODEL_PT, map_location="cpu")


def load_lora_adapter_checkpoint(model, checkpoint_dir: str | Path):
    from peft import set_peft_model_state_dict

    checkpoint_dir = Path(checkpoint_dir)
    peft_model = model.qwen_vl_interface.model
    set_peft_model_state_dict(peft_model, _load_adapter_state(checkpoint_dir), adapter_name="default")
    model.action_model.load_state_dict(_load_action_state(checkpoint_dir))
    return model
