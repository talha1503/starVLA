#!/usr/bin/env python
"""Inspect a trained StarVLA VLA: action-head decision + a VLM reasoning trace.

For N sampled frames from a rl_games dataset (default: flappy_200ep), this:
  1. runs the **action head** (the policy's *real* decision), and
  2. generates a free-form **reasoning trace** from the same VLM backbone.

Output is a CSV with one row per sample:
    image_frame_path, true_label, action_from_vlm, reasoning_trace
(plus a few extra diagnostic columns: action_id, action_probs, latency_ms).

IMPORTANT — honesty caveat
    The action comes from the action head on the `🔍` token's hidden state. The
    reasoning text is produced by a *separate* `generate()` pass on the same
    backbone, so it is **post-hoc narration**, not a causal chain-of-thought: the
    model was action-SFT'd (small LM-head loss), not trained to reason. Use it to
    *probe* what the VLM can say, not as a faithful explanation of the action.

Run this on a GPU box with the trained checkpoint and the starVLA env installed.
Example:
    python examples/rl_games/scripts/inspect_reasoning_trace.py \
        --ckpt-path /path/to/openvla_flappy_mixed_checkpoint \
        --dataset-subdir flappy_fix_latency_0_200ep \
        --split val --num-samples 20 \
        --output-csv /workspace/outputs/reasoning_trace_flappy.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# action_id -> label, in the order listed in the flappy prompt ("NOOP, FLAP").
ENV_ACTION_LABELS = {
    "flappy": ["NOOP", "FLAP"],
    "demon_attack": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt-path",
        default="",
        help="Local trained StarVLA (QwenOFT) checkpoint dir. Optional if --hf-repo-id is given.",
    )
    p.add_argument("--output-csv", required=True, help="Where to write the results CSV.")
    # --- optional: auto-download the checkpoint from HuggingFace ---
    p.add_argument(
        "--hf-repo-id",
        default="",
        help="HF model repo to download the checkpoint from, e.g. talha15032/openvla_bridge_flappy_latency_mixed_exp2.",
    )
    p.add_argument(
        "--hf-include",
        default="",
        help="Glob pattern of files to download from the repo (like `hf download --include`), "
        "e.g. 'steps_5000_state/**'. Default: download the whole repo.",
    )
    p.add_argument("--hf-revision", default=None, help="Optional repo revision/branch/commit.")
    p.add_argument(
        "--download-dir",
        default="",
        help="Where to download the HF checkpoint. Default: <csv dir>/_ckpt/<repo-name>.",
    )
    p.add_argument(
        "--ckpt-subpath",
        default="",
        help="Subdirectory inside the downloaded repo that is the actual checkpoint dir to load "
        "(e.g. 'steps_5000_state'). Default: repo root.",
    )
    p.add_argument("--hf-token", default=None, help="HF token (else uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN env).")
    # The synced HF checkpoint repo only contains the model state (steps_N_state/),
    # NOT config.yaml / dataset_statistics.json. read_mode_config needs both at the
    # run dir, so supply them (from the training run dir, e.g.
    # results/Checkpoints/<run_id>/config.yaml). The script arranges the layout
    # read_mode_config expects (run_dir/config.yaml + run_dir/checkpoints/steps_N_state).
    p.add_argument(
        "--config-yaml",
        default="",
        help="Path to the run's config.yaml. If omitted for an HF-downloaded ckpt, it is reconstructed.",
    )
    p.add_argument(
        "--dataset-statistics-json",
        default="",
        help="Path to the run's dataset_statistics.json. If omitted, a stub is written (fine for discrete actions).",
    )
    # --- auto-reconstruction of config.yaml / dataset_statistics.json (when not provided) ---
    # The synced HF repo lacks these; if the original run dir is gone we recompose
    # config.yaml from the same Hydra config the run used and stub the stats.
    p.add_argument("--base-vlm", default="", help="Local path to the base Qwen3-VL model (for from_pretrained).")
    p.add_argument(
        "--base-vlm-repo",
        default="StarVLA/Qwen3-VL-4B-Instruct-Action",
        help="HF repo to download the base VLM from when --base-vlm is not given.",
    )
    p.add_argument("--base-vlm-dir", default="", help="Where to download the base VLM. Default: <csv dir>/_base_vlm.")
    p.add_argument("--recon-model", default="openvla", help="Hydra model group for config reconstruction.")
    p.add_argument("--recon-init", default="bridge", help="Hydra init group for config reconstruction.")
    p.add_argument("--recon-mode", default="mixed_latency", help="Hydra mode group for config reconstruction.")
    p.add_argument("--recon-config-name", default="train", help="Hydra config name for reconstruction.")
    p.add_argument("--recon-run-id", default="reconstructed_run", help="Dummy run_id for reconstruction.")
    p.add_argument("--recon-workspace-dir", default="/workspace", help="workspace_dir for reconstruction.")
    p.add_argument(
        "--recon-override",
        action="append",
        default=[],
        help="Extra Hydra override(s) for reconstruction (repeatable), e.g. architecture-affecting launch overrides.",
    )
    p.add_argument("--dataset-name", default="latency-sensitive-bench/flappy_200ep")
    p.add_argument("--dataset-subdir", default="flappy_fix_latency_0_200ep")
    p.add_argument("--split", default="val")
    p.add_argument("--env-name", default="flappy", choices=sorted(ENV_ACTION_LABELS))
    p.add_argument("--num-samples", type=int, default=20, help="Number of frames to sample/run.")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--frames-dir", default="", help="Dir to dump sampled frame PNGs. Default: <csv dir>/frames")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--action-condition",
        action="store_true",
        help="Condition the reasoning text on the action head's chosen action (cleaner, still post-hoc).",
    )
    p.add_argument(
        "--reasoning-prompt-template",
        default=(
            "{instruction}\n\nLook carefully at the current game frame. Think step by step about "
            "what is happening, then state the single best next action and why."
        ),
        help="Template for the free (non action-conditioned) reasoning prompt. {instruction} is filled in.",
    )
    p.add_argument(
        "--action-condition-template",
        default=(
            "{instruction}\n\nThe chosen next action is: {action}. Looking at the current game frame, "
            "explain step by step why this is a good action."
        ),
        help="Template used when --action-condition is set. {instruction} and {action} are filled in.",
    )
    return p.parse_args()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def ensure_base_vlm(args, default_root: Path) -> str:
    """Return a local base-VLM dir, downloading --base-vlm-repo if needed."""
    if args.base_vlm:
        p = Path(args.base_vlm).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--base-vlm path does not exist: {p}")
        return str(p)
    from huggingface_hub import snapshot_download

    name = args.base_vlm_repo.split("/")[-1]
    target = Path(args.base_vlm_dir).expanduser().resolve() if args.base_vlm_dir else default_root / "_base_vlm" / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading base VLM {args.base_vlm_repo} -> {target} ...")
    local = snapshot_download(repo_id=args.base_vlm_repo, repo_type="model", local_dir=str(target), token=args.hf_token)
    return str(Path(local).resolve())


def reconstruct_run_config(run_dir: Path, base_vlm: str, args) -> None:
    """Recompose config.yaml (same Hydra config the run used) + write stats into run_dir."""
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hydra import compose, initialize_config_dir
    from starVLA.training.rl_games import apply_action_spec, apply_model_alias, validate_rl_games_config

    config_dir = repo_root / "examples" / "rl_games" / "config"
    overrides = [
        f"model={args.recon_model}",
        f"env={args.env_name}",
        f"init={args.recon_init}",
        f"mode={args.recon_mode}",
        f"run_id={args.recon_run_id}",
        f"workspace_dir={args.recon_workspace_dir}",
        f"paths.base_model_dir={base_vlm}",
        f"framework.qwenvl.base_vlm={base_vlm}",
        *args.recon_override,
    ]
    print(f"Reconstructing config.yaml via Hydra compose: model={args.recon_model} env={args.env_name} "
          f"init={args.recon_init} mode={args.recon_mode}")
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=args.recon_config_name, overrides=overrides)
    try:
        validate_rl_games_config(cfg)
    except Exception as exc:  # noqa: BLE001 - validation may need run-only fields we don't use
        print(f"[warn] validate_rl_games_config raised (continuing): {exc}")
    apply_model_alias(cfg)
    apply_action_spec(cfg)

    config_path = run_dir / "config.yaml"
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(config=OmegaConf.create(container), f=str(config_path))
    except Exception as exc:  # noqa: BLE001 - fall back to unresolved
        print(f"[warn] could not fully resolve config ({exc}); saving unresolved.")
        OmegaConf.save(config=cfg, f=str(config_path))

    stats_path = run_dir / "dataset_statistics.json"
    if args.dataset_statistics_json:
        shutil.copyfile(Path(args.dataset_statistics_json).expanduser(), stats_path)
    else:
        stats_path.write_text(json.dumps({}), encoding="utf-8")
    print(f"Wrote {config_path} and {stats_path}")


def resolve_ckpt_path(args, default_root: Path) -> str:
    """Return a local checkpoint dir, downloading from HuggingFace if requested.

    If --hf-repo-id is set, snapshot the repo (optionally filtered by --hf-include)
    into --download-dir, then point at --ckpt-subpath inside it. Otherwise use the
    local --ckpt-path as-is.
    """
    if not args.hf_repo_id:
        if not args.ckpt_path:
            raise SystemExit("Provide either --ckpt-path (local) or --hf-repo-id (to download).")
        return str(Path(args.ckpt_path).expanduser().resolve())

    from huggingface_hub import snapshot_download

    repo_name = args.hf_repo_id.split("/")[-1]
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else default_root / "_ckpt" / repo_name
    download_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [args.hf_include] if args.hf_include else None
    print(f"Downloading {args.hf_repo_id} (include={allow_patterns or 'ALL'}) -> {download_dir} ...")
    local_root = snapshot_download(
        repo_id=args.hf_repo_id,
        repo_type="model",
        local_dir=str(download_dir),
        allow_patterns=allow_patterns,
        revision=args.hf_revision,
        token=args.hf_token,
    )
    state_dir = Path(local_root) / args.ckpt_subpath if args.ckpt_subpath else Path(local_root)
    state_dir = state_dir.resolve()
    if not state_dir.exists():
        raise SystemExit(f"Downloaded checkpoint path does not exist: {state_dir} (check --ckpt-subpath).")
    if not any((state_dir / n).exists() for n in ("model.safetensors", "pytorch_model.bin")):
        raise SystemExit(
            f"{state_dir} has no model.safetensors/pytorch_model.bin. "
            "Point --ckpt-subpath at the state dir (e.g. 'steps_5000_state')."
        )

    # The synced repo lacks config.yaml / dataset_statistics.json and is flat, so
    # build the layout read_mode_config expects:
    #   <loadable>/config.yaml, <loadable>/dataset_statistics.json,
    #   <loadable>/checkpoints/<state_dir>/model.safetensors   (parents[1] == <loadable>)
    loadable = (Path(local_root) / "_loadable").resolve()
    (loadable / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.config_yaml:
        # User supplied the real run files -> use them as-is.
        shutil.copyfile(Path(args.config_yaml).expanduser(), loadable / "config.yaml")
        if args.dataset_statistics_json:
            shutil.copyfile(Path(args.dataset_statistics_json).expanduser(), loadable / "dataset_statistics.json")
        else:
            (loadable / "dataset_statistics.json").write_text(json.dumps({}), encoding="utf-8")
    else:
        # No config provided -> auto-reconstruct (download base VLM if needed, recompose config, stub stats).
        print("config.yaml not provided; auto-reconstructing from the launch config ...")
        base_vlm = ensure_base_vlm(args, default_root)
        reconstruct_run_config(loadable, base_vlm, args)

    link = loadable / "checkpoints" / state_dir.name
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(state_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(state_dir, link)
    print(f"Arranged loadable checkpoint at: {link}")
    return str(link)


def load_framework(ckpt_path: str, device: str):
    import torch
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    return model, torch


def pil_from_row(row) -> Image.Image:
    img = row["image"]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def decode_action_head(framework, torch, pil: Image.Image, instruction: str, labels: list[str]):
    """Run the real policy decision (action head) and return (label, id, probs)."""
    example = {"image": [pil], "lang": instruction}
    with torch.inference_mode():
        out = framework.predict_action(examples=[example])
    vec = np.asarray(out["normalized_actions"])[0, 0]  # (D,)
    n = len(labels)
    head = np.asarray(vec[:n], dtype=np.float64)
    action_id = int(np.argmax(head))
    label = labels[action_id] if action_id < n else str(action_id)
    probs = _softmax(head)
    return label, action_id, probs


def generate_reasoning(framework, torch, pil: Image.Image, reasoning_prompt: str, max_new_tokens: int) -> str:
    """Free-form text from the VLM backbone (post-hoc narration)."""
    iface = framework.qwen_vl_interface
    inputs = iface.build_qwenvl_inputs(images=[[pil]], instructions=[reasoning_prompt])
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = iface.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    new_tokens = generated[:, input_len:]
    text = iface.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return " ".join(text.split()).strip()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    labels = ENV_ACTION_LABELS[args.env_name]

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else out_csv.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset_name} [{args.dataset_subdir}] split={args.split} ...")
    ds = load_dataset(
        args.dataset_name,
        data_dir=args.dataset_subdir,
        split=args.split,
        cache_dir=args.cache_dir,
        verification_mode="no_checks",
    )
    n = min(int(args.num_samples), len(ds))
    ds = ds.shuffle(seed=int(args.sample_seed)).select(range(n))
    print(f"Sampled {n} frames.")

    ckpt_path = resolve_ckpt_path(args, default_root=out_csv.parent)
    print(f"Loading model from {ckpt_path} ...")
    framework, torch = load_framework(ckpt_path, args.device)

    rows_out = []
    for i in range(n):
        row = ds[i]
        pil = pil_from_row(row)
        instruction = str(row.get("prompt", ""))
        true_label = str(row.get("action_text", row.get("action_id", "")))

        frame_path = frames_dir / f"sample_{i:04d}.png"
        pil.save(frame_path)

        action_label, action_id, probs = decode_action_head(framework, torch, pil, instruction, labels)

        if args.action_condition:
            reasoning_prompt = args.action_condition_template.format(instruction=instruction, action=action_label)
        else:
            reasoning_prompt = args.reasoning_prompt_template.format(instruction=instruction)
        reasoning = generate_reasoning(framework, torch, pil, reasoning_prompt, args.max_new_tokens)

        rows_out.append(
            {
                "image_frame_path": str(frame_path),
                "true_label": true_label,
                "action_from_vlm": action_label,
                "reasoning_trace": reasoning,
                "action_id": action_id,
                "action_probs": ",".join(f"{labels[j]}={probs[j]:.3f}" for j in range(len(labels))),
                "latency_ms": row.get("latency_ms", ""),
            }
        )
        print(f"[{i + 1}/{n}] true={true_label} pred={action_label}  reasoning[:80]={reasoning[:80]!r}")

    fieldnames = [
        "image_frame_path",
        "true_label",
        "action_from_vlm",
        "reasoning_trace",
        "action_id",
        "action_probs",
        "latency_ms",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    acc = np.mean([r["true_label"] == r["action_from_vlm"] for r in rows_out]) if rows_out else 0.0
    print(f"\nWrote {len(rows_out)} rows -> {out_csv}")
    print(f"Frames -> {frames_dir}")
    print(f"Action-head agreement with dataset label: {acc:.1%}")


if __name__ == "__main__":
    main()
