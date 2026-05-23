#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from examples.rl_games.data_conversion.latency_prompt_map import build_latency_prompt_map
from examples.rl_games.data_conversion.verify_flappy_dataset import _load_train_split


EXPECTED_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
]
REQUIRED_PROMPT_PARTS = ["Deadly Corridor", *EXPECTED_ACTIONS]


def _load_first_available(
    dataset_name: str,
    cache_dir: str | None,
    column_options: tuple[list[str] | None, ...],
):
    last_exc: Exception | None = None
    for columns in column_options:
        try:
            return _load_train_split(dataset_name, cache_dir, columns=columns)
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no column options provided")


def _action_vector(row: dict[str, Any]) -> list[float]:
    raw_action = row.get("action", row.get("actions"))
    if raw_action is not None:
        values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
        if len(values) != len(EXPECTED_ACTIONS):
            raise ValueError(f"Deadly Corridor action must have {len(EXPECTED_ACTIONS)} values, got {len(values)}")
        if any(value not in (0.0, 1.0) for value in values):
            raise ValueError(f"Deadly Corridor action values must be binary, got {values}")
        return values

    if "action_text" in row and row.get("action_text") is not None:
        text = str(row["action_text"]).upper()
        return [1.0 if label in text else 0.0 for label in EXPECTED_ACTIONS]

    raise ValueError("Deadly Corridor rows must include either multibinary `action` or `action_text`")


def verify_dataset(
    dataset_name: str,
    *,
    rows: int = 200,
    cache_dir: str | None = None,
    strict: bool = False,
    allow_mixed_latency_prompts: bool = False,
) -> bool:
    try:
        ds = _load_first_available(
            dataset_name,
            cache_dir,
            (
                ["prompt", "action_text", "latency_raw_frames", "latency_ms"],
                ["prompt", "action", "latency_raw_frames", "latency_ms"],
                ["prompt", "actions", "latency_raw_frames", "latency_ms"],
                ["prompt", "action_text"],
                ["prompt", "action"],
                ["prompt", "actions"],
                None,
            ),
        )
    except Exception as exc:
        print(f"ERROR: could not load dataset {dataset_name}: {exc}")
        if strict:
            raise
        return False

    if len(ds) == 0:
        print("ERROR: dataset has zero train rows.")
        if strict:
            raise ValueError("dataset has zero train rows")
        return False

    sample_n = min(rows, len(ds))
    ok = True

    prompts = {str(ds[i]["prompt"]) for i in range(sample_n)}
    if allow_mixed_latency_prompts:
        try:
            prompt_ds = _load_first_available(
                dataset_name,
                cache_dir,
                (
                    ["split", "prompt", "latency_raw_frames", "latency_ms"],
                    ["prompt", "latency_raw_frames", "latency_ms"],
                    None,
                ),
            )
            mapping = build_latency_prompt_map(prompt_ds)
            if len(mapping) <= 1:
                raise ValueError(f"expected more than one latency prompt, got {len(mapping)}")
            print("Latency prompt map:")
            print(json.dumps(mapping, indent=2))
        except Exception as exc:
            print(f"ERROR: invalid mixed-latency prompt mapping: {exc}")
            ok = False
    else:
        invalid_prompts = [
            prompt for prompt in prompts
            if not all(part in prompt for part in REQUIRED_PROMPT_PARTS)
        ]
        if invalid_prompts:
            print("ERROR: prompt does not contain the expected Deadly Corridor action vocabulary.")
            print(f"  sampled prompts: {sorted(prompts)}")
            print(f"  required parts: {REQUIRED_PROMPT_PARTS}")
            ok = False

    seen_active = set()
    for i in range(sample_n):
        try:
            action = _action_vector(ds[i])
        except Exception as exc:
            print(f"ERROR: invalid action at row {i}: {exc}")
            ok = False
            break
        seen_active.update(idx for idx, value in enumerate(action) if value)

    if not seen_active:
        print("WARNING: sampled rows contain only no-op Deadly Corridor actions.")

    if ok:
        print(f"Deadly Corridor dataset verification passed for {dataset_name} ({sample_n} sampled rows).")
    elif strict:
        raise ValueError("Deadly Corridor dataset verification failed")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-mixed-latency-prompts", "--allow_mixed_latency_prompts", action="store_true")
    args = parser.parse_args()

    try:
        ok = verify_dataset(
            args.dataset_name,
            rows=args.rows,
            cache_dir=args.cache_dir,
            strict=args.strict,
            allow_mixed_latency_prompts=args.allow_mixed_latency_prompts,
        )
    except Exception:
        if args.strict:
            raise
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
