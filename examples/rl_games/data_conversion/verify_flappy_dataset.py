#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ACTIONS = ["NOOP", "FLAP"]
EXPECTED_PROMPT = (
    "You are playing Flappy Bird. "
    "Pass through the pipe gaps and stay alive. "
    "Choose the action: NOOP, FLAP."
)


def _local_parquet_files(dataset_name: str) -> list[str] | None:
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.exists():
        return None
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"dataset_name={dataset_name!r} exists but is not a parquet file")
        return [str(dataset_path)]

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"dataset_name={dataset_name!r} exists but contains no parquet files")

    train_files = [
        parquet_file
        for parquet_file in parquet_files
        if any("train" in part.lower() for part in parquet_file.relative_to(dataset_path).parts)
    ]
    return [str(parquet_file) for parquet_file in (train_files or parquet_files)]


def _load_train_split(dataset_name: str, cache_dir: str | None, columns: list[str] | None = None):
    from datasets import load_dataset

    def _filter_internal_split(ds):
        if "split" in ds.column_names:
            return ds.filter(lambda row: str(row["split"]).lower() == "train")
        return ds

    local_files = _local_parquet_files(dataset_name)
    if local_files is not None:
        load_columns = list(columns) if columns is not None else None
        if load_columns is not None and "split" not in load_columns:
            load_columns.append("split")
        try:
            ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=load_columns)
        except (ValueError, KeyError):
            if columns is None:
                raise
            ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=columns)
        return _filter_internal_split(ds)

    try:
        ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=columns)
        return _filter_internal_split(ds)
    except (ValueError, KeyError):
        load_columns = list(columns or [])
        if "split" not in load_columns:
            load_columns.append("split")
        ds_all = load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=load_columns)
        return ds_all.filter(lambda row: str(row["split"]).lower() == "train")


def build_latency_prompt_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_latency: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "split" in row and str(row["split"]).lower() != "train":
            continue
        if "latency" not in row or "prompt" not in row:
            raise KeyError(f"row is missing latency/prompt columns; available columns: {sorted(row.keys())}")
        latency = int(row["latency"])
        prompt = str(row["prompt"])
        latency_ms = row.get("latency_ms")
        current = by_latency.get(latency)
        if current is None:
            by_latency[latency] = {
                "latency": latency,
                "latency_ms": latency_ms,
                "prompt": prompt,
            }
            continue
        if current["prompt"] != prompt or current.get("latency_ms") != latency_ms:
            raise ValueError(f"inconsistent prompt/latency_ms values for latency={latency}")
    return {str(k): by_latency[k] for k in sorted(by_latency)}


def verify_dataset(
    dataset_name: str,
    *,
    rows: int = 200,
    cache_dir: str | None = None,
    strict: bool = False,
    allow_mixed_latency_prompts: bool = False,
) -> bool:
    try:
        for columns in (
            ["prompt", "action_id", "action_text", "latency", "latency_ms"],
            ["prompt", "action_id", "action_text"],
            ["prompt", "action_id"],
            None,
        ):
            try:
                ds = _load_train_split(dataset_name, cache_dir, columns=columns)
                break
            except Exception:
                if columns is None:
                    raise
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
            mapping = build_latency_prompt_map(ds)
            print("Latency prompt map:")
            print(json.dumps(mapping, indent=2))
        except Exception as exc:
            print(f"ERROR: invalid mixed-latency prompt mapping: {exc}")
            ok = False
    else:
        if len(prompts) != 1 or next(iter(prompts)) != EXPECTED_PROMPT:
            print("ERROR: prompt does not match expected Flappy prompt.")
            print(f"  sampled prompts: {sorted(prompts)}")
            print(f"  expected: {EXPECTED_PROMPT!r}")
            ok = False

    action_id_to_text: dict[int, set[str]] = defaultdict(set)
    seen_ids = set()
    has_action_text = "action_text" in ds.column_names
    for i in range(sample_n):
        action_id = int(ds[i]["action_id"])
        seen_ids.add(action_id)
        if has_action_text:
            action_id_to_text[action_id].add(str(ds[i]["action_text"]))

    if not seen_ids or min(seen_ids) < 0 or max(seen_ids) >= len(EXPECTED_ACTIONS):
        print(f"ERROR: action_id values must be in [0, {len(EXPECTED_ACTIONS) - 1}], saw {sorted(seen_ids)}")
        ok = False

    if has_action_text:
        for action_id, texts in sorted(action_id_to_text.items()):
            expected = EXPECTED_ACTIONS[action_id] if action_id < len(EXPECTED_ACTIONS) else None
            if texts != {expected}:
                print(f"ERROR: action_id={action_id} maps to {sorted(texts)}, expected {expected!r}")
                ok = False

    if ok:
        print(f"Flappy dataset verification passed for {dataset_name} ({sample_n} sampled rows).")
    elif strict:
        raise ValueError("Flappy dataset verification failed")
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
