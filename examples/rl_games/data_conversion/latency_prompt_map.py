from __future__ import annotations

from typing import Any, Iterable


def build_latency_prompt_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_latency: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "split" in row and str(row["split"]).lower() != "train":
            continue
        if "latency_raw_frames" not in row or "prompt" not in row:
            raise KeyError(f"row is missing latency_raw_frames/prompt columns; available columns: {sorted(row.keys())}")
        latency_raw_frames = int(row["latency_raw_frames"])
        prompt = str(row["prompt"])
        latency_ms = row.get("latency_ms")
        current = by_latency.get(latency_raw_frames)
        if current is None:
            by_latency[latency_raw_frames] = {
                "latency_raw_frames": latency_raw_frames,
                "latency_ms": latency_ms,
                "prompt": prompt,
            }
            continue
        if current["prompt"] != prompt or current.get("latency_ms") != latency_ms:
            raise ValueError(f"inconsistent prompt/latency_ms values for latency_raw_frames={latency_raw_frames}")
    return {str(k): by_latency[k] for k in sorted(by_latency)}
