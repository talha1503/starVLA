from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_flash_attention_patch_removes_mrope_position_ids_only_for_fa2():
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    from starVLA.model.modules.vlm.QWen3 import _patch_qwen3vl_flash_attention_position_ids

    calls = []
    cls = qwen3_vl.Qwen3VLTextAttention
    original_forward = cls.forward
    try:
        original_patch_marker = cls._starvla_flash_attention_position_ids_patched
        had_patch_marker = True
    except AttributeError:
        had_patch_marker = False

    def fake_forward(self, *args, **kwargs):
        calls.append(kwargs)
        return "ok"

    cls.forward = fake_forward
    if had_patch_marker:
        del cls._starvla_flash_attention_position_ids_patched

    try:
        _patch_qwen3vl_flash_attention_position_ids()

        fa2_attention = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2")
        )
        cls.forward(fa2_attention, position_ids="mrope-position-ids", other="kept")
        assert "position_ids" not in calls[-1]
        assert calls[-1]["other"] == "kept"

        sdpa_attention = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        cls.forward(sdpa_attention, position_ids="mrope-position-ids", other="kept")
        assert calls[-1]["position_ids"] == "mrope-position-ids"
        assert calls[-1]["other"] == "kept"
    finally:
        cls.forward = original_forward
        if had_patch_marker:
            cls._starvla_flash_attention_position_ids_patched = original_patch_marker
        elif hasattr(cls, "_starvla_flash_attention_position_ids_patched"):
            del cls._starvla_flash_attention_position_ids_patched


if __name__ == "__main__":
    test_flash_attention_patch_removes_mrope_position_ids_only_for_fa2()
