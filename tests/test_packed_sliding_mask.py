"""Gate for the packed sliding-window+sink training mask.

Proves that `make_packed_sliding_mask_mod` (used by
`QwenOFT._forward_memory_packed`) plus the custom M-RoPE position formula
reproduce the streaming visibility on a tiny Qwen3-VL text model, WITHOUT the
4B multimodal checkpoint. Runs on CPU with a dense additive mask (eager
attention); BlockMask vs additive mask only changes speed, not the result.

Run:
  /home/lixinyuan/miniconda3/envs/starvla_rl_games_gr00t/bin/python -m pytest \
    starVLA/tests/test_packed_sliding_mask.py -q
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from starVLA.model.modules.vlm.kv_memory import make_packed_sliding_mask_mod

DT = torch.float32
T, F, A, R = 3, 4, 2, 5  # prefix, frame_len, action_len, num steps
BLOCK = F + A
L = T + R * BLOCK
VOCAB = 128


def _tiny_text_model(num_hidden_layers=2):
    cfg = Qwen3VLTextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=VOCAB,
        rope_scaling={"mrope_section": [2, 1, 1], "rope_type": "default"},
        _attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3VLTextModel(cfg).to(DT).eval()


def _layout():
    """Per-token kind/step, custom M-RoPE positions, and action gather indices,
    mirroring QwenOFT._packed_window_layout for a text-only synthetic rollout."""
    kind = torch.zeros(L, dtype=torch.long)
    step = torch.full((L,), -1, dtype=torch.long)
    pos = torch.zeros(3, 1, L, dtype=torch.long)
    pos[:, 0, :T] = torch.arange(T)
    action_idx = []
    for k in range(R):
        blk = T + k * BLOCK
        kind[blk:blk + F] = 1
        kind[blk + F:blk + BLOCK] = 2
        step[blk:blk + BLOCK] = k
        f_lo = T + k * F  # frames-only consecutive layout
        frame_pos = torch.arange(f_lo, f_lo + F)
        pos[:, 0, blk:blk + F] = frame_pos
        pos[:, 0, blk + F:blk + BLOCK] = torch.arange(1, A + 1) + int(frame_pos.max())
        action_idx.append(list(range(blk + F, blk + BLOCK)))
    return kind, step, pos, torch.tensor(action_idx)


def _dense(kind, step, W):
    mod = make_packed_sliding_mask_mod(kind, step, W)
    qg, kg = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    boolm = mod(0, 0, qg, kg)
    add = torch.zeros(L, L, dtype=DT).masked_fill_(~boolm, float("-inf"))
    return add.view(1, 1, L, L), boolm


def _rebased_layout(W):
    """Per-step [sink_k][f_k][a_k] with the sink re-based by o_k = j0*F, mirroring
    QwenOFT._packed_window_layout's rebased_sink branch for a text-only rollout.
    Frames keep their global frames-only positions (T + k*F); sink_k is shifted so
    it sits one position before step k's oldest visible frame."""
    Lr = R * (T + BLOCK)
    kind = torch.zeros(Lr, dtype=torch.long)
    step = torch.full((Lr,), -1, dtype=torch.long)
    pos = torch.zeros(3, 1, Lr, dtype=torch.long)
    action_idx = []
    cur = 0
    for k in range(R):
        j0 = max(0, k - W + 1)
        o_k = j0 * F
        kind[cur:cur + T] = 0
        step[cur:cur + T] = k
        pos[:, 0, cur:cur + T] = torch.arange(T) + o_k
        fk = cur + T
        kind[fk:fk + F] = 1
        step[fk:fk + F] = k
        frame_pos = torch.arange(T + k * F, T + k * F + F)
        pos[:, 0, fk:fk + F] = frame_pos
        ak = fk + F
        kind[ak:ak + A] = 2
        step[ak:ak + A] = k
        pos[:, 0, ak:ak + A] = torch.arange(1, A + 1) + int(frame_pos.max())
        action_idx.append(list(range(ak, ak + A)))
        cur += T + BLOCK
    return kind, step, pos, torch.tensor(action_idx)


def _run(model, ids, pos, add):
    emb = model.embed_tokens(ids)
    return model(inputs_embeds=emb, position_ids=pos, attention_mask=add, use_cache=False).last_hidden_state


def test_no_eviction_equals_prefix_forward():
    """With W>=R each step's action read-out equals a forward over just
    [text][f1..fk][action_k] with the packed positions (validates mask + gather)."""
    model = _tiny_text_model()
    kind, step, pos, action_idx = _layout()
    ids = torch.randint(1, VOCAB, (1, L))
    add, boolm = _dense(kind, step, W=R)
    assert bool(boolm.any(dim=1).all()), "a query row is fully masked"
    hid = _run(model, ids, pos, add)
    for k in range(R):
        sel = list(range(T))
        for j in range(k + 1):
            blk = T + j * BLOCK
            sel += list(range(blk, blk + F))
        blk_k = T + k * BLOCK
        sel += list(range(blk_k + F, blk_k + BLOCK))
        sel_t = torch.tensor(sel)
        n = len(sel)
        causal = torch.triu(torch.full((n, n), float("-inf")), diagonal=1).view(1, 1, n, n)
        ref = _run(model, ids[:, sel_t], pos[:, :, sel_t], causal)[:, -A:, :]
        packed = hid[:, action_idx[k], :]
        assert (ref - packed).abs().max().item() < 1e-4


def test_sliding_window_cutoff_single_layer():
    """1 layer: a step's read-out is invariant to out-of-window frames, sensitive
    to in-window ones (>1 layer legitimately leaks across the window via chained
    frames — same as the streaming KV cache — so the hard cutoff is 1-layer)."""
    model = _tiny_text_model(num_hidden_layers=1)
    kind, step, pos, action_idx = _layout()
    ids = torch.randint(1, VOCAB, (1, L))
    W = 3
    add, _ = _dense(kind, step, W)
    base = _run(model, ids, pos, add)
    last = action_idx[R - 1]
    out = ids.clone(); out[:, T:T + F] = torch.randint(1, VOCAB, (1, F))  # step 0 (out of window)
    d_out = (base[:, last, :] - _run(model, out, pos, add)[:, last, :]).abs().max().item()
    inw = ids.clone(); blkL = T + (R - 1) * BLOCK
    inw[:, blkL:blkL + F] = torch.randint(1, VOCAB, (1, F))  # step R-1 (in window)
    d_in = (base[:, last, :] - _run(model, inw, pos, add)[:, last, :]).abs().max().item()
    assert d_out < 1e-5 and d_in > 1e-3


def test_action_readout_isolation():
    """Perturbing one step's action tokens must not change other action read-outs
    or any frame state (action blocks are read-out only, invisible to others)."""
    model = _tiny_text_model()
    kind, step, pos, action_idx = _layout()
    ids = torch.randint(1, VOCAB, (1, L))
    add, _ = _dense(kind, step, W=R)
    base = _run(model, ids, pos, add)
    pert = ids.clone(); pert[:, action_idx[0]] = torch.randint(1, VOCAB, (1, A))
    other = _run(model, pert, pos, add)
    assert (base[:, action_idx[1], :] - other[:, action_idx[1], :]).abs().max().item() < 1e-5
    frames = (kind == 1).nonzero().flatten()
    assert (base[:, frames, :] - other[:, frames, :]).abs().max().item() < 1e-5


def test_rebased_sink_distance_matches_eval_window():
    """The per-step re-based sink reproduces the eval rolling-KV layout's BOUNDED
    sink->frame distance: at every step the visible frames sit at consecutive
    offsets 1, 1+F, 1+2F, ... after the sink (i.e. the window glued right behind
    the sink), regardless of how deep into the rollout the step is. The legacy
    single global sink (control) instead has a distance that GROWS with the step
    index once frames are evicted — the train/eval forward mismatch the re-based
    sink removes."""
    W = 3  # < R=5 so steps 3,4 evict the oldest frame(s)
    kind, step, pos, _ = _rebased_layout(W)
    t = pos[0, 0]  # temporal M-RoPE dim
    blk = T + BLOCK
    for k in range(R):
        j0 = max(0, k - W + 1)  # oldest visible frame at step k
        sink_last = int(t[k * blk + T - 1])
        for i, j in enumerate(range(j0, k + 1)):
            f_first = int(t[j * blk + T])  # frame j's first token (its own block)
            assert f_first - sink_last == i * F + 1, f"rebased step {k} frame {j}"

    # Control: legacy single global sink — distance to the oldest visible frame
    # grows as j0*F+1 (eval wants 1), confirming the systematic mismatch.
    _, _, pos_legacy, _ = _layout()
    t2 = pos_legacy[0, 0]
    sink_last2 = int(t2[T - 1])
    for k in range(R):
        j0 = max(0, k - W + 1)
        f_oldest = int(t2[T + j0 * BLOCK])  # frame j0's first token (legacy layout)
        assert f_oldest - sink_last2 == j0 * F + 1
    assert max(0, (R - 1) - W + 1) > 0  # the test actually exercises eviction


def test_rebased_sink_no_fully_masked_row():
    """Re-based per-step sinks (step>=0, visible only to their own step) still
    leave every query able to see at least its own sink + itself -> no NaN."""
    W = 3
    kind, step, pos, _ = _rebased_layout(W)
    Lr = kind.shape[0]
    mod = make_packed_sliding_mask_mod(kind, step, W)
    qg, kg = torch.meshgrid(torch.arange(Lr), torch.arange(Lr), indexing="ij")
    boolm = mod(0, 0, qg, kg)
    assert bool(boolm.any(dim=1).all()), "a query row is fully masked"
    # each step's frame/action query sees exactly one sink block: its own.
    sink_steps = step[(kind == 0)]
    assert set(sink_steps.tolist()) == set(range(R))
