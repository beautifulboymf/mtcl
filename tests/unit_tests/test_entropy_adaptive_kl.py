# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entropy-adaptive KL + entropy-based chunk selection: the properties the inline loss
in fsdp_actor_worker.py must satisfy.

These re-implement the exact formulas the loss uses (the loss is inline, not callable),
and pin the contracts that make the mechanism real: if entropy-adaptivity silently
collapses to a single fixed direction, or chunk-selection stops being mean-1, the run
looks fine while doing something other than the design.
"""

import torch


def _adaptive_kl(ls, lt, fwd_scale=1.0):
    """Replica of the entropy_adaptive branch. ls/lt: log-probs [..., V]."""
    pt = lt.exp()
    ps = ls.exp()
    kl_rev = (ps * (ls - lt)).sum(dim=-1)
    kl_fwd = (pt * (lt - ls)).sum(dim=-1)
    t_ent = -(pt * lt).sum(dim=-1)
    ent_max = torch.log(torch.tensor(float(pt.shape[-1])))
    w = (t_ent / ent_max).clamp(0.0, 1.0)
    return kl_rev + w * fwd_scale * kl_fwd, kl_rev, kl_fwd, w


def _logsoftmax(logits):
    return torch.log_softmax(logits, dim=-1)


def test_confident_teacher_is_pure_reverse():
    # teacher near one-hot (entropy ~0) -> w ~0 -> kl ~ reverse only.
    V = 256
    lt = _logsoftmax(torch.tensor([[[20.0] + [0.0] * (V - 1)]]))  # very peaked
    ls = _logsoftmax(torch.randn(1, 1, V))
    kl, kl_rev, kl_fwd, w = _adaptive_kl(ls, lt)
    assert w.item() < 0.05, f"peaked teacher should give w~0, got {w.item()}"
    assert torch.allclose(kl, kl_rev, atol=1e-3), "confident teacher -> reverse only"


def test_unsure_teacher_adds_forward():
    # teacher uniform (max entropy) -> w ~1 -> kl ~ reverse + fwd_scale*forward.
    V = 256
    lt = _logsoftmax(torch.zeros(1, 1, V))  # uniform -> entropy = log V
    ls = _logsoftmax(torch.randn(1, 1, V))
    kl, kl_rev, kl_fwd, w = _adaptive_kl(ls, lt, fwd_scale=1.0)
    assert w.item() > 0.98, f"uniform teacher should give w~1, got {w.item()}"
    assert torch.allclose(kl, kl_rev + kl_fwd, atol=1e-3), "unsure teacher -> reverse + forward"


def test_fwd_scale_zero_is_pure_reverse_everywhere():
    V = 64
    lt = _logsoftmax(torch.randn(2, 3, V))
    ls = _logsoftmax(torch.randn(2, 3, V))
    kl, kl_rev, _, _ = _adaptive_kl(ls, lt, fwd_scale=0.0)
    assert torch.allclose(kl, kl_rev, atol=1e-5)


def test_weight_monotone_in_teacher_entropy():
    # The design invariant is that the FORWARD MIXING WEIGHT w rises with teacher
    # entropy (w=0 confident -> reverse only; w=1 unsure -> +forward). The product
    # w*kl_fwd is NOT guaranteed monotone (a flatter teacher has smaller forward KL),
    # so we pin w itself, which is what the switch is keyed on.
    V = 128
    base = torch.linspace(0, 6, V).reshape(1, 1, V)
    ls = _logsoftmax(torch.randn(1, 1, V))
    ws = []
    for temp in [0.02, 1.0, 5.0, 1e6]:  # rising teacher entropy (flatter)
        lt = _logsoftmax(base / temp)
        _, _, _, w = _adaptive_kl(ls, lt)
        ws.append(w.item())
    assert ws == sorted(ws), f"forward-mixing weight must rise with teacher entropy: {ws}"
    assert ws[0] < 0.2 and ws[-1] > 0.98, f"endpoints should span ~[0,1]: {ws}"


def test_grad_flows_to_student_only():
    V = 32
    lt = _logsoftmax(torch.randn(1, 2, V))
    slog = torch.randn(1, 2, V, requires_grad=True)
    ls = _logsoftmax(slog)
    kl, _, _, _ = _adaptive_kl(ls, lt.detach())
    kl.sum().backward()
    assert slog.grad is not None and slog.grad.abs().sum() > 0


def _tip_score(ls, lt, sad):
    """Replica of the TIP soft-OR score (arXiv:2604.14084 Eq.2-5) aggregated to per-chunk.

    ls/lt: [B, chunks*sad, V]. Returns per-chunk score s in [0,1].
    """
    kl_tok = (ls.exp() * (ls - lt)).sum(dim=-1)  # d = KL(student||teacher), Eq.3
    psd = ls.exp()
    s_ent = -(psd * ls).sum(dim=-1)
    emax = torch.log(torch.tensor(float(ls.shape[-1])))
    h = s_ent / emax  # Eq.2, student entropy normalized
    d = kl_tok
    h = (h - h.min()) / (h.max() - h.min()).clamp_min(1e-6)  # per-batch min-max
    d = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
    B = kl_tok.shape[0]
    h_c = h.reshape(B, -1, sad).mean(dim=-1)
    d_c = d.reshape(B, -1, sad).mean(dim=-1)
    return h_c + d_c - h_c * d_c  # Eq.5 soft-OR


def test_soft_or_is_parameter_free_range():
    V, sad, chunks, B = 64, 7, 5, 3
    ls = _logsoftmax(torch.randn(B, chunks * sad, V))
    lt = _logsoftmax(torch.randn(B, chunks * sad, V))
    s = _tip_score(ls, lt, sad)
    assert (s >= -1e-6).all() and (s <= 1.0 + 1e-6).all(), "soft-OR score must lie in [0,1]"


def test_soft_or_nonzero_when_either_signal_nonzero():
    # confident-but-wrong: h~0 (peaked student) but d>0 (disagrees) -> s must be > 0
    # (the Q3 recovery that an entropy-only score is blind to).
    V, sad = 64, 4
    s0 = torch.full((sad, V), -10.0); s0[:, 0] = 10.0   # student peaked bin 0
    t0 = torch.full((sad, V), -10.0); t0[:, 5] = 10.0   # teacher peaked bin 5 (disagree)
    s1 = torch.full((sad, V), -10.0); s1[:, 0] = 10.0   # student peaked bin 0
    t1 = torch.full((sad, V), -10.0); t1[:, 0] = 10.0   # teacher agrees
    ls = _logsoftmax(torch.cat([s0, s1]).unsqueeze(0))
    lt = _logsoftmax(torch.cat([t0, t1]).unsqueeze(0))
    s = _tip_score(ls, lt, sad)
    assert s[0, 0] > s[0, 1], "confident-but-wrong chunk must score above confident-and-agreeing"
    assert s[0, 0] > 0.0, "confident-but-wrong must be nonzero (Q3 recovery)"


def test_unsure_chunk_scores_high():
    V, sad = 64, 4
    s0 = torch.zeros(sad, V)  # student uniform (unsure)
    t0 = torch.zeros(sad, V)
    s1 = torch.full((sad, V), -10.0); s1[:, 0] = 10.0
    t1 = torch.full((sad, V), -10.0); t1[:, 0] = 10.0
    ls = _logsoftmax(torch.cat([s0, s1]).unsqueeze(0))
    lt = _logsoftmax(torch.cat([t0, t1]).unsqueeze(0))
    s = _tip_score(ls, lt, sad)
    assert s[0, 0] > s[0, 1], "unsure chunk must score above the solved chunk"


def _chunk_weight(ls, lt, sad, kappa=1.0):
    """Our soft reweight w = 1 + kappa*(s/mean s), s = soft-OR (TIP Eq.5). Returns [B,chunks]."""
    s = _tip_score(ls, lt, sad)
    cw = 1.0 + kappa * (s / s.mean().clamp_min(1e-6))
    return cw / cw.mean().clamp_min(1e-6)  # mean-1


def test_chunk_weight_is_mean_one():
    V, sad, chunks, B = 64, 7, 5, 3
    ls = _logsoftmax(torch.randn(B, chunks * sad, V))
    lt = _logsoftmax(torch.randn(B, chunks * sad, V))
    cw = _chunk_weight(ls, lt, sad)
    assert abs(cw.mean().item() - 1.0) < 1e-4, "chunk weights must be mean-1 (no loss-scale drift)"


def test_kappa_zero_is_uniform():
    V, sad, chunks, B = 32, 3, 4, 2
    ls = _logsoftmax(torch.randn(B, chunks * sad, V))
    lt = _logsoftmax(torch.randn(B, chunks * sad, V))
    cw = _chunk_weight(ls, lt, sad, kappa=0.0)
    assert torch.allclose(cw, torch.ones_like(cw), atol=1e-5), "kappa=0 -> uniform (ablation off)"


def test_important_chunk_gets_more_weight_no_token_dropped():
    # confident-but-wrong chunk must get MORE weight than solved chunk, and NO weight
    # is exactly zero (our design reweights, never drops -- unlike TIP's TopK).
    V, sad = 64, 4
    s0 = torch.full((sad, V), -10.0); s0[:, 0] = 10.0   # student confident bin 0
    t0 = torch.full((sad, V), -10.0); t0[:, 5] = 10.0   # teacher bin 5 (disagree)
    s1 = torch.full((sad, V), -10.0); s1[:, 0] = 10.0   # solved: confident + agrees
    t1 = torch.full((sad, V), -10.0); t1[:, 0] = 10.0
    ls = _logsoftmax(torch.cat([s0, s1]).unsqueeze(0))
    lt = _logsoftmax(torch.cat([t0, t1]).unsqueeze(0))
    cw = _chunk_weight(ls, lt, sad, kappa=1.0)
    assert cw[0, 0] > cw[0, 1], "confident-but-wrong chunk must be up-weighted vs solved"
    assert (cw > 0).all(), "no chunk is dropped (soft reweight, not TopK)"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
