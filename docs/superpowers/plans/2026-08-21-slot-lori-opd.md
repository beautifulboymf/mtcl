# slot-LoRI joint 4-teacher OPD 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把学生的单份 r=128 LoRA 换成 4 个子空间严格正交的 slot（每 suite 一个），梯度按 suite 硬路由，打破 joint 多 teacher OPD "只搬运不增加" 的守恒。

**Architecture:** 每个被 LoRA 的 `nn.Linear` 换成 `SlotLoRALinear` = frozen base + `SlotProj`（持有自由矩阵 Z，前向里现算 Ā=(ZZᵀ)^(-1/2)Z，正交性进计算图）+ `SlotOut`（持有 B，逐 slot 算贡献后对非归属 slot 整条 detach）。四个 slot 前向永远全开，产物 merge 成单个 HF 模型。

**Tech Stack:** PyTorch 2.6 / FSDP1 (`use_orig_params=False`) / peft 0.11.1（teacher、anchor 仍走 PEFT，学生不走）/ pytest 9.0.3 / RLinf embodied OPD pipeline。

**Spec:** `docs/superpowers/specs/2026-08-21-slot-lori-opd-design.md`

**测试怎么跑：** 全部是纯 CPU torch 单测，不需要 GPU / ray / LIBERO。

```bash
cd /home/fanruochen/CL/RLinf
/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_*.py -v
```

**提交约定：** 作者 beautifulboymf，**不加 Co-Authored-By**。当前分支 `openvla-oft-opd-distill`。**只 commit，不 push**（remote `mtcl` 由用户自己推）。

---

---

## ⚠️ 接线契约（Task 7 落地后，Task 9/13 以此为准）

**actor 取 gate**：

```python
from rlinf.models import find_slot_gate
gate = find_slot_gate(self.model)     # 存在 model._slot_gate，getattr 能穿透 FSDP root
with gate.scoped(ids):                 # ids 用 model._slot_order 作 suite_order 算出
    loss = ...                         # 学生前向
    loss.backward()                    # ★ 必须在同一作用域内
```

`model._slot_gate` 是普通属性（`SlotGate` 不是 `nn.Module`），不进 state dict / FSDP / optimizer。另存 `model._slot_order`（tuple），让路由顺序与 slot 索引顺序是同一个值。

**rollout worker 也会被注入 slot**：`huggingface_worker.py:95` 对 `cfg.actor.model` 做 deepcopy 后走同一个 `get_model`，所以它拿到自己的 **strict gate**，而生成时没有路由 → `gate.current()` 会 raise。这是 strict 的正确行为，处理办法是把生成包进 `gate.ungated()`（rollout 本来就该跑全 slot 合并后的策略，不做路由）。

**`actor.model.slot_lora` 的确切键**（Task 13 的 YAML 按此写）：

| key | 必填 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | — | `false` | 假值走原 PEFT 路径 |
| `slot_ranks` | **是** | — | `{suite_name: rank}` 映射；给列表会被拒 |
| `slot_order` | **是** | — | 固定 slot **索引**顺序的 suite 名列表；给字符串会被拒 |
| `a_scale_mode` | 否 | `match_mt4` | 另可 `unit`，其余值 raise |
| `a_scale_ref_rank` | 否 | `128` | |
| `orth_eps` | 否 | `1e-6` | |

rank 列表按 `[slot_ranks[s] for s in slot_order]` 取，**绝不依赖映射的迭代顺序**。以下情况会 loud `ValueError`：必填键缺失/为 null、`slot_order` 有重复、`slot_order` 里的名字没有对应 rank、`slot_ranks` 里有不在 `slot_order` 中的 suite、以及 `lora_path` 与 `enabled: true` 同时给（那个 adapter 会被静默忽略）。

---

## ⚠️ API 变更（Task 2 返工后，Task 3/4/6/9 以此为准）

原计划把 per-sample gate 用 `ContextVar` 传递，**这是错的**：autograd 引擎在 CUDA 上用每设备的 worker 线程跑反向节点，而本仓库会开 gradient checkpointing（`fsdp_model_manager.py:253`），重算发生在反向、在那个线程上。实测复现：作用域在主线程仍开着，重算里读到的是 `None` → 门控整个不生效、四个 slot 梯度互相串、**不报错也不 NaN**，曲线完全正常。

现在改为**共享 holder 对象**（属性访问不受线程和重算影响），`slot_gate()` / `get_slot_gate()` 已删除：

```python
from rlinf.models.slot_lora import SlotGate

gate = SlotGate(num_slots=len(slot_ranks), strict=True)   # 每个模型一个，注入时创建，所有 gated 模块共享同一引用
# num_slots 是第一个位置参数，strict 必须按关键字传；SlotGate(True) 会 raise（bool 被显式拒绝，
# 否则它会绑到 num_slots 上、悄悄建出一个单 slot 的 gate）。
# 每个共享该 gate 的 SlotOut 都必须满足 len(slot_ranks) == gate.num_slots，否则构造期 raise ——
# 否则 gate 说 4 而 B 只有 3 个 slot 时，id=3 能过范围校验却匹配不到任何 slot，同一个静默失败换个门进来。
# num_slots=None 时 gate 拒绝安装任何非 None 的路由（单 suite 无路由的 run 不受影响）：跳过范围校验
# 等于把保证悄悄降级，而『越界 id 与 -1 不可区分』正是这条校验存在的理由。

class SlotOut(nn.Module):
    def __init__(self, out_features, slot_ranks, gate, ...):
        self.gate = gate          # 只存引用，不在这里读
    def forward(self, h):         # 注意：没有 gate_ids 参数了
        gate_ids = self.gate.current()      # strict 且未设置时 raise

# 调用方（fsdp_actor_worker）：
with gate.scoped(ids):            # ids: LongTensor[B]，B = 本 micro-batch，-1 = 无 slot 拥有
    loss = student(**batch)
    loss.backward()               # ★ backward 必须在作用域内 —— checkpoint 会重跑 forward
```

- `gate.current()` 严格读取（strict 下未设置就 raise）；`gate.current_unchecked()` 是唯一的逃生口，只给 metrics 用，故意起得难看以便在 diff 里显眼。
- `SlotGate(num_slots=None, strict=False)` 用于单 suite 无路由的 run；`gate.ungated()` 在 strict 模型里开一个不门控的窗口（eval / rollout）。
- **strict 还堵住了 holder 本身堵不住的一个洞**：作用域在 `.backward()` 之前就退出。这种情况现在 raise，而不是静默地重算成 ungated。
- ids 的 device 归调用方管（读路径每次前向要跑 200-400 次，不能在那里 `.to(device)`）。
- `self.gate = gate` 是普通对象属性，不增加 children 也不增加 parameter，所以 `utils.py:306` 的 FSDP 叶子判定仍然成立。

下面 Task 3/4 正文里凡是 `SlotOut.forward(h, gate_ids)`、`get_slot_gate()`、`slot_gate(...)` 的写法，按本节替换。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `rlinf/models/slot_lora/__init__.py` | 对外导出 |
| `rlinf/models/slot_lora/orth.py` | `orthogonalize()`（Löwdin 对称正交化）+ `orth_error()` |
| `rlinf/models/slot_lora/modules.py` | `SlotProj` / `SlotOut` / `SlotLoRALinear` + gate 上下文 |
| `rlinf/models/slot_lora/routing.py` | `match_suite_ids()` 纯函数：prompt 文本 → suite 索引 |
| `rlinf/models/slot_lora/inject.py` | `inject_slot_lora()` 模块替换 + `enable_slot_diag()` / `collect_slot_diag()` |
| `rlinf/models/__init__.py` | 接线：`is_lora` 且 `slot_lora.enabled` 时走 slot 注入而非 `get_peft_model` |
| `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` | optimizer 分组：`slot_A`(wd=0) / `slot_B` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `_route_prepare()`、学生前向前设 gate、交替调度、slot 指标 |
| `examples/embodiment/config/libero_mt4slot_6gpu.yaml` | R1 的配置 |
| `examples/embodiment/incremental_sft/opd_mt4slot.sh` | R1 启动脚本 |
| `opd_distill/scripts/convert_oft_slot_ckpt.py` | slot checkpoint → merged HF 模型 |
| `tests/unit_tests/test_slot_lora_orth.py` | Task 1 |
| `tests/unit_tests/test_slot_lora_modules.py` | Task 2-4 |
| `tests/unit_tests/test_slot_lora_routing.py` | Task 5 |
| `tests/unit_tests/test_slot_lora_inject.py` | Task 6 |

---

### Task 1: 正交化原语

**Files:**
- Create: `rlinf/models/slot_lora/__init__.py`
- Create: `rlinf/models/slot_lora/orth.py`
- Test: `tests/unit_tests/test_slot_lora_orth.py`

- [ ] **Step 1: 写失败的测试**

`tests/unit_tests/test_slot_lora_orth.py`：

```python
import torch

from rlinf.models.slot_lora.orth import orth_error, orthogonalize


def _z(rows=16, cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=torch.float64)


class TestOrthogonalize:
    def test_rows_are_orthonormal(self):
        a = orthogonalize(_z())
        eye = torch.eye(a.shape[0], dtype=a.dtype)
        assert torch.allclose(a @ a.T, eye, atol=1e-10)

    def test_scale_invariant(self):
        z = _z()
        assert torch.allclose(orthogonalize(3.7 * z), orthogonalize(z), atol=1e-10)

    def test_row_blocks_are_mutually_orthogonal(self):
        a = orthogonalize(_z(rows=16))
        cross = a[:6] @ a[6:].T
        assert cross.abs().max().item() < 1e-10

    def test_preserves_row_space(self):
        # projecting Z onto span(A rows) must return Z exactly
        z = _z()
        a = orthogonalize(z)
        assert torch.allclose(z @ a.T @ a, z, atol=1e-8)

    def test_is_differentiable(self):
        z = _z().requires_grad_(True)
        orthogonalize(z).sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_preserves_input_dtype(self):
        a = orthogonalize(_z().to(torch.bfloat16))
        assert a.dtype == torch.bfloat16

    def test_orth_error_is_zero_after_orthogonalization(self):
        assert orth_error(orthogonalize(_z())).item() < 1e-10

    def test_orth_error_is_positive_for_raw_matrix(self):
        assert orth_error(_z()).item() > 1.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_orth.py -v
```
Expected: FAIL —— `ModuleNotFoundError: No module named 'rlinf.models.slot_lora'`

- [ ] **Step 3: 实现**

`rlinf/models/slot_lora/__init__.py`：

```python
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
"""slot-LoRI: per-suite LoRA slots on mutually orthogonal input subspaces."""

from rlinf.models.slot_lora.orth import orth_error, orthogonalize

__all__ = ["orthogonalize", "orth_error"]
```

`rlinf/models/slot_lora/orth.py`（同样的 Apache 头，下略）：

```python
import torch


def orthogonalize(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Loewdin (symmetric) orthogonalization:  A = (Z Zᵀ)^(-1/2) Z.

    The rows of the result are orthonormal and span exactly the same subspace as
    the rows of ``z``. Among all row-orthonormal matrices this is the one closest
    to ``z`` in Frobenius norm, so a gradient step taken on ``z`` is preserved to
    first order.

    Differentiable end to end: this is how the orthogonality constraint enters the
    computation graph, instead of being a post-``optimizer.step()`` projection.

    The Gram matrix and its inverse square root are always computed in fp32 --
    bf16 has ~3 decimal digits, far too few for an eigendecomposition -- and the
    result is cast back to the input dtype.

    Note A is invariant to the scale of Z (orthogonalize(cZ) == orthogonalize(Z)),
    which is why the Z parameter group MUST use weight_decay=0: decay would shrink
    Z toward zero, worsening the conditioning of Z Zᵀ, with no effect on A.
    """
    dtype = z.dtype
    z32 = z.float()
    gram = z32 @ z32.transpose(-2, -1)
    evals, evecs = torch.linalg.eigh(gram)
    inv_sqrt = (evecs * evals.clamp_min(eps).rsqrt().unsqueeze(-2)) @ evecs.transpose(
        -2, -1
    )
    return (inv_sqrt @ z32).to(dtype)


def orth_error(a: torch.Tensor) -> torch.Tensor:
    """‖A Aᵀ − I‖_F -- the直接 correctness check on orthogonalize()."""
    a32 = a.float()
    gram = a32 @ a32.transpose(-2, -1)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return (gram - eye).norm()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_orth.py -v
```
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/slot_lora/__init__.py rlinf/models/slot_lora/orth.py tests/unit_tests/test_slot_lora_orth.py
git commit -m "feat(slot-lora): differentiable Loewdin orthogonalization primitive"
```

---

### Task 2: gate 上下文 + SlotProj

**Files:**
- Create: `rlinf/models/slot_lora/modules.py`
- Modify: `rlinf/models/slot_lora/__init__.py`
- Test: `tests/unit_tests/test_slot_lora_modules.py`

- [ ] **Step 1: 写失败的测试**

`tests/unit_tests/test_slot_lora_modules.py`（本任务只加这两个类，Task 3/4 会继续往同一文件追加）：

```python
import torch

from rlinf.models.slot_lora.modules import SlotProj, get_slot_gate, slot_gate


class TestSlotGate:
    def test_default_is_none(self):
        assert get_slot_gate() is None

    def test_context_sets_and_restores(self):
        ids = torch.tensor([0, 1])
        with slot_gate(ids):
            assert get_slot_gate() is ids
        assert get_slot_gate() is None

    def test_restores_on_exception(self):
        try:
            with slot_gate(torch.tensor([0])):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert get_slot_gate() is None


class TestSlotProj:
    def _proj(self, in_features=32, total_rank=8, scale=1.0):
        torch.manual_seed(0)
        return SlotProj(in_features, total_rank, scale, dtype=torch.float64)

    def test_is_a_leaf_module_so_fsdp_wraps_it_alone(self):
        # rlinf/hybrid_engines/fsdp/utils.py:306 wraps modules with no children,
        # a .weight, and weight.requires_grad -- that is what gives the module its
        # own flat param (uniform requires_grad) AND materializes the full Z inside
        # its own forward, which orthogonalize() needs.
        p = self._proj()
        assert list(p.named_children()) == []
        assert p.weight is not None
        assert p.weight.requires_grad

    def test_weight_shape(self):
        assert self._proj(in_features=32, total_rank=8).weight.shape == (8, 32)

    def test_orth_weight_rows_are_orthonormal(self):
        a = self._proj().orth_weight()
        assert torch.allclose(a @ a.T, torch.eye(8, dtype=a.dtype), atol=1e-10)

    def test_forward_applies_scale(self):
        p = self._proj(scale=1.0)
        x = torch.randn(4, 32, dtype=torch.float64)
        base = p(x)
        p.scale = 2.5
        assert torch.allclose(p(x), 2.5 * base, atol=1e-12)

    def test_forward_shape(self):
        out = self._proj(in_features=32, total_rank=8)(torch.randn(4, 7, 32, dtype=torch.float64))
        assert out.shape == (4, 7, 8)

    def test_gradient_reaches_z(self):
        p = self._proj()
        p(torch.randn(4, 32, dtype=torch.float64)).sum().backward()
        assert p.weight.grad is not None
        assert torch.isfinite(p.weight.grad).all()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -v
```
Expected: FAIL —— `ImportError: cannot import name 'SlotProj'`

- [ ] **Step 3: 实现**

`rlinf/models/slot_lora/modules.py`（Apache 头 + 以下内容）：

```python
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.slot_lora.orth import orthogonalize

# Per-sample slot ownership for the CURRENT student forward: LongTensor[B] holding the
# slot index each sample belongs to, or -1 for "no slot owns this sample".
# A ContextVar rather than a plain global because the actor worker is an asyncio actor.
_SLOT_GATE: ContextVar = ContextVar("rlinf_slot_gate", default=None)


def get_slot_gate():
    return _SLOT_GATE.get()


@contextmanager
def slot_gate(gate_ids):
    """Scope the per-sample slot routing to one student forward."""
    token = _SLOT_GATE.set(gate_ids)
    try:
        yield
    finally:
        _SLOT_GATE.reset(token)


class SlotProj(nn.Module):
    """Holds Z; produces the row-orthonormal Ā = (Z Zᵀ)^(-1/2) Z used by every slot.

    Deliberately a LEAF module (no child modules, one ``.weight``): that is the test in
    ``rlinf/hybrid_engines/fsdp/utils.py:306``, so FSDP gives it its own flat parameter.
    Two consequences we depend on: requires_grad is uniform inside that flat param
    (required with ``use_orig_params=False``), and the FULL Z is all-gathered when this
    module's forward runs -- ``orthogonalize`` needs every row at once.
    """

    def __init__(self, in_features, total_rank, scale, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.in_features = int(in_features)
        self.total_rank = int(total_rank)
        self.scale = float(scale)
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.empty(self.total_rank, self.in_features, dtype=dtype, device=device)
        )
        # std = 1/sqrt(d_in) gives unit-ish row norms and a well-conditioned Z Zᵀ.
        # The actual scale is irrelevant to Ā (orthogonalize is scale-invariant).
        nn.init.normal_(self.weight, mean=0.0, std=self.in_features**-0.5)
        self._collect_diag = False
        self._diag = None

    def orth_weight(self):
        a = orthogonalize(self.weight, eps=self.eps)
        if self._collect_diag:
            with torch.no_grad():
                # The Gram MUST come from an fp32 recomputation, not from `a` cast down
                # to bf16. Measured: as cond(Z) goes 10 -> 20 the fp32 orthogonality error
                # degrades 16x (6.8e-5 -> 1.07e-3) while a bf16 reading moves only from
                # 0.01887 to 0.01894 -- the rounding floor hides the entire slide, and by
                # the time bf16 moves (cond 30 -> 0.054) the result is already garbage.
                # Failure here is a cliff, not a slope, so the sentinel must watch the
                # slope. One extra NS call (~1.3 ms) on ONE module per training step.
                a32 = orthogonalize(self.weight.detach().float(), eps=self.eps)
                self._diag = {"gram": a32 @ a32.transpose(-2, -1), "scale": self.scale}
            self._collect_diag = False
        return a

    def forward(self, x):
        return F.linear(x, self.orth_weight() * self.scale)
```

`rlinf/models/slot_lora/__init__.py` 追加：

```python
from rlinf.models.slot_lora.modules import SlotProj, get_slot_gate, slot_gate

__all__ = ["orthogonalize", "orth_error", "SlotProj", "get_slot_gate", "slot_gate"]
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -v
```
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/slot_lora/modules.py rlinf/models/slot_lora/__init__.py tests/unit_tests/test_slot_lora_modules.py
git commit -m "feat(slot-lora): SlotProj (Z-parameterized orthonormal projection) + gate context"
```

---

### Task 3: SlotOut —— 门控语义（本计划最容易写错的一块）

**Files:**
- Modify: `rlinf/models/slot_lora/modules.py`
- Modify: `rlinf/models/slot_lora/__init__.py`
- Test: `tests/unit_tests/test_slot_lora_modules.py`

在 `h` 上乘门是**错的**：B 是一整块参数，∂L/∂B_j = outer(grad_out, h_j) 对所有 j 都非零。必须逐 slot 算出贡献 `c_k`，再把非归属 slot 的**整条贡献** detach 掉。`torch.where(mask, c, c.detach())` 前向数值不变（四个 slot 的完整和），反向只走归属 slot。

- [ ] **Step 1: 写失败的测试**

追加到 `tests/unit_tests/test_slot_lora_modules.py`：

```python
from rlinf.models.slot_lora.modules import SlotOut


class TestSlotOut:
    RANKS = (3, 5)  # slot0 -> columns 0:3, slot1 -> columns 3:8

    def _out(self, out_features=6):
        torch.manual_seed(1)
        m = SlotOut(out_features, self.RANKS, dtype=torch.float64)
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))  # B must be non-zero to test gating
        return m

    def _h(self, batch=4):
        return torch.randn(batch, sum(self.RANKS), dtype=torch.float64, requires_grad=True)

    def test_offsets_and_total_rank(self):
        m = self._out()
        assert m.total_rank == 8
        assert m.offsets == (0, 3)

    def test_b_is_zero_initialized(self):
        assert torch.count_nonzero(SlotOut(6, self.RANKS).weight) == 0

    def test_forward_value_is_the_full_slot_sum_regardless_of_gate(self):
        m, h = self._out(), self._h()
        ungated = m(h, None)
        gated = m(h, torch.tensor([0, 1, 0, -1]))
        assert torch.allclose(ungated, gated, atol=1e-12)

    def test_forward_equals_a_plain_linear_when_ungated(self):
        m, h = self._out(), self._h()
        assert torch.allclose(m(h, None), F.linear(h, m.weight), atol=1e-12)

    def test_gradient_reaches_only_the_owning_slot_columns(self):
        m, h = self._out(), self._h()
        m(h, torch.tensor([0, 0, 0, 0])).sum().backward()
        assert m.weight.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(m.weight.grad[:, 3:8]) == 0

    def test_gradient_to_h_is_blocked_outside_the_owning_block(self):
        m, h = self._out(), self._h()
        m(h, torch.tensor([1, 1, 1, 1])).sum().backward()
        assert torch.count_nonzero(h.grad[:, 0:3]) == 0
        assert h.grad[:, 3:8].abs().sum() > 0

    def test_unrouted_samples_produce_no_gradient_at_all(self):
        m, h = self._out(), self._h()
        m(h, torch.tensor([-1, -1, -1, -1])).sum().backward()
        assert torch.count_nonzero(m.weight.grad) == 0
        assert torch.count_nonzero(h.grad) == 0

    def test_mixed_batch_routes_each_sample_to_its_own_slot(self):
        m, h = self._out(), self._h()
        # only sample 0 (slot 0) and sample 1 (slot 1) may contribute
        m(h, torch.tensor([0, 1, -1, -1])).sum().backward()
        assert torch.count_nonzero(h.grad[0, 3:8]) == 0
        assert h.grad[0, 0:3].abs().sum() > 0
        assert torch.count_nonzero(h.grad[1, 0:3]) == 0
        assert h.grad[1, 3:8].abs().sum() > 0
        assert torch.count_nonzero(h.grad[2]) == 0
        assert torch.count_nonzero(h.grad[3]) == 0

    def test_works_with_a_sequence_dimension(self):
        m = self._out()
        h = torch.randn(4, 7, 8, dtype=torch.float64, requires_grad=True)
        out = m(h, torch.tensor([0, 0, 1, 1]))
        assert out.shape == (4, 7, 6)
        out.sum().backward()
        assert torch.count_nonzero(h.grad[0, :, 3:8]) == 0

    def test_rejects_gate_of_wrong_length(self):
        m, h = self._out(batch=4), self._h(batch=4)
        try:
            m(h, torch.tensor([0, 1]))
        except AssertionError:
            return
        raise AssertionError("expected an assertion on mismatched gate length")
```

文件顶部补 `import torch.nn.functional as F`。

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -k SlotOut -v
```
Expected: FAIL —— `ImportError: cannot import name 'SlotOut'`

- [ ] **Step 3: 实现**

追加到 `rlinf/models/slot_lora/modules.py`：

```python
class SlotOut(nn.Module):
    """Holds B (d_out x R) and sums the per-slot contributions with a gradient gate.

    ``ΔW = Σ_k B_k Ā_k`` where B_k is the k-th column block of B. The forward VALUE is
    always the full sum, so the acting policy is the fully merged model -- there is no
    train/inference mismatch and no inference-time routing.

    The gate must be applied to the whole per-slot contribution, never to ``h``:
    ∂L/∂B_j = outer(grad_out, h_j) is non-zero for every j regardless of what h looks
    like, so masking h leaks gradient into every slot's B. Detaching the contribution
    blocks the path into both B_j and Ā_j at once.

    Also a LEAF module, for the same FSDP reason as SlotProj.
    """

    def __init__(self, out_features, slot_ranks, dtype=None, device=None):
        super().__init__()
        self.out_features = int(out_features)
        self.slot_ranks = tuple(int(r) for r in slot_ranks)
        self.total_rank = sum(self.slot_ranks)
        offsets, running = [], 0
        for r in self.slot_ranks:
            offsets.append(running)
            running += r
        self.offsets = tuple(offsets)
        # B = 0 at init -> ΔW = 0 -> the model starts byte-equivalent to its base.
        self.weight = nn.Parameter(
            torch.zeros(self.out_features, self.total_rank, dtype=dtype, device=device)
        )
        self._collect_diag = False
        self._diag = None

    def forward(self, h, gate_ids=None):
        if gate_ids is not None:
            assert gate_ids.shape[0] == h.shape[0], (
                f"slot gate length {gate_ids.shape[0]} != batch {h.shape[0]}; the gate "
                "must be built from THIS micro-batch, before the student forward"
            )
        out = None
        for k, (off, r) in enumerate(zip(self.offsets, self.slot_ranks)):
            c = F.linear(h[..., off : off + r], self.weight[:, off : off + r])
            if gate_ids is not None:
                owns = (gate_ids == k).view(gate_ids.shape[0], *([1] * (c.dim() - 1)))
                c = torch.where(owns, c, c.detach())
            out = c if out is None else out + c
        if self._collect_diag:
            self._stash_diag()
        return out

    @torch.no_grad()
    def _stash_diag(self):
        """Small per-slot quantities for the orthogonality / capacity metrics.

        ‖ΔW_k‖_F = scale * ‖B_k‖_F because Ā_k has orthonormal rows, and
        ⟨ΔW_s, ΔW_t⟩_F = scale² * tr( (B_tᵀB_s) (Ā_sĀ_tᵀ) ) -- both need only r x r
        matrices, never the d_out x d_in ΔW itself.
        """
        b = self.weight.detach().float()
        blocks = [
            b[:, off : off + r] for off, r in zip(self.offsets, self.slot_ranks)
        ]
        self._diag = {
            "norms": [blk.norm() for blk in blocks],
            "cross": {
                (s, t): blocks[t].transpose(-2, -1) @ blocks[s]
                for s in range(len(blocks))
                for t in range(s + 1, len(blocks))
            },
        }
        self._collect_diag = False
```

`__init__.py` 的导入与 `__all__` 加上 `SlotOut`。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -v
```
Expected: 19 passed

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/slot_lora/modules.py rlinf/models/slot_lora/__init__.py tests/unit_tests/test_slot_lora_modules.py
git commit -m "feat(slot-lora): SlotOut with per-sample contribution-level gradient gating"
```

---

### Task 4: SlotLoRALinear

**Files:**
- Modify: `rlinf/models/slot_lora/modules.py`
- Modify: `rlinf/models/slot_lora/__init__.py`
- Test: `tests/unit_tests/test_slot_lora_modules.py`

- [ ] **Step 1: 写失败的测试**

追加：

```python
from rlinf.models.slot_lora.modules import SlotLoRALinear


class TestSlotLoRALinear:
    RANKS = (3, 5)

    def _layer(self, in_features=16, out_features=6, scale=1.0):
        torch.manual_seed(2)
        base = torch.nn.Linear(in_features, out_features, dtype=torch.float64)
        return SlotLoRALinear(base, self.RANKS, scale)

    def test_output_equals_base_at_init(self):
        layer = self._layer()
        x = torch.randn(4, 16, dtype=torch.float64)
        assert torch.allclose(layer(x), layer.base(x), atol=1e-12)

    def test_base_weights_are_frozen(self):
        layer = self._layer()
        assert layer.base.weight.requires_grad is False
        assert layer.base.bias.requires_grad is False

    def test_only_z_and_b_are_trainable(self):
        layer = self._layer()
        trainable = sorted(n for n, p in layer.named_parameters() if p.requires_grad)
        assert trainable == ["slot_A.weight", "slot_B.weight"]

    def test_output_matches_base_plus_delta_weight(self):
        layer = self._layer()
        with torch.no_grad():
            layer.slot_B.weight.copy_(torch.randn_like(layer.slot_B.weight))
        x = torch.randn(4, 16, dtype=torch.float64)
        expected = layer.base(x) + x @ layer.delta_weight().to(x.dtype).T
        assert torch.allclose(layer(x), expected, atol=1e-9)

    def test_delta_weight_is_zero_at_init(self):
        assert self._layer().delta_weight().abs().max().item() == 0.0

    def test_reads_the_gate_from_the_context(self):
        layer = self._layer()
        with torch.no_grad():
            layer.slot_B.weight.copy_(torch.randn_like(layer.slot_B.weight))
        x = torch.randn(4, 16, dtype=torch.float64)
        with slot_gate(torch.tensor([0, 0, 0, 0])):
            layer(x).sum().backward()
        assert torch.count_nonzero(layer.slot_B.weight.grad[:, 3:8]) == 0

    def test_match_mt4_scale_reproduces_the_peft_row_norm(self):
        # PEFT init_lora_weights="gaussian" draws A with std = 1/r, so its rows have
        # norm sqrt(d_in)/r. Ā's rows have norm 1, hence this compensating scale.
        layer = self._layer(in_features=16, scale=(16**0.5) / 128)
        assert abs(layer.slot_A.scale - (16**0.5) / 128) < 1e-12
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -k SlotLoRALinear -v
```
Expected: FAIL —— `ImportError: cannot import name 'SlotLoRALinear'`

- [ ] **Step 3: 实现**

追加到 `modules.py`：

```python
class SlotLoRALinear(nn.Module):
    """A frozen nn.Linear plus K per-suite LoRA slots on mutually orthogonal subspaces.

    Replaces PEFT's lora.Linear on the STUDENT only. Teachers and the dual-KL anchor keep
    using PEFT -- nothing about them changes.
    """

    def __init__(self, base: nn.Linear, slot_ranks, scale, eps=1e-6):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        ref = base.weight
        self.slot_A = SlotProj(
            base.in_features,
            sum(int(r) for r in slot_ranks),
            scale,
            eps,
            dtype=ref.dtype,
            device=ref.device,
        )
        self.slot_B = SlotOut(
            base.out_features, slot_ranks, dtype=ref.dtype, device=ref.device
        )

    def forward(self, x):
        return self.base(x) + self.slot_B(self.slot_A(x), get_slot_gate())

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """scale * B Ā -- what gets added into the base weight at merge time (fp32)."""
        a = self.slot_A.orth_weight().float() * self.slot_A.scale
        return self.slot_B.weight.float() @ a
```

`__init__.py` 导出 `SlotLoRALinear`。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_modules.py -v
```
Expected: 26 passed

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/slot_lora/modules.py rlinf/models/slot_lora/__init__.py tests/unit_tests/test_slot_lora_modules.py
git commit -m "feat(slot-lora): SlotLoRALinear wrapper with merge-ready delta_weight()"
```

---

### Task 5: prompt → suite 的纯函数

**Files:**
- Create: `rlinf/models/slot_lora/routing.py`
- Modify: `rlinf/models/slot_lora/__init__.py`
- Test: `tests/unit_tests/test_slot_lora_routing.py`

语义必须和 `fsdp_actor_worker.py:1284-1291` 现有的 teacher 路由**逐字一致**（小写化 + 子串匹配 + 首个命中即停），否则同一个样本的 teacher 和 slot 会配错。

- [ ] **Step 1: 写失败的测试**

`tests/unit_tests/test_slot_lora_routing.py`：

```python
from rlinf.models.slot_lora.routing import match_suite_ids

ROUTE = {
    "pick up the black bowl": "libero_spatial",
    "put the cream cheese": "libero_object",
    "open the top drawer": "libero_goal",
}
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


class TestMatchSuiteIds:
    def test_maps_each_prompt_to_its_suite_index(self):
        texts = ["In: pick up the black bowl and place it", "put the cream cheese in the basket"]
        assert match_suite_ids(texts, ROUTE, ORDER) == [0, 1]

    def test_is_case_insensitive_and_strips(self):
        assert match_suite_ids(["  OPEN THE TOP DRAWER now "], ROUTE, ORDER) == [2]

    def test_unknown_prompt_is_minus_one(self):
        assert match_suite_ids(["do something else entirely"], ROUTE, ORDER) == [-1]

    def test_suite_missing_from_order_is_minus_one(self):
        assert match_suite_ids(["open the top drawer"], ROUTE, ["libero_spatial"]) == [-1]

    def test_empty_batch(self):
        assert match_suite_ids([], ROUTE, ORDER) == []

    def test_first_match_wins(self):
        route = {"alpha": "libero_spatial", "beta": "libero_object"}
        assert match_suite_ids(["alpha beta"], route, ORDER) == [0]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_routing.py -v
```
Expected: FAIL —— `ModuleNotFoundError: rlinf.models.slot_lora.routing`

- [ ] **Step 3: 实现**

`rlinf/models/slot_lora/routing.py`：

```python
from typing import Dict, List, Sequence


def match_suite_ids(
    texts: Sequence[str],
    prompt_to_suite: Dict[str, str],
    suite_order: Sequence[str],
) -> List[int]:
    """Map decoded rollout prompts to slot indices; -1 means "no slot owns this sample".

    Matching semantics are copied verbatim from the teacher routing in
    ``fsdp_actor_worker.py`` (lower-case, strip, substring, first hit wins) so that a
    sample's teacher and its slot can never disagree.
    """
    index_of = {suite: i for i, suite in enumerate(suite_order)}
    ids: List[int] = []
    for text in texts:
        lowered = text.strip().lower()
        found = -1
        for instruction, suite in prompt_to_suite.items():
            if instruction in lowered:
                found = index_of.get(suite, -1)
                break
        ids.append(found)
    return ids
```

`__init__.py` 导出 `match_suite_ids`。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_routing.py -v
```
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/slot_lora/routing.py rlinf/models/slot_lora/__init__.py tests/unit_tests/test_slot_lora_routing.py
git commit -m "feat(slot-lora): prompt-to-slot routing helper mirroring teacher routing"
```

---

### Task 6: 模块注入 + 诊断收集

**Files:**
- Create: `rlinf/models/slot_lora/inject.py`
- Modify: `rlinf/models/slot_lora/__init__.py`
- Test: `tests/unit_tests/test_slot_lora_inject.py`

- [ ] **Step 1: 写失败的测试**

`tests/unit_tests/test_slot_lora_inject.py`：

```python
import torch
import torch.nn as nn

from rlinf.models.slot_lora.inject import (
    collect_slot_diag,
    enable_slot_diag,
    inject_slot_lora,
)
from rlinf.models.slot_lora.modules import SlotLoRALinear

TARGETS = ["q_proj", "o_proj"]
RANKS = (2, 3)


class Block(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, dtype=torch.float64)
        self.o_proj = nn.Linear(dim, dim, dtype=torch.float64)
        self.mlp = nn.Linear(dim, dim, dtype=torch.float64)  # NOT a target


class Toy(nn.Module):
    def __init__(self, n_blocks=2, dim=16):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(n_blocks))

    def forward(self, x):
        for b in self.blocks:
            x = b.mlp(b.o_proj(b.q_proj(x)))
        return x


class TestInjectSlotLora:
    def test_replaces_every_target_module(self):
        model = Toy()
        replaced = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert len(replaced) == 4
        assert isinstance(model.blocks[0].q_proj, SlotLoRALinear)
        assert isinstance(model.blocks[1].o_proj, SlotLoRALinear)

    def test_leaves_non_target_modules_alone(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert isinstance(model.blocks[0].mlp, nn.Linear)
        assert not isinstance(model.blocks[0].mlp, SlotLoRALinear)

    def test_does_not_rewrap_the_frozen_base(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        inner = model.blocks[0].q_proj.base
        assert isinstance(inner, nn.Linear)
        assert not isinstance(inner, SlotLoRALinear)

    def test_output_is_unchanged_at_init(self):
        model, x = Toy(), torch.randn(3, 16, dtype=torch.float64)
        before = model(x)
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert torch.allclose(model(x), before, atol=1e-12)

    def test_only_slot_params_are_trainable(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        for name, param in model.named_parameters():
            expected = ".slot_A." in name or ".slot_B." in name
            assert param.requires_grad is expected, name

    def test_match_mt4_scale_depends_on_in_features(self):
        model = Toy(dim=16)
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="match_mt4", ref_rank=128)
        assert abs(model.blocks[0].q_proj.slot_A.scale - (16**0.5) / 128) < 1e-12

    def test_rejects_unknown_scale_mode(self):
        try:
            inject_slot_lora(Toy(), RANKS, TARGETS, scale_mode="nope")
        except ValueError:
            return
        raise AssertionError("expected ValueError on unknown scale_mode")


class TestSlotDiagnostics:
    def _model(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        for m in model.modules():
            if isinstance(m, SlotLoRALinear):
                with torch.no_grad():
                    m.slot_B.weight.copy_(torch.randn_like(m.slot_B.weight))
        return model

    def test_enable_then_collect_returns_metrics(self):
        model = self._model()
        assert enable_slot_diag(model) is True
        model(torch.randn(3, 16, dtype=torch.float64))
        diag = collect_slot_diag(model)
        assert diag["slot/orth_err"] < 1e-6
        assert abs(diag["slot/cos_0_1"]) < 1e-6
        assert diag["slot/dw_norm_0"] > 0.0
        assert diag["slot/dw_norm_1"] > 0.0

    def test_collect_without_enable_returns_empty(self):
        assert collect_slot_diag(self._model()) == {}

    def test_enable_returns_false_without_slots(self):
        assert enable_slot_diag(Toy()) is False
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_inject.py -v
```
Expected: FAIL —— `ModuleNotFoundError: rlinf.models.slot_lora.inject`

- [ ] **Step 3: 实现**

`rlinf/models/slot_lora/inject.py`：

```python
import torch
import torch.nn as nn

from rlinf.models.slot_lora.modules import SlotLoRALinear, SlotOut, SlotProj


def inject_slot_lora(
    model: nn.Module,
    slot_ranks,
    target_modules,
    scale_mode: str = "match_mt4",
    ref_rank: int = 128,
    eps: float = 1e-6,
):
    """Replace every targeted nn.Linear with a SlotLoRALinear and freeze everything else.

    ``scale_mode="match_mt4"`` sets s = sqrt(d_in)/ref_rank per module, which reproduces
    the row norm of the PEFT gaussian-initialized A used by the mt4 baseline, so ΔW moves
    at the same rate on step 1 and the learning rate is not a hidden variable in the
    comparison. ``"unit"`` leaves the orthonormal rows at norm 1.
    """
    if scale_mode not in ("match_mt4", "unit"):
        raise ValueError(f"unknown scale_mode {scale_mode!r}")
    targets = set(target_modules)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in targets or not isinstance(child, nn.Linear):
                continue
            scale = (
                (child.in_features**0.5) / ref_rank
                if scale_mode == "match_mt4"
                else 1.0
            )
            setattr(parent, child_name, SlotLoRALinear(child, slot_ranks, scale, eps))
            replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    for name, param in model.named_parameters():
        param.requires_grad_(".slot_A." in name or ".slot_B." in name)
    return replaced


def enable_slot_diag(model: nn.Module) -> bool:
    # NOTE (from Task 4): arm through SlotLoRALinear.arm_diag(), NOT by finding "the first
    # SlotProj" and "the first SlotOut" separately. The cosine pairs Ā_tĀ_sᵀ with B_tᵀB_s and
    # they MUST come from the same layer; separate searches pair by module registration order,
    # and in a 7B model full of same-shaped projections a mispairing has the right shape and
    # the wrong numbers, with nothing to raise. i.e.:
    #   layer = next(m for m in model.modules() if isinstance(m, SlotLoRALinear))
    #   layer.arm_diag()          # arms both children and clears the previous _diag
    """Arm the diagnostics on ONE SlotLoRALinear; they are computed inside its forward.

    Doing it inside the forward is what keeps this free under FSDP: the module's full
    parameters are already all-gathered there, so no extra collective is needed. Module
    registration order (base, slot_A, slot_B) guarantees the first SlotProj and the first
    SlotOut belong to the same layer.
    """
    proj = next((m for m in model.modules() if isinstance(m, SlotProj)), None)
    out = next((m for m in model.modules() if isinstance(m, SlotOut)), None)
    if proj is None or out is None:
        return False
    proj._collect_diag = True
    out._collect_diag = True
    proj._diag = None
    out._diag = None
    return True


def collect_slot_diag(model: nn.Module) -> dict:
    """Read what the armed forward stashed. Empty dict if nothing was armed/ran."""
    proj = next((m for m in model.modules() if isinstance(m, SlotProj)), None)
    out = next((m for m in model.modules() if isinstance(m, SlotOut)), None)
    if proj is None or out is None or proj._diag is None or out._diag is None:
        return {}
    gram = proj._diag["gram"]
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    metrics = {"slot/orth_err": (gram - eye).norm().item()}

    offsets, ranks = out.offsets, out.slot_ranks
    norms = out._diag["b_norms"]   # NOT "norms" -- SlotOut._stash_diag stores it as b_norms
    for k, norm in enumerate(norms):
        metrics[f"slot/dw_norm_{k}"] = (proj._diag["scale"] * norm).item()
    for (s, t), p_ts in out._diag["cross"].items():
        # SLICE ORDER MATTERS AND IS EASY TO GET WRONG. cross[(s,t)] = B_tᵀB_s has shape
        # (r_t, r_s), so it pairs with gram[t_slice, s_slice] -- NOT [s_slice, t_slice].
        # With unequal per-slot ranks the wrong order raises a shape error; with EQUAL
        # ranks (a perfectly plausible future config) both slicings have the same shape
        # and the wrong one silently computes tr(B_sᵀB_t Ā_sĀ_tᵀ), a different quantity.
        # Verified numerically against a deliberately non-orthogonal A: a true orthonormal
        # Ā makes both slicings ~0 and cannot tell them apart.
        q_st = gram[
            offsets[t] : offsets[t] + ranks[t], offsets[s] : offsets[s] + ranks[s]
        ]
        # ELEMENTWISE, not einsum("ij,ji->"). cross[(s,t)] is (r_t, r_s) and pairs with
        # gram[t_slice, s_slice] of the same shape, so the contraction is a plain Hadamard
        # sum. The einsum form transposes one of them: for unequal ranks it raises, for
        # EQUAL ranks it silently computes Σ P⊙Qᵀ, a different quantity.
        inner = (p_ts.to(q_st.dtype) * q_st).sum()
        # ⟨ΔW_s, ΔW_t⟩_F = scale² · Σ(B_tᵀB_s ⊙ Ā_tĀ_sᵀ) and ‖ΔW_k‖_F = scale · ‖B_k‖_F,
        # so scale² cancels in the cosine and must NOT be applied here. It does NOT cancel
        # in dw_norm_k above, which is why that one carries an explicit factor.
        denom = (norms[s] * norms[t]).clamp_min(1e-12)
        metrics[f"slot/cos_{s}_{t}"] = (inner / denom).item()
    proj._diag = None
    out._diag = None
    return metrics
```

`__init__.py` 导出 `inject_slot_lora` / `enable_slot_diag` / `collect_slot_diag`。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_inject.py -v
```
Expected: 10 passed

- [ ] **Step 5: 全量回归 + 提交**

```bash
cd /home/fanruochen/CL/RLinf
/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_*.py -v
git add rlinf/models/slot_lora/inject.py rlinf/models/slot_lora/__init__.py tests/unit_tests/test_slot_lora_inject.py
git commit -m "feat(slot-lora): module injection and collective-free orthogonality diagnostics"
```
Expected: 51 passed

---

### Task 7: 接进 get_model

**Files:**
- Modify: `rlinf/models/__init__.py`（`if cfg.is_lora:` 分支，第 224 行起）

学生和 rollout 侧共用这一条路径：rollout worker 在 `rlinf/workers/rollout/hf/huggingface_worker.py:95` 里 `deepcopy(cfg.actor.model)` 后调同一个 `get_model`，所以配置放在 `actor.model.slot_lora` 下，rollout 自动得到同构模型，权重同步（裸 state_dict 键名匹配）零改动。

- [ ] **Step 1: 加分支**

在 `rlinf/models/__init__.py` 中，找到

```python
    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
```

在 `if cfg.is_lora:` 之后、`from peft import ...` 之前插入：

```python
        # `cfg.is_lora` MUST stay True on the slot path. The per-leaf FSDP wrap policy that
        # gives slot_A/slot_B their own flat params is only registered when is_lora is set
        # (rlinf/hybrid_engines/fsdp/strategy/fsdp.py:163). Without it the trainable slot
        # params land in the enclosing transformer layer's flat param together with the
        # FROZEN base, and FSDP refuses to flatten mixed requires_grad under
        # use_orig_params=False. That is a loud failure, but an avoidable one.
        _slot_cfg = cfg.get("slot_lora", None)
        if _slot_cfg is not None and _slot_cfg.get("enabled", False):
            # slot-LoRI: K per-suite LoRA slots on mutually orthogonal input subspaces,
            # replacing PEFT entirely on this model. Teachers/anchor still use PEFT.
            import sys as _sys

            from rlinf.models.slot_lora import inject_slot_lora

            _ranks_map = dict(_slot_cfg["slot_ranks"])
            _order = list(_slot_cfg["slot_order"])
            _ranks = [int(_ranks_map[s]) for s in _order]
            _inj = inject_slot_lora(
                model,
                _ranks,
                SLOT_LORA_TARGET_MODULES,
                scale_mode=_slot_cfg.get("a_scale_mode", "match_mt4"),
                ref_rank=int(_slot_cfg.get("a_scale_ref_rank", 128)),
                eps=float(_slot_cfg.get("orth_eps", 1e-6)),
            )
            # The FSDP leaf-wrap guarantee SlotProj depends on has a fourth condition
            # that is easy to miss: rlinf/hybrid_engines/fsdp/utils.py:306 also requires
            # `getattr(module, "_to_lora", True) is True`, and tag_vlm_subtree(model,
            # False) (rlinf/models/__init__.py:362) stamps _to_lora=False on EVERY
            # module. Only the pi0 branch calls it today, so the slot path is clean --
            # but if that ever changes, Z would be folded into someone else's flat param
            # and the forward would no longer see the full Z, with NO error raised.
            # Convert that silent failure into a loud one.
            for _m in model.modules():
                if type(_m).__name__ in ("SlotProj", "SlotOut"):
                    assert getattr(_m, "_to_lora", True) is True, (
                        f"{type(_m).__name__} was tagged _to_lora=False; FSDP would not "
                        "give it its own flat param and orthogonalize() would see a "
                        "sharded Z"
                    )
            # inject_slot_lora returns a frozen dataclass SlotInjection(paths, gate) --
            # deliberately NOT a tuple or a list: a NamedTuple would let `len(result)`
            # silently return 2 (the field count) in the very log line that reports how
            # many layers were adapted, and would let `paths, gate = inject(...)` keep
            # working by accident. Use `_inj.paths` / `_inj.gate` explicitly.
            #
            # The GATE MUST BE KEPT: the actor installs per-micro-batch routing through it
            # (`with gate.scoped(ids):`). Stash it where the actor can reach it, or
            # re-derive it as `next(m for m in model.modules() if isinstance(m, SlotOut)).gate`
            # (safe -- every layer shares the one instance).
            _replaced = _inj.paths
            _ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _sys.stderr.write(
                f"[slot-lora] injected {len(_replaced)} SlotLoRALinear; "
                f"order={_order} ranks={_ranks} R={sum(_ranks)}; "
                f"scale_mode={_slot_cfg.get('a_scale_mode', 'match_mt4')}; "
                f"trainable params={_ntr}\n"
            )
            if hasattr(model, "value_head"):
                for param in model.value_head.parameters():
                    param.requires_grad = True
            return model
```

并在文件顶层（`import` 之后）加上 target 列表，与 PEFT 分支里的 `target_modules` 逐字相同：

```python
# Same target list as the PEFT LoraConfig below, so slot-LoRI touches exactly the
# modules the mt4 baseline's LoRA touched.
SLOT_LORA_TARGET_MODULES = [
    "proj", "qkv", "fc1", "fc2",                      # vision
    "q", "kv", "fc3", "out_proj",                     # projector
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",   # llm
]
```

- [ ] **Step 2: 冒烟验证（不加载 7B，用玩具 config）**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python - <<'PY'
import torch, torch.nn as nn
from rlinf.models import SLOT_LORA_TARGET_MODULES
from rlinf.models.slot_lora import inject_slot_lora

class B(nn.Module):
    def __init__(s):
        super().__init__(); s.q_proj = nn.Linear(8, 8); s.k_proj = nn.Linear(8, 8); s.other = nn.Linear(8, 8)
m = B()
n = inject_slot_lora(m, [2, 3], SLOT_LORA_TARGET_MODULES, scale_mode="unit")
print("replaced:", n)
assert len(n) == 2 and "other" not in " ".join(n)
print("OK")
PY
```
Expected: `replaced: ['q_proj', 'k_proj']` 然后 `OK`

- [ ] **Step 3: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/models/__init__.py
git commit -m "feat(slot-lora): wire slot injection into get_model behind actor.model.slot_lora"
```

---

### Task 8: optimizer 参数分组（slot_A 的 weight_decay 必须为 0）

**Files:**
- Modify: `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`（`build_optimizer`，约 479-520 行）

Ā 对 Z 的尺度不变，所以 weight decay 对函数**零影响**，只会把 Z 拉向 0、恶化 ZZᵀ 的条件数。必须单独分组关掉。分组还同时给 Task 10 的交替调度提供抓手（按 `group["name"]` 找组）。

- [ ] **Step 1: 改分类逻辑**

把

```python
                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)
```

改成

```python
                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    elif ".slot_A." in name:
                        params_slot_a.append(param)
                    elif ".slot_B." in name:
                        params_slot_b.append(param)
                    else:
                        params_actor.append(param)
```

在该循环前初始化 `params_slot_a, params_slot_b = [], []`，并给现有两个 group 补上 `"name"`：

```python
        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {"params": params_actor, "lr": self._cfg.optim.lr, "betas": betas, "name": "actor"}
            )
        if len(params_slot_a) > 0:
            # weight_decay=0: A is invariant to the scale of Z, so decay cannot change the
            # function -- it only drives Z toward zero and makes Z Zᵀ ill-conditioned.
            param_groups.append(
                {
                    "params": params_slot_a,
                    "lr": self._cfg.optim.lr,
                    "betas": betas,
                    "weight_decay": 0.0,
                    "name": "slot_A",
                }
            )
        if len(params_slot_b) > 0:
            param_groups.append(
                {"params": params_slot_b, "lr": self._cfg.optim.lr, "betas": betas, "name": "slot_B"}
            )
        if len(params_critic) > 0:
            param_groups.append(
                {"params": params_critic, "lr": self._cfg.optim.value_lr, "betas": betas, "name": "critic"}
            )
```

- [ ] **Step 2: 验证 AdamW 接受额外键并遵守 per-group wd**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python - <<'PY'
import torch
a = torch.nn.Parameter(torch.ones(3)); b = torch.nn.Parameter(torch.ones(3))
opt = torch.optim.AdamW(
    [{"params": [a], "lr": 0.1, "weight_decay": 0.0, "name": "slot_A"},
     {"params": [b], "lr": 0.1, "name": "slot_B"}], weight_decay=0.5)
a.grad = torch.zeros(3); b.grad = torch.zeros(3)
opt.step()
print("slot_A (wd=0), unchanged:", a.data)
print("slot_B (wd=0.5), decayed:", b.data)
assert torch.allclose(a.data, torch.ones(3)); assert (b.data < 1.0).all()
print("OK")
PY
```
Expected: `slot_A` 保持 1.0，`slot_B` 被衰减，最后 `OK`

- [ ] **Step 3: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
git commit -m "feat(slot-lora): separate slot_A/slot_B optimizer groups, zero weight decay on Z"
```

---

### Task 9: 学生前向之前完成路由（顺序陷阱）

**Files:**
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py`

学生前向在 `self.model(...)`（约 2131 行），`_teacher_forward` 在其后（约 2155 行），而 `self._last_groups` 是在 `_teacher_forward` 里才写的。直接拿 `_last_groups` 当 gate 会用到**上一个 micro-batch** 的路由，整批错位。必须把 decode+匹配抽出来在学生前向之前跑。

- [ ] **Step 1: 加 `_route_prepare`**

在 `_teacher_forward` 定义之前插入：

```python
    def _route_prepare(self, forward_inputs):
        """Decode this micro-batch's prompts ONCE and derive both routings.

        Sets, for the CURRENT micro-batch:
          * ``self._last_groups``       teacher path -> sample indices (what _teacher_forward uses)
          * ``self._slot_gate_ids``     LongTensor[B] slot index per sample, -1 = unrouted
          * ``self._slot_fallback``     fraction of samples that matched no suite

        MUST be called before the STUDENT forward: the student runs first, so the slot gate
        cannot come from _teacher_forward's own bookkeeping without being one micro-batch stale.
        """
        from rlinf.models.slot_lora import match_suite_ids

        route = getattr(self, "teacher_prompt_to_suite", None)
        models = getattr(self, "teacher_models", None)
        ids = forward_inputs["input_ids"]
        bsz = ids.shape[0]
        self._slot_gate_ids = None
        self._slot_fallback = 0.0
        if not route or not models or len(models) <= 1:
            self._last_groups = None
            return
        tok = getattr(self, "_route_tokenizer", None)
        if tok is None:
            proc = getattr(self.teacher_model, "input_processor", None)
            tok = getattr(proc, "tokenizer", None) if proc is not None else None
            self._route_tokenizer = tok
        if tok is None:
            self._last_groups = None
            return

        texts = tok.batch_decode(ids, skip_special_tokens=True)
        suite_order = self._slot_suite_order()
        slot_ids = match_suite_ids(texts, route, suite_order)

        default_path = next(iter(models))
        groups = {}
        for i, sid in enumerate(slot_ids):
            suite = suite_order[sid] if sid >= 0 else None
            path = (
                self.teacher_suite_to_path.get(suite, default_path)
                if suite
                else default_path
            )
            groups.setdefault(path, []).append(i)
        self._last_groups = groups
        self._slot_fallback = sum(1 for s in slot_ids if s < 0) / max(bsz, 1)
        if getattr(self, "_slot_enabled", False):
            self._slot_gate_ids = torch.as_tensor(
                slot_ids, dtype=torch.long, device=ids.device
            )

    def _slot_suite_order(self):
        """The slot index -> suite mapping. Single source of truth, cached."""
        order = getattr(self, "_slot_order_cache", None)
        if order is None:
            scfg = self.cfg.actor.model.get("slot_lora", None)
            if scfg is not None and scfg.get("enabled", False):
                order = list(scfg["slot_order"])
            else:
                order = sorted((getattr(self, "teacher_suite_to_path", None) or {}).keys())
            self._slot_order_cache = order
        return order
```

- [ ] **Step 2: 让 `_teacher_forward` 复用而不是重算**

在 `_teacher_forward` 里，把从 `texts = _tok.batch_decode(...)` 到 `self._last_groups = groups` 这一段替换为：

```python
        if getattr(self, "_last_groups", None) is None:
            self._route_prepare(forward_inputs)
        groups = self._last_groups
        if not groups:
            return self.teacher_model(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )
```

（`_route_prepare` 已经把 `self._last_groups` 设好；这里只做兜底，保证单独调用 `_teacher_forward` 的旧路径行为不变。）

- [ ] **Step 3: 在学生前向外面套 gate**

在 `run_training` 的 micro-batch 循环里，把

```python
                    with self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
```

改为

```python
                    # Route BEFORE the student forward -- see _route_prepare's docstring.
                    self._last_groups = None
                    if self.cfg.algorithm.adv_type == "opd" and forward_inputs is not None:
                        self._route_prepare(forward_inputs)
                    with slot_gate(getattr(self, "_slot_gate_ids", None)), self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
```

文件顶部加 `from rlinf.models.slot_lora import slot_gate`，并在 `__init__` 里（`self.teacher_model = None` 附近）加：

```python
        _scfg = self.cfg.actor.model.get("slot_lora", None)
        self._slot_enabled = bool(_scfg is not None and _scfg.get("enabled", False))
        self._slot_gate_ids = None
        self._slot_fallback = 0.0
```

- [ ] **Step 4: 验证 import 不炸**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -c "
import rlinf.workers.actor.fsdp_actor_worker as m
assert hasattr(m.FSDPActorWorker if hasattr(m,'FSDPActorWorker') else object, '__name__') or True
import inspect, rlinf.workers.actor.fsdp_actor_worker as w
src = inspect.getsource(w)
assert '_route_prepare' in src and 'slot_gate(' in src
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/workers/actor/fsdp_actor_worker.py
git commit -m "fix(slot-lora): compute prompt routing before the student forward, not after"
```

---

### Task 10: 交替调度 B,B,A（每个 step 以 B 收尾）

**Files:**
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py`（`run_training` 里 `self.optimizer_step()` 调用处，约 2645 行）

- [ ] **Step 1: 加调度辅助方法**

加到 worker 类里：

```python
    def _slot_capture_lrs(self):
        """Snapshot the scheduler-set lr of every group at the START of a training step.

        The alternating schedule zeroes a group's lr right before each optimizer_step; the
        snapshot is what it restores from, so a zeroed lr can never leak into the next step.
        """
        self._slot_lrs = [g["lr"] for g in self.optimizer.param_groups]

    def _slot_apply_phase(self, update_index: int, is_last_update_of_step: bool):
        """Freeze one factor by setting its lr to 0 and dropping its grads.

        Returns the phase actually applied ('A', 'B', or '-' when slots are off).

        Grads must be cleared, not just lr-zeroed: AdamW would otherwise keep folding the
        frozen factor's gradients into exp_avg / exp_avg_sq, so the first step after it
        thaws would move on momentum accumulated while it was supposed to be still.
        """
        if not getattr(self, "_slot_enabled", False):
            return "-"
        schedule = self.cfg.actor.model.slot_lora.get("alt_schedule", None)
        if not schedule:
            return "-"
        phase = "B" if is_last_update_of_step else schedule[update_index % len(schedule)]
        if not hasattr(self, "_slot_lrs"):
            self._slot_capture_lrs()
        for group, lr in zip(self.optimizer.param_groups, self._slot_lrs):
            name = group.get("name", "actor")
            if name not in ("slot_A", "slot_B"):
                continue
            active = name == f"slot_{phase}"
            group["lr"] = lr if active else 0.0
            if not active:
                for p in group["params"]:
                    p.grad = None
        return phase
```

- [ ] **Step 2: 在 run_training 里接上**

`run_training` 开头（`self._dw_refresh_weights()` 之后）加：

```python
        self._slot_capture_lrs()
        self._slot_update_index = getattr(self, "_slot_update_index", 0)
        _slot_total_updates = update_epoch * (rollout_size // batch_size_per_rank)
        _slot_done_updates = 0
```

把

```python
                grad_norm, lr_list = self.optimizer_step()
```

改为

```python
                _slot_done_updates += 1
                _slot_phase = self._slot_apply_phase(
                    self._slot_update_index,
                    is_last_update_of_step=(_slot_done_updates == _slot_total_updates),
                )
                self._slot_update_index += 1
                grad_norm, lr_list = self.optimizer_step()
```

并把 `_slot_phase` 记进 metrics：

```python
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if _slot_phase != "-":
                    data["slot/phase_is_A"] = 1.0 if _slot_phase == "A" else 0.0
```

- [ ] **Step 3: 单测调度逻辑**

新建 `tests/unit_tests/test_slot_lora_schedule.py`：

```python
def phase_for(update_index, is_last, schedule="BBA"):
    return "B" if is_last else schedule[update_index % len(schedule)]


class TestAlternatingSchedule:
    def test_bba_cycle(self):
        assert [phase_for(i, False) for i in range(6)] == list("BBABBA")

    def test_last_update_of_a_step_is_forced_to_b(self):
        assert phase_for(2, True) == "B"

    def test_two_thirds_of_updates_train_b(self):
        phases = [phase_for(i, False) for i in range(45)]
        assert phases.count("B") == 30 and phases.count("A") == 15
```

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_schedule.py -v
```
Expected: 3 passed

- [ ] **Step 4: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/workers/actor/fsdp_actor_worker.py tests/unit_tests/test_slot_lora_schedule.py
git commit -m "feat(slot-lora): B,B,A alternating schedule with forced B ending per step"
```

---

### Task 11: 指标

**Files:**
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py`

- [ ] **Step 1: 每个 training step 布一次诊断**

在 `run_training` 里 `self._slot_capture_lrs()` 之后加：

```python
        if getattr(self, "_slot_enabled", False):
            from rlinf.models.slot_lora import enable_slot_diag

            enable_slot_diag(self.model)
```

- [ ] **Step 2: 步末收集**

在 `self.lr_scheduler.step()` 之前、metrics 汇总之前加：

```python
        if getattr(self, "_slot_enabled", False):
            from rlinf.models.slot_lora import collect_slot_diag

            _sd = collect_slot_diag(self.model)
            _sd["slot/route_fallback_frac"] = float(getattr(self, "_slot_fallback", 0.0))
            for _k, _v in _sd.items():
                append_to_dict(metrics, {_k: _v})
```

- [ ] **Step 3: 上线后要盯的数（写进 PR/日志说明，不是代码）**

| 指标 | 期望 | 不符合说明 |
|---|---|---|
| `slot/orth_err`（**fp32 口径**） | < 1e-3 | 1e-3~5e-2 = Z 正在退化；> 5e-2 = 已塌，结果无效。**不能读 bf16 的 Ā**：cond(Z) 10→20 时 fp32 误差涨 16 倍而 bf16 读数纹丝不动（0.01887→0.01894） |
| `slot/cos_s_t`（6 个） | \|·\| < 1e-3 | > 5e-2 = 塌。与 orth_err 复用同一个 fp32 gram |
| `slot/route_fallback_frac` | **必须恰为 0** | 非 0 = 路由表有洞。这类样本被「恰好第一个加载的」teacher 打分并计入 loss，梯度却进不了任何 slot —— 硬错误，停下修表，不要接着看结果 |
| `slot/phase_is_A` | 约 1/3 | 交替调度没生效 |
| `actor/dynw_*` | 随步数变化 | 恒为 1.000 = 动态权重仍是死的 |
| `slot/dw_norm_k` | 逐步增长，long 最大 | 某个 slot 恒 0 = 该 suite 从未被路由到 |

- [ ] **Step 4: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add rlinf/workers/actor/fsdp_actor_worker.py
git commit -m "feat(slot-lora): log orthogonality, per-slot capacity and routing metrics"
```

---

### Task 12: checkpoint → merged HF 模型

**Files:**
- Create: `opd_distill/scripts/convert_oft_slot_ckpt.py`
- Create: `opd_distill/scripts/convert_oft_slot_ckpt.sh`

- [ ] **Step 1: 复制现有转换器作为骨架**

```bash
cd /home/fanruochen/CL/RLinf/opd_distill/scripts
cp convert_oft_lora_ckpt.py convert_oft_slot_ckpt.py
cp convert_oft_lora_ckpt.sh convert_oft_slot_ckpt.sh
```

- [ ] **Step 2: 换掉 PEFT 包装与合并两段**

在 `convert_oft_slot_ckpt.py` 里，把 `get_peft_model(...)` + `merge_and_unload()` 那两步换成：

```python
def wrap_with_slots(model, slot_ranks, scale_mode, ref_rank, eps):
    """Rebuild the EXACT training-time structure so the checkpoint keys line up."""
    from rlinf.models import SLOT_LORA_TARGET_MODULES
    from rlinf.models.slot_lora import inject_slot_lora

    replaced = inject_slot_lora(
        model, slot_ranks, SLOT_LORA_TARGET_MODULES,
        scale_mode=scale_mode, ref_rank=ref_rank, eps=eps,
    )
    print(f"    injected {len(replaced)} SlotLoRALinear")
    return model


def merge_slots(model):
    """W <- W + scale * B Ā, then swap each SlotLoRALinear back to its plain nn.Linear.

    Ā is recomputed from the stored Z with the SAME fp32 routine used in training, so the
    merged model reproduces the training-time forward exactly. Z is what lives in the
    checkpoint -- never Ā, and never a random seed to be replayed.
    """
    import torch

    from rlinf.models.slot_lora.modules import SlotLoRALinear

    merged = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, SlotLoRALinear):
                continue
            with torch.no_grad():
                delta = child.delta_weight()
                base = child.base
                base.weight.add_(delta.to(base.weight.dtype))
            setattr(parent, child_name, base)
            merged += 1
    print(f"    merged {merged} slot layers into the base weights")
    return model
```

在 `main()` 里：`build_base_model(...)` 之后调 `wrap_with_slots(...)`，`load_state_dict` 之后调 `merge_slots(model)`，随后沿用原脚本的 `save_pretrained` + 拷贝 aux 文件那几步不变。CLI 新增参数：

```python
    # `scale` is persisted in the checkpoint via SlotProj.get_extra_state, so the
    # converter must VERIFY the value it derives from these flags against the stored one
    # and abort on mismatch. Recomputing it blindly is how training and conversion end up
    # silently disagreeing by a constant factor: the merge still succeeds, the model still
    # runs, and every delta is simply the wrong size.
    ap.add_argument("--slot-ranks", default="128,64,48,16",
                    help="comma-separated, in slot_order (10,goal,spatial,object)")
    ap.add_argument("--slot-scale-mode", default="match_mt4", choices=["match_mt4", "unit"])
    ap.add_argument("--slot-ref-rank", type=int, default=128)
    ap.add_argument("--slot-eps", type=float, default=1e-6)
```

`load_state_dict` 之后必须保留原脚本那三行断言（`unexpected == 0`、`missing == 0`），它们是键名对齐的唯一保险。

`convert_oft_slot_ckpt.sh` 里把调用的 py 文件名换成 `convert_oft_slot_ckpt.py`，并把 `--lora-rank` 换成 `--slot-ranks "$SLOT_RANKS"`（默认 `128,64,48,16`）。

**合并的验收容差（Task 4 实测）**：合并后的模型每层相对训练时 forward 的相对误差约 **3.4e-3**，这是把 ΔW 舍入进 bf16 base weight 的地板（`‖Ā_bf16 − Ā_fp32‖/‖Ā‖ = 1.66e-3`），不是 merge 错误。**不要期待逐位相等**，用 ~1e-2 的相对容差。另外 `delta_weight()` 必须走 forward 同一条路线（`orth_weight()`），不要在转换器里用 fp32 重新推导 Ā —— 转换器的职责是复现被训练的那个模型，不是改进它（实测同路线 3.70e-3 vs fp32 重推 4.05e-3，同路线在每个宽度每个 seed 都更近）。

- [ ] **Step 3: 用玩具模型验证 merge 数学**

```bash
cd /home/fanruochen/CL/RLinf && /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python - <<'PY'
import torch, torch.nn as nn
from rlinf.models.slot_lora.modules import SlotLoRALinear

base = nn.Linear(16, 6, dtype=torch.float64)
layer = SlotLoRALinear(base, [2, 3], scale=0.7)
with torch.no_grad():
    layer.slot_B.weight.copy_(torch.randn_like(layer.slot_B.weight))
x = torch.randn(5, 16, dtype=torch.float64)
before = layer(x)

merged = nn.Linear(16, 6, dtype=torch.float64)
with torch.no_grad():
    merged.weight.copy_(layer.base.weight + layer.delta_weight())
    merged.bias.copy_(layer.base.bias)
print("max abs diff:", (merged(x) - before).abs().max().item())
assert torch.allclose(merged(x), before, atol=1e-9)
print("OK")
PY
```
Expected: diff ~1e-15，然后 `OK`

- [ ] **Step 4: 提交**

```bash
cd /home/fanruochen/CL/RLinf
git add opd_distill/scripts/convert_oft_slot_ckpt.py opd_distill/scripts/convert_oft_slot_ckpt.sh
git commit -m "feat(slot-lora): converter that merges slot checkpoints into a plain HF model"
```

---

### Task 13: R1 配置与启动脚本

**Files:**
- Create: `examples/embodiment/config/libero_mt4slot_6gpu.yaml`
- Create: `examples/embodiment/incremental_sft/opd_mt4slot.sh`

- [ ] **Step 1: 配置**

从 mt4 用的配置起手：

```bash
cd /home/fanruochen/CL/RLinf
cp examples/embodiment/config/libero_seqcl_opd_2gpu.yaml examples/embodiment/config/libero_mt4slot_6gpu.yaml
```

在 `actor.model` 下加（注意是 `actor.model`，不是 `actor`——rollout worker deepcopy 的是 `cfg.actor.model`）：

```yaml
    slot_lora:
      enabled: true
      slot_order: [libero_10, libero_goal, libero_spatial, libero_object]
      slot_ranks: {libero_10: 128, libero_goal: 64, libero_spatial: 48, libero_object: 16}
      a_scale_mode: match_mt4
      a_scale_ref_rank: 128
      alt_schedule: "BBA"
      orth_eps: 1.0e-6
```

`algorithm` 下加：

```yaml
  distill_dyn_weight: 1.0
  distill_w_ema: 0.9
  distill_w_min: 0.25
  distill_w_max: 4.0
```

`actor.teacher_map` 换成 mt4 的四个专家：

```yaml
  teacher_map:
    libero_spatial: "/share/fanruochen-local/outputs/inc_sft_opd/base_stats130::/share/fanruochen-local/outputs/seqcl_rlspat_opd/spatial_cat_r160"
    libero_object:  "/share/fanruochen-local/outputs/inc_sft_opd/base_stats130::/share/fanruochen-local/outputs/inc_sft_opd/sft_object_base3_cont/openvla-7b-base+libero_object_no_noops+b32+lr-0.0003+lora-r32+dropout-0.0--image_aug/adapters/step_500"
    libero_goal:    "/share/fanruochen-local/outputs/inc_sft_opd/base_stats130::/share/fanruochen-local/outputs/inc_sft_opd/teachers_r160/goal_r160"
    libero_10:      "/share/fanruochen-local/outputs/inc_sft_opd/lwf_long_e1000_merged::/share/fanruochen-local/outputs/seqcl_long_opd130/adapter/long_opd130"
```

**不要**设 `anchor_lambda` / `distill_fail_alpha`：mt4 / mt4w2 的配置里就没有这两项，加了对照就不成立。

- [ ] **Step 2: 启动脚本**

`examples/embodiment/incremental_sft/opd_mt4slot.sh`，照 `run_serial_cycle.sh` 的前置检查（显存 < 5000 MiB **且** 利用率采样 3 次多数 < 20%，加 `df`）：

```bash
#!/bin/bash
# opd_mt4slot.sh -- R1: slot-LoRI joint 4-teacher OPD from the mt4 starting point.
# Controls are mt4 (0.745) and mt4w2 (0.750); everything except the student's LoRA
# structure is held at mt4's values.
set -uo pipefail
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
O=/share/fanruochen-local/outputs
START="$O/inc_sft_opd/lwf_long_e1000_merged"   # mt4's exact starting model
TAG="${TAG:-mt4slot}"
GPUS="${GPUS:-0,1,2,3,4,5}"
STEPS="${STEPS:-15}"
PORT="${PORT:-63000}"

[ -f "$START/model.safetensors.index.json" ] || { echo "ABORT: start is not an HF model dir: $START"; exit 1; }

# Memory-only GPU checks miss compute-bound tenants (2026-08-20: <2% memory, 73-97% util).
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 5000 )) || { echo "ABORT: GPU$g holds ${used} MiB -- not ours to take"; exit 1; }
  busy=0
  for _ in 1 2 3; do
    util=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    (( util < 20 )) || busy=$((busy+1)); sleep 2
  done
  (( busy < 2 )) || { echo "ABORT: GPU$g at ${util}% util -- not ours to take"; exit 1; }
done
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= 120 )) || { echo "ABORT: only ${free}G free (this run writes ~29G)"; exit 1; }
echo "== df =="; df -h /share/fanruochen-local | tail -1

set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

SEQCL_GPUS="$GPUS" SEQCL_STUDENT_PATH="$START" \
SEQCL_ACTIVE_SUITES='[libero_spatial,libero_object,libero_goal,libero_10]' \
SEQCL_SUITE_WEIGHTS='null' SEQCL_MAX_STEPS="$STEPS" SEQCL_CURRENT_SUITE="$TAG" \
ISO_RAY_PORT="$PORT" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_mt4slot_6gpu \
    algorithm.rollout_epoch=3 \
    env.train.total_num_envs=48 \
    actor.micro_batch_size=8 \
    actor.global_batch_size=192
echo "SLOT_TRAIN_DONE $(date '+%F %T')"
```

`chmod +x` 之。

- [ ] **Step 3: 提交**

```bash
cd /home/fanruochen/CL/RLinf
chmod +x examples/embodiment/incremental_sft/opd_mt4slot.sh
git add examples/embodiment/config/libero_mt4slot_6gpu.yaml examples/embodiment/incremental_sft/opd_mt4slot.sh
git commit -m "feat(slot-lora): R1 config and launcher matched to the mt4 baseline"
```

---

### Task 14: 交付给用户手动启动

heavy run（训练、eval rollout、大写盘）**不自动跑** —— 把命令交给用户。

- [ ] **Step 1: 全量单测**

```bash
cd /home/fanruochen/CL/RLinf
/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python -m pytest tests/unit_tests/test_slot_lora_*.py -v
```
Expected: 54 passed

- [ ] **Step 2: 把这段交给用户，由用户自己跑**

```bash
cd /home/fanruochen/CL/RLinf
bash /share/fanruochen-local/dev/scripts/safe_run.sh \
  bash -c 'TAG=mt4slot GPUS=0,1,2,3,4,5 STEPS=15 PORT=63000 \
    bash examples/embodiment/incremental_sft/opd_mt4slot.sh' \
  2>&1 | tee /share/fanruochen-local/outputs/opd_mt4slot_driver.log
```

- [ ] **Step 3: 前三步必须人工核对（不对就停，别等跑完）**

1. 启动日志里 `[slot-lora] injected N SlotLoRALinear ... R=256`，N 与 mt4 的 LoRA 层数一致；
2. `slot/orth_err` < 1e-3、6 个 `slot/cos_*` 全 < 1e-3（**fp32 口径**）—— 超过 5e-2 就是正交性塌了，整个 run 无意义；落在 1e-3~5e-2 之间说明 Z 在退化，别硬跑；
3. `slot/route_fallback_frac` **恰为 0**（非 0 是硬错误，不是警告——见 Task 11 的表）；
4. `actor/dynw_*` 从第 2 步起不再是 1.000；
5. **步 1-2 慢 5-6 倍是 inductor 预热，不是 bug**，第 3 步之前不要判断性能、也不要给 ETA。

- [ ] **Step 4: 训练结束后转换 + 评测（同样交给用户）**

```bash
CUDA_VISIBLE_DEVICES="" bash /home/fanruochen/CL/RLinf/opd_distill/scripts/convert_oft_slot_ckpt.sh \
  <full_weights.pt> /share/fanruochen-local/outputs/seqcl_mt4slot/converted/mt4slot \
  /share/fanruochen-local/outputs/inc_sft_opd/lwf_long_e1000_merged
```

然后按现有 post-hoc 口径评测：**temp 1.0、50 env、四个 suite 分别评**，与 mt4 (0.745) / mt4w2 (0.750) 对比。**训练内 per-suite SR 一律不作数**（实测两个方向上都偏差过 ±0.33），只用来看进程是否活着。

---

## 自查

**Spec 覆盖**：Z 参数化 → Task 1-2；逐 slot detach 门控 → Task 3；SlotLoRALinear + merge → Task 4/12；路由（含顺序陷阱）→ Task 5/9；per-slot rank → Task 6/13；尺度对齐 → Task 4/6；FSDP 叶子模块约束 → Task 2/3；配置挂在 `actor.model` 保证 rollout 同构 → Task 7/13；slot_A 的 wd=0 → Task 8；BBA + 每 step B 收尾 + 冻结组清 grad → Task 10；观测量 → Task 6/11；转换 → Task 12；实验/判据 → Task 13/14。**YAGNI 三项（B 稀疏 mask、数据驱动 A 初始化、推理期路由）无对应任务，符合预期。**

**命名一致性**：`slot_A` / `slot_B` 作为参数名与 optimizer group 名在 Task 4/8/10 一致；`orth_weight()` 在 Task 2/4/12 一致；`match_suite_ids` 在 Task 5/9 一致；`inject_slot_lora` 签名 `(model, slot_ranks, target_modules, scale_mode, ref_rank, eps)` 在 Task 6/7/12 一致；`enable_slot_diag` / `collect_slot_diag` 在 Task 6/11 一致。
