# slot-LoRI joint 多 teacher OPD — 设计文档

日期：2026-08-21
状态：设计已确认，待写实现计划
相关：`project_mt4_4teacher_opd_result`（要打破的对照）、LoRI（arXiv 2504.07448, COLM 2025）

---

## 1. 问题

joint 4-teacher routed OPD 跑了两次，都只**搬运**能力、不**增加**能力：

| 运行 | 变化 | 四 suite 均值 |
|---|---|---|
| mt4 | long +0.28 / goal −0.20 | 0.745 |
| mt4w2 | goal +0.12 / long −0.10 | 0.750（0.12 SE = 没动） |

两次的涨跌互为镜像，spatial / object 从不动。诊断：四个 teacher 的 KL 梯度在**同一个 backward** 里落到**同一份 r=128 的学生 LoRA** 上，彼此没有任何隔离手段，最终解是四路梯度的折中，而不是四份能力的叠加。

优化本身没问题（distill_loss 1.090→0.364，opd_kl 1.238→0.657 单调下降），所以瓶颈不在"学不动"，在"学到的互相覆盖"。

## 2. 方案

把学生的单份 LoRA 拆成 **4 个 slot**，每个 suite 一个，**子空间严格正交**，梯度按 suite 硬路由。

ΔW = Σ_k B_k Ā_k，其中 Ā_k 是 Ā 的第 k 个行块，Ā 的所有行两两正交。

正交为什么解决问题：两个 slot 更新的 Frobenius 内积

    ⟨ΔW_s, ΔW_t⟩_F = tr( B_sᵀ B_t · Ā_t Ā_sᵀ )

只要 Ā_t Ā_sᵀ = 0，内积恒为 0，**与 B 无关**。B 怎么训都不会让两个 slot 互相干扰。这是构造出来的性质，不依赖数据、不依赖训练过程。

**四个 slot 在前向中永远全部激活**，推理不做路由，最终 merge 成一个 HF 模型。这不是 MoE / adapter zoo：slot 只是训练期的梯度隔离手段，产物仍是单模型。

### 2.1 Z 参数化（正交性进计算图）

不把 Ā 当参数，而是存自由矩阵 **Z ∈ ℝ^{R×d_in}**（R = Σ r_k），前向里现算

    Ā = (Z Zᵀ)^(-1/2) · Z          （对称/Löwdin 正交化）

整个式子对 Z 可微（`torch.linalg.eigh` 支持反向），autograd 直接给出约束梯度，**不需要任何 optimizer 之后的投影/retraction 步骤**。正交性由构造保证，任何时刻 ĀĀᵀ = I。

选它而不是"训完再正交化"的理由：后者是 optimizer.step() 之后的事后处理，不进计算图，训练目标和实际参数更新之间有一层不可见的修正；Z 参数化没有这个缝。

数值细节：
- 逆平方根用 **coupled Newton–Schulz 迭代**算，不用 `eigh`。这不是备选方案而是默认，理由是实测的（A100，R=256，d=4096，200 层）：`torch.linalg.eigh` 单次 6.87 ms、**且是一次硬 device-host 同步**（排队 155 ms 的 GPU 工作后调用，Python 侧阻塞 156 ms 才返回；同样情况下 matmul 只要 0.01 ms）。200 层前向 eigh 要 1316 ms，而这点计算量只值一次 4096³ matmul 的 6.5 ms —— 约 500 倍的效率差，外加每层一次同步会把 FSDP 的 all-gather/compute overlap 全部打掉。按本设计的 micro 8 / global_batch 192，单次优化器更新 24 个 micro-batch，光正交化就约 31 s，一个 step 约 95 s。batch 化没用（256×256 的 batched syevd 在 cuSOLVER 上没有快路径，实测 1236 ms 几乎不变）。
  NS(8) 实测前向 211 ms、前反向 766 ms（eigh 1316/1409），排队后 1.52 ms 返回、**无同步**，而 bf16 输出精度与 eigh 打平（orth_error 0.00949 vs 0.00960，两者都已压在 bf16 舍入地板上）。
- 附带修掉一个 eigh 特有的坑：`eigh` 的反向含 `1/(λ_i − λ_j)`，**谱严格简并时梯度是 NaN 而前向完全正常**（实测：Z 的行本身正交时前向 orth_error 5e-15、梯度 NaN）。NS 在同一输入上梯度有限。因此 Z **必须**高斯初始化，绝不能用 `nn.init.orthogonal_`。
- 全过程在 **`torch.autocast(enabled=False)` 且 TF32 关闭**里做（两者都是保存-禁用-恢复）。TF32 是第二条独立的 fp32 泄漏路径，本仓库有两处进程级打开它；实测开着 TF32 时 bf16 地板从 0.0189/0.0094 抬到 0.0292/0.0248，fp32 输入更是从 7.9e-6 掉到 2.03e-2 —— 会把「健康」和「已塌」两档挤到分不开。autocast 是按 op 拦截的，显式 cast 成 fp32 的张量照样被 `matmul` 降回 bf16：实测在 `autocast(bfloat16)` 内 orth_error 2.5e-2、外面 4.4e-6，差 4 个数量级且不报任何错。学生前向恰好就在 `self.amp_context` 里（`fsdp_actor_worker.py:2130`）。
- 精度按 **floor** 处理：bf16/fp16 提升到 fp32，fp32/float64 用自己的 dtype，最后 cast 回输入 dtype。
- NS 的归一化让**尺度不变性严格成立**（Z→cZ 时 M→c²M，归一化把 c 约掉），不再依赖对特征值做绝对阈值 clamp —— 旧写法的绝对 `eps` 会在 Z 整体缩小时破坏尺度不变性（实测 c=1e-4：偏差 9.0e-3、orth_error 0.447），而尺度不变性正是 wd=0 的论据。
- R > d_in 会让 Gram 秩亏、静默返回非正交结果（实测 R=64/d=32：orth_error 5.657、不报错）。加一条纯 shape 守卫直接 raise。
- Ā 对 Z 的**尺度不变**（Z → cZ 得到同一个 Ā）。因此 **Z 这一组的 weight_decay 必须设 0**：wd 只会把 Z 拉向 0、恶化 ZZᵀ 的条件数，对函数没有任何影响。这是一个必须显式处理的坑（全局 wd = 0.01）。
- Z 初始化：对每个被 LoRA 的权重矩阵独立采一个 R×d_in 高斯阵。随机高斯阵满秩，正交化后即为一组随机正交基；不需要额外 QR。

### 2.2 尺度对齐

现有学生 LoRA：`r=128, lora_alpha=128 → scaling = 1.0`，`init_lora_weights="gaussian"`（A 的元素 std = 1/r）。所以 mt4 里 A 的行范数 ≈ √d_in / 128。

Ā 的行是单位向量（范数 1），直接用会让 ΔW 每步比 mt4 大若干倍，混淆"结构变了"和"学习率变了"。因此乘一个固定常数：

    Ā_used = s · Ā,   s = √d_in / 128   （逐模块，按该模块的 d_in 算）

其余 scaling 保持 1.0。这样 step 1 的 ΔW 量级与 mt4 一致，消融里学习率不是变量。配置项 `slot_a_scale_mode: match_mt4 | unit`，默认 `match_mt4`。

副作用（接受并记录）：slot k 的每步步长随 √r_k 增长，即 rank 大的 slot 不仅子空间大、步子也略大。这与"给 long 多投入"的意图同向。

### 2.3 per-slot rank

底座 `lwf_long_e1000_merged` 的 post-hoc 起点（temp 1.0 / 50 env）：long 0.56、goal 0.68、spatial 0.76、object 0.96，均值 0.740。按赤字分配容量：

| slot | suite | 起点 SR | 赤字 | rank |
|---|---|---|---|---|
| slot_10 | libero_10 (long) | 0.56 | 0.44 | 128 |
| slot_goal | libero_goal | 0.68 | 0.32 | 64 |
| slot_spatial | libero_spatial | 0.76 | 0.24 | 48 |
| slot_object | libero_object | 0.96 | 0.04 | 16 |

R = 256。d_in 最小的被 LoRA 模块（vision 侧约 1024）也远大于 256，正交基一定存在。

## 3. 模块结构

新建 `SlotLoRALinear`，替换学生侧的 PEFT LoRA 注入（teacher / anchor 侧完全不动，仍走 PEFT）。

```
SlotLoRALinear(base_linear, slot_ranks, scale)
├── base   : nn.Linear，frozen（requires_grad=False）
├── slot_A : SlotProj  — weight = Z (R × d_in)，trainable，叶子模块
└── slot_B : SlotOut   — weight = B (d_out × R)，trainable，零初始化，叶子模块
```

前向：

```
h  = slot_A(x)                    # 内部：Ā = orth(Z); return F.linear(x, s·Ā)  -> (..., R)
out = base(x) + slot_B(h, gates)  # 内部：按 rank 块切 h，逐 slot 算贡献后带门求和
```

`SlotOut.forward(h, gates)`：

```
c_k = F.linear(h[..., off_k:off_k+r_k], B[:, off_k:off_k+r_k])      # slot k 的贡献
out = Σ_k [ c_k                if 该样本属于 suite k
          | c_k.detach()       otherwise ]
```

**为什么必须逐 slot 算贡献再 detach，而不是在 h 上乘门**（这条论证 2026-08-21 被变异测试纠正过一次，记下正确版本）：在 h 上乘 mask 的**梯度路由其实是对的** —— ∂L/∂B_j = Σ_i grad_out_i ⊗ (mask_i⊙h_i)_j，非归属块为 0。错的是**前向值**：out_i 只剩 owner slot 的贡献，模型不再是四个 slot 合并后的整体，训练和推理对不上，「全 slot 永远激活」这个前提直接没了。实测变异（把实现换成 mask h）只有「前向值等于全 slot 和」那一条测试失败，**所有梯度测试都通过** —— 所以那条测试是这一整类错误实现的唯一防线，不可删。
`torch.where(owns, c, c.detach())` 同时满足两边：前向 `c` 与 `c.detach()` 数值相同所以输出仍是完整和，反向只走 owner。

**门控的开销（已实测，暂不优化）**：逐 slot 循环比单次 `F.linear` 多约 390 ms/micro-batch（B=8/T=512/R=256/K=4/d_out=4096，按 300 个模块折算：480 ms vs 92 ms），一个 training step（72 个 micro-batch）约 28 s，开 gradient checkpointing 前向重跑后翻倍。相对 mt4 约 14 分钟/步是 3-6%，先不动。若 profiling 显示这块吃掉了步时间，有一个现成的等价替换：`out = full + (m - m.detach())`，其中 `full = F.linear(h, W)`（no_grad）、`m = F.linear(h * mask, W)`，前向 bit-exact、fp64 下梯度与循环版 max|diff| = 0.0，kernel 数从约 40 降到约 4（实测 210 ms vs 480 ms）。显存不构成理由：`torch.where` 版与零拷贝 autograd.Function 实测峰值完全相同（116.3 MiB），因为 `where` 的 backward 只 save condition，中间张量在重绑定时立即释放。

### 3.1 FSDP 约束

`use_orig_params=False`（`rlinf/config.py:419`），同一个 flat param 内 requires_grad 必须一致。现有 LoRA 逐叶子包策略（`rlinf/hybrid_engines/fsdp/utils.py:306`）的判据是"无子模块 + 有 `.weight` + `weight.requires_grad`"。

`SlotProj` / `SlotOut` 都满足（无子模块、持有 `.weight`、可训），会各自成为独立 FSDP 单元 → requires_grad 均匀，且**在各自 forward 内部 weight 已被 all-gather 成完整张量**，这正是 `Ā = orth(Z)` 需要的（必须拿到全部 R 行才能算 ZZᵀ）。

frozen 的 `base` 是叶子但 `weight.requires_grad=False`，不会被单独包，随所在 transformer 层一起被包 —— 与现在 PEFT 的情形一致。

## 4. 路由

复用现成的 teacher 路由表：`self.teacher_prompt_to_suite`（`fsdp_actor_worker.py:1176` 构建，`:1230 _teacher_forward` 使用，分组结果写在 `self._last_groups`，`:1300`）。

- teacher 用哪张表分 suite，slot 就用同一张表 —— **同一个样本的 teacher 和 slot 必然配对**，不存在"用 goal 的 teacher 去更新 long 的 slot"。
- **顺序陷阱（必须处理）**：学生前向在 `fsdp_actor_worker.py:2131`，`_teacher_forward` 在 `:2155` —— teacher 在**后**。所以 `self._last_groups`（`:1300` 才写入）在学生前向时是**上一个 micro-batch 的**，直接拿来当 gate 会整体错位一个 micro-batch。必须把"decode prompt → suite"抽成 `_route_prepare(forward_inputs)`，在**学生前向之前**调用，`_teacher_forward` 复用其结果（顺带省掉一次 `batch_decode`）。
- 每个 micro-batch：`_route_prepare` 产出的 per-sample gate 张量通过**共享 holder** 交给所有 `SlotOut`（`with gate.scoped(ids):`）。**不能用 `ContextVar`** —— autograd 在 CUDA 上用每设备 worker 线程跑反向，gradient checkpointing 的重算发生在那里，ContextVar 会读到 `None` 从而静默地整个不门控（已实测复现）。holder 是普通对象属性，不受线程和重算影响。
- **作用域必须同时覆盖 forward 和 backward**：checkpoint 会重跑 forward，若作用域在 `.backward()` 前就退出，strict 模式下会 raise（而不是静默 ungated）。
- prompt 匹配不上时的 fallback：与 teacher 侧**完全一致**（落到 default），并记录 `route_fallback_frac`。这四个 suite 的 40 个任务 prompt 都在表里，预期为 0；非 0 说明路由表有洞，必须先修再看结果。

## 5. 交替训练调度

按**优化器更新**交替，不按训练步交替（一次 rollout 内有 `global_batch 192` × 3 次更新）：

    B, B, A, B, B, A, ... 且最后一次更新必为 B

- `slot_alt_schedule: "BBA"`（默认），`null` 表示 A、B 联合训练（消融用）。
- 冻结方式：**把对应参数组的 lr 设为 0**，不改 `requires_grad`（`use_orig_params=False` 下中途改 requires_grad 会破坏 FSDP flat param）。
- 同时把被冻结那组的 `p.grad = None` 再调 `optimizer.step()`：否则 AdamW 的 exp_avg / exp_avg_sq 会在冻结阶段继续吸收梯度，等该组解冻时第一步用的是上一阶段积下来的动量。
- 需要在 `fsdp_model_manager.py:496` 的 `build_optimizer` 里把 slot 参数拆成 `slot_A` / `slot_B` 两个 param group，`slot_A` 组 `weight_decay=0`。
- B 收尾：调度按 BBA 循环，但**每个训练 step 的最后一次优化器更新强制为 B**（若循环恰好落在 A，该次改为 B）。比"只保证 run 的最后一次是 B"更强，且因为 checkpoint 是按 step 存的，**每个存下来的 checkpoint 都是 B 收尾**。

理由：B 固定时 ΔW 对 A 线性，反之亦然，交替 = 坐标下降，每个子问题条件数更好（AltLoRA 报告过同样的收益）。注意**交替本身不提供正交性**——正交性完全来自 Z 参数化；交替只影响优化质量。

## 6. 动态蒸馏力度

已有实现 `_dw_refresh_weights`（`fsdp_actor_worker.py:1967`）：每步用各 suite 的 KL(student‖teacher) 做 EMA，权重 ∝ (KL_k / mean)^γ，clamp 到 [`distill_w_min`, `distill_w_max`]。它按 KL 走、不按训练内 SR 走 —— 训练内 per-suite SR 已被证明在两个方向上偏差可达 ±0.33，**任何在线自适应都不得以它为输入**。

它在 mt4w2 里因 `run_training` 重复定义被遮蔽而全程失效（权重恒为 1.000），调用点现已在生效的那个定义里。本设计首次真正启用：

    distill_dyn_weight: 1.0 / distill_w_ema: 0.9 / distill_w_min: 0.25 / distill_w_max: 4.0

在 slot 结构下它的语义更干净：各 suite 梯度落在不同 B_k 上，per-suite 权重 = **per-slot 步长倍率**，不再改变合成梯度的方向。

另外保留静态旋钮：per-suite 采样权重（`SEQCL_SUITE_WEIGHTS`），给 long 多分 env。

## 7. 产物与转换

- checkpoint 键变为 `...slot_A.weight`（Z）/ `...slot_B.weight`（B）。**存 Z 不存 Ā**，也不靠重放随机种子。
- 转换（`opd_distill/scripts/convert_oft_lora_ckpt.py` 与 `/share/fanruochen-local/dev/scripts/extract_lora_adapter.py`）需增加分支：读 Z → 用与训练**完全相同**的 fp32 过程算 Ā → `W ← W + s·B·Ā` → 存 HF 模型。
- 产物仍是一个 15G 的 merged HF 模型，**评测脚本零改动**。
- `RLINF_CONVERT_VALUE_HEAD=False` 照旧。

## 8. 配置项

配置必须挂在 **`actor.model`** 下面，不能挂在 `actor` 下面：rollout worker 在 `rlinf/workers/rollout/hf/huggingface_worker.py:95` 里 `deepcopy(cfg.actor.model)` 后调同一个 `get_model`，所以放在 `actor.model` 里 rollout 侧会**自动**得到结构完全相同的模型，权重同步（走裸 state_dict 的键名匹配，`fsdp_actor_worker.py:1630 get_rollout_state_dict`）零改动即可工作。

```yaml
actor:
 model:
  slot_lora:
    enabled: true
    slot_ranks: {libero_spatial: 48, libero_object: 16, libero_goal: 64, libero_10: 128}
    a_scale_mode: match_mt4        # match_mt4 | unit
    alt_schedule: "BBA"            # null = 联合训练
    orth_eps: 1.0e-6
algorithm:
  distill_dyn_weight: 1.0
  distill_w_ema: 0.9
  distill_w_min: 0.25
  distill_w_max: 4.0
```

## 9. 观测量

**正确性（必须先看，不对就是实现有 bug，结果无意义）**
- `slot/orth_err` = ‖ĀĀᵀ − I‖_F，**必须在 fp32 的 Ā 上算，不能读模型实际用的那个 bf16 Ā**。原因是实测的盲区：Z 的条件数从 10 涨到 20 时，fp32 误差劣化 16 倍（6.8e-5 → 1.07e-3），而 bf16 读数从 0.01887 只动到 0.01894 —— 整个劣化过程被 bf16 的舍入地板完全盖住，等它动的时候（cond 30 → 0.054，fp32 0.051）已经掉下悬崖了。失效是**断崖不是斜坡**，所以监控必须用能看见斜坡的那个量。
  实现：`SlotProj` 被 arm 的那一步，额外用 fp32 重算一次 Ā 来算 gram（一次 NS 调用约 1.3 ms，每个 training step 只在**一个**模块上做一次，可忽略）。
  判据（fp32 口径）：**< 1e-3 健康；1e-3 ~ 5e-2 = Z 正在退化，查 Z 的条件数；> 5e-2 = 正交性已塌，结果无效**。高斯初始化给出 cond(Z)≈3，`iters=12` 的验证包线约到 cond(Z)≈20，也就是只有约一个数量级的漂移余量 —— 这条曲线要每步看。
  另记一个 bf16 口径的参考值（不作判据）：R=256 时地板为 0.0189（d_in=1024）/ 0.0094（d_in=4096）；若 TF32 未被关掉会抬到 0.0292 / 0.0248
- `slot/cos_st` = cos⟨ΔW_s, ΔW_t⟩_F 全部 6 个 pair。与 orth_err 同源（复用同一个 fp32 gram），判据 **|cos| < 1e-3 健康、> 5e-2 为塌**。**不要展开 ΔW**（d_out×d_in 太贵）：用 ⟨ΔW_s,ΔW_t⟩_F = tr(B_sᵀB_t · Ā_tĀ_sᵀ) 只算 R×R 的小矩阵，Ā 的 Gram 前向里已经有了
- `slot/route_fallback_frac`，预期 0
- `slot/dynw_*`，必须随步数变化；恒为 1.000 = 机制没接上

**研究量**
- 每个 slot 的 ‖ΔW_k‖_F 随步数（容量用了多少，long 是否真的用得比别人多）
- 四条 per-suite KL(student‖teacher) 曲线
- `distill_loss` / `opd_kl` 总曲线，与 mt4 的 1.090→0.364 / 1.238→0.657 对齐比较

## 10. 实验

底座：**`inc_sft_opd/lwf_long_e1000_merged`** —— 与 mt4 的起点逐字节相同（已核 `opd_mt4_driver.log` 的 `model_path`），因此 mt4（0.745）与 mt4w2（0.750）**直接作为对照，零额外算力**。选它而不是更强的 `lwf_long_e2000_merged`（0.835）的原因就是这个：e2000 上没有任何匹配的 joint-OPD 对照，要自己再跑一次 15 步才有守恒带。

teacher / 步数 / env / batch 全部沿用 mt4：4 个按 prompt 路由的 per-suite expert，long 的 teacher 是 `lwf_long_e1000_merged::long_opd130`（与学生同血统，其 base 就是学生起点），另外三个是 `base_stats130::<per-suite adapter>`；15 步、6 GPU、envs 48、group_size 4、rollout_epoch 3、micro 8、global_batch 192。`anchor_lambda=0`、`distill_fail_alpha` 不设 —— mt4 / mt4w2 的配置里就没有这两项，保持一致。

| 跑次 | 配置 | 问题 |
|---|---|---|
| R1 | slot-LoRI + BBA + dynw | 主结果 |
| R2（条件性） | slot-LoRI + 联合训练（`alt_schedule: null`） | 交替值不值 |
| R3（条件性） | 单 LoRA r=128 + dynw（其余同 mt4） | 收益来自 slot，还是来自"dynw 第一次真的生效" |

R2/R3 只在 R1 有信号时才跑。R3 之所以仍有必要：mt4 / mt4w2 都配了 `distill_dyn_weight: 1.0`，但机制被 `run_training` 重复定义遮蔽而全程失效（权重恒为 1.000），所以那两次实际是**均匀权重**；R1 是"slot + 活的 dynw"，R3 把这两个因素分开。

**判据**（post-hoc 50 env、temp 1.0、四 suite，唯一可信口径）：
1. 四 suite 均值是否脱离 0.745 ± 0.04 的守恒带（起点 0.740，mt4 0.745，mt4w2 0.750）；
2. 是否仍出现"一涨一等量跌"的镜像。

均值涨了但仍是镜像 → 只是换了个搬运方式；均值涨且各 suite 不再互相抵消 → 结构有效。

## 11. 风险

- **可塑性**：Ā 的行被约束为正交，slot k 不能转进兄弟已占的 ≤224 维；但 d_in=4096 中还有 3840 维空闲，且自身 r_k 维内可任意旋转。判定信号：`distill_loss` 下降明显慢于 mt4（1.09→0.36）。
- **Z 的病态**：wd 必须为 0；仍需监控 ZZᵀ 的最小特征值，逼近 `orth_eps` 说明 Z 退化，应周期性行归一化（不改变 Ā）。
- **正交化的前向开销**（已实测，见 §2.1）：换成 Newton–Schulz 后 200 层 211 ms 前向、无同步。仍有一条未做的优化：Ā 是参数 Z 的纯函数、与激活无关，而 BBA 调度的 B 阶段 Z 被冻结（lr=0），此时 Ā 是常量却每个 micro-batch 重算 24 次。若前向开销仍显著，在 B 阶段缓存 Ā 即可，数学完全等价。先不做（YAGNI），实测到再说。
- **orth_err 若进入 1e-3~5e-2 带的处置**：先把 `iters` 从 12 提到 16，不要动别的（实测 cond(Z)=20 时 iters 11→6.8e-2、12→1.07e-3、13→1.62e-4）。`iters` 上界也有风险：秩亏的 Z 上零空间分量按 1.5^iters 增长，实测 iters≈50 就变 NaN，所以 16 是安全的、40 以上不是。
- **Z 的病态是静默失败**：NS 和 eigh 都一样，Z 病态时函数正常返回一个非正交的结果，不报错。唯一的哨兵是 `slot/orth_err`（见 §9 的阈值），必须每步看。
- **显存**：与 2026-08-20 夜里的 OOM 无关（那是 student+teacher+anchor 三份 7B 挤单卡）。slot 总参数量约为原 r=128 单 LoRA 的两倍，可忽略。
- **前置条件**：GPU 0-3 当前被其他租户占用（78-80% util）。启动前按 `run_serial_cycle.sh` 的双重预检（显存 + 利用率采样 3 次）确认，并 `df -h /share/fanruochen-local`（当前 97% / 806G，一段 OPD 约写 29G）。

## 12. 明确不做（YAGNI）

- **LoRI 的 B 稀疏 mask**：slot 之间参数本就不相交，mask 防遗忘的作用冗余；其唯一剩余收益是省参数量，而参数量不是瓶颈。代价是每 suite 多一段校准 OPD。留作 v2。
- **数据驱动的 A 初始化**（LoRA-GA / PiSSA 式，对 teacher 初始梯度做 SVD）：四 suite 输入高度重叠，主方向需再做跨 slot 正交化，引入"谁先占方向"的排序偏置，正好把要消除的不对称塞回来；且 on-policy 下 teacher 的梯度方向随策略移动而变。Z 参数化下 A 本来就能随数据学，此路的收益被覆盖。
- **重新拉平四 suite 的血统**（回到共同祖先重训四段）：底座固定为 mt4 的起点 `lwf_long_e1000_merged`，血统不对称原样保留，只用 rank 分配 + 采样权重 + dynw 来应对。这样换来的是 mt4 / mt4w2 两个现成对照。
- **推理期 slot 路由**：产物必须是单模型。
