# 数据无关的双-KL BASE 锚:在多-teacher OPD 中保住 VLA 泛化

**设计文档 · 2026-07-22 · OpenVLA-OFT + LIBERO**
状态:设计待审(未开始实现/训练)

---

## 0. 一句话

在 on-policy 蒸馏(OPD)把一个强 task-teacher 蒸进 VLA 学生的同时,加一个**数据无关(data-free)、熵门控的 BASE forward-KL 锚**,让学生学会新任务却不丢失 base 策略的广泛泛化;并证明**锚的 support 类型(base 自 rollout)才是保泛化的杠杆,而不是输入的广度**。

---

## 1. 定位与新意(来自 2026-07-14 文献调研)

**已被别人做掉、不能当卖点的:**
- **VLA-OPD**(arXiv:2603.26666):就是我们的 OPD 骨架(frozen expert 给 student 自 rollout 打分、reverse-KL、OpenVLA-OFT/pi0+LIBERO),而且**已经报告 OPD 比 SFT 更保 held-out 任务**。→ "OPD 学新不忘旧"不再新。
- **RETAIN / VLA 参数合并**(arXiv:2512.08333,ICLR 2026):微调 VLA 权重 × base 做 WiSE-FT 插值,单模型"继承 base 通才 + 学新任务",OOD 上打过两个父模型。→ 一个训练-free 的插值已做到"learn-better AND forget-less"。**必须打败它。**
- **SDFT**(arXiv:2601.19897)/ **MOPD**(arXiv:2606.30406):"把 base 当锚 teacher 做 reverse-KL 持续学习"、"多-teacher OPD 合并"都在 LLM 上发过。→ 纯"多-teacher CL / 把 base 当普通 teacher"没有新意。

**我们的新意所在(必须落在机制 + 非平凡发现,不是新算法):**
1. **熵门控双-KL**:治的是一个**有文献支撑的已知病因**——mode-seeking reverse-KL 会剥掉 base 的广覆盖(**EOPD** arXiv:2603.07079 量化:reverse-KL OPD 只留 6.8% 高熵 token vs teacher 18.5%)。我们用 base 的 **forward-KL(mode-covering)** 逐-token 门控地补回这层覆盖。没人在 VLA / action policy 上做过。
2. **support 类型 > 广度**(方向 B):我们自己的负结果是"在各种合成输入上保泛化都 cap 在 ~0.18、与广度无关";文献(静态生成式 replay 会 off-manifold 错位、PRISM/DDA 的 precision-recall 分解)指向真正的杠杆是**输入是否 on-policy/on-manifold**。我们用 **base 自 rollout** 作锚 support,并用消融**正面推翻"广度无关"这个旧结论**——证明是 support 类型(on-policy base)而非广度决定成败。这个"负结果→机制→翻案"本身就是一个非平凡发现。

**一句话故事(投稿口径):** 多-teacher OPD 会因 mode-seeking 丢 base 泛化;我们用数据无关、base 自 rollout support 上的**熵门控双-KL 锚**,在严格不碰原数据下同时**学新更好且泛化更少遗忘**,Pareto 打过 RETAIN 权重插值 / reverse-LiNeS / plain VLA-OPD。

---

## 2. 假设(可证伪)

- **H-A(机制)**:在 plain OPD 上加熵门控 BASE forward-KL 锚,能在**相同新任务 SR** 下取得**更高的泛化保留率**,Pareto 支配 {plain OPD, RETAIN 合并, reverse-LiNeS, WiSE-FT}。
- **H-B(support)**:锚的 support 用 **base 自 rollout** 时泛化保留显著高于 off-policy probe / 随机合成;且**广度不是主因**(窄 base-rollout ≈ 宽 base-rollout ≫ off-policy)——推翻旧"breadth-invariant ~0.18"。
- **H-null(诚实反向假设)**:若 A 打不过 RETAIN 插值,或 base-rollout support 不优于 off-policy,则本方向对 VLA 不成立,如实报告(接受 RETAIN/事后法为更优)。

---

## 3. 实验设定

**平台**:OpenVLA-OFT(离散 action token:8 chunk × 7 dim = 56 个 token,每个 256-bin 分类分布 → 熵、forward/reverse KL 精确可算)。

**三个模型(norm 全对齐——这是我们 OPD 的血泪教训,`unnorm_key` 必须一致):**
| 角色 | 权重 | 是否训练 |
|---|---|---|
| 学生 πθ | init = `RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`(130 通才) | LoRA rank 32 训练 |
| task-teacher π_T | `RLinf-OpenVLAOFT-GRPO-LIBERO-object`(单套件专家 ~0.98) | 冻结 |
| base π_B | = 学生初始权重的冻结副本(130 通才本身) | 冻结 |

**单轮 CL**:distill object 一个 teacher(object:通才 ~0.7 → 专家 ~0.98 = "学新");held-out = spatial/goal/long + object 的 PRO 扰动。

**两条数据流(都自生成、严格 data-free):**
1. **任务流**:学生 on-policy rollout 在 `libero_object` prompt 上(标准 OPD support)。
2. **锚流**:**base π_B** 在 **`libero_90`(90 任务,宽、on-manifold、与 4 个评测套件不相交)** 上 rollout,采集 `(state, base 软动作分布 p_B)` 对。**★ 一次性预生成、复用**(base 冻结 → 锚目标是静态的;省算力且红线友好,避免每步重复 rollout)。

---

## 4. 算法:熵门控双-KL

记每个 token 位置的 student/teacher/base 分类分布为 `p_θ, p_T, p_B`(V=256 bins),base 该位置的熵 `H = -Σ_v p_B(v) log p_B(v)`,`H_max = log V`。

**总损失**
```
L  =  L_task            +   λ · L_anchor
```

**任务项(在学生 on-policy 任务 rollout 上,reverse-KL / mode-seeking / 学 teacher):**
```
L_task = E_{任务 rollout} Σ_pos  D_KL( p_θ ‖ p_T )
```
→ **已实现**,复用现有 OPD:`advantages.py` 的 per-token reward `r_t = log p_T(a_t) − log p_θ(a_t)` + policy-gradient(= on-policy reverse-KL 梯度估计,MiniLLM 式)。不改。

**锚项(在 base 预生成 rollout 上,forward-KL / mode-covering / 保 base 广覆盖):**
```
L_anchor = E_{base rollout} Σ_pos  g(H_pos) · D_KL( p_B ‖ p_θ )
         = E_{base rollout} Σ_pos  g(H_pos) · Σ_v p_B(v) log( p_B(v) / p_θ(v) )
```
→ base 是固定采样分布,`p_B` 是静态软标签 → 这是一个**直接可微的 KD 软标签损失**(不需 policy-gradient),比任务项简单。`D_KL(p_B‖p_θ)` 在 `p_θ→0` 而 `p_B>0` 处爆炸 → **强制学生覆盖 base 的全部支持**(含低概率的泛化尾巴),正是治 reverse-KL 剥覆盖的病。

**门控 `g(H)`(A 的新意;方向当消融扫,不预设赢家):**
- **(i) 低熵加权**(默认主设):`g = ((H_max − H)/H_max)^κ` —— base 置信(给确定动作)处锚更狠 = 保住承载泛化的确定行为。
- **(ii) 高熵加权**(EOPD 式):`g = (H/H_max)^κ` —— 高熵处上 mode-covering,保住易被抹平的低概率覆盖。
- **(ungated)**:`g ≡ 1`(= 输入分离的双-KL,当消融基线,隔离门控贡献)。
- `κ`(门控锐度)、`λ`(锚强度)为超参。

**需要的 infra 增量(实现时明确):**
- **加载第二个冻结模型 base π_B**(现 OPD 只加载一个 teacher `teacher_model_path`)→ 加 `base_model_path` + `base_unnorm_key`。
- **预生成 base 锚集**:一次 base rollout on `libero_90`(≤50 env,safe_run)→ 存 `(state, p_B soft logits)`(体积可控,存本地 outputs/,非 checkpoint)。
- **actor 加锚损失分支**:读锚集、算 `g(H)·D_KL(p_B‖p_θ)`、加到现有 `L_task` 上(env 开关,默认 0 = 现有 run 字节不变)。

---

## 5. 评测协议(命门:防泄漏 + 诚实指标)

**主成功率 = `success_once`**(领域标准 —— LIBERO/VLA-OPD/RETAIN 都报 once,为**与文献可比**必须用它;用户明确纠正 2026-07-22)。`success_at_end`(结束仍保持)eval 时一并记录,仅作 solve-then-drift 的**补充诊断**,不作 headline。**全部 ≤50 env、GPU-pin、EGL、safe_run。**

**泛化保留率(主指标):** `GenRet(axis) = SR_once_πθ(axis) / SR_once_BASE(axis)`。1.0=完全保住,<1=遗忘,>1=反涨。
BASE once 分母(已测):object 0.64 / spatial 0.66 / goal 0.40 / long 0.72(avg 0.605)。

**两条泛化轴 + 防泄漏:**
| 轴 | 训练时锚看到? | 评测数据 | 泄漏? |
|---|---|---|---|
| **PRO 扰动(主 = 真零样本泛化)** | 锚只在**标准**(未扰动)prompt 上 | object 的 PRO 扰动(object-appearance / swap / lan) | **无**(锚从不见扰动)✓ |
| **held-out 套件(次 = 广度/遗忘)** | 锚 = `libero_90`,与 4 套件**不相交** | spatial / goal / long(未 distill) | **无**(90 ∩ 4套件 = ∅)✓ |

> 注:base=130 通才本就会 4 套件 → held-out 套件轴严格说测的是"distill object 会不会拖垮 spatial/goal/long"(广度/遗忘),**PRO 扰动才是真正的零样本泛化**(base 未针对扰动训练;其画像 object-appearance≈0.5、swap/lan≈1.0,有明确可保空间)。两轴都报。

**新任务轴**:object 的 at_end SR(学得如何)。

**Headline 图**:x=新任务 SR,y=泛化保留率(PRO 为主),散点标 {BASE, plain-OPD, RETAIN, reverse-LiNeS, WiSE-FT, A, B 各 arm},证明 A/B 落在右上、Pareto 支配。

---

## 6. 基线与消融

**必须打败的基线(大多是事后/合并,**不需训练**,算力便宜):**
| 基线 | 怎么得到 | 训练? |
|---|---|---|
| BASE(130 通才) | 现成 | 否(保留率=1.0 参照 + 新任务下限) |
| plain OPD | 只 `L_task`(现有配方) | 是(1 run) |
| RETAIN 合并 | plain-OPD 学生 × BASE 权重插值,α∈{.25,.5,.75} | 否(合并) |
| reverse-LiNeS | 把 plain-OPD 的 task-vector 按深度反向缩放拉回 base | 否(事后) |
| WiSE-FT | 均匀权重插值 | 否(合并) |

**消融(定义 A 与 B):**
- **A 门控**:(i) 低熵 vs (ii) 高熵 vs (ungated) —— 隔离门控方向与贡献。
- **B support**(同一双-KL,只换锚流 state 分布,目标恒为 `p_B`):
  1. **base 自 rollout**(on-policy,on-manifold)← 押的赢家
  2. student 自 rollout(on-policy 但漂移)
  3. off-policy 固定 probe(别的任务 demo 状态)← 旧失败模式
  4. 随机/扰动合成(宽但 off-manifold)← breadth-but-noise
- **广度对照**(翻案关键):窄 base-rollout(只 object)vs 宽 base-rollout(libero_90)—— 若两者≈且都≫off-policy → 证明 **support 类型是杠杆、广度不是**。
- `λ ∈ {1,2,4}`、`κ` 小扫。

---

## 7. 成功判据(单 seed 先行)

- **A 成立**:熵门控双-KL 在**相同新任务 SR**(object at_end ≥ ~0.9)下,PRO 保留率**严格高于全部基线**(尤其 RETAIN);或相同保留率下新任务 SR 更高。即 Pareto 支配。
- **B 成立**:base-rollout support 的 PRO 保留率**显著高于** off-policy-probe 与随机合成;且窄 vs 宽 base-rollout ≈(广度不是主因)。
- **翻案成立**:上一条同时使旧"breadth-invariant ~0.18"结论被 support-type 维度解释。
- 任一不成立 → 如实报告 H-null,不硬凑。

---

## 8. 算力计划与红线安全(硬约束——服务器不抗造)

**逐条红线(全程):**
- **磁盘**:每次存盘前 `df -h /share/fanruochen-local`;convert 后立删 DCP raw;只留 best;锚集不是 checkpoint(小)。**逼近 1T / 掉到 <60G 立即停。**
- **GPU**:只用空闲的**连续卡对**(FSDP 需连续);启动前 `nvidia-smi -i` profile;**绝不碰别租户进程**;VRAM<85% 不硬开。
- **watchdog**:所有 heavy job 包 `safe_run.sh`(nice/ionice + 盘/RAM/iowait/进程爆炸自动杀);save_interval 只在末尾(避免每步 32G 触发 IO_BLOCK)。
- **评测**:≤50 env 顺序、GPU-pin、EGL;不并发爆 RAM。
- **规模**:**单 seed 先行**;每 run 2-GPU、LoRA rank32、~10–20 步(几小时);**分阶段设门、不并发爆盘**;heavy run 我给命令 / 你确认,不擅自跑([[feedback_manual_run_high_risk]])。

**训练 run 预算(分阶段,只在过门后进下一阶段):**
- **Stage 0(只读 + 1 次 base rollout)**:核对资产;预生成 base 锚集(1 次 base rollout on libero_90,≤50env);post-hoc eval **BASE** 在 PRO+held-out(建保留率分母)。**0 训练 run。**
- **Stage 1(核心,~2 训练 run)**:plain-OPD(1)+ A-gated 默认(i)(1);RETAIN/LiNeS/WiSE-FT 全是事后合并(0 训练)。全 eval → **门:A-gated Pareto 打过全部基线?** 不过则调 λ/κ 或停。
- **Stage 2(B support 消融,~3 训练 run)**:A-gated 换 support = {student-rollout, off-policy-probe, random} 各 1。**门:base-rollout 是否最优 + 广度对照。**
- **Stage 3(精修,~2–3 run)**:门控 (ii)、λ/κ 扫、ungated。取 Pareto 占优者。
- 达标 → 才考虑 **多 seed 复核** 与 **扩到多-teacher 顺序**(下一篇/下一阶段设计)。

---

## 9. 风险与开放问题

- **门控方向 (i) vs (ii) 未知** → 设计成消融,数据决定(不预设)。
- **infra 增量真实**:第二冻结模型 + 预生成锚集 + actor 锚损失分支;需自己的调试回合(默认 env=0 字节不变,保护现有 run)。
- **held-out 套件轴非真零样本**(base 已会)→ 定位为广度/遗忘;PRO 才是零样本泛化 headline。
- **可能打不过 RETAIN**:RETAIN 已很强且 train-free;若我们只 marginal 胜,需要靠 data-free + on-policy 的**叙事**与**多-teacher 可扩展性**(RETAIN 是两两插值,teacher 多了要合并)差异化。
- **plasticity 独立失败**(Dohare, Nature 2024):加 base 锚可能"学不动"不只是锚太强 → λ 扫 + 门控让锚只在 base 确定处发力,给任务学习让路。

---

## 10. 关键文件(实现时)

- 复用:`rlinf/algorithms/advantages.py`(OPD reverse-KL,不改)、`rlinf/workers/actor/fsdp_actor_worker.py`(加 base 模型 + 锚损失)、`rlinf/workers/rollout/hf/huggingface_worker.py`(base 锚集生成)、`convert_oft_lora_ckpt.py`、`eval_step20_4suite.sh`(改 PRO + held-out)。
- 新增:锚集生成脚本、双-KL 训练 config(env 开关默认关)、RETAIN/reverse-LiNeS/WiSE-FT 合并脚本、Pareto 汇总脚本。
