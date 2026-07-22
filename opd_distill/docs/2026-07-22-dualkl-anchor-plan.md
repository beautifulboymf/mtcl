# 双-KL BASE 锚 · 实现计划

> **For agentic workers:** implement task-by-task; checkbox `- [ ]` tracks progress. Design = `opd_distill/docs/2026-07-22-dualkl-anchor-design.md`.

**Goal:** 在 OpenVLA-OFT OPD 里加一个数据无关、熵门控的 BASE forward-KL 锚,学新任务同时保住 base 泛化,Pareto 打过 RETAIN/LiNeS/WiSE-FT/plain-OPD。

**Architecture:** 复用现有 OPD distill 机制(`fsdp_actor_worker.py` 已有 forward/reverse/jsd KL + `distill_conf_tau` 熵门控)。新增:第二个冻结 base 模型 + 一个 base-anchor 项 `loss = L_task + λ·g(H_B)·D_KL(p_B‖p_θ)`。V1=锚在学生任务 rollout 态(小改);V2=锚在 base 宽 rollout(libero_90)态。

**Tech stack:** RLinf FSDP actor, OpenVLA-OFT 离散 token, LIBERO(+PRO), EGL GPU render, safe_run 硬 watchdog。

**红线(每个 run 前后强制):** `df -h /share/fanruochen-local`;safe_run 包裹;≤50 env 顺序;单卡对连续空闲(启动前 `nvidia-smi -i` profile,VRAM<85% 不开);绝不碰别租户;convert 后立删 DCP raw;单 seed 先行;heavy run 不并发爆盘。

---

## Stage 0 — BASE 基线(retention 分母)· 无 infra 改动 · 低风险

**产出:** BASE(=130-Base-Lora)在 4 套件标准 + object 的 PRO 扰动 上的 `success_at_end` SR,作为 GenRet 分母。

- [ ] **0.1** 建 PRO eval 配置(object,perturbation 分 object/swap/lan)。基于现有 `libero_object_g2_eval.yaml`,eval 时 `LIBERO_TYPE=pro LIBERO_PERTURBATION=<p>`。
- [ ] **0.2** 建 held-out 套件 eval:spatial/goal/long g2 配置(已有)。
- [ ] **0.3** 顺序 eval BASE(`RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`,unnorm=libero_130_no_noops_trajall)在:object 标准 + object PRO(object/swap/lan)+ spatial/goal/long 标准。50 env,safe_run,GPU-pin 空闲卡。
  - Run: `bash examples/embodiment/eval_embodiment.sh <cfg> LIBERO rollout.model.model_path=$BASE actor.model.model_path=$BASE actor.model.is_lora=False actor.model.unnorm_key=libero_130_no_noops_trajall env.eval.total_num_envs=50`(PRO 加 `LIBERO_TYPE=pro LIBERO_PERTURBATION=<p>` 前缀)。
  - 记 `eval/success_at_end`(+once)入 `outputs/dualkl/baseline_BASE.md`。
- [ ] **0.4** 同法 eval **plain-OPD 学生的 object-teacher**(`RLinf-OpenVLAOFT-GRPO-LIBERO-object`)在 object 标准 + PRO — 给"新任务天花板"与"专家泛化画像"参照。
- [ ] **验证门:** BASE 在 object 标准 ~0.7、PRO object-appearance ~0.5、swap/lan ~1.0、held-out 套件各有非零 SR(与 memory 画像一致)。数字落表。

---

## Stage 1a — infra:第二冻结 base + anchor 项(V1,锚在任务态)· 小改 ~40 行

**Files:** Modify `rlinf/workers/actor/fsdp_actor_worker.py`(base 加载 ~1033-1067 附近;anchor 项 ~1601 后、loss 组装 ~1632)。Config 新增 `actor.base_model_path`/`actor.base_unnorm_key` + `algorithm.anchor_lambda`/`anchor_kl`/`anchor_gate`/`anchor_gate_tau`(全默认关 → 现有 run 字节不变)。

- [ ] **1a.1** 加 `_load_base_model`(拷 `_load_teacher_model`,用 `base_model_path`/`base_unnorm_key`,`self.base_model` 冻结 eval)。在 init 处 `if self.cfg.actor.get("base_model_path"): self._load_base_model()`。
- [ ] **1a.2** 在 OPD distill 块(~1601 后,`opd_distill_loss` 已算好)加 anchor 项:
  ```python
  anchor_loss = None
  _alam = float(self.cfg.algorithm.get("anchor_lambda", 0.0))
  if _alam > 0.0 and getattr(self, "base_model", None) is not None and "action_logits" in output_dict:
      with torch.no_grad(), self.amp_context:
          base_out = self.base_model(forward_inputs=forward_inputs, compute_logprobs=False, use_cache=False, **kwargs)
      lb = torch.log_softmax(base_out["action_logits"].float().detach(), dim=-1)   # base soft targets
      ls2 = torch.log_softmax(output_dict["action_logits"].float(), dim=-1)         # student (grad)
      pb = lb.exp()
      # PRESERVE = mode-COVERING forward-KL D_KL(p_B || p_theta): student must cover base support
      akl_tok = (pb * (lb - ls2)).sum(dim=-1)                                       # [B, chunks*sad]
      # entropy gate g(H_base): (i) low-ent = base confident -> preserve harder; (ii) high-ent
      _gate = self.cfg.algorithm.get("anchor_gate", "none")
      _gtau = float(self.cfg.algorithm.get("anchor_gate_tau", 1.0))
      if _gate in ("low_ent", "high_ent"):
          with torch.no_grad():
              b_ent = -(pb * lb).sum(dim=-1)
              emax = torch.log(torch.tensor(float(pb.shape[-1]), device=pb.device))
              conf = (1.0 - b_ent / emax).clamp(0, 1)          # 1 = confident/low-ent
              gate = conf.pow(_gtau) if _gate == "low_ent" else (1.0 - conf).pow(_gtau)
          akl_tok = akl_tok * gate
      if loss_mask is not None:
          anchor_loss = (akl_tok * mtok).sum() / mtok.sum().clamp_min(1.0)
      else:
          anchor_loss = akl_tok.mean()
  ```
- [ ] **1a.3** loss 组装(~1632)改:
  ```python
  loss = opd_distill_loss
  if anchor_loss is not None:
      loss = loss + _alam * anchor_loss
      metrics_data["actor/anchor_loss"] = anchor_loss.detach().item()
  ```
- [ ] **1a.4 SMOKE(红线安全,≤2min):** 拷 `libero_object_opd_faithful130_2gpu.yaml` → `libero_object_dualkl_2gpu.yaml`,加 `actor.base_model_path=<130-Base-Lora>`、`actor.base_unnorm_key=libero_130_no_noops_trajall`、`algorithm.anchor_lambda=1.0 anchor_kl=forward anchor_gate=low_ent anchor_gate_tau=1.0`,`max_steps=1`,单卡对空闲。safe_run 跑 1 步。
  - **验证门:** 日志出现 `actor/anchor_loss=<非0>` 且无 crash(base 加载成功、anchor 项进 loss)。`anchor_lambda=0` 时行为 == plain-OPD(字节等价性 sanity)。
- [ ] **1a.5** commit(作者 beautifulboymf,本地不 push):`feat: data-free dual-KL BASE anchor (V1, task-state support)`。

---

## Stage 1b — 核心实验:plain-OPD vs V1-dualKL(锚在任务态)· 2 训练 run

**Files:** `libero_object_dualkl_2gpu.yaml`(上)+ merge 基线脚本。

- [ ] **1b.1** 训练 **plain-OPD**(=现有 faithful130,`anchor_lambda=0`,object teacher,10 步)→ 已有配方,1 run,safe_run。convert best → HF。删 DCP raw。
- [ ] **1b.2** 训练 **V1-dualKL**(`anchor_lambda=1` low_ent forward,object teacher,10 步)→ 1 run,safe_run。convert best。删 DCP raw。
- [ ] **1b.3** merge 基线(**无训练**):
  - RETAIN/WiSE-FT:`merge_wise.py`——`θ = (1-α)·θ_base + α·θ_plainOPD`,α∈{.25,.5,.75}(对 merged HF safetensors 逐张量插值)。
  - reverse-LiNeS:`merge_lines.py`——task-vector=θ_plainOPD−θ_base,按层深度**反向**缩放(浅层保留、深层拉回)后加回 base。
- [ ] **1b.4** 全体 post-hoc eval(50env,safe_run,GPU-pin):BASE / plain-OPD / V1-dualKL / RETAIN(3α)/ reverse-LiNeS,在 object 标准 + object PRO(3扰动)+ spatial/goal/long。记 `success_at_end`。
- [ ] **1b.5** 算 GenRet = SR / SR_BASE;画 Pareto(x=object SR,y=PRO 保留率)。表+图入 `outputs/dualkl/stage1_results.md`。
- [ ] **验证门 H-A:** V1-dualKL 在相同 object SR 下 PRO 保留率 > plain-OPD,且(目标)≥ RETAIN/LiNeS。**过 → 进 Stage 2;不过 → 扫 anchor_lambda∈{2,4}/gate∈{high_ent,none}/anchor_gate_tau,或如实报 H-null。**

---

## Stage 2 — V2:base 宽 rollout 锚支持 + B 消融 · 只在 Stage1b 过门后做

**目标:** 证明锚 support = base 自 rollout(宽 libero_90)优于 off-policy/随机,且 support 类型 > 广度。

- [ ] **2.1** 预生成 base 锚集(**一次性**,红线安全):base(130-Base-Lora)在 `libero_90` rollout,dump `(obs, base action_logits)` → `outputs/dualkl/anchor_libero90.pt`。≤50env,safe_run。基于 rollout worker 加 dump 钩子或复用 eval rollout。
- [ ] **2.2** actor 支持"锚集流":`anchor_source=preload` 时,每步从锚集采一 minibatch,在其 obs 上算 anchor_loss(base soft target 已存,student 前向取 logits)。混入现有 batch loss。
- [ ] **2.3** 训练 V2-dualKL(base-宽-rollout 锚)+ 消融 support 各 arm:{base-宽-rollout, student-rollout(=V1), off-policy-probe, random-synth} + 广度对照{窄 object base-rollout vs 宽 libero_90}。每 arm 1 run,safe_run,分阶段不并发。
- [ ] **2.4** post-hoc eval 全 arm(同 1b.4 协议)。
- [ ] **验证门 H-B:** base-rollout support 的 PRO 保留率 > off-policy/random;窄≈宽 ≫ off-policy(support 类型是杠杆、广度不是,翻案旧 breadth-invariant)。

---

## Stage 3 — 精修(过 H-A/H-B 后)

- [ ] **3.1** 门控方向 (i) low_ent vs (ii) high_ent 对比;anchor_kl forward vs jsd;λ/τ 扫。取 Pareto 占优。
- [ ] **3.2** 多 seed(≥2)复核 headline。
- [ ] **3.3** 汇总 `outputs/dualkl/FINAL.md` + Pareto 图 → 交用户。

---

## 进度记录(execution 时更新)

- 状态:**Stage 0 起步**(2026-07-22)。
- 空闲卡快照:启动每步前 `nvidia-smi` 现查。
- checkpoint 清理:convert 后即删 DCP;废弃 arm 整目录删(删前 df + 记录)。
